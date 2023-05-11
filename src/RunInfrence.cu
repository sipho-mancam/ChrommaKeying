/*
 ============================================================================
 Name        : RunInfrence.cu
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <string>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 16;

const char* INPUT_BLOB_NAME_RESNET = "data";
const char* OUTPUT_BLOB_NAME_RESNET = "prob";
inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }
using namespace nvinfer1;

static Logger gLogger;


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


template<typename T, bool isBGR>
__global__ void gpuTensorNormMean( T* input, int iWidth, float* output, int oWidth, int oHeight, float2 scale, float multiplier, float min_value, const float3 mean, const float3 stdDev)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);
//	printf("%d %f\n",step);

	const T px = input[ dy *iWidth + dx ];

	const float3 rgb = isBGR ? make_float3(px.z, px.y, px.x)
						: make_float3(px.x, px.y, px.z);

	output[n * 0 + m] = ((rgb.x * multiplier + min_value) - mean.x) / stdDev.x;
	output[n * 1 + m] = ((rgb.y * multiplier + min_value) - mean.y) / stdDev.y;
	output[n * 2 + m] = ((rgb.z * multiplier + min_value) - mean.z) / stdDev.z;
//	output[n * 0 + m] = 1.0;
//	output[n * 1 + m] = 1.0;
//	output[n * 2 + m] = 1.0;
}

template<bool isBGR>
cudaError_t launchTensorNormMean( void* input, size_t inputWidth, size_t inputHeight,
						    float* output, size_t outputWidth, size_t outputHeight,
						    const float2& range, const float3& mean, const float3& stdDev )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	std::cout <<"mul"<< multiplier <<"range:"<< range.x <<std::endl;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	//if( format == IMAGE_RGB8 )
	gpuTensorNormMean<uchar3, isBGR><<<gridDim, blockDim>>>((uchar3*)input, inputWidth, output, outputWidth, outputHeight, scale, multiplier, range.x, mean, stdDev);
//	else if( format == IMAGE_RGBA8 )
//		gpuTensorNormMean<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>((uchar4*)input, inputWidth, output, outputWidth, outputHeight, scale, multiplier, range.x, mean, stdDev);
//	else if( format == IMAGE_RGB32F )
//		gpuTensorNormMean<float3, isBGR><<<gridDim, blockDim, 0, stream>>>((float3*)input, inputWidth, output, outputWidth, outputHeight, scale, multiplier, range.x, mean, stdDev);
//	else if( format == IMAGE_RGBA32F )
//		gpuTensorNormMean<float4, isBGR><<<gridDim, blockDim, 0, stream>>>((float4*)input, inputWidth, output, outputWidth, outputHeight, scale, multiplier, range.x, mean, stdDev);
//	else
//		return cudaErrorInvalidValue;
	cudaError_t cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "gpuTensorNormMean failed!");
			return cudaStatus;
		}
	return cudaGetLastError();
}



// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights1(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d1(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d1(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d1(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{stride, stride});
        IScaleLayer* bn3 = addBatchNorm2d1(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME_RESNET, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights1("/home/jurie/cuda-workspace/RunInfrence/resnet18_best.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d1(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");

    IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
    IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");

    IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
    IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");

    IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
    IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

    IPoolingLayer* pool2 = network->addPoolingNd(*relu9->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStrideNd(DimsHW{1, 1});

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), OUTPUT_SIZE, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME_RESNET);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}





void doInferenceCuda(IExecutionContext& context, void* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME_RESNET);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME_RESNET);
    assert(outputIndex==1);
    assert(inputIndex==0);
    buffers[0]=input;
    // Create GPU buffers on device
   // CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
  //  CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto start = std::chrono::system_clock::now();

    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    //CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}





void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME_RESNET);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME_RESNET);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));


    auto start = std::chrono::system_clock::now();


    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


std::vector <std::string> myimages;

using namespace std;
void list_dir(const char *path) {
   struct dirent *entry;
   DIR *dir = opendir(path);

   if (dir == NULL) {
      return;
   }
   while ((entry = readdir(dir)) != NULL) {
//   cout << entry->d_name << endl;
   string name =entry->d_name;
   if(name.find(".bmp")!=-1)
	   myimages.push_back(name);


   }
   closedir(dir);
}
float3 *resnet_data=0;
uchar3 *resnet_data_uchar=0;
IRuntime* runtime=0;
ICudaEngine* engine;
IExecutionContext* context;



void InitResnet18()
{
	char *trtModelStream{nullptr};
	size_t size{0};
	std::ifstream file("/home/jurie/resnet18_test.engine", std::ios::binary);

	if (file.good()) {
		std::cout << "open" << std::endl;
		file.seekg(0, file.end);
		size = file.tellg();
		std::cout << size << std::endl;
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
		}
	 else {
		return ;
	}



	runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
	assert(engine != nullptr);
	context = engine->createExecutionContext();
	assert(context != nullptr);
	cudaMalloc(&resnet_data,3*INPUT_H*INPUT_W* sizeof(float));
	delete[] trtModelStream;
}


void DestroyResnet18()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();

}


int Classify(cv::Mat img_size)
{
			//cv::Mat img_size = cv::imread("/home/jurie/Pictures/resnet_test_2/"+n);//classIndex:8classMax:2.84357
		cv::Mat out_float;

			cv::cuda::GpuMat imgmatgpu;
			cv::cuda::GpuMat imgmatgpu_size;
			imgmatgpu.create( img_size.cols,img_size.rows,CV_8UC3);
			//imgmatgpu.step=244;
			cudaMemcpy(imgmatgpu.data,img_size.data,img_size.cols*img_size.rows*3,cudaMemcpyHostToDevice);
			launchTensorNormMean<true>(( void* )imgmatgpu.data,  size_t(imgmatgpu.rows),size_t(imgmatgpu.cols),
					( float* )resnet_data, 224,  224,
					make_float2(0.0f, 1.0f),make_float3(0.485f, 0.456f, 0.406f), make_float3(0.229f, 0.224f, 0.225f));

			cv::cuda::GpuMat  destination3(224,224,CV_32FC1,resnet_data);
		 //   std::cout << destination3.step <<" "<<destination0.step<<std::endl;
			destination3.download(out_float);
			cv::imshow("outfloat3",out_float);
			// Run inference
			static float prob[OUTPUT_SIZE];
			for (int i = 0; i < 1; i++) {
			//	doInference(*context, data, prob, 1);
				doInferenceCuda(*context, resnet_data, prob, 1);
			}
			// Print histogram of the output distribution
			 int classIndex = -1;
			float classMax = -1.0f;
			std::cout << "\nOutput:\n\n";
			for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
			{
			//	std::cout << prob[i] << ", ";
				const float value = prob[i];
					if( value > classMax )
					{
						classIndex = i;
						classMax   = value;
					}
			}
			std::cout  << "classIndex:" << classIndex << 	"classMax:"   <<classMax <<std::endl;
			return classIndex;

}

int main1(int argc, char** argv)
{


	list_dir("/home/jurie/Pictures/resnet_test");

	InitResnet18();

    // create a model using the API directly and serialize it to a stream
//    char *trtModelStream{nullptr};
//    size_t size{0};
//
//    if (std::string(argv[1]) == "-s") {
//        IHostMemory* modelStream{nullptr};
//        APIToModel(1, &modelStream);
//        assert(modelStream != nullptr);
//
//        std::ofstream p("/home/jurie/Documents/version6/tensorrtx/resnet/build2/resnet18_test.engine", std::ios::binary);
//        if (!p)
//        {
//            std::cerr << "could not open plan output file" << std::endl;
//            return -1;
//        }
//        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
//        modelStream->destroy();
//        return 1;
//    } else if (std::string(argv[1]) == "-d") {
//        std::ifstream file("/home/jurie/Documents/version6/tensorrtx/resnet/build2/resnet18_test.engine", std::ios::binary);
//
//        if (file.good()) {
//        	std::cout << "open" << std::endl;
//            file.seekg(0, file.end);
//            size = file.tellg();
//            std::cout << size << std::endl;
//            file.seekg(0, file.beg);
//            trtModelStream = new char[size];
//            assert(trtModelStream);
//            file.read(trtModelStream, size);
//            file.close();
//        }
//    } else {
//        return -1;
//    }






//



	for(string n:myimages)
	{


		std::cout  << std::endl;
		Classify(cv::imread("/home/jurie/Pictures/resnet_test/"+n));
		cv::waitKey(-1);


	}


    return 0;
}

