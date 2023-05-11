#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include <thread>         // std::thread
#include <mutex>
#include <queue>
#include <condition_variable>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;
bool bInitYolo=false;
static Logger gLogger;
const static int kOutputSize1 = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSize2 = 32 * (kInputH / 4) * (kInputW / 4);

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir, std::string& labels_filename) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.33;
            gw = 0.25;
        } else if (net[0] == 's') {
            gd = 0.33;
            gw = 0.50;
        } else if (net[0] == 'm') {
            gd = 0.67;
            gw = 0.75;
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
        } else if (net[0] == 'x') {
            gd = 1.33;
            gw = 1.25;
        } else if (net[0] == 'c' && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 5) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        labels_filename = std::string(argv[4]);
    } else {
        return false;
    }
    return true;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1/*, float** cpu_output_buffer2*/) {
  assert(engine->getNbBindings() == 3);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex1 = engine->getBindingIndex(kOutputTensorName);
  const int outputIndex2 = engine->getBindingIndex("proto");
  assert(inputIndex == 0);
  assert(outputIndex1 == 1);
  assert(outputIndex2 == 2);

  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer1, kBatchSize * kOutputSize1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer2, kBatchSize * kOutputSize2 * sizeof(float)));

  // Alloc CPU buffers
  *cpu_output_buffer1 = new float[kBatchSize * kOutputSize1];
  //*cpu_output_buffer2 = new float[kBatchSize * kOutputSize2];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1/*, float* output2*/, int batchSize) {
  context.enqueue(batchSize, buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output1, buffers[1], batchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
 // CUDA_CHECK(cudaMemcpyAsync(output2, buffers[2], batchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;

  engine = build_seg_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);

  assert(engine != nullptr);

  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // Close everything down
  engine->destroy();
  builder->destroy();
  config->destroy();
  serialized_engine->destroy();
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

//void process_mask_to_final_thread(cv::cuda::GpuMat *seg_mat,int batchindex, const float* proto,
//		int proto_size, std::vector<Detection>* dets)
//{
//
//
//
//
//}

IExecutionContext* m_context = nullptr;
cudaStream_t m_stream;
IRuntime* m_runtime = nullptr;
ICudaEngine* m_engine = nullptr;
float* m_gpu_buffers[3];
float* m_cpu_output_buffer1 = nullptr;
float* m_gpu_output_buffer2 = nullptr;
cv::cuda::GpuMat m_mask_mat_gpu;
cv::cuda::GpuMat m_mask_mat_gpu_scaled;


int InitYolov5()
{
	std::string wts_name = "";
	std::string engine_name = "/home/jurie/Documents/Computer Vision/Workspace/CudaChromaUbuntu/src/res/yolov5s-seg-27.engine";
	std::string labels_filename = "/home/jurie/Documents/Computer Vision/Workspace/CudaChromaUbuntu/src/res/labels.txt";
	float gd = 0.0f, gw = 0.0f;
	std::string img_dir="/home/jurie/Pictures/seg_test";


	//
	//	  // Create a model using the API directly and serialize it to a file
	//	  if (!wts_name.empty()) {
	//	    serialize_engine(kBatchSize, gd, gw, wts_name, engine_name);
	//	    return 0;
	//	  }

	  // Deserialize the engine from file
	 // IRuntime* runtime = nullptr;
	//  ICudaEngine* engine = nullptr;
	 // IExecutionContext* context = nullptr;
	  deserialize_engine(engine_name, &m_runtime, &m_engine, &m_context);
	  //cudaStream_t stream;
	  CUDA_CHECK(cudaStreamCreate(&m_stream));





	  prepare_buffers(m_engine, &m_gpu_buffers[0], &m_gpu_buffers[1], &m_gpu_buffers[2], &m_cpu_output_buffer1);


	  m_mask_mat_gpu.create(160,960,CV_32FC1);//full mask for 3840 * 542
	  m_mask_mat_gpu.step=960*sizeof(float);

	  m_mask_mat_gpu_scaled.create(160*4,960*4,CV_32FC1);//full mask for 3840 * 542
	  m_mask_mat_gpu_scaled.step=(960*4)*sizeof(float);

	bInitYolo=true;
	std::cout << "Yoloinit Done" <<std::endl;

	return 0;


}


int main_test(int argc, char** argv) {
  cudaSetDevice(kGpuId);

	std::string wts_name = "";
	std::string engine_name = "res/yolov5s-seg-27.engine";
	std::string labels_filename = "res/labels.txt";
	float gd = 0.0f, gw = 0.0f;
	std::string img_dir="/home/jurie/Pictures/seg_test";


  std::vector<std::string> file_names;
   if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
     std::cerr << "read_files_in_dir failed." << std::endl;
     return -1;
   }

//  std::string img_dir;
//  if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir, labels_filename)) {
//    std::cerr << "arguments not right!" << std::endl;
//    std::cerr << "./yolov5_seg -s [.wts] [.engine] [n/s/m/l/x or c gd gw]  // serialize model to plan file" << std::endl;
//    std::cerr << "./yolov5_seg -d [.engine] ../images coco.txt  // deserialize plan file, read the labels file and run inference" << std::endl;
//    return -1;
//  }

  // Create a model using the API directly and serialize it to a file
  if (!wts_name.empty()) {
    serialize_engine(kBatchSize, gd, gw, wts_name, engine_name);
    return 0;
  }

  // Deserialize the engine from file
  IRuntime* runtime = nullptr;
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  deserialize_engine(engine_name, &runtime, &engine, &context);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Init CUDA preprocessing
  cuda_preprocess_init(kMaxInputImageSize);

  // Prepare cpu and gpu buffers
  float* gpu_buffers[3];
  float* cpu_output_buffer1 = nullptr;
  float* gpu_output_buffer2 = nullptr;
  prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &gpu_buffers[2], &cpu_output_buffer1);

  // Read images from directory
//  std::vectoinference time: 26ms
//  total time including mask: 64ms
//  inference time: 26ms
//  total time including mask: 115ms
//  inference time: 27ms
//  total time including mask: 43msr<std::string> file_names;
//  if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
//    std::cerr << "read_files_in_dir failed." << std::endl;
//    return -1;
//  }

  // Read the txt file for classnames
  std::ifstream labels_file(labels_filename, std::ios::binary);
  if (!labels_file.good()) {
    std::cerr << "read " << labels_filename << " error!" << std::endl;
    return -1;
  }
  std::unordered_map<int, std::string> labels_map;
  read_labels(labels_filename, labels_map);
  assert(kNumClass == labels_map.size());


  //cv::cuda::GpuMat mask_mat_gpu(135,960,CV_32FC1);//full mask for


  cv::cuda::GpuMat mask_mat_gpu(160,960,CV_32FC1);//full mask for 3840 * 542
  mask_mat_gpu.step=960*sizeof(float);

  // batch predict

  for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
    // Get a batch of images
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;
    for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
      cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
      img_batch.push_back(img);
      img_name_batch.push_back(file_names[j]);
    }


  //  while(1)
   // {
    auto start_pre = std::chrono::system_clock::now();
    // Preprocess
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer1/*, cpu_output_buffer2*/, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // NMS
    auto start_mask = std::chrono::system_clock::now();
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer1, img_batch.size(), kOutputSize1, kConfThresh, kNmsThresh);
    auto end_nms = std::chrono::system_clock::now();
	std::vector<std::thread*> threadlist;
    // Draw result and save image
    for (size_t b = 0; b < img_name_batch.size(); b++) {
    	std::vector<Detection> *res = &res_batch[b];

      //process_mask_to_final(&mask_mat_gpu,b,&cpu_output_buffer2[b * kOutputSize2], kOutputSize2, res);


    	gpu_output_buffer2=(float* )gpu_buffers[2];

      std::thread *first = new std::thread(process_mask_to_final,&mask_mat_gpu,b,&gpu_output_buffer2[b * kOutputSize2], kOutputSize2, res);
      threadlist.push_back(first);
     // break;

    }
	std::for_each(threadlist.begin(), threadlist.end(),[](std::thread* &th) {th->join();});
    auto end_pre = std::chrono::system_clock::now();
    std::cout << "total time including mask: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - start_pre).count() << "ms" << std::endl;
    std::cout << "total nms: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_nms - start_mask).count() << "ms" << std::endl;
    std::cout << "total mask: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - end_nms).count() << "ms" << std::endl;

    //}


    cv::Mat mask_mat_cpu;
    
    mask_mat_gpu.download(mask_mat_cpu);//full mask for

    cv::imshow("mask",mask_mat_cpu);
    cv::resize(mask_mat_cpu, mask_mat_cpu, cv::Size(960*4, 640));
    std::cout << mask_mat_cpu.cols <<" "<< mask_mat_cpu.rows<< std::endl;
  //  cv::waitKey(-1);

    cv::Mat tt=cv::imread("/home/jurie/Pictures/202092_1928_10544.bmp");

    std::cout << tt.cols <<" "<<tt.rows<<std::endl;

    	for (int x = 0; x < tt.cols; x++) {
    		for (int y = 0; y < tt.rows ; y++) {
    			float val = mask_mat_cpu.at<float>(y, x);

    			if (val <= 0.5)
    				continue;
    			tt.at<cv::Vec3b>(y, x)[0] = tt.at<cv::Vec3b>(y, x)[0]/2*val;
    			tt.at<cv::Vec3b>(y, x)[1] = tt.at<cv::Vec3b>(y, x)[0]/2*val;
    			tt.at<cv::Vec3b>(y, x)[2] = tt.at<cv::Vec3b>(y, x)[0]/2*val;
    		}
    	}

    cv::imshow("img",tt);
    cv::waitKey(-1);






//    break;
  }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  CUDA_CHECK(cudaFree(gpu_buffers[2]));
  delete[] cpu_output_buffer1;

  cuda_preprocess_destroy();
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}
//void doInference_from_cuda(IExecutionContext& context, cudaStream_t& stream,
//		void **buffers, void *remote_buffers, float* output, int batchSize) {
//
//	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//
//	buffers[0]= remote_buffers;
////	CUDA_CHECK(
////			cudaMemcpyAsync(buffers[0], remote_buffers,
////					batchSize * 3 * kInputH * kInputW * sizeof(float),
////					cudaMemcpyDeviceToDevice, 0));
//	context.enqueue(batchSize, buffers, stream, nullptr);
//	CUDA_CHECK(
//			cudaMemcpyAsync(output, buffers[1],
//					batchSize * kOutputSize1 * sizeof(float),
//					cudaMemcpyDeviceToHost, 0));
//	cudaStreamSynchronize(stream);
//}
float *GetSegmentedMask()
{
	return (float*)m_mask_mat_gpu_scaled.data;
}

std::vector<Detection> doInference_YoloV5(void *remote_buffers,
		float fnms) {

	int fcount = kBatchSize;
	std::vector < std::vector < Detection >> batch_res(fcount);
	std::vector<Detection> Balls;
	std::vector<Detection> Persons;
	std::vector<Detection> all_Together;
	if (!bInitYolo)
		return all_Together;
	m_mask_mat_gpu.setTo(cv::Scalar(0));
	m_gpu_buffers[0]=(float *)remote_buffers;
	infer(*m_context, m_stream, (void**)m_gpu_buffers, m_cpu_output_buffer1/*, cpu_output_buffer2*/, kBatchSize);

    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, m_cpu_output_buffer1, kBatchSize, kOutputSize1, kConfThresh, kNmsThresh);
    std::vector<std::thread*> threadlist;

    for(int b=0;b<kBatchSize;b++)
    {
		std::vector<Detection> *res = &res_batch[b];
		m_gpu_output_buffer2=(float* )m_gpu_buffers[2];

		std::thread *first = new std::thread(process_mask_to_final,&m_mask_mat_gpu,b,&m_gpu_output_buffer2[b * kOutputSize2], kOutputSize2, res);
		threadlist.push_back(first);
    }
    std::for_each(threadlist.begin(), threadlist.end(),[](std::thread* &th) {th->join();});

    cv::Mat m_mask_mat_cpu_scaled, m_mask_mat_cpu;

    m_mask_mat_gpu_scaled.download(m_mask_mat_cpu_scaled);
    m_mask_mat_gpu.download(m_mask_mat_cpu);

    cv::resize(m_mask_mat_cpu, m_mask_mat_cpu_scaled, cv::Size(960*4, 160*4));


	if(1)
	{
		// cv::Mat cuda_display
    // m_mask_mat_gpu_scaled.download(cuda_display);
		cv::imshow("mat", m_mask_mat_cpu_scaled);
	}

	return all_Together;
}
