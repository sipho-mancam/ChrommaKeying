#include "yoloAPI.hpp"

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

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

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2) {
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
  *cpu_output_buffer2 = new float[kBatchSize * kOutputSize2];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize) {
  context.enqueue(batchSize, buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output1, buffers[1], batchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(output2, buffers[2], batchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
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


YoloAPI::YoloAPI(std::string engineN, std::string wtsN, std::string labels, std::string imgD)
{
    this->engine_name = engineN;
    this->wts_name = wtsN;
    this->img_dir = imgD;  
    this->labels_filename = labels;
    this->init(); 
}


void YoloAPI::init()
{
    this->deserialize();
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    prepare_buffers(this->engine, &this->gpu_buffers[0], &this->gpu_buffers[1], &this->gpu_buffers[2], &this->cpu_output_buffer1, &this->cpu_output_buffer2);
    this->read_files();

}

void YoloAPI::deserialize()
{
    deserialize_engine(this->engine_name, &this->runtime, &this->engine, &this->context);
    if(this->engine == nullptr)
    {
        throw("Engine Error exception");
    }
}

void YoloAPI::read_files()
{
    if (read_files_in_dir(this->img_dir.c_str(), this->file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        throw("read_files_in_dir failed.");
    }

    // Read the txt file for classnames
    std::ifstream labels_file(this->labels_filename, std::ios::binary);
    if (!labels_file.good()) {
        std::cerr << "read " << this->labels_filename << " error!" << std::endl;
        throw("read_files_in_dir failed.");
    }
    read_labels(this->labels_filename, this->labels_map);
    assert(kNumClass == this->labels_map.size());

}

void YoloAPI::preprocessor(std::vector<cv::Mat>& img_batch)
{
    // Preprocess
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);
}

void YoloAPI::rInfer()
{
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer1, cpu_output_buffer2, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}


void YoloAPI::postProcess(std::vector<cv::Mat> &img_batch, std::vector<std::string> &img_name_batch)
{
     // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer1, img_batch.size(), kOutputSize1, kConfThresh, kNmsThresh);

    // Draw result and save image
    for (size_t b = 0; b < img_batch.size(); b++) {
        auto& res = res_batch[b];
        cv::Mat img = img_batch[b];

        auto masks = process_mask(&cpu_output_buffer2[b * kOutputSize2], kOutputSize2, res);
        draw_mask_bbox(img, res, masks);
        cv::imwrite("_" + img_name_batch[b], img);
    }
}


void YoloAPI::run(std::vector<cv::Mat>&img_batch, std::vector<std::string>& img_name_batch)
{
    this->preprocessor(img_batch);
    // Run inference
    this->rInfer();
   
    this->postProcess(img_batch, img_name_batch);
 
}


YoloAPI::~YoloAPI()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    CUDA_CHECK(cudaFree(gpu_buffers[2]));
    delete[] cpu_output_buffer1;
    delete[] cpu_output_buffer2;
    cuda_preprocess_destroy();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}