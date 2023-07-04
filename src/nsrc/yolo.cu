/*
 * yolo.cpp
 *
 *  Created on: 12 Jun 2023
 *      Author: jurie
 */

#include "yolo.hpp"
#include <fstream>

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

using namespace nvinfer1;

static Logger gLogger;





YoloMask::YoloMask(IPipeline *obj): IMask(obj) // @suppress("Class members should be properly initialized")
{
	char *cwd = getenv("CWD");
	std::string rootDir(cwd);
	std::string engine_name = rootDir+"/res/yolo-seg-4.engine";

	api = new YoloAPI(obj, std::string("/home/jurie/Documents/Computer Vision/tensorrtx/yolov5/build/yolov5x-latest.engine"));

	this->outputBufferDetections = nullptr;
	this->outputBufferMask = nullptr;
	this->batchData = nullptr;
	this->context = nullptr;
	this->engine = nullptr;
	this->runtime = nullptr;
	this->stream;
	this->cudaStatus = cudaStreamCreate(&stream);
	this->checkCudaError("create cuda stream", "Yolo Mask Constructor");
	this->started = false;
	this->loaded = false;
	this->maskOutCpu = new float[kBatchSize * kOutputSize2];
	this->detectionsOutCpu = new float[kBatchSize * kOutputSize1];
	memset(this->detectionsOutCpu, 0, kBatchSize * kOutputSize1*sizeof(float));
	memset(this->maskOutCpu, 0 , kBatchSize * kOutputSize2*sizeof(float));
	this->initialize();
}

void YoloMask::initialize()
{
//	this->started = false;
//	cudaSetDevice(kGpuId);

//

//	std::ifstream engine_file(engine_name, std::ios::binary);
//
//	if(!engine_file)
//	{
//		std::cerr<<"Engine File doesn't exist: \n"<<"Path: "<<engine_name<<std::endl;
//		return;
//	}
//
//	deserialize_engine(engine_name, &runtime, &engine, &context);
//
//	assert(engine->getNbBindings() == 3);
//	// In order to bind the buffers, we need to know the names of the input and output tensors.
//	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
//	const int inputIndex = engine->getBindingIndex(kInputTensorName);
//	const int outputIndex1 = engine->getBindingIndex(kOutputTensorName);
//	const int outputIndex2 = engine->getBindingIndex("proto"); // mask
//	assert(inputIndex == 0);
//	assert(outputIndex1 == 1);
//	assert(outputIndex2 == 2);

//	initYolo();

	this->started = true;
}

void YoloMask::__cutToPanels()
{
	// input is fixed to width*2 and height/2 (Interlacing problem)
	int n = 3;
	int width = this->frame.cols/(2*n);
	int overlappingFactor = width/n+ width%n;
	int lastEnd = 0;

	for(int i=0; i<(n+1); i++)
	{
		cv::Rect roi(cv::Point(lastEnd, 0), cv::Size(width, this->frame.rows));
		cv::Mat tImg = this->frame(roi);
		cv::resize(tImg, tImg, cv::Size(640, 640));
		this->img_batch.push_back(tImg);
		lastEnd += width-overlappingFactor;
	}

}

void YoloMask::prepareImages()
{
	// convert to RGB,
	// cut to 4 panels
	// send it to yolo
	static int tracker = 0;
	std::cout<<tracker<<std::endl;
	this->img_batch.clear();
	this->convertToRGB();
	cv::cuda::GpuMat mat(cv::Size(this->iWidth, this->iHeight), CV_8UC3, this->rgbVideo);

	mat.download(this->frame);
	this->frame.create(cv::Size(this->iWidth*2, this->iHeight/2), CV_8UC3);
	this->__cutToPanels();


}

void YoloMask::preprocess()
{
	int src_height  = this->iHeight;
	int src_width = this->iWidth;

	int dst_width = kInputW;
	int dst_height = kInputH;

	AffineMatrix s2d, d2s;
	float scale = std::min(kInputH / (float)src_height, kInputW / (float)src_width);

	s2d.value[0] = scale;
	s2d.value[1] = 0;
	s2d.value[2] = -scale * src_width  * 0.5  + dst_width * 0.5;
	s2d.value[3] = 0;
	s2d.value[4] = scale;
	s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

	cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
	cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
	cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

	memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

	int jobs = kInputH * kInputW;
	int threads = 256;
	int blocks = ceil(jobs / (float)threads);

	cv::cuda::GpuMat rgbData(this->iHeight, this->iWidth, CV_8UC3, this->rgbVideo, this->iWidth*sizeof(uchar3));

	warpaffine_kernel<<<blocks, threads, 0, stream>>>(
	  rgbData.ptr(), src_width * 3, src_width,
	  src_height, this->gpuBuffs[0], dst_width,
	  dst_height, 128, d2s, jobs);
}

void YoloMask::runInference()
{
//	float *buffers[3];
//	buffers[0] = this->batchData;
//	buffers[1] = this->outputBufferDetections;
//	buffers[2] = this->outputBufferMask;


	context->enqueue(kBatchSize,(void**)this->gpuBuffs, stream, nullptr);

	this->cudaStatus = cudaMemcpyAsync(this->detectionsOutCpu, this->gpuBuffs[1], kBatchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream);
	this->checkCudaError("Copy memory", "cpu memory");
	this->cudaStatus = cudaMemcpyAsync(this->maskOutCpu, this->gpuBuffs[2], kBatchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
	this->checkCudaError("Copy memory", "cpu memory");
	cudaDeviceSynchronize();
	this->cudaStatus = cudaGetLastError();
	this->checkCudaError("synchronize device", "cpu memory");
}

void YoloMask::postprocess()
{
	// NMS
//	std::vector<std::vector<Detection>> res_batch;
//	batch_nms(res_batch, this->detectionsOutCpu, kBatchSize, kOutputSize1, kConfThresh, kNmsThresh);
//
//	for (size_t b = 0; b < kBatchSize; b++)
//	{
////		auto& res = res_batch[b];
////		auto masks = process_mask_s(&this->maskOutCpu[b * kOutputSize2], kOutputSize2, res);
//	}


}


void YoloMask::create()
{
	if(!this->loaded) return;
	if(!this->started)return;
	this->preprocess();
	this->runInference();
	this->postprocess();
}

uchar* YoloMask::output()
{
	return this->maskBuffer;
}

bool YoloMask::isMask()
{
	return this->mask;
}
void YoloMask::load(float* bD, float* oD, float* oM)
{
	this->batchData = bD;
	this->outputBufferDetections = oD;
	this->outputBufferMask = oM;
	this->loaded = true;
}


void YoloMask::getBatch()
{
	cudaSetDevice(kGpuId);
	this->prepareImages();
	api->run(this->img_batch);
}


char *cwd = getenv("CWD");
std::string rootDir(cwd);
std::string engine_name = rootDir+"/res/yolo-seg-4.engine";
  // Deserialize the engine from file

IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
IExecutionContext* context = nullptr;
cudaStream_t stream;
// Prepare cpu and gpu buffers
float* gpu_buffers[3];
float* cpu_output_buffer1 = nullptr;
float* cpu_output_buffer2 = nullptr;


void initYolo()
{
	cudaSetDevice(kGpuId);
	CUDA_CHECK(cudaStreamCreate(&stream));
	deserialize_engine(engine_name, &runtime, &engine, &context);
	 // Init CUDA preprocessing
	cuda_preprocess_init(kMaxInputImageSize);
	prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &gpu_buffers[2], &cpu_output_buffer1, &cpu_output_buffer2);
}


void yoloRun(std::vector<cv::Mat> img_batch) {


  int counter = 0;

  // Preprocess
  cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
//  std::cout<<"I execute"<<std::endl;
   auto start = std::chrono::system_clock::now();
   infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer1, cpu_output_buffer2, kBatchSize);
   auto end = std::chrono::system_clock::now();
   std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // NMS
   std::vector<std::vector<Detection>> res_batch;
   batch_nms(res_batch, cpu_output_buffer1, img_batch.size(), kOutputSize1, kConfThresh, kNmsThresh);


	// Draw result and save image
   for (size_t b = 0; b < img_batch.size(); b++) {
	  auto& res = res_batch[0];
	  cv::Mat img = img_batch[b];

	  auto masks = process_mask(&cpu_output_buffer2[b * kOutputSize2], kOutputSize2, res);
//
	  draw_mask_bbox(img, res, masks);
//	  break;
//	  cv::imwrite("_" + img_name_batch[b], img);
	}


  // Release stream and buffers
//  cudaStreamDestroy(stream);
//  CUDA_CHECK(cudaFree(gpu_buffers[0]));
//  CUDA_CHECK(cudaFree(gpu_buffers[1]));
//  CUDA_CHECK(cudaFree(gpu_buffers[2]));
//  delete[] cpu_output_buffer1;
//  delete[] cpu_output_buffer2;
//  cuda_preprocess_destroy();
//  // Destroy the engine
//  context->destroy();
//  engine->destroy();
//  runtime->destroy();

}





