/*
 * yolo.cpp
 *
 *  Created on: 12 Jun 2023
 *      Author: jurie
 */

#include "yolo.hpp"
#include <fstream>

using namespace nvinfer1;

static Logger gLogger;


void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, int batchSize) {
  context.enqueue(batchSize, buffers, stream, nullptr);
  cudaStreamSynchronize(stream);
}


YoloMask::YoloMask(IPipeline *obj): IMask(obj) // @suppress("Class members should be properly initialized")
{
	this->outputBufferDetections = nullptr;
	this->outputBufferMask = nullptr;
	this->batchData = nullptr;
	this->context = nullptr;
	this->engine = nullptr;
	this->runtime = nullptr;
	this->stream;
	this->cudaStatus = cudaStreamCreate(&stream);
	this->checkCudaError("create cuda stream", "Yolo Mask Constructor");
}

void YoloMask::initialize()
{
	cudaSetDevice(kGpuId);
	char *cwd = getenv("CWD");
	std::string rootDir(cwd);
	std::string engine_name = rootDir+"/res/yolov5s-seg-27.engine";

	std::ifstream engine_file(engine_name, std::ios::binary);

	if(!engine_file)
	{
		std::cerr<<"Engine File doesn't exist: \n"<<"Path: "<<engine_name<<std::endl;
		return;
	}

	deserialize_engine(engine_name, &runtime, &engine, &context);

	assert(engine->getNbBindings() == 3);
	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine->getBindingIndex(kInputTensorName);
	const int outputIndex1 = engine->getBindingIndex(kOutputTensorName);
	const int outputIndex2 = engine->getBindingIndex("proto");
	assert(inputIndex == 0);
	assert(outputIndex1 == 1);
	assert(outputIndex2 == 2);
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

	warpaffine_kernel<<<blocks, threads, 0, stream>>>(
	  (uchar*)this->rgbVideo, src_width * 3, src_width,
	  src_height, this->batchData, dst_width,
	  dst_height, 128, d2s, jobs);
}

void YoloMask::runInference()
{
	float *buffers[3];
	buffers[0] = this->batchData;
	buffers[1] = this->outputBufferDetections;
	buffers[2] = this->outputBufferMask;

	infer(*context, stream, (void**)buffers, kBatchSize);
}



void YoloMask::create()
{

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
}

