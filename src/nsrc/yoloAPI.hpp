
#ifndef SRC_NSRC_YOLO_HPP_
#define SRC_NSRC_YOLO_HPP_

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

static Logger gLogger;
const static int kOutputSize1 = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSize2 = 32 * (kInputH / 4) * (kInputW / 4);

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir, std::string& labels_filename);
void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);
void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2);
void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize);

// class IPipeline
// {
// protected:
// 	uint4* video;
// 	uint4* fill;
// 	uint4* key, *augVideo;
// 	uchar3* rgbVideo;
// 	cudaStream_t stream;
// 	cudaError_t cudaStatus;
// 	long int frameSizePacked, frameSizeUnpacked, rowLength;
// 	int iWidth, iHeight;
// 	std::mutex* mtx;

// public:
// 	IPipeline();
// 	IPipeline(IPipeline*);
// 	virtual ~IPipeline() = default;
// 	virtual void create(){}
// 	virtual void update() {}
// //	virtual void output() = 0;
// 	virtual void init() {}
// 	virtual void convertToRGB();
// 	virtual void convertToRGB(uint4* src);
// 	virtual void rgbToYUYV();

// 	uint4* getVideo(){return this->video;}
// 	uint4* getFill(){ return this->fill;}
// 	uint4* getKey(){ return this->key;}
// 	uchar3* getRGB(){return this->rgbVideo;}
// 	std::mutex* getMutex(){return this->mtx;}
// 	long int getFrameSize(){return this->frameSizeUnpacked;}
// 	long int getPFrameSize(){return this->frameSizePacked;}
// 	int getWidth(){return this->iWidth;}
// 	int getHeight(){return this->iHeight;}
// 	cudaStream_t getStream(){return this->stream;}

// 	void setMutex(std::mutex* m){this->mtx = m;}
// 	void checkCudaError(std::string action, std::string loc);

// };





#endif /* SRC_NSRC_YOLO_HPP_ */
