/*
 * yolo.hpp
 *
 *  Created on: 12 Jun 2023
 *      Author: jurie
 */

#ifndef SRC_NSRC_YOLO_HPP_
#define SRC_NSRC_YOLO_HPP_

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include "interfaces.hpp"
//#include <NvInfer.hpp>




using namespace nvinfer1;

struct AffineMatrix {
  float value[6];
};

const static int kOutputSize1 = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSize2 = 32 * (kInputH / 4) * (kInputW / 4);

__global__ void warpaffine_kernel(uint8_t* , int , int ,int , float* , int ,int , uint8_t , AffineMatrix ,int );
void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, int batchSize);
void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, int batchSize);
void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2);
void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize);
void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);
void yoloRun(std::vector<cv::Mat> res);
void initYolo();

//class YoloAPI;



#endif /* SRC_NSRC_YOLO_HPP_ */
