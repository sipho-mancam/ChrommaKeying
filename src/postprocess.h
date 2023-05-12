#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);

void batch_nms(std::vector<std::vector<Detection>>& batch_res, float *output, int batch_size, int output_size, float conf_thresh, float nms_thresh = 0.5);
void nms_combine(std::vector<Detection>& res, float *m_prob, float conf_thresh, float nms_thresh = 0.5,int iBatchCount=0,int iOutputSize=0) ;
void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch);

void process_mask_to_final(cv::cuda::GpuMat *seg_mat,int batchindex,const float* proto, int proto_size, std::vector<Detection>* dets);
//std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets);
std::vector<cv::Mat*> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets);
void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat*>& masks, std::unordered_map<int, std::string>& labels_map);

void create_mask(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks, std::unordered_map<int, std::string>& labels_map);
