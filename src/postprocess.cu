#include "postprocess.h"
#include "utils.h"
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>
#include <queue>
#include <condition_variable>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudaoptflow.hpp>
//#include <format>

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
	float l, r, t, b;
	float r_w = kInputW / (img.cols * 1.0);
	float r_h = kInputH / (img.rows * 1.0);
	if (r_h > r_w) {
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (kInputH - r_w * img.rows) / 2;
		b = bbox[1] + bbox[3] / 2.f - (kInputH - r_w * img.rows) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	} else {
		l = bbox[0] - bbox[2] / 2.f - (kInputW - r_h * img.cols) / 2;
		r = bbox[0] + bbox[2] / 2.f - (kInputW - r_h * img.cols) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;
	}
	return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

static float iou(float lbox[4], float rbox[4]) {
	float interBox[] = { (std::max)(lbox[0] - lbox[2] / 2.f,
			rbox[0] - rbox[2] / 2.f), //left
	(std::min)(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
	(std::max)(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
	(std::min)(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
			};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

static bool cmp(const Detection& a, const Detection& b) {
	return a.conf > b.conf;
}

int GetOffset(int b)
{
	switch(b){
		case 0:
			return 0;
			break;
		case 1:
			return 427;
			break;

		case 2:
			return 853;
			break;

		case 3:
			return 1280;
			break;

		case 4:
				return 0+1920;
				break;
			case 5:
				return 427+1920;
				break;

			case 6:
				return 853+1920;
				break;

			case 7:
				return 1280+1920;
				break;
		default :
			return 0;
			break;
		}
	return 0;
}

void nms(std::vector<Detection>& res, float* output /*Output Buffer from "infer" func*/, float conf_thresh,
		float nms_thresh) {
	int det_size = sizeof(Detection) / sizeof(float);
	std::map<float, std::vector<Detection>> m;
	// collect all the output boxes into a map
	for (int i = 0; i < output[0] && i < kMaxNumOutputBbox; i++) {
		if (output[1 + det_size * i + 4] <= conf_thresh)
			continue;

		// load the ouput from the inference into a Detection object.
		// The float is aligned with the Detection structure using "alignas" on struct def.
		// That's why the copy below is possible.
		Detection det;
		memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));

		// check if the detections belonging to this class are already appended
		if (m.count(det.class_id) == 0)
			// if no such detection, create a new instance of such and id
			m.emplace(det.class_id, std::vector<Detection>());
		// append the detection to those matching it
		m[det.class_id].push_back(det);
	}
	// iterate over all the detections packed by class
	for (auto it = m.begin(); it != m.end(); it++) {
		auto& dets = it->second; // vector from the map.
		// sort the detections in the list of detections per class using confidence.
		std::sort(dets.begin(), dets.end(), cmp); // sort the detections by confidence

		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {

				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

void batch_nms(std::vector<std::vector<Detection>>& res_batch, float *output, int batch_size, int output_size, float conf_thresh, float nms_thresh) {

	res_batch.resize(batch_size);

	for (int i = 0; i < batch_size; i++) {
		nms(res_batch[i], &output[i * output_size], conf_thresh, nms_thresh);
	}

}

void draw_bbox(std::vector<cv::Mat>& img_batch,
		std::vector<std::vector<Detection>>& res_batch) {
	for (size_t i = 0; i < img_batch.size(); i++) {
		auto& res = res_batch[i];
		cv::Mat img = img_batch[i];
		for (size_t j = 0; j < res.size(); j++) {
			cv::Rect r = get_rect(img, res[j].bbox);
			cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
			cv::putText(img, std::to_string((int) res[j].class_id),
					cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
					cv::Scalar(0xFF, 0xFF, 0xFF), 2);
		}
	}
}

static cv::Rect get_downscale_rect(float bbox[4], float scale) {
	float left = bbox[0] - bbox[2] / 2;
	float top = bbox[1] - bbox[3] / 2;
	float right = bbox[0] + bbox[2] / 2;
	float bottom = bbox[1] + bbox[3] / 2;
	left /= scale;
	top /= scale;
	right /= scale;
	bottom /= scale;
	return cv::Rect(round(left), round(top), round(right - left),
			round(bottom - top));
}

cv::Mat scale_mask(cv::Mat mask, cv::Mat img) {
	int x, y, w, h;
	float r_w = kInputW / (img.cols * 1.0);
	float r_h = kInputH / (img.rows * 1.0);
	if (r_h > r_w) {
		w = kInputW;
		h = r_w * img.rows;
		x = 0;
		y = (kInputH - h) / 2;
	} else {
		w = r_h * img.cols;
		h = kInputH;
		x = (kInputW - w) / 2;
		y = 0;
	}
	cv::Rect r(x, y, w, h);
	cv::Mat res;
	cv::resize(mask(r), res, img.size());
	return res;
}

std::condition_variable cond_var;
std::mutex m;

inline __device__ __host__ int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void createmask_to_final_GPU_Para(float *maskDownload0, float *mask,int iOffset,
		int height, int width, int iStep, float* proto, int proto_size,int x_s,int y_s,int width_s,int height_s) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= (width) || y >= (height))
		return;

	if (x >= (width_s+x_s) || y >= (height_s+y_s))
		return;

	if (x <= x_s || y <= y_s)
			return;
//
//	printf("%d %d %d %d %d %d \n",x,y,x_s,y_s,width_s,height_s);
//	printf("%d\n",width);
//
	float e = 0.0f;
	for (int j = 0; j < 32; j++) {
		e += mask[j] * proto[j * proto_size / (32) + y * width + x];
	}
	e = 1.0f / (1.0f + expf(-e));
	float *maskDownload_val = &maskDownload0[x + y * 960+iOffset];
	if(e>0.5)
	if(*maskDownload_val < e)
		*maskDownload_val = 1;
	//*maskDownload_val = 1.0;
	// if(e>0.5)
	//	 printf("%f\n",e);

}

void createmask_to_final_GPU(cv::cuda::GpuMat *seg_mat,int batchindex, int x, Detection dets,
		 const float* proto_gpu,int proto_size, cv::Rect rn) {

	cv::cuda::GpuMat mask = cv::cuda::GpuMat(32,1 , CV_32F);
//	cv::cuda::GpuMat proto_GPU = cv::cuda::GpuMat( proto_size,1, CV_32F);

	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(mask.data, dets.mask, sizeof(float) * 32,
			cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return;
	}

	dim3 block(16, 16);
	dim3 grid(iDivUp((kInputH / 4), block.x), iDivUp((kInputW / 4), block.y));

	int iOffset=0;
	switch(batchindex)
	{
	case 0:
		iOffset=0;
			break;
	case 1:
		iOffset=428/4;
			break;
	case 2:
		iOffset=854/4;
			break;
	case 3:
		iOffset=1278/4;
			break;
	case 4:
		iOffset=1920/4;
			break;
	case 5:
		iOffset=(1920+428)/4;
			break;
	case 6:
		iOffset=(1920+854)/4;
			break;
	case 7:
		iOffset=(1920+1278)/4;
			break;
	}
	createmask_to_final_GPU_Para <<<grid, block>>>  ((float *) seg_mat->data,
			(float *) mask.data,iOffset, (kInputH / 4), (kInputW / 4),
			seg_mat->cols, (float *) proto_gpu, proto_size,rn.x,rn.y,rn.width,rn.height);



	//cv::cuda::resize(mask_mat_gpu, mask_mat_gpu, cv::Size(kInputW, kInputH));
	//cv::Mat mask_mat(mask_mat_gpu);
//
	//cv::imshow("test",mask_mat);
	//cv::waitKey(-1);
//	auto r = get_rect(*img, dets.bbox);
//	cv::Mat img_mask = scale_mask(mask_mat, *img);
//	r = get_rect(*img, dets.bbox);
//	for (int x = r.x; x < r.x + r.width; x++) {
//		for (int y = r.y; y < r.y + r.height; y++) {
//			float val = img_mask.at<float>(y, x);
//
//			if (val <= 0.5)
//				continue;
//			img->at<cv::Vec3b>(y, x)[0] = img->at<cv::Vec3b>(y, x)[0]/2;
//			img->at<cv::Vec3b>(y, x)[1] = img->at<cv::Vec3b>(y, x)[0]/2;
//			img->at<cv::Vec3b>(y, x)[2] = img->at<cv::Vec3b>(y, x)[0]/2;
//		}
//	}
//	cv::imwrite("/home/jurie/Pictures/test/"+std::to_string(x)+"test.bmp	", *img);
//	std::this_thread::sleep_for(std::chrono::milliseconds(100));

}

void createmask_to_final(cv::cuda::GpuMat *seg_mat,int batchindex, int x, Detection dets,
		 const float* proto_gpu, int proto_size) {

	auto r1 = get_downscale_rect(dets.bbox, 4);
	createmask_to_final_GPU(seg_mat,batchindex, x, dets, proto_gpu, proto_size,r1);

	return ;
}

void createmask(int x, Detection dets, const float* proto, int proto_size,
		std::vector<cv::Mat*> *masks) {
	std::cout << "create mask " << x << std::endl;
	cv::Mat *mask_mat = new cv::Mat(kInputH / 4, kInputW / 4, CV_32FC1);
	auto r = get_downscale_rect(dets.bbox, 4);
	for (int x = r.x; x < r.x + r.width; x++) {
		for (int y = r.y; y < r.y + r.height; y++) {
			float e = 0.0f;
			for (int j = 0; j < 32; j++) {
				e += dets.mask[j]* proto[j * proto_size / (32) + y * mask_mat->cols + x];
			}

			e = 1.0f / (1.0f + expf(-e));
			//        std::cout<<proto_size/(32*80)<<std::endl;
			//        return masks;
			mask_mat->at<float>(y, x) = e;
		}
	}

	std::string ss;
	// std::cout << std::format("filename_{}.bmp",x);
	cv::resize(*mask_mat, *mask_mat, cv::Size(kInputW, kInputH));
	// cv::imwrite()
	//    std::unique_lock<std::mutex> lock{m};
	masks->push_back(mask_mat);
	//    lock.unlock();

}

void process_mask_to_final(cv::cuda::GpuMat *seg_mat,int batchindex, const float* proto,
		int proto_size, std::vector<Detection>* dets) {

	std::vector<std::thread*> threadlist;
	int iTest = 0;
	for (size_t i = 0; i < dets->size(); i++) {

		std::thread *first = new std::thread(createmask_to_final,seg_mat,batchindex, iTest++,
				(*dets)[i], proto, proto_size);
		threadlist.push_back(first);

	}
	std::for_each(threadlist.begin(), threadlist.end(),[](std::thread* &th) {th->join();});
}

std::vector<cv::Mat*> process_mask(const float* proto, int proto_size,
		std::vector<Detection>& dets) {
	std::vector<cv::Mat*> masks;
//  std::cout<<"End"<<std::endl;

	std::vector<std::thread*> threadlist;
//  std::thread first (createmask,5);
	//first.join();
	int iTest = 0;
	for (size_t i = 0; i < dets.size(); i++) {

		std::thread *first = new std::thread(createmask, iTest++, dets[i],
				proto, proto_size, &masks);
		threadlist.push_back(first);


	}

	std::for_each(threadlist.begin(), threadlist.end(),
			[](std::thread* &th) {th->join();});

	return masks;
}

void create_mask(cv::Mat& img, std::vector<Detection>& dets,
		std::vector<cv::Mat>& masks,
		std::unordered_map<int, std::string>& labels_map) {
	static std::vector<uint32_t> colors = { 0xFF3838, 0xFF9D97, 0xFF701F,
			0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17, 0x3DDB86, 0x1A9334,
			0x00D4BB, 0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF, 0x0018EC,
			0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7 };
	for (size_t i = 0; i < dets.size(); i++) {
		cv::Mat img_mask = scale_mask(masks[i], img);
		auto color = colors[(int) dets[i].class_id % colors.size()];
		auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF,
				color >> 16 & 0xFF);

		cv::Rect r = get_rect(img, dets[i].bbox);
		for (int x = r.x; x < r.x + r.width; x++) {
			for (int y = r.y; y < r.y + r.height; y++) {
				float val = img_mask.at<float>(y, x);
				if (val <= 0.5)
					continue;
				img.at<cv::Vec3b>(y, x)[0] = img.at<cv::Vec3b>(y, x)[0] / 2
						+ bgr[0] / 2;
				img.at<cv::Vec3b>(y, x)[1] = img.at<cv::Vec3b>(y, x)[1] / 2
						+ bgr[1] / 2;
				img.at<cv::Vec3b>(y, x)[2] = img.at<cv::Vec3b>(y, x)[2] / 2
						+ bgr[2] / 2;
			}
		}

		cv::rectangle(img, r, bgr, 2);

		// Get the size of the text
		cv::Size textSize = cv::getTextSize(
				labels_map[(int) dets[i].class_id] + " "
						+ to_string_with_precision(dets[i].conf),
				cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
		// Set the top left corner of the rectangle
		cv::Point topLeft(r.x, r.y - textSize.height);

		// Set the bottom right corner of the rectangle
		cv::Point bottomRight(r.x + textSize.width, r.y + textSize.height);

		// Set the thickness of the rectangle lines
		int lineThickness = 2;

		// Draw the rectangle on the image
		cv::rectangle(img, topLeft, bottomRight, bgr, -1);

		cv::putText(img,
				labels_map[(int) dets[i].class_id] + " "
						+ to_string_with_precision(dets[i].conf),
				cv::Point(r.x, r.y + 4), cv::FONT_HERSHEY_PLAIN, 1.2,
				cv::Scalar::all(0xFF), 2);

	}
}

void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets,
		std::vector<cv::Mat*>& masks,
		std::unordered_map<int, std::string>& labels_map) {
	static std::vector<uint32_t> colors = { 0xFF3838, 0xFF9D97, 0xFF701F,
			0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17, 0x3DDB86, 0x1A9334,
			0x00D4BB, 0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF, 0x0018EC,
			0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7 };
	for (size_t i = 0; i < dets.size(); i++) {
		cv::Mat img_mask = scale_mask(*masks[i], img);
		auto color = colors[(int) dets[i].class_id % colors.size()];
		auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF,
				color >> 16 & 0xFF);

		cv::Rect r = get_rect(img, dets[i].bbox);
		for (int x = r.x; x < r.x + r.width; x++) {
			for (int y = r.y; y < r.y + r.height; y++) {
				float val = img_mask.at<float>(y, x);

				if (val <= 0.5)
					continue;
//        img.at<cv::Vec3b>(y, x)[0] = 255;
//        img.at<cv::Vec3b>(y, x)[1] = 255;
//        img.at<cv::Vec3b>(y, x)[2] = 255;

				img.at<cv::Vec3b>(y, x)[0] = img.at<cv::Vec3b>(y, x)[0] / 2
						+ bgr[0] / 2;
				img.at<cv::Vec3b>(y, x)[1] = img.at<cv::Vec3b>(y, x)[1] / 2
						+ bgr[1] / 2;
				img.at<cv::Vec3b>(y, x)[2] = img.at<cv::Vec3b>(y, x)[2] / 2
						+ bgr[2] / 2;

			}
		}

		//   cv::rectangle(img, r, bgr, 2);

		// Get the size of the text
		//   cv::Size textSize = cv::getTextSize(labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf), cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
		// Set the top left corner of the rectangle
//    cv::Point topLeft(r.x, r.y - textSize.height);

		// Set the bottom right corner of the rectangle
		//   cv::Point bottomRight(r.x + textSize.width, r.y + textSize.height);

		// Set the thickness of the rectangle lines
		//   int lineThickness = 2;

		// Draw the rectangle on the image
		//   cv::rectangle(img, topLeft, bottomRight, bgr, -1);

		// cv::putText(img, labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf), cv::Point(r.x, r.y + 4), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar::all(0xFF), 2);

	}
}

