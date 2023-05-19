#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yololayer.h"

using namespace nvinfer1;


cv::Rect get_rect_scale( float bbox[4],double scale) {
//    int l, r, t, b;
//    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
//    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
//    if (r_h > r_w) {
//        l = bbox[0] - bbox[2] / 2.f;
//        r = bbox[0] + bbox[2] / 2.f;
//        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
//        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
//        l = l / r_w;
//        r = r / r_w;
//        t = t / r_w;
//        b = b / r_w;
//    } else {
//        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
//        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
//        t = bbox[1] - bbox[3] / 2.f;
//        b = bbox[1] + bbox[3] / 2.f;
//        l = l / r_h;
//        r = r / r_h;
//        t = t / r_h;
//        b = b / r_h;
//    }
    return cv::Rect(bbox[0]-(bbox[2]/2*scale),bbox[1]-((bbox[3])/2.0*scale),bbox[2]*scale,bbox[3]*scale);
}


cv::Rect get_rect( float bbox[4],int iOffset) {
//    int l, r, t, b;
//    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
//    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
//    if (r_h > r_w) {
//        l = bbox[0] - bbox[2] / 2.f;
//        r = bbox[0] + bbox[2] / 2.f;
//        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
//        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
//        l = l / r_w;
//        r = r / r_w;
//        t = t / r_w;
//        b = b / r_w;
//    } else {
//        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
//        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
//        t = bbox[1] - bbox[3] / 2.f;
//        b = bbox[1] + bbox[3] / 2.f;
//        l = l / r_h;
//        r = r / r_h;
//        t = t / r_h;
//        b = b / r_h;
//    }
    return cv::Rect(bbox[0]-bbox[2]/2+iOffset,bbox[1]-(bbox[3])/2,bbox[2],bbox[3]);
}

cv::Rect get_rect_varsize_from_point( int x,int y,float iSize) {

	int xtemp=x-iSize/2;
	int ytemp=y-iSize/2;
	if((xtemp)<0)
		x=0;
	else
		if((xtemp)>(1920*2-iSize))
			x=(1920*2-iSize);
		else
			x=xtemp;

	if(ytemp<0)
			y=0;
		else
			if((ytemp)>(1080/2-iSize))
				y=(1080/2-iSize);
			else
				y=ytemp;



	cv::Rect re=cv::Rect(x,y,iSize,iSize);
	//std::cout << x<<" "<< y<<" "<<" " << re << std::endl;


    return cv::Rect(x,y,iSize,iSize);
}


cv::Rect get_rect_varsize_save( float bbox[4],int iXOffset,int iYOffset,int iSize) {


	int x=bbox[0]+iXOffset;
	int y=bbox[1]+iYOffset;
	int iWidth=x-iSize/2;
	int iHeight=y-iSize/2;

	if((iWidth)<0)
			x=0;
		else
			if((iWidth)>(1920*2-iSize))
				x=(1920*2-iSize);
			else
				x=iWidth;

		if((iHeight)<0)
				y=0;
			else
				if((iHeight)>(1080/2-iSize))
					y=(1080/2-iSize);
				else
					y=iHeight;

    return cv::Rect(x,y,iSize,iSize);
}

cv::Rect get_rect_varsize( float bbox[4],int iOffset,float iSize) {

    return cv::Rect(bbox[0]-iSize/2+iOffset,bbox[1]-iSize/2,iSize,iSize);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo_Big::Detection& a, const Yolo_Big::Detection& b) {
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
void nms(std::vector<Yolo_Big::Detection>& res, float *m_prob, float conf_thresh, float nms_thresh = 0.5,int iBatchCount=0,int iOutputSize=0) {

	  std::map<float, std::vector<Yolo_Big::Detection>> m;
for(int x=0;x<iBatchCount;x++)
{
	   int det_size = sizeof(Yolo_Big::Detection) / sizeof(float);
	   float *output=m_prob+x*iOutputSize;


	    for (int i = 0; i < output[0] && i < Yolo_Big::MAX_OUTPUT_BBOX_COUNT; i++) {

	        if (output[1 + det_size * i + 4] <= conf_thresh) continue;

	        Yolo_Big::Detection det;
	        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
	        if (m.count(det.class_id) == 0)
	        	m.emplace(det.class_id, std::vector<Yolo_Big::Detection>());
	      //  else
	        det.bbox[0]=det.bbox[0]+GetOffset(x);
	        m[det.class_id].push_back(det);
	    }
}

	    for (auto it = m.begin(); it != m.end(); it++) {
	        //std::cout << it->second[0].class_id << " --- " << std::endl;
	        auto& dets = it->second;

	        std::sort(dets.begin(), dets.end(), cmp);

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

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
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

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

ILayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname) {
    ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, Yolo_Big::INPUT_H / 2, Yolo_Big::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, Yolo_Big::INPUT_H / 2, Yolo_Big::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, Yolo_Big::INPUT_H / 2, Yolo_Big::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, Yolo_Big::INPUT_H / 2, Yolo_Big::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor* inputTensors[] = { cv3->getOutput(0), cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}

ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    ITensor* inputTensors[] = { y1, cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    return cv3;
}

ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k1, k1 });
    pool1->setPaddingNd(DimsHW{ k1 / 2, k1 / 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k2, k2 });
    pool2->setPaddingNd(DimsHW{ k2 / 2, k2 / 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k3, k3 });
    pool3->setPaddingNd(DimsHW{ k3 / 2, k3 / 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });

    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = Yolo_Big::CHECK_COUNT * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto *p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

IPluginV2Layer* addYolo_BigLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("Yolo_BigLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);
    PluginField plugin_fields[2];
    int netinfo[4] = {Yolo_Big::CLASS_NUM, Yolo_Big::INPUT_W, Yolo_Big::INPUT_H, Yolo_Big::MAX_OUTPUT_BBOX_COUNT};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    int scale = 8;
    std::vector<Yolo_Big::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Yolo_Big::YoloKernel kernel;
        kernel.width = Yolo_Big::INPUT_W / scale;
        kernel.height = Yolo_Big::INPUT_H / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("Yolo_Biglayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto Yolo_Big = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return Yolo_Big;
}
#endif

