/*
 * interfaces.hpp
 *
 *  Created on: May 31, 2023
 *      Author: sipho-mancam
 */

#ifndef SRC_NSRC_INTERFACES_HPP_
#define SRC_NSRC_INTERFACES_HPP_

#include <p-processor.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <InputLoopThrough.h>
#include <cuda_runtime.h>
#include <YUVUChroma.cuh>
#include <stdio.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "events.hpp"
#include "ui.hpp"


#define CUDA_LOOKUP_SIZE 1073741824
using namespace nvinfer1;


class IPipeline
{
protected:
	uint4* video;
	uint4* fill;
	uint4* key, *augVideo;
	uchar3* rgbVideo;
	cudaStream_t stream;
	cudaError_t cudaStatus;
	long int frameSizePacked, frameSizeUnpacked, rowLength;
	int iWidth, iHeight;
	std::mutex* mtx;

public:
	IPipeline();
	IPipeline(IPipeline*);
	virtual ~IPipeline() = default;
	virtual void create(){}
	virtual void update() {}
//	virtual void output() = 0;
	virtual void init() {}
	virtual void convertToRGB();
	virtual void convertToRGB(uint4* src);
	virtual void rgbToYUYV();

	uint4* getVideo(){return this->video;}
	uint4* getFill(){ return this->fill;}
	uint4* getKey(){ return this->key;}
	uchar3* getRGB(){return this->rgbVideo;}
	std::mutex* getMutex(){return this->mtx;}
	long int getFrameSize(){return this->frameSizeUnpacked;}
	long int getPFrameSize(){return this->frameSizePacked;}
	int getWidth(){return this->iWidth;}
	int getHeight(){return this->iHeight;}
	cudaStream_t getStream(){return this->stream;}

	void setMutex(std::mutex* m){this->mtx = m;}
	void checkCudaError(std::string action, std::string loc);

};

class IMask: public IPipeline
{
protected:
	bool mask;
	uchar* maskBuffer;
	uchar3* maskRGB;
public:
	IMask(IPipeline *obj):IPipeline(obj)
	{
		this->mask = false;
		this->maskBuffer = nullptr;
		this->maskRGB = nullptr;
		this->init();
	}
	void erode(int);
	void dilate(int);
	void openMorph(int size);
	virtual ~IMask() = default;
	virtual bool isMask() = 0;
	virtual uchar* output() = 0;
	void init() override;
	void load(uchar* m){this->maskBuffer = m;}
	void load(uchar* m, uchar3* rgb){this->maskBuffer = m; this->maskRGB = rgb;}
	uchar* getMask(){return this->maskBuffer;}
	virtual void toRGB(uchar3* rgb){};
	uchar3* getMaskRGB(){return this->maskRGB;}

};


class Preview :IPipeline
{
private:
	uchar3* rgbData;
	cv::cuda::GpuMat mat;
	cv::Mat prev;
public:
	Preview(IPipeline *obj): IPipeline(obj)
	{
		this->rgbData = nullptr;
		this->mat.create(this->iHeight, this->iWidth, CV_8UC3);
		this->mat.step = 5760;
	}

	void load(uchar3* rgb){ this->rgbData = rgb;}
	void load(uchar* mask){ this->rgbData = (uchar3*)mask;}

	void preview(std::string windowHandle)
	{
		if(rgbData==nullptr)return;

		mat.data = (uchar*)this->rgbData;
		mat.download(this->prev);
		cv::imshow(windowHandle, this->prev);
		cv::waitKey(5);
	}
};

/*
 * Video is received as YCbCr from decklink and unpacked to yuyv
 * rgbVideo contains the received video rgbOutput
 */

class Input : public IPipeline
{
private:
	VideoIn* input;
	bool in;
	uchar2* pVideo, *pKey, *pFill;

public:
	Input(VideoIn* i);
	void init() override; // initialize cuda variables
	bool isOutput(){return in;}
	void run(); // receive video and copy it to gpu
	void load(uchar2* pv, uchar2* pk, uchar2* pf);
	uchar2* getPVideo(){return this->pVideo;}
	uchar2* getPKey(){return this->pKey;}
	uchar2* getPFill(){return this->pFill;}
};



class Preprocessor: public IPipeline
{
private:
	uchar2* pVideo, *pKey, *pFill;

public:
	Preprocessor(IPipeline* , uchar2* video, uchar2*key, uchar2*fill);
	Preprocessor(uchar2* uvideo, uchar2* ukey, uchar2* fill);
	void unpack(); // unpack yuv to yuyv
	void create() override; // Some more pre-processing logic
	void init() override;
	void reload(uchar2* pVideo, uchar2* pKey, uchar2* pFill);
	void load(uint4 *v, uint4 *k, uint4* f, uint4* av, uchar3* rgb);
};


class SnapShot: public IPipeline
{
private:
	uchar3* videoSnapShot;
	uint4* frozenVideo;
	bool taken;
	IPipeline *base;
public:
	SnapShot(IPipeline* obj): IPipeline(obj)
	{
		this->videoSnapShot=nullptr;
		this->base = obj;
		taken = false;
		this->frozenVideo = nullptr;
	}

	void load(uchar3* v, uint4* sv){ this->videoSnapShot = v; this->frozenVideo = sv;}
	void takeSnapShot()
	{
		taken = false;
		this->cudaStatus = cudaMemcpy(this->videoSnapShot, this->rgbVideo, this->iHeight*this->iWidth*sizeof(uchar3), cudaMemcpyDeviceToDevice);
		assert((this->cudaStatus==cudaSuccess));

		this->cudaStatus = cudaMemcpy(this->frozenVideo, this->augVideo, this->frameSizeUnpacked, cudaMemcpyDeviceToDevice);
		assert((this->cudaStatus==cudaSuccess));

		taken  = true;
	}
	bool isSnaped(){return this->taken;}
	uchar3* getSnapShot(){return this->videoSnapShot;} // this will return the last snapshot taken
	uint4* getFrozenVideo(){return this->frozenVideo;}
};


class LookupTable: public IPipeline
{
private:
	uchar* lookupBuffer;
	bool loaded;
	uint4* snapShot;

public:

	LookupTable(IPipeline *obj);
	void create() override;
	void load(uchar* lb, uint4* snap){this->lookupBuffer = lb; this->snapShot = snap;}
	void setSnap(uint4* snap){this->snapShot = snap;}

	void update(bool clickEn, MouseData md, std::unordered_map<std::string, int> ws);
	uchar* output(){return this->lookupBuffer;}
	bool isLoaded(){return loaded;}
	void clearTable();
};


class ChrommaMask: public IMask
{
private:
	LookupTable* table;
public:
	ChrommaMask(IPipeline *obj, LookupTable* t);
	void create() override;
	bool isMask() {return this->mask;}
	void update(){}
	uchar* output() override;
	void toRGB(uchar3* rgb) override;
	void toRGB();
};



class YoloAPI : public IPipeline
{
private:
    // uchar3 *mask;
    std::vector<cv::Mat> masks;
    std::string engine_name;

    cv::Mat merged_mask;

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream;

    // Prepare cpu and gpu buffers
    float* gpu_buffers[3];
    float* cpu_output_buffer1 = nullptr;
    float* cpu_output_buffer2 = nullptr;

    void init();
    void deserialize();
    void preprocessor(std::vector<cv::Mat>& img_batch);
    void rInfer();
    void postProcess(std::vector<cv::Mat> &img_batch);
    void mergeBatch(); // this will merge the masks into a single mask
public:
    YoloAPI(IPipeline *obj, std::string engineF);
    void run(std::vector<cv::Mat>&img_batch);
   ~YoloAPI();
};



class YoloMask: public IMask
{
private:
	float *batchData; // buffer containing normalized image data put together in a batch of 8. (GPU Memory) planar data(rrrgggbbb)
	float *outputBufferMask, *outputBufferDetections;
	float *maskOutCpu, *detectionsOutCpu;
	float **gpuBuffs;

	YoloAPI *api;

	std::vector<cv::Mat> img_batch;
	cv::Mat frame;

	bool started, loaded;

	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	cudaStream_t stream;

	void preprocess();
	void runInference();
	void initialize();
	void postprocess();
	void prepareImages();
	void __cutToPanels();
public:
	YoloMask(IPipeline *obj);
	void create() override;
	uchar* output() override;
	bool isMask()override;
	void load(float* d, float*, float*);
	void load(float **gpuB){this->gpuBuffs = gpuB;}

	void getBatch();


//	void test()
//	{
//		this->prepareImages();
//	}

};



class MaskOut: public IMask
{
private:
	ChrommaMask* chromma;
	YoloMask* yolo;
public:
	MaskOut(ChrommaMask* cm, YoloMask* y): IMask(cm)
	{
		this->yolo = y;
		this->chromma = cm;
	}
	void create(); // combine chroma and yolov mask
	uchar* output();
};



class Keyer: public IPipeline
{
private:
	uchar* finalMask;
	double4 parabolic;
public:
	Keyer(IPipeline* obj, uchar* finalMask);
	void create(int blend); // combines the key, fill and video to video
};


class Pipeline
{
private:
	std::unordered_map<std::string, IPipeline*> pipelineObjects;
	std::vector<std::string> availableObjects;
	WindowsContainer *container;

	Input *input;
	Preprocessor *preproc;
	SnapShot * snapShot;
	LookupTable *lookup;
	ChrommaMask *chrommaMask;
	YoloMask *yoloMask;
	Keyer *keyer;
	int event;
	std::mutex* mtx;
public:
	Pipeline(WindowsContainer* cont, std::mutex* m)
	{
		this->container = cont;
		this->input = nullptr;
		this->preproc = nullptr;
		this->snapShot = nullptr;
		this->lookup = nullptr;
		this->chrommaMask = nullptr;
		this->yoloMask = nullptr;
		this->keyer = nullptr;
		event = -1;
		mtx = m;
	}
	~Pipeline(); // iterates over all objects and frees them from memory.
	void run(); // run the pipeline
	void load();
	void addPipelineObject(IPipeline* obj, std::string name);
	void assertObjects();
	void viewPipeline(); // prints all the objects in the pipeline in the order they appear
};

void startPipeline(Pipeline *obj);
void allocateMemory(void** devptr, long int size);


#endif /* SRC_NSRC_INTERFACES_HPP_ */
