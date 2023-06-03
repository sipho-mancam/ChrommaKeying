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
//	virtual void convertToRGB(){}

	uint4* getVideo(){return this->video;}
	uint4* getFill(){ return this->fill;}
	uint4* getKey(){ return this->key;}
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
public:
	IMask(IPipeline *obj):IPipeline(obj)
	{
		this->mask = false;
		this->maskBuffer = nullptr;
		this->init();
	}
	void erode(int);
	void dilate(int);
	virtual ~IMask() = default;
	virtual bool isMask() = 0;
	void init() override;
	virtual uchar* output() = 0;
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

	void preview(std::string windowHandle)
	{
		if(rgbData==nullptr)return;

		mat.data = (uchar*)this->rgbData;
		mat.download(this->prev);
		cv::imshow(windowHandle, this->prev);

		cv::waitKey(5);
	}


};

/****
 * Video is received as yuv from decklink and unpacked to yuyv
 * rgbVideo contains the received video rgbOutput
 *
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
	void convertToRGB(); // converts from yuyv to RGB
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
};


class YoloMask: public IMask
{
public:
	YoloMask(IPipeline *obj):IMask(obj){}
	void create() override;
	uchar* output() override;
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

public:

};






inline void allocateMemory(void** devptr, long int size);
void startPipeline();






#endif /* SRC_NSRC_INTERFACES_HPP_ */
