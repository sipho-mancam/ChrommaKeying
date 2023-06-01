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


class IPipeline
{
protected:
	uint4* video;
	uint4* fill;
	uint4* key, *augVideo;
	cudaStream_t stream;
	cudaError_t cudaStatus;
	long int frameSizePacked, frameSizeUnpacked;
	int iWidth, iHeight;
	std::mutex* mtx;

public:
	IPipeline();
	IPipeline(IPipeline*);
	virtual ~IPipeline() = default;
	virtual void create() = 0;
	virtual void update() = 0;
//	virtual void output() = 0;
	virtual void init() = 0;

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


class Preview
{
public:
	Preview()
	{

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
	bool isInput(){return in;}
	void run(); // receive video and copy it to gpu
};



class Preprocessor: public IPipeline
{
private:
	uchar3* rgbVideo;
	uchar2* pVideo, *pKey, *pFill;


public:
	Preprocessor(IPipeline* , uchar2* video, uchar2*key, uchar2*fill);
	Preprocessor(uchar2* uvideo, uchar2* ukey, uchar2* fill);
	void unpack(); // unpack yuv to yuyv
	void convertToRGB(); // converts from yuyv to RGB
	void create() override; // Some more pre-processing logic
	void init() override;
};


class SnapShot: public IPipeline
{
private:
	uint4* videoSnapShot;
	bool taken;
public:
	SnapShot(IPipeline* obj): IPipeline(obj)
	{
		this->videoSnapShot=nullptr;
		this->cudaStatus = cudaMalloc((void**)&videoSnapShot, this->frameSizeUnpacked);
		assert((this->cudaStatus==cudaSuccess));
		taken = false;
	}

	void takeSnapShot()
	{
		taken = false;
		this->cudaStatus = cudaMemcpyAsync(this->videoSnapShot, this->augVideo, this->frameSizeUnpacked, cudaMemcpyDeviceToDevice, NULL);
		assert((this->cudaStatus==cudaSuccess));
		taken  = true;
	}

	bool isSnaped(){return this->taken;}
	uint4* getSnapShot(){return this->videoSnapShot;} // this will return the last snapshot taken
};


class LookupTable: public IPipeline
{
private:
	uchar* lookupBuffer;
	bool loaded;

public:

	LookupTable(IPipeline *obj);
	void create() override;
	void update(bool init , bool clickEn, MouseData md, WindowSettings ws);
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


class Mask: public IMask
{
private:
	ChrommaMask* chromma;
	YoloMask* yolo;
public:
	Mask(ChrommaMask* cm, YoloMask* y): IMask(cm)
	{
		this->yolo = y;
		this->chromma = cm;
	}

	void create(); // combine chroma and yolov mask
	uchar* output();
};

class ChrommaKey: public IPipeline
{
private:


public:

};












#endif /* SRC_NSRC_INTERFACES_HPP_ */
