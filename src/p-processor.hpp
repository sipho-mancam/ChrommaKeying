#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <InputLoopThrough.h>
#include <thread>
#include <cuda_runtime.h>
#include <cassert>
#include <YUVUChroma.cuh>

#define MAX_LOOK_UP 3
#define CUDA_LOOKUP_SIZE 1073741824  // 1024*1024*1024
#define SIZE_ULONG4_CUDA 16
#define MAX_PATH 260


class PipelineObj{
private:
	int id = 0;

protected:
	cudaError_t cudaStatus;
	cudaStream_t stream;
	long int frameSizePacked;
	long int frameSizeUnpacked;
	int iWidth;
	int iHeight;
	std::mutex* mtx;

	virtual void cudaCleanup() = 0;
	virtual void cudaInit() = 0;
	int getId(){return this->id;}
	bool toCuda(void* src, void* dst, long int size);

public:
	cudaStream_t getCudaStream(){return this->stream;}
	void checkCudaError(std::string action, std::string loc);
	long int getFrameSize(){return this->frameSizeUnpacked;}
	int getWidth(){ return this->iWidth;}
	int getHeight(){return this->iHeight;}
	virtual ~PipelineObj() = default;
	void setMutex(std::mutex *m){this->mtx = m;}
	std::mutex* getMutex(){return this->mtx;}
};



class Processor: public PipelineObj{
private:
	int iDelayFrames;
	uchar2* yPackedCudaVideo, *yPackedCudaFill, *yPackedCudaKey;
	uchar2* yUnpackedCudaVideo, *yUnpackedCudaFill, *yUnpackedCudaKey;
	uchar3* cudaRGB;
	VideoIn* deckLinkInput;

public:
	Processor()
	{
		this->deckLinkInput = new VideoIn;
		// Wait for DeckLink Device to receive the first frame and initialize
		while(this->deckLinkInput->m_sizeOfFrame == -1)
			std::this_thread::sleep_for(std::chrono::milliseconds(40));

		this->frameSizePacked = this->deckLinkInput->m_sizeOfFrame;
		this->frameSizeUnpacked = this->deckLinkInput->m_iFrameSizeUnpacked;
		this->iWidth = this->deckLinkInput->m_iWidth;
		this->iHeight = this->deckLinkInput->m_iHeight;
		this->iDelayFrames = 1;

		this->yPackedCudaFill = nullptr;
		this->yPackedCudaKey = nullptr;
		this->yPackedCudaVideo = nullptr;

		this->yUnpackedCudaFill = nullptr;
		this->yUnpackedCudaKey = nullptr;
		this->yUnpackedCudaVideo = nullptr;
		this->cudaRGB = nullptr;

		this->cudaInit();
		this->mtx = nullptr;
	}
	Processor(VideoIn *vi)
	{
		assert((vi!=nullptr));
		this->deckLinkInput = vi;

		// Wait for DeckLink Device to receive the first frame and initialize
		while(this->deckLinkInput->m_sizeOfFrame == -1)
			std::this_thread::sleep_for(std::chrono::milliseconds(40));

		this->frameSizePacked = this->deckLinkInput->m_sizeOfFrame;
		this->frameSizeUnpacked = this->deckLinkInput->m_iFrameSizeUnpacked;
		this->iWidth = this->deckLinkInput->m_iWidth;
		this->iHeight = this->deckLinkInput->m_iHeight;
		this->iDelayFrames = 1;

		this->yPackedCudaFill = nullptr;
		this->yPackedCudaKey = nullptr;
		this->yPackedCudaVideo = nullptr;

		this->yUnpackedCudaFill = nullptr;
		this->yUnpackedCudaKey = nullptr;
		this->yUnpackedCudaVideo = nullptr;
		this->cudaRGB = nullptr;
		this->mtx = nullptr;

		this->cudaInit();
	}

	~Processor()
	{
		this->cudaCleanup();
	}
	// Initialize all the cuda memory (Allocate and Set if necessary)
	void sendDataTo(); // send packed key and fill to cuda.
	void unpackYUV(); // launch kernels to unpack yuv data and place in buffers above
	void snapshot(cv::cuda::GpuMat* RGBData);
	VideoIn* getVideoIn(){return this->deckLinkInput;}
	uchar2* getVideo(){return this->yUnpackedCudaVideo;}
	uchar2* getKey(){return this->yUnpackedCudaKey;}
	uchar2* getFill(){return this->yUnpackedCudaFill;}

	void run();
	void cudaCleanup() override;
	void cudaInit() override;
	void cudaReset();

};


class ChrommaKey : public PipelineObj {
private:

	Processor* proc;
	uchar**  chromaGeneratedMask;
	uchar**  lookupTable;
	uchar2* video, *key, *fill;


public:
	ChrommaKey(Processor *p)
	{
		assert((p!=nullptr));
		this->proc = p;
		this->chromaGeneratedMask = nullptr;
		this->lookupTable = nullptr;
		this->stream = this->proc->getCudaStream();
		this->iWidth = this->proc->getHeight();
		this->iHeight = this->proc->getWidth();
		this->frameSizeUnpacked = this->proc->getFrameSize();
		this->video = this->proc->getVideo();
		this->key = this->proc->getVideo();
		this->fill = this->proc->getFill();
	}

	void cudaInit() override;
	void cudaCleanup() override;
	void generateChrommaMask();
	void erodeAndDilate(int, int);
	void updateLookup(bool init , bool clickEn, MouseData md, WindowSettings ws);


	~ChrommaKey()
	{
		this->cudaCleanup();
	}


};
