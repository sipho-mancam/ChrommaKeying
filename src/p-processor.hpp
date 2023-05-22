#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <InputLoopThrough.h>
#include <thread>
#include <cuda_runtime.h>

#define MAX_LOOK_UP 3
#define CUDA_LOOKUP_SIZE 1073741824  // 134217728 1024*1024*1024
#define SIZE_ULONG4_CUDA 16

class PipeLineObj{
private:
	int id = 0;

public:
	PipeLineObj()
	{

	}

	int getId(){return this->id;}
};

class Processor{

private:
	long int frameSizePacked;
	long int frameSizeUnpacked;
	int iWidth;
	int iHeight;
	int iDelayFrames;
	cudaError_t cudaStatus;
	cudaStream_t stream;

	uchar2 *yPackedCudaVideo, *yPackedCudaFill, *yPackedCudaKey;
	uchar2* yUnpackedCudaVideo, *yUnpackedCudaFill, *yUnpackedCudaKey;
	uchar3* cudaRGB;
	uchar* chromaGeneratedMask[3];
	uchar* LookUpDataArry[MAX_LOOK_UP];
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
	}
	Processor(VideoIn *vi)
	{
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

		this->cudaInit();
	}

	~Processor()
	{
		this->cudaCleanup();
	}

	void cudaCleanup();
	void cudaInit(); // Initialize all the cuda memory (Allocate and Set if necessary)
	bool toCuda(void* src, void* dst, long int size);
	void sendDataTo(); // send packed key and fill to cuda.
	void unpackYUV(); // launch kernels to unpack yuv data and place in buffers above
	void snapshot(cv::cuda::GpuMat* RGBData);
	VideoIn* getVideoIn(){return this->deckLinkInput;}
	void cudaReset();

	void run();

};
