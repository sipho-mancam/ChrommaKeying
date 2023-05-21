#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <DeckLinkAPI.h>
#include <thread>

#define MAX_LOOK_UP 3

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

	uchar2 *yPackedCudaVideo, *yPackedCudaFill, *yPackedCudaKey;
	uchar2* yUnpackedCudaVideo, *yUnpackedCudaFill, *yUnpackedCudaKey;
	uchar* chromaGeneratedMask[3];
	uchar* LookUpDataArry[MAX_LOOK_UP];
	VideoIn deckLinkInput;

public:
	Processor()
	{
		// Wait for DeckLink Device to receive the first frame and initialize
		while(this->deckLinkInput.m_sizeOfFrame == -1)std::this_thread::sleep_for(std::chrono::milliseconds(40));


		this->frameSizePacked = this->deckLinkInput.m_sizeOfFrame;
		this->frameSizeUnpacked = this->deckLinkInput.m_iFrameSizeUnpacked;
		this->iWidth = this->deckLinkInput.m_iWidth;
		this->iHeight = this->deckLinkInput.m_iHeight;

		this->yPackedCudaFill = nullptr;
		this->yPackedCudaKey = nullptr;
		this->yPackedCudaVideo = nullptr;

		this->yUnpackedCudaFill = nullptr;
		this->yUnpackedCudaKey = nullptr;
		this->yUnpackedCudaVideo = nullptr;


	}

	void cudaInit(); // Initialize all the cuda memory (Allocate and Set if necessary)







};
