/*
 * pipeline.cpp
 *
 *  Created on: 31 May 2023
 *      Author: jurie
 */
//#include <p-processor.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <InputLoopThrough.h>
#include <cuda_runtime.h>
#include <YUVUChroma.cuh>
#include <stdio.h>
#include <opencv2/cudafilters.hpp>
#include "interfaces.hpp"

/**** Utils *****/
inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }


IPipeline::IPipeline()
{
	this->fill = nullptr;
	this->key = nullptr;
	this->video = nullptr;
	this->augVideo = nullptr;
	this->frameSizePacked = 0;
	this->frameSizeUnpacked = 0;
	this->rowLength = 0;
	this->iHeight = 0;
	this->iWidth = 0;
	this->cudaStatus ;
	this->cudaStatus = cudaStreamCreate(&this->stream);
	this->mtx = nullptr;
}

IPipeline::IPipeline(IPipeline* pObj)
{
	assert(pObj!=nullptr);
	this->fill = pObj->fill;
	this->key = pObj->key;
	this->video = pObj->video;
	this->augVideo = pObj->augVideo;
	this->frameSizePacked = pObj->frameSizePacked;
	this->frameSizeUnpacked = pObj->frameSizeUnpacked;
	this->iHeight = pObj->iHeight;
	this->iWidth = pObj->iWidth;
	this->cudaStatus ;
	this->cudaStatus = cudaStreamCreate(&this->stream);
	this->mtx = pObj->mtx;
	this->rowLength  = pObj->rowLength;
}

void IPipeline::checkCudaError(std::string action, std::string loc)
{
	if(this->cudaStatus != cudaSuccess)
	{
		std::cerr<<"[Error]: Failed to "<< action<<" to "<< loc <<" \n"
				<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)
		<<std::endl;
	}
}


Input::Input(VideoIn* i): IPipeline()
{
	this->input = i;
	this->in = false;
	this->pFill = nullptr;
	this->pKey = nullptr;
	this->pVideo = nullptr;
	this->init();
}

void Input::init()
{
	while(this->input->m_sizeOfFrame == -1)
		std::this_thread::sleep_for(std::chrono::milliseconds(40));
	input->WaitForFrames(-1);

	input->imagelistVideo.ClearAll(0);
	input->imagelistFill.ClearAll(0);
	input->imagelistKey.ClearAll(0);
	input->ImagelistOutput.ClearAll(1);

	this->cudaStatus = cudaMalloc((void**)&this->pVideo, this->input->m_sizeOfFrame);
	this->checkCudaError("Allocate Memory", " packedVideo");

	this->cudaStatus = cudaMalloc((void**)&this->pKey, this->input->m_sizeOfFrame);
	this->checkCudaError("Allocate Memory", " packedKey");

	this->cudaStatus = cudaMalloc((void**)&this->pFill, this->input->m_sizeOfFrame);
	this->checkCudaError("Allocate Memory", " packedVideo");

	this->iHeight = this->input->m_iHeight;
	this->iWidth = this->input->m_iWidth;
	this->frameSizePacked = this->input->m_sizeOfFrame;
	this->frameSizeUnpacked = this->input->m_iFrameSizeUnpacked;
	this->rowLength = this->input->m_RowLength;


}


void Input::run()
{
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->rowLength/16, block.x), iDivUp(this->iHeight, block.y));
	const int srcAlignedWidth = this->rowLength/16;
	const int dstAlignedWidth = this->iWidth/2;

	input->WaitForFrames(1);
	static void* videoFrame;
	this->in = false;
	if(videoFrame)
		free(videoFrame);
	videoFrame = this->input->imagelistVideo.GetFrame(true);
	void* keyFrame = this->input->imagelistKey.GetFrame(true);
	void* fillFrame = this->input->imagelistFill.GetFrame(true);

	this->cudaStatus = cudaMemcpy(this->pVideo, videoFrame, this->frameSizePacked, cudaMemcpyHostToDevice);
	this->checkCudaError("copy memory", " pVideo");
	assert((this->cudaStatus == cudaSuccess));

	this->cudaStatus = cudaMemcpy(this->pKey, keyFrame, this->frameSizePacked, cudaMemcpyHostToDevice);
	this->checkCudaError("copy memory", " pKey");
	assert((this->cudaStatus == cudaSuccess));

	this->cudaStatus = cudaMemcpy(this->pFill, fillFrame, this->frameSizePacked, cudaMemcpyHostToDevice);
	this->checkCudaError("copy memory", " pFill");
	assert((this->cudaStatus == cudaSuccess));

	if(fillFrame)
		free(fillFrame);
	if(keyFrame)
		free(keyFrame);
	this->in = true;
}




Preprocessor::Preprocessor(uchar2* video, uchar2*key, uchar2*fill) // these variables must be GPU pointers
{
	this->pVideo = video;
	this->pKey = key;
	this->pFill = fill;
	this->augVideo = nullptr;
	this->rgbVideo = nullptr;
	this->init();
}

Preprocessor::Preprocessor(IPipeline* obj, uchar2* video, uchar2*key, uchar2*fill): IPipeline(obj)
{
	this->pVideo = obj->pVideo;
	this->pVideo = video;
	this->pKey = key;
	this->pFill = fill;
	this->augVideo = nullptr;
	this->rgbVideo = nullptr;
	this->init();
}

void Preprocessor::init()
{
	this->cudaStatus = cudaMalloc((void**)&this->video, this->frameSizeUnpacked);
	this->checkCudaError("Allocate Memory", " video buffer");
	assert((this->cudaStatus == cudaSuccess));

	this->cudaStatus = cudaMalloc((void**)&this->key, this->frameSizeUnpacked);
	this->checkCudaError("Allocate Memory", " key buffer");
	assert((this->cudaStatus == cudaSuccess));

	this->cudaStatus = cudaMalloc((void**)&this->fill, this->frameSizeUnpacked);
	this->checkCudaError("Allocate Memory", " fill buffer");
	assert((this->cudaStatus == cudaSuccess));

	this->cudaStatus = cudaMalloc((void**)&this->augVideo,this->frameSizeUnpacked);
	this->checkCudaError("Allocate Memory", " augmented video buffer");
	assert((this->cudaStatus == cudaSuccess));

	this->cudaStatus = cudaMalloc((void**)&this->rgbVideo, this->iHeight*this->iWidth*sizeof(uchar3));
	this->checkCudaError("Allocate Memory", " RGB Video buffer");
	assert((this->cudaStatus == cudaSuccess));
}

void Preprocessor::load(uchar2* pv, uchar2* pk, uchar2* pf)
{
	this->pVideo = pv; this->pKey = pk; this->pFill = pf;
}

void Preprocessor::unpack()
{
	// Unpacked yuv to yuyv
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->rowLength/16, block.x), iDivUp(this->iHeight, block.y));
	const int srcAlignedWidth = this->rowLength/16;
	const int dstAlignedWidth = this->iWidth/2;

	// Unpack yuv video from decklink and store it in yUnpackedCudaVideo
	yuyvPackedToyuyvUnpacked <<<grid, block>>>(
			(uint4*)this->pVideo,
			(uint4*)this->video,
			srcAlignedWidth,
			dstAlignedWidth,
			this->iHeight
		);

	// Unpack yuv key from decklink and store it in yUnpackedCudaKey
	yuyvPackedToyuyvUnpacked <<<grid, block, 0, this->stream>>>(
				(uint4*)this->pKey,
				(uint4*)this->key,
				srcAlignedWidth,
				dstAlignedWidth,
				this->iHeight
			);


	// Unpack yuv fill from decklink and store it in yUnpackedCudaFill
	yuyvPackedToyuyvUnpacked <<<grid, block>>>(
				(uint4*)this->pFill,
				(uint4*)this->fill,
				srcAlignedWidth,
				dstAlignedWidth,
				this->iHeight
			);

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize device", " at unpacking");

	this->cudaStatus = cudaMemcpyAsync(this->augVideo, this->video, this->frameSizeUnpacked, cudaMemcpyDeviceToHost,this->stream);
	this->checkCudaError("copy data", " augmented video buffer");

}

void Preprocessor::convertToRGB()
{
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->rowLength/16, block.x), iDivUp(this->iHeight, block.y));

	yuyvUmPackedToRGB<<<block, grid>>>(
			this->augVideo,
			this->rgbVideo,
			this->iWidth,
			this->iWidth,
			this->iHeight,
			this->key
		);
	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("Synchronize device", " host");
}

void Preprocessor::create()
{

}

LookupTable::LookupTable(IPipeline *obj): IPipeline(obj)
{
	this->lookupBuffer = nullptr;
	this->loaded = false;
}

void LookupTable::create()
{
	this->cudaStatus = cudaMalloc((void**)&this->lookupBuffer, (long int)4*pow(2, 10)*sizeof(uchar));
	assert((this->cudaStatus==cudaSuccess));
	this->loaded = false;
}

void LookupTable::update(bool init, bool clickEn, MouseData md, WindowSettings ws)
{
//	std::cout<<"[info]: Update Lookup started"<<std::endl;

	if(!clickEn)return;

	if (md.bHandleLDown)
	{
		this->loaded = false;
		int maxRecSize = 200;
		float ScalingValue = maxRecSize*1.0/ws.m_iOuter_Diam*1.0;

		const dim3 block(16, 16);
		const dim3 grid(
				iDivUp((ws.m_iOuter_Diam+ws.m_iUV_Diam)*2, block.x),
				iDivUp((ws.m_iOuter_Diam+ws.m_iUV_Diam)*2, block.y)
				);

		uchar* ptrLookUpDataToUse = this->lookupBuffer;

		this->mtx->lock();
		std::cout<<"[info]: Thread locked ..."<<std::endl;

		for (int x = (md.iXUpDynamic / 2); x<(md.iXDownDynamic/2); x++)
		{
			for (int y = md.iYUpDynamic; y < md.iYDownDynamic; y=y+2)
			{
				UpdateLookupFrom_XY_Posision_Diffrent_Scaling <<<grid, block>>> (
						(uint4*)this->augVideo,
						ptrLookUpDataToUse,
						x, y,
						(this->iHeight / 2),
						ws.m_iOuter_Diam*2,
						ws.m_iUV_Diam*2,
						ws.m_iLum_Diam,
						ScalingValue,
						maxRecSize
						);
			}
		}

		this->cudaStatus = cudaDeviceSynchronize();
		this->checkCudaError("synchronize host", " kernel: updateLookupFromMouse");
		assert(this->cudaStatus==cudaSuccess);
		this->loaded = true;
		this->mtx->unlock();
	}
//	std::cout<<"[info]: LookupTable updated successfully"<<std::endl;
}

void IMask::init()
{
	this->cudaStatus = cudaMalloc((void**)this->maskBuffer, this->iHeight*this->iWidth*sizeof(uchar));
	assert(this->cudaStatus==cudaSuccess);
}

void IMask::erode(int size)
{
	cv::cuda::GpuMat chrommaMaskInput;
	cv::cuda::GpuMat chrommaMaskOutput(this->iWidth/2,this->iHeight*2,CV_8UC1, this->maskBuffer,Mat::CONTINUOUS_FLAG);
	chrommaMaskOutput.step=this->iWidth*2;

	// erode output mask
	int an = size;
	cv::Mat element = getStructuringElement(MORPH_ELLIPSE, Size(an*2+1, an*2+1), Point(an, an));
	Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter(MORPH_ERODE, chrommaMaskInput.type(), element);
	erodeFilter->apply(chrommaMaskInput, chrommaMaskOutput);
}

void IMask::dilate(int size)
{
	cv::cuda::GpuMat chrommaMaskInput;
	cv::cuda::GpuMat chrommaMaskOutput(this->iWidth/2,this->iHeight*2,CV_8UC1, this->maskBuffer,Mat::CONTINUOUS_FLAG);
	chrommaMaskOutput.step=this->iWidth*2;

	// Dilate the output mask
	int an = size;
	cv::Mat element = getStructuringElement(MORPH_ELLIPSE, Size(an*2+1, an*2+1), Point(an, an));
	Ptr<cv::cuda::Filter> erodeFilter2 = cv::cuda::createMorphologyFilter(MORPH_DILATE, chrommaMaskInput.type(), element);
	erodeFilter2->apply(chrommaMaskInput, chrommaMaskOutput);
}

ChrommaMask::ChrommaMask(IPipeline* obj, LookupTable* t): IMask(obj)
{
	this->table = t;
}

void ChrommaMask::create()
{
	if(!table->isLoaded())return;
	this->mtx->lock();
	this->mask = false;
	const int dstAlignedWidth = this->iWidth;
	const int srcAlignedWidth = this->iWidth/2;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(srcAlignedWidth, block.x), iDivUp(this->iHeight, block.y));

	yuyv_Unpacked_GenerateMask <<<grid, block, 0, this->stream>>> (
			(uint4*)this->augVideo,
			this->maskBuffer,
			this->table->output(),
			this->iWidth,
			this->iHeight,
			srcAlignedWidth,
			dstAlignedWidth,
			0
			);

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize host", "yuyvGenerateMask");
	assert((this->cudaStatus == cudaSuccess));

	this->mask = true;

	this->mtx->unlock();
}

uchar* ChrommaMask::output()
{
	if(!this->mask)return nullptr;
	this->create();
	this->update();
	return this->maskBuffer;
}

void YoloMask::create()
{

}

uchar* YoloMask::output()
{
	return this->maskBuffer;
}

void MaskOut::create()
{
	// perform some mask creation here ...
}

uchar* MaskOut::output()
{
	return this->chromma->output();
}




