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
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>
#include "interfaces.hpp"
#include <ui.hpp>
#include "yolo.hpp"
#include "color_balance.hpp"
#include "contours.hpp"
#include "cuda_runtime_api.h"

/**** Utils *****/
inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

__global__ void gammaCorrect(uint4* unpackedVideo, int srcAlignedWidth, int height, double gamma)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
			return;

	uint4* pixelValue = &unpackedVideo[y*srcAlignedWidth+x];

	pixelValue->w = pow(((double)pixelValue->w*1.0/1024), gamma) * 1024;
	pixelValue->y = pow(((double)pixelValue->y*1.0/1024), gamma) * 1024;

}

IPipeline::IPipeline()
{
	this->fill = nullptr;
	this->key = nullptr;
	this->video = nullptr;
	this->augVideo = nullptr;
	this->rgbVideo = nullptr;
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
	this->rgbVideo = pObj->rgbVideo;
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

void IPipeline::convertToRGB()
{

	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->iWidth/2, block.x), iDivUp(this->iHeight, block.y));


	yuyvUnpackedToRGB<<<grid, block>>>(
			this->augVideo,
			this->rgbVideo,
			this->iWidth/2,
			this->iWidth,
			this->iHeight,
			this->key
		);

	this->cudaStatus = cudaGetLastError();
	this->checkCudaError("Launch Kernel", "Device");

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("Synchronize device", " host");

}

void IPipeline::convertToRGB(uint4* src)
{

	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->iWidth/2, block.x), iDivUp(this->iHeight, block.y));


	yuyvUnpackedToRGB<<<grid, block>>>(
			src,
			this->rgbVideo,
			this->iWidth/2,
			this->iWidth,
			this->iHeight,
			this->key
		);

	this->cudaStatus = cudaGetLastError();
	this->checkCudaError("Launch Kernel", "Device");

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("Synchronize device", " host");

}

void IPipeline::rgbToYUYV()
{

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

	this->iHeight = this->input->m_iHeight;
	this->iWidth = this->input->m_iWidth;
	this->frameSizePacked = this->input->m_sizeOfFrame;
	this->frameSizeUnpacked = this->input->m_iFrameSizeUnpacked;
	this->rowLength = this->input->m_RowLength;
}

void Input::load(uchar2* pv, uchar2* pk, uchar2* pf)
{
	this->pVideo = pv; this->pKey = pk; this->pFill = pf;
}


void Input::run(int delay)
{
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->rowLength/16, block.x), iDivUp(this->iHeight, block.y));
	const int srcAlignedWidth = this->rowLength/16;
	const int dstAlignedWidth = this->iWidth/2;

	input->WaitForFrames(delay);
	static void* videoFrame;
	this->in = false;
	if(videoFrame)
		free(videoFrame);
	videoFrame = this->input->imagelistVideo.GetFrame(true);
	if(!videoFrame)
	{
		this->in  = false;
		return;
	}
	void* keyFrame = this->input->imagelistKey.GetFrame(true);
	if(!keyFrame)
	{
		this->in  = false;
		return;
	}
	void* fillFrame = this->input->imagelistFill.GetFrame(true);
	if(!fillFrame)
	{
		this->in  = false;
		return;
	}

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

void Input::sendOut(uint4* output)
{
	assert(output!=nullptr);
	uint4* data = new uint4[this->frameSizePacked];

	this->cudaStatus = cudaMemcpy(data, output, this->frameSizePacked, cudaMemcpyDeviceToHost);
	assert(this->cudaStatus==cudaSuccess);
	this->input->ImagelistOutput.AddFrame((void*)data);

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
	this->pVideo = video;
	this->pKey = key;
	this->pFill = fill;
	this->augVideo = nullptr;
	this->rgbVideo = nullptr;
	this->init();
}

void Preprocessor::init()
{

}

void Preprocessor::reload(uchar2* pv, uchar2* pk, uchar2* pf)
{
	this->pVideo = pv; this->pKey = pk; this->pFill = pf;
}

void Preprocessor::load(uint4* v, uint4* k, uint4* f, uint4* av, uchar3* rgb)
{
	this->video = v; this->key = k; this->fill = f; this->augVideo = av; this->rgbVideo = rgb;
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
			this->video,
			srcAlignedWidth,
			dstAlignedWidth,
			this->iHeight
		);
	this->cudaStatus = cudaGetLastError();
	this->checkCudaError("Launch Kernel", "Device");
	// Unpack yuv key from decklink and store it in yUnpackedCudaKey
	yuyvPackedToyuyvUnpacked <<<grid, block>>>(
				(uint4*)this->pKey,
				this->key,
				srcAlignedWidth,
				dstAlignedWidth,
				this->iHeight
			);
	this->cudaStatus = cudaGetLastError();
	this->checkCudaError("Launch Kernel", "Device");
	// Unpack yuv fill from decklink and store it in yUnpackedCudaFill
	yuyvPackedToyuyvUnpacked <<<grid, block>>>(
				(uint4*)this->pFill,
				this->fill,
				srcAlignedWidth,
				dstAlignedWidth,
				this->iHeight
			);
	this->cudaStatus = cudaGetLastError();
	this->checkCudaError("Launch Kernel", "Device");

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize device", " at unpacking");

	this->cudaStatus = cudaMemcpy(this->augVideo, this->video, this->frameSizeUnpacked, cudaMemcpyDeviceToDevice);
	this->checkCudaError("copy data", " augmented video buffer");
}



void Preprocessor::create()
{
//	cv::cuda::GpuMat input(this->iHeight, this->iWidth, CV_32SC4, this->augVideo, this->frameSizeUnpacked/this->iHeight);
//	cv::cuda::GpuMat output;//(this->iHeight, this->iWidth, CV_32SC4);
//
//	Ptr<Filter> gaus = cv::cuda::createGaussianFilter(input.type(), input.type(), cv::Size(13,13), -1);
//	gaus->apply(input, output);
	const int srcAlignedWidth = this->iWidth/2;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(srcAlignedWidth, block.x), iDivUp(this->iHeight, block.y));
	gammaCorrect<<<grid, block>>>(
			this->augVideo,
			srcAlignedWidth,
			this->iHeight,
			0.45
	);

	this->cudaStatus = cudaDeviceSynchronize();
	assert(this->cudaStatus==cudaSuccess);

//	std::vector<cv::cuda::GpuMat> res;
//	cv::cuda::split(input, res);
//
//	Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE();

}

LookupTable::LookupTable(IPipeline *obj): IPipeline(obj)
{
	this->lookupBuffer = nullptr;
	this->loaded = false;
	this->snapShot = nullptr;
}

void LookupTable::create()
{
	this->loaded = false;
}

void LookupTable::update(bool clickEn, MouseData md, std::unordered_map<std::string, int> ws)
{
	if(!clickEn)return;

	if (md.bHandleLDown)
	{
		this->loaded = false;
		int maxRecSize = 200;
		float ScalingValue = maxRecSize*1.0/ws[WINDOW_TRACKBAR_OUTER_DIAM]*1.0;

		const dim3 block(16, 16);
		const dim3 grid(
						iDivUp((ws[WINDOW_TRACKBAR_OUTER_DIAM]+ws[WINDOW_TRACKBAR_UV_DIAM])*2, block.x),
						iDivUp((ws[WINDOW_TRACKBAR_OUTER_DIAM]+ws[WINDOW_TRACKBAR_UV_DIAM])*2, block.y)
						);

		for (int x = (md.iXUpDynamic / 2); x<(md.iXDownDynamic /2); x++)
		{
			for (int y = md.iYUpDynamic; y < md.iYDownDynamic; y=y+2)
			{
				UpdateLookupFrom_XY_Posision_Diffrent_Scaling <<<grid, block>>> (
						this->snapShot,
						this->lookupBuffer,
						x, y,
						(this->iWidth / 2),
						ws[WINDOW_TRACKBAR_OUTER_DIAM]*2,
//						ws[WINDOW_TRACKBAR_UV_DIAM]*2,
						2,
						ws[WINDOW_TRACKBAR_LUM],
						ScalingValue,
						255
						);
				this->cudaStatus = cudaGetLastError();
				this->checkCudaError("Launch kernel", "Device");
			}
		}

		this->cudaStatus = cudaDeviceSynchronize();
		this->checkCudaError("synchronize host", " kernel: updateLookupFromMouse");
		assert(this->cudaStatus==cudaSuccess);
		this->loaded = true;
	}
}

void LookupTable::clearTable()
{
	this->cudaStatus = cudaMemset(this->lookupBuffer, 0, this->iWidth*this->iHeight*sizeof(uchar));
	assert(this->cudaStatus==cudaSuccess);
}

void IMask::init()
{

}

void IMask::erode(int size)
{
	cv::cuda::GpuMat chrommaMaskInput(this->iHeight,this->iWidth,CV_8UC1, this->maskBuffer, this->iWidth*sizeof(uchar));
	cv::cuda::GpuMat chrommaMaskOutput;

	// erode output mask
	int an = size;
	cv::Mat element = getStructuringElement(MORPH_RECT, Size(an*2+1, an*2+1), Point(an, an));
	Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter(MORPH_ERODE, chrommaMaskInput.type(), element);
	erodeFilter->apply(chrommaMaskInput, chrommaMaskOutput);

	chrommaMaskOutput.copyTo(chrommaMaskInput);
}

void IMask::dilate(int size)
{
	cv::cuda::GpuMat chrommaMaskInput(this->iHeight,this->iWidth,CV_8UC1, this->maskBuffer, this->iWidth*sizeof(uchar));
	cv::cuda::GpuMat chrommaMaskOutput;

	// Dilate the output mask
	int an = size;
	cv::Mat element = getStructuringElement(MORPH_ELLIPSE, Size(an*2+1, an*2+1));
	Ptr<cv::cuda::Filter> erodeFilter2 = cv::cuda::createMorphologyFilter(MORPH_DILATE, chrommaMaskInput.type(), element);
	erodeFilter2->apply(chrommaMaskInput, chrommaMaskOutput);

	chrommaMaskOutput.copyTo(chrommaMaskInput);
}

void IMask::openMorph(int size)
{
	cv::cuda::GpuMat chrommaMaskInput(this->iHeight,this->iWidth,CV_8UC1, this->maskBuffer, this->iWidth*sizeof(uchar));
	cv::cuda::GpuMat chrommaMaskOutput;

	int an = size;
	cv::Mat element = getStructuringElement(MORPH_RECT, Size(an*2+1, an*2+1));
	Ptr<cv::cuda::Filter> openingFilter = cv::cuda::createMorphologyFilter(MORPH_OPEN, chrommaMaskInput.type(), element, cv::Point(-1, -1), 2);
	openingFilter->apply(chrommaMaskInput, chrommaMaskOutput);

	chrommaMaskOutput.copyTo(chrommaMaskInput);

}

ChrommaMask::ChrommaMask(IPipeline* obj, LookupTable* t): IMask(obj)
{
	this->table = t;
}

void ChrommaMask::create()
{
	if(!table->isLoaded())return;
	this->mask = false;
	const int dstAlignedWidth = this->iWidth;
	const int srcAlignedWidth = this->iWidth/2;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(srcAlignedWidth, block.x), iDivUp(this->iHeight, block.y));

	yuyv_Unpacked_GenerateMask <<<grid, block>>> (
			(uint4*)this->augVideo,
			this->maskBuffer,
			this->table->output(),
			this->iWidth,
			this->iHeight,
			srcAlignedWidth,
			dstAlignedWidth,
			0
			);
	this->cudaStatus = cudaGetLastError();
	assert(this->cudaStatus==cudaSuccess);

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize host", "yuyvGenerateMask");
	assert((this->cudaStatus == cudaSuccess));

	this->mask = true;
}

uchar* ChrommaMask::output()
{
	this->create();
	if(!this->mask)return nullptr;
	this->update(); // clean it up and post-process it.
	return this->maskBuffer;
}

void ChrommaMask::toRGB(uchar3* rgb/*Cuda Pointer*/)
{
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->iWidth/2, block.x), iDivUp(this->iHeight, block.y));
	const int dstAlignedWidth = this->iWidth;

	Msk2RGB <<<grid, block>>> (
			this->maskBuffer,
			this->maskBuffer,
			this->maskBuffer,
			rgb,
			this->iWidth/2, // source aligned width
			dstAlignedWidth,
			this->iHeight
			);
	this->cudaStatus = cudaGetLastError();
	assert(this->cudaStatus==cudaSuccess);

	this->cudaStatus = cudaDeviceSynchronize();
	assert(this->cudaStatus==cudaSuccess);
}

void ChrommaMask::toRGB()
{
	if(this->maskRGB==nullptr) return;
	if(!this->mask) return;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->iWidth/2, block.x), iDivUp(this->iHeight, block.y));
	const int dstAlignedWidth = this->iWidth;

	Msk2RGB <<<grid, block>>> (
			this->maskBuffer,
			this->maskBuffer,
			this->maskBuffer,
			this->maskRGB,
			this->iWidth/2, // source aligned width
			dstAlignedWidth,
			this->iHeight
			);
	this->cudaStatus = cudaGetLastError();
	assert(this->cudaStatus==cudaSuccess);

	this->cudaStatus = cudaDeviceSynchronize();
	assert(this->cudaStatus==cudaSuccess);
}


void MaskOut::create()
{
	// perform some mask creation here ...
}

uchar* MaskOut::output()
{
	return this->chromma->output();
}


Keyer::Keyer(IPipeline* obj, uchar* mask): IPipeline(obj)
{
	this->finalMask = mask;
	this->parabolic = calc_parabola_vertex(0, 0, 512, 1, 1024, 0);
	this->packed = nullptr;
}

void Keyer::create(int blend=480)
{

	const int dstAlignedWidth = (this->iWidth / 2);
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(dstAlignedWidth, block.x), iDivUp(this->iHeight, block.y));
	const int maskWidth = this->iWidth;

	keyAndFill<<<grid, block>>>(
			this->video, // Remember to replace with video after testing...
			this->fill,
			this->key,
			this->iWidth,
			this->iHeight,
			dstAlignedWidth,
			maskWidth,
			this->finalMask,
			blend,
			this->parabolic
		);
	this->cudaStatus = cudaGetLastError();
	assert(this->cudaStatus==cudaSuccess);

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize kernel", "Device");
	assert(this->cudaStatus==cudaSuccess);

}


void Keyer::pack()
{
	assert(this->packed!=nullptr);
	const int srcAlignedWidth = (this->iWidth / 2);
	const int dstAlignedWidth = this->rowLength/16;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(srcAlignedWidth, block.x), iDivUp(this->iHeight, block.y));

	yuyvUnPackedToyuyvpacked<<<grid, block>>>(
			this->packed,
			this->video,
			dstAlignedWidth,
			srcAlignedWidth,
			this->iHeight
	);

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize kernel", "Device");
	assert(this->cudaStatus==cudaSuccess);
}

void Pipeline::run()
{
	this->load();
	this->viewPipeline();
	KeyingWindow *keyingWindow = (KeyingWindow*)this->container->getWindow(WINDOW_NAME_KEYING);
	assert(keyingWindow!=nullptr);
	SettingsWindow *settings = (SettingsWindow*)this->container->getWindow(WINDOW_NAME_SETTINGS);
	assert(settings != nullptr);
	WindowI *maskPreview = this->container->getWindow(WINDOW_NAME_MASK);
	WindowI *outputWindow = this->container->getWindow(WINDOW_NAME_OUTPUT);
	WindowI *main = this->container->getWindow(WINDOW_NAME_MAIN);

	Preview prev(this->preproc);

	ColorBalancer colB;

	ContourDetector cd;


	int outputCounter = 0;
	cv::cuda::GpuMat rgb;
	rgb.create(preproc->getHeight(), preproc->getWidth(), CV_8UC3);
	rgb.step = 5760;

	cv::Mat dd;

	while(event!= WINDOW_EVENT_EXIT)
	{
		event = this->container->getEvent();
		input->run(settings->getTrackbarValues()[WINDOW_TRACKBAR_DELAY]);

		if(input->isOutput())
		{
			outputCounter = 0;
			preproc->reload(input->getPVideo(), input->getPKey(), input->getPFill());
			preproc->unpack();
			preproc->create();
//			preproc->convertToRGB();
////
//			rgb.data = (uchar*)preproc->getRGB();
////
//			rgb.download(dd);
//
//			colB.setImage(&dd);
//			colB.Brightness(settings->getTrackbarValues()[WINDOW_TRACKBAR_BRIGHTNESS]);
//			colB.saturation(settings->getTrackbarValues()[WINDOW_TRACKBAR_SAT]);
//			colB.finish();
//
//			rgb.upload(dd);
//			yoloMask->getBatch();

//			prev.load(preproc->getRGB());
//			prev.preview(main->getHandle());

			switch(event)
			{
			case WINDOW_EVENT_CAPTURE:
				snapShot->convertToRGB();

//				rgb.data = (uchar*)snapShot->getRGB();
//				rgb.download(dd);
//
//				colB.setImage(&dd);
//				colB.Brightness(settings->getTrackbarValues()[WINDOW_TRACKBAR_BRIGHTNESS]);
//				colB.saturation(settings->getTrackbarValues()[WINDOW_TRACKBAR_SAT]);
//				colB.finish();
//
//				rgb.upload(dd);

				snapShot->takeSnapShot();

				keyingWindow->loadImage(snapShot->getSnapShot());
				keyingWindow->captured();
				keyingWindow->show();

				chrommaMask->output();
				if(chrommaMask->isMask())
				{
//					cd.setImage(chrommaMask->getMask());
//					bool changed = true;
//					cd.detect(changed);
					chrommaMask->dilate(settings->getTrackbarValues()[WINDOW_TRACKBAR_DILATE]);
					chrommaMask->erode(settings->getTrackbarValues()[WINDOW_TRACKBAR_ERODE]);
					chrommaMask->openMorph(settings->getTrackbarValues()[WINDOW_TRACKBAR_ERODE]);
					chrommaMask->toRGB();

					prev.load(chrommaMask->getMaskRGB());
					prev.preview(maskPreview->getHandle());
//					keyer->create(settings->getTrackbarValues()[WINDOW_TRACKBAR_BLENDING]);
				}

				break;

			case WINDOW_EVENT_SAVE_IMAGE:

				break;

			}

			if(chrommaMask->isMask())
			{

				chrommaMask->output();
				chrommaMask->dilate(settings->getTrackbarValues()[WINDOW_TRACKBAR_DILATE]);
				chrommaMask->erode(settings->getTrackbarValues()[WINDOW_TRACKBAR_ERODE]);
//				chrommaMask->openMorph(settings->getTrackbarValues()[WINDOW_TRACKBAR_ERODE]);

				#ifndef DEBUG
				chrommaMask->toRGB();
				prev.load(chrommaMask->getMaskRGB());
				prev.preview(maskPreview->getHandle());
				keyer->create(settings->getTrackbarValues()[WINDOW_TRACKBAR_BLENDING]);
				keyer->pack();
				input->sendOut(keyer->getOutput());
//				keyer->convertToRGB(keyer->getVideo());
//				prev.load(keyer->getRGB());
//				prev.preview(outputWindow->getHandle());
				#endif
			}

			if(keyingWindow->isCaptured())// frame is captured
			{
				keyingWindow->update();
				lookup->update(keyingWindow->isCaptured(), keyingWindow->getMD(), settings->getTrackbarValues());
			}
		}
		else
		{
			outputCounter ++;
		}
	}
}

void Pipeline::addPipelineObject(IPipeline* obj, std::string name)
{
	try{
		this->pipelineObjects[name] = obj;
		this->availableObjects.push_back(name);
	}catch(std::exception &e){

	}
}


void Pipeline::load()
{
	try{
		preproc = (Preprocessor*)this->pipelineObjects[OBJECT_PREPROCESSOR];
		snapShot = (SnapShot*) this->pipelineObjects[OBJECT_SNAPSHOT];
		lookup = (LookupTable*) this->pipelineObjects[OBJECT_LOOKUP];
		chrommaMask = (ChrommaMask*)this->pipelineObjects[OBJECT_CHROMMA_MASK];
		input = (Input*) this->pipelineObjects[OBJECT_INPUT];
		yoloMask = (YoloMask*)this->pipelineObjects[OBJECT_YOLO_MASK];
		keyer = (Keyer*)this->pipelineObjects[OBJECT_KEYER];
	}catch(std::exception& err){
		// Do some logging here
		std::cerr<<"[Error]: Failed to load objects into the pipeline"<<std::endl;
		exit(-1);
	}
	this->assertObjects();
}

void Pipeline::assertObjects()
{
	assert(preproc!=nullptr);
	assert(snapShot!=nullptr);
	assert(lookup!=nullptr);
	assert(chrommaMask!=nullptr);
	assert(input!=nullptr);
	assert(yoloMask!=nullptr);
	assert(keyer!=nullptr);
	assert(mtx!=nullptr);
}

void Pipeline::viewPipeline()
{
	std::cout<<"[info]: Printing all available objects in the Pipeline: "<<std::endl;
	for(std::string& obj: this->availableObjects)
	{
		std::cout<<"\tName:\t"<<obj<<std::endl;
	}
}




void allocateMemory(void** devptr, long int size)
{
	cudaError_t cudaStatus = cudaMalloc(devptr, size);
	assert(cudaStatus==cudaSuccess);
}

void startPipeline(Pipeline* pipeline)
{
	pipeline->run();
}


