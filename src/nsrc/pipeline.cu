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
#include <ui.hpp>

/**** Utils *****/
inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }


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

//	std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
//	std::cout<<"Video: "<<videoFrame<<std::endl;
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
		float ScalingValue = maxRecSize*1.0/ws["Outer Diam"]*1.0;

		const dim3 block(16, 16);
		const dim3 grid(
						iDivUp((ws["Outer Diam"]+ws["UV Diam"])*2, block.x),
						iDivUp((ws["Outer Diam"]+ws["UV Diam"])*2, block.y)
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
						1,10,5,
//						ws["Outer Diam"]*2,
//						ws["UV Diam"]*2,
//						ws["E Lum"],
						ScalingValue,
						maxRecSize
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


Keyer::Keyer(IPipeline* obj, uchar* mask): IPipeline(obj)
{
	this->finalMask = mask;
	this->parabolic = calc_parabola_vertex(0, 0, 512, 1, 1024, 0);
}

void Keyer::create()
{

	const int dstAlignedWidth = (this->iWidth / 2);
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(dstAlignedWidth, block.x), iDivUp(this->iHeight, block.y));
	const int maskWidth = this->iWidth;

	keyAndFill<<<grid, block>>>(
			this->augVideo, // Remember to replace with video after testing...
			this->fill,
			this->key,
			this->iWidth,
			this->iHeight,
			dstAlignedWidth,
			maskWidth,
			this->finalMask,
			480,
			this->parabolic
		);
	this->cudaStatus = cudaGetLastError();
	assert(this->cudaStatus==cudaSuccess);

	this->cudaStatus = cudaDeviceSynchronize();
	assert(this->cudaStatus==cudaSuccess);

}



inline void allocateMemory(void** devptr, long int size)
{
	cudaError_t cudaStatus = cudaMalloc(devptr, size);
	assert(cudaStatus==cudaSuccess);
}

void startPipeline()
{
	uchar *chrommaLookupBuffer, *chrommaMask;
	uchar2 *pVideo, *pKey, *pFill;
	uchar3* rgbVideo, *vSnapshot, *maskRGB;
	uint4 *video, *key, *fill, *aVideo, *snapShotV;

	VideoIn decklink;

	Input *in = new Input(&decklink);

	/*************************************************************************************
	 * This is for memory alignment                                                      *
	 * It seems allocating device memory inside an object causes misalignment,           *
	 * Reason:                                                                           *
	 * 	still need to read more and find out why.                                        *
	 * Solution:                                                                         *
	 * 	Declare memory outside and load it inside, but keep the rest of the flow fixed.  *
	 *************************************************************************************/
	allocateMemory((void**)&pVideo,in->getPFrameSize());
	allocateMemory((void**)&pKey, in->getPFrameSize());
	allocateMemory((void**)&pFill, in->getPFrameSize());

	allocateMemory((void**)&video, in->getFrameSize());
	allocateMemory((void**)&key, in->getFrameSize());
	allocateMemory((void**)&fill, in->getFrameSize());
	allocateMemory((void**)&aVideo, in->getFrameSize());
	allocateMemory((void**)&snapShotV, in->getFrameSize());
	allocateMemory((void**)&vSnapshot, in->getWidth()*in->getHeight()*sizeof(uchar3));
	allocateMemory((void**)&rgbVideo, in->getWidth()*in->getHeight()*sizeof(uchar3));
	allocateMemory((void**)&maskRGB, in->getWidth()*in->getHeight()*sizeof(uchar3));

	allocateMemory((void**)&chrommaMask, in->getWidth()*in->getHeight()*sizeof(uchar));
	allocateMemory((void**)&chrommaLookupBuffer, 1024*1024*1024);

	WindowsContainer uiContainer;

	WindowI mainWindow("Main"); // plays the video playback

	KeyingWindow keyingWindow("Keying Window", in->getWidth(), in->getHeight()); // keying window

	SettingsWindow settings("Setting"); // settings

	WindowI maskPreview("Mask Preview");

	keyingWindow.enableMouse();

	uiContainer.addWindow(&mainWindow);
	uiContainer.addWindow(&keyingWindow);
	uiContainer.addWindow(&settings);

	in->load(pVideo, pKey, pFill);

	in->run();

	if(in->isOutput())
	{
		Preprocessor *pp = new Preprocessor(in, in->getPVideo(), in->getPKey(), in->getPFill());
		pp->load(video, key, fill, aVideo, rgbVideo);

		SnapShot *ss = new SnapShot(pp);
		ss->load(vSnapshot, snapShotV);

		Preview *prev = new Preview(ss);
		prev->load(ss->getSnapShot());

		LookupTable *lt = new LookupTable(ss);
		lt->load(chrommaLookupBuffer, snapShotV);

		ChrommaMask *cm = new ChrommaMask(pp, lt);
		cm->load(chrommaMask, maskRGB);

		Keyer *keyer = new Keyer(pp, cm->getMask());

		while(uiContainer.dispatchKey() != 27)
		{
			in->run();
			pp->reload(in->getPVideo(), in->getPKey(), in->getPFill());
			pp->unpack();
			pp->convertToRGB();

			prev->load(pp->getRGB());
			prev->preview(mainWindow.getHandle());

			if(uiContainer.getKey() == 'q')
			{
				ss->takeSnapShot();
				keyingWindow.loadImage(ss->getSnapShot());
				keyingWindow.show();
				cm->output();
				if(cm->isMask())
				{
					cm->toRGB();
					cv::cuda::GpuMat mat;
					mat.create(pp->getHeight(), pp->getWidth(), CV_8UC3);
					mat.step = 5760;
					mat.data = (uchar*)cm->getMaskRGB();

					cv::Mat prev;
					mat.download(prev);
					cv::imshow("Mask Preview", prev);
				}
			}

			if(cm->isMask())
			{
				keyer->create();
				keyer->convertToRGB();
				prev->load(keyer->getRGB());
				prev->preview(mainWindow.getHandle());
			}

			if(keyingWindow.isCaptured())// frame is captured
			{
				lt->update(keyingWindow.isCaptured(), keyingWindow.getMD(), settings.getTrackbarValues());
			}

			keyingWindow.update();
		}
	}

	std::cout<<"Pipeline finished"<<std::endl;
}


