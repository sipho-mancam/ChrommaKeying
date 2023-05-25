
#include <p-processor.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <InputLoopThrough.h>
#include <cuda_runtime.h>
#include <YUVUChroma.cuh>
#include <stdio.h>
#include <opencv2/cudafilters.hpp>


/**** Utils *****/
inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

/***************/
void PipelineObj::checkCudaError(std::string action, std::string loc)
{
	if(this->cudaStatus != cudaSuccess)
	{
		std::cerr<<"[Error]: Failed to "<< action<<" to "<< loc <<" \n"
				<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)<<std::endl;
	}
}

bool PipelineObj::toCuda(void* src, void* dst, long int size)
{
	this->cudaStatus = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	return this->cudaStatus != cudaSuccess;
}


void Processor::cudaInit()
{
	cudaStatus = cudaMalloc((void**)&this->yPackedCudaFill, this->frameSizePacked);
	this->checkCudaError("Allocate memory", " yPackedCudaFill");
	assert(this->cudaStatus==cudaSuccess);

	cudaStatus = cudaMalloc((void**)&this->yPackedCudaKey, this->frameSizePacked);
	this->checkCudaError("Allocate memory", " yPackedCudaKey");
	assert(this->cudaStatus==cudaSuccess);

	cudaStatus = cudaMalloc((void**)&this->yPackedCudaVideo, this->frameSizePacked);
	this->checkCudaError("Allocate memory", " yPackedCudaVideo");
	assert(this->cudaStatus==cudaSuccess);

	cudaStatus = cudaMalloc((void**)&this->yUnpackedCudaFill, this->frameSizeUnpacked);
	this->checkCudaError("Allocate memory", " yUnpackedCudaFill");
	assert(this->cudaStatus==cudaSuccess);

	cudaStatus = cudaMalloc((void**)&this->yUnpackedCudaKey, this->frameSizeUnpacked);
	this->checkCudaError("Allocate memory", " yUnpackedCudaKey");
	assert(this->cudaStatus==cudaSuccess);

	cudaStatus = cudaMalloc((void**)&this->yUnpackedCudaVideo, this->frameSizeUnpacked);
	this->checkCudaError("Allocate memory", " yUnpackedCudaVideo");
	assert(this->cudaStatus==cudaSuccess);

	cudaStatus = cudaMalloc((void**)&this->cudaRGB, this->iWidth*this->iHeight*sizeof(uchar3));
	this->checkCudaError("Allocate memory", " cudaRGB");
	assert(this->cudaStatus==cudaSuccess);

	cudaStatus = cudaMalloc((void**)&this->videoSnapshot, this->frameSizeUnpacked);
	this->checkCudaError("Allocate memory", "videoSnapshot");
	assert(this->cudaStatus==cudaSuccess);

	// log some shit here ...
	std::cout<<"[Info]: Finished initializing cuda variables\n"<<std::endl;
}


void Processor::sendDataTo(bool pop = true)
{
	// read video from the deck Link card and send it to cuda
	// retrive frame if there's one to retrieve
	this->deckLinkInput->WaitForFrames(this->iDelayFrames);
	bool popVid = this->iDelayFrames <= this->deckLinkInput->imagelistVideo.GetFrameCount();
	static void* videoFrame;

	if(videoFrame)
		free(videoFrame);

	videoFrame = this->deckLinkInput->imagelistVideo.GetFrame(pop);
	void* keyFrame = this->deckLinkInput->imagelistKey.GetFrame(pop);
	void* fillFrame = this->deckLinkInput->imagelistFill.GetFrame(pop);


	if(videoFrame && keyFrame && fillFrame)
	{
		if(this->toCuda((void*)videoFrame,(void*)this->yPackedCudaVideo, this->frameSizePacked))
		{
			this->checkCudaError("Copy data", " yPackedCudaVideo");
			assert(this->cudaStatus==cudaSuccess);
		}

		if(this->toCuda(keyFrame, this->yPackedCudaKey, this->frameSizePacked))
		{
			this->checkCudaError("Copy data", " yPackedCudaKey");
			assert(this->cudaStatus==cudaSuccess);
		}

		if(this->toCuda(fillFrame, this->yPackedCudaFill, this->frameSizePacked))
		{
			this->checkCudaError("Copy data", " yPackedCudaFill");
			assert(this->cudaStatus==cudaSuccess);
		}
	}

	if(fillFrame)
		free(fillFrame);

	if(keyFrame)
		free(keyFrame);

//	std::cout<<"[info]: Done sending data to cuda"<<std::endl;
}


void Processor::unpackYUV()
{
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->deckLinkInput->m_RowLength/SIZE_ULONG4_CUDA, block.x), iDivUp(this->iHeight, block.y));
	const int srcAlignedWidth = this->deckLinkInput->m_RowLength/SIZE_ULONG4_CUDA;
	const int dstAlignedWidth = this->iWidth/2;

	// Unpack yuv video from decklink and store it in yUnpackedCudaVideo
	yuyvPackedToyuyvUnpacked <<<grid, block, 0, this->stream>>>(
			(uint4*)this->yPackedCudaVideo,
			(uint4*)this->yUnpackedCudaVideo,
			srcAlignedWidth,
			dstAlignedWidth,
			this->iHeight
		);

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize device", " at unpacking");
	// Unpack yuv key from decklink and store it in yUnpackedCudaKey
	yuyvPackedToyuyvUnpacked <<<grid, block, 0, this->stream>>>(
				(uint4*)this->yPackedCudaKey,
				(uint4*)this->yUnpackedCudaKey,
				srcAlignedWidth,
				dstAlignedWidth,
				this->iHeight
			);
	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize device", " at unpacking");

	// Unpack yuv fill from decklink and store it in yUnpackedCudaFill
	yuyvPackedToyuyvUnpacked <<<grid, block, 0, this->stream>>>(
				(uint4*)this->yPackedCudaFill,
				(uint4*)this->yUnpackedCudaFill,
				srcAlignedWidth,
				dstAlignedWidth,
				this->iHeight
			);

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize device", " at unpacking");
}

void Processor::snapshot(cv::cuda::GpuMat* RGBData)
{
	if(this->mtx)
		this->mtx->lock();

	this->takeSnapShot();

	this->cudaReset();
	const int srcAlignedWidth = this->deckLinkInput->m_RowLength/SIZE_ULONG4_CUDA;
	const int dstAlignedWidth = this->iWidth/2;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(dstAlignedWidth, block.x), iDivUp(this->iHeight, block.y));
//

	uint4* video2;
	cudaMalloc(&video2, this->frameSizeUnpacked);

	yuyvUmPackedToRGB_lookup <<<grid, block , 0, this->stream>>> (
			(uint4*)this->videoSnapshot,
			this->cudaRGB,
			dstAlignedWidth,
			this->iWidth,
			this->iHeight,
//			(uint4*)this->yUnpackedCudaKey,
			video2,
			nullptr // this variable is not used in the function
		);
//
	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize device", " at yUyVUnpackedToRGB");

	this->cudaStatus = cudaMemcpy(RGBData->data, (uchar*)this->cudaRGB, this->iWidth*this->iHeight*sizeof(uchar3), cudaMemcpyDeviceToDevice);
	this->checkCudaError("copy memory", " from cudaRGB to RGBData->data");

//	std::cout<<"[info]: Finished copying snapshot\n";
	cudaFree(video2);

	if(this->mtx)
		this->mtx->unlock();
}

void Processor::takeSnapShot()
{
	this->cudaStatus = cudaMemcpy(this->videoSnapshot, this->yUnpackedCudaVideo,  this->frameSizeUnpacked, cudaMemcpyDeviceToDevice);
	this->checkCudaError("Copy memory", "videoSnapshot");
}

uchar2* Processor::getSnapShot()
{
	return this->videoSnapshot;
}

void Processor::cudaReset()
{
	this->cudaStatus = cudaMemset(this->cudaRGB, 0, this->iHeight*this->iWidth*sizeof(uchar));
	this->checkCudaError("reset", "cudaRGB");
}

void Processor::cudaCleanup()
{
	this->cudaStatus = cudaFree(this->yPackedCudaFill);
	this->checkCudaError("free", "yPackedCudaFill");

	this->cudaStatus = cudaFree(this->yPackedCudaVideo);
	this->checkCudaError("free", "yPackedCudaVideo");

	this->cudaStatus = cudaFree(this->yPackedCudaKey);
	this->checkCudaError("free", "yUnpackedCudaKey");

	this->cudaStatus = cudaFree(this->yUnpackedCudaFill);
	this->checkCudaError("free", "yUnpackedCudaFill");

	this->cudaStatus = cudaFree(this->yUnpackedCudaVideo);
	this->checkCudaError("free", "yUnpackedCudaVideo");

	this->cudaStatus = cudaFree(this->yUnpackedCudaKey);
	this->checkCudaError("free", "yUnpackadCudaKey");

	this->cudaStatus = cudaFree(this->cudaRGB);
	this->checkCudaError("free", "cudaRGB");

}

void Processor::run()
{
	if(this->mtx)
		this->mtx->lock();

	this->sendDataTo();
	this->unpackYUV();
	if(this->mtx)
		this->mtx->unlock();
}


void ChrommaKey::cudaInit()
{
	this->lookupTable = new uchar*[MAX_LOOK_UP];
	this->chromaGeneratedMask = new uchar*[MAX_LOOK_UP];

	this->cudaStatus = cudaMalloc(&this->maskDown, this->iHeight*this->iWidth*sizeof(uchar));
	this->checkCudaError("allocate memory", "maskDown");

	int i = 0;
	for(i=0; i<MAX_LOOK_UP; i++)
	{
		this->cudaStatus = cudaMalloc((void**)&this->chromaGeneratedMask[i], this->iHeight*this->iWidth*sizeof(uchar));
		this->checkCudaError("Allocate memory", "chromeGeneratedMask child");
		assert(this->cudaStatus==cudaSuccess);

		this->cudaStatus = cudaMalloc((void**)&this->lookupTable[i], CUDA_LOOKUP_SIZE);
		this->checkCudaError("Allocate memory", "LookupTable child");
		assert(this->cudaStatus==cudaSuccess);
	}

}

void ChrommaKey::cudaCleanup()
{
	int i = 0;

	for(i=0; i<MAX_LOOK_UP; i++)
	{
		this->cudaStatus = cudaFree(this->chromaGeneratedMask[i]);
		this->checkCudaError("Free memory", "chromeGeneratedMask child");

		this->cudaStatus = cudaFree(this->lookupTable[i]);
		this->checkCudaError("Free memory", "LookupTable child");
	}

	this->cudaStatus = cudaFree(this->chromaGeneratedMask);
	this->checkCudaError("Free memory", "chromeGeneratedMask");

	this->cudaStatus = cudaFree(this->lookupTable);
	this->checkCudaError("Free memory", "LookupTable");

	std::cout<<"[Info]: Finished initializing cuda variables\n"<<std::endl;
}

void ChrommaKey::generateChrommaMask()
{
	this->mtx->lock();
	this->video = this->proc->getSnapShot();

	const int dstAlignedWidth = this->iWidth;
	const int srcAlignedWidth = this->iWidth/2;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(srcAlignedWidth, block.x), iDivUp(this->iHeight, block.y));

	yuyv_Unpacked_GenerateMask <<<grid, block, 0, this->stream>>> (
			(uint4*)this->video,
			this->chromaGeneratedMask[0],
			this->lookupTable[0],
			this->iWidth,
			this->iHeight,
			srcAlignedWidth,
			dstAlignedWidth,
			0
			);

	this->cudaStatus = cudaDeviceSynchronize();
	this->checkCudaError("synchronize host", "yuyvGenerateMask");
	assert((this->cudaStatus == cudaSuccess));
	this->mtx->unlock();
}

void ChrommaKey::maskPreview( cv::cuda::GpuMat& mask , int index)
{
	this->generateChrommaMask();
	// do some thread safety here...

	assert(this->cudaStatus==cudaSuccess);

	this->cudaStatus = cudaMemcpy(maskDown, this->chromaGeneratedMask[0], this->iHeight*this->iWidth*sizeof(uchar), cudaMemcpyDeviceToDevice);
	this->checkCudaError("copy memory", "maskDown");

	mask.data=maskDown;
}

void ChrommaKey::erodeAndDilate(int iErode, int iDilate)
{
	cv::cuda::GpuMat chrommaMaskInput;
	cv::cuda::GpuMat chrommaMaskOutput(this->iWidth/2,this->iHeight*2,CV_8UC1, this->chromaGeneratedMask[0],Mat::CONTINUOUS_FLAG);
	chrommaMaskOutput.step=this->iWidth*2;

	// erode output mask
	int an = iErode;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(an*2+1, an*2+1), Point(an, an));
	Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter(MORPH_ERODE, chrommaMaskInput.type(), element);
	erodeFilter->apply(chrommaMaskInput, chrommaMaskOutput);

	// Dilate the output mask
	an = iDilate;
	element = getStructuringElement(MORPH_ELLIPSE, Size(an*2+1, an*2+1), Point(an, an));
	Ptr<cv::cuda::Filter> erodeFilter2 = cv::cuda::createMorphologyFilter(MORPH_DILATE, chrommaMaskInput.type(), element);
	erodeFilter2->apply(chrommaMaskInput, chrommaMaskOutput);
}

void ChrommaKey::updateLookup(bool clickEn,bool pb, MouseData md, WindowSettings ws)
{

	std::cout<<"[info]: Update Lookup started"<<std::endl;
//

	// get the snaphost
	this->video = this->proc->getSnapShot();
//	if(!init)return;
	if(!clickEn)return;

	if (md.bHandleLDown)
	{

		int maxRecSize = 200;
		float ScalingValue = maxRecSize*1.0/ws.m_iOuter_Diam*1.0;

		const dim3 block(16, 16);
		const dim3 grid(
				iDivUp((ws.m_iOuter_Diam+ws.m_iUV_Diam)*2, block.x),
				iDivUp((ws.m_iOuter_Diam+ws.m_iUV_Diam)*2, block.y)
				);

		uchar* ptrLookUpDataToUse = this->lookupTable[0];

		if(pb)
		{
			ptrLookUpDataToUse = this->lookupTable[1];
		}

		this->mtx->lock();
		std::cout<<"[info]: Thread locked ..."<<std::endl;

		for (int x = (md.iXUpDynamic / 2); x<(md.iXDownDynamic/2); x++)
		{
			for (int y = md.iYUpDynamic; y < md.iYDownDynamic; y=y+2)
			{
				UpdateLookupFrom_XY_Posision_Diffrent_Scaling <<<grid, block>>> (
						(uint4*)this->video,
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

		this->mtx->unlock();


	}
	std::cout<<"[info]: LookupTable updated successfully"<<std::endl;
//
}
