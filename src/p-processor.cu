
#include <p-processor.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <InputLoopThrough.h>
#include <cuda_runtime.h>
#include <YUVUChroma.cuh>
#include <stdio.h>


/**** Utils *****/
inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

/***************/
void PipelineObj::checkCudaError(std::string action, std::string loc)
{
	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to "<< action<<" to"<< loc <<" \n"
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
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&this->yPackedCudaFill, this->frameSizePacked);
	if(cudaStatus != cudaSuccess){
		this->checkCudaError("Allocate memory", " yPackedCudaFill");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yPackedCudaKey, this->frameSizePacked);
	if(cudaStatus != cudaSuccess){
		this->checkCudaError("Allocate memory", " yPackedCudaKey");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yPackedCudaVideo, this->frameSizePacked);
	if(cudaStatus != cudaSuccess){
		this->checkCudaError("Allocate memory", " yPackedCudaVideo");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yUnpackedCudaFill, this->frameSizeUnpacked);
	if(cudaStatus != cudaSuccess){
		this->checkCudaError("Allocate memory", " yUnpackedCudaFill");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yUnpackedCudaKey, this->frameSizeUnpacked);
	if(cudaStatus != cudaSuccess){
		this->checkCudaError("Allocate memory", " yUnpackedCudaKey");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yUnpackedCudaVideo, this->frameSizeUnpacked);
	if(cudaStatus != cudaSuccess){
		this->checkCudaError("Allocate memory", " yUnpackedCudaVideo");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->cudaRGB, this->iWidth*this->iHeight*sizeof(uchar3));
	if(cudaStatus != cudaSuccess){
		this->checkCudaError("Allocate memory", " cudaRGB");
		exit(-1);
	}

	std::cout<<"[Info]: Finished initializing cuda variables\n"<<std::endl;
}


void Processor::sendDataTo()
{
	// read video from the deck Link card and send it to cuda
	// retrive frame if there's one to retrieve
	this->deckLinkInput->WaitForFrames(this->iDelayFrames);
	bool popVid = this->iDelayFrames <= this->deckLinkInput->imagelistVideo.GetFrameCount();
	static void* videoFrame;

	if(videoFrame)
		free(videoFrame);

	videoFrame = this->deckLinkInput->imagelistVideo.GetFrame(true);
	void* keyFrame = this->deckLinkInput->imagelistKey.GetFrame(true);
	void* fillFrame = this->deckLinkInput->imagelistFill.GetFrame(true);

	cudaError_t cudaStatus;

	if(videoFrame && keyFrame && fillFrame)
	{
		if(this->toCuda((void*)videoFrame,(void*)this->yPackedCudaVideo, this->frameSizePacked))
		{
			this->checkCudaError("Copy data", " yPackedCudaVideo");
			exit(-1);
		}

		if(this->toCuda(keyFrame, this->yPackedCudaKey, this->frameSizePacked))
		{
			this->checkCudaError("Copy data", " yPackedCudaKey");
			exit(-1);
		}

		if(this->toCuda(fillFrame, this->yPackedCudaFill, this->frameSizePacked))
		{
			this->checkCudaError("Copy data", " yPackedCudaFill");
			exit(-1);
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

	this->cudaReset();
	const int srcAlignedWidth = this->deckLinkInput->m_RowLength/SIZE_ULONG4_CUDA;
	const int dstAlignedWidth = this->iWidth/2;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(dstAlignedWidth, block.x), iDivUp(this->iHeight, block.y));
//

	uint4* video2;
	cudaMalloc(&video2, this->frameSizeUnpacked);

	yuyvUmPackedToRGB_lookup <<<grid, block , 0, this->stream>>> (
			(uint4*)this->yUnpackedCudaVideo,
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
	int i = 0;
	this->cudaStatus = cudaMalloc(this->chromaGeneratedMask, 3*sizeof(uchar*));
	this->checkCudaError("Allocate memory", "chromeGeneratedMask");

	for(i=0; i<3; i++)
	{
		this->cudaStatus = cudaMalloc(this->chromaGeneratedMask, this->iHeight*this->iWidth*sizeof(uchar));
		this->checkCudaError("Allocate memory", "chromeGeneratedMask");
	}
}
