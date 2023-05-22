
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



void Processor::cudaInit()
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&this->yPackedCudaFill, this->frameSizePacked);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "Failed to Allocate Memory for: yPackedCudaFill");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yPackedCudaKey, this->frameSizePacked);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "Failed to Allocate Memory for: yPackedCudaKey");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yPackedCudaVideo, this->frameSizePacked);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "Failed to Allocate Memory for: yPackedCudaVideo");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yUnpackedCudaFill, this->frameSizeUnpacked);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "Failed to Allocate Memory for: yPackedCudaFill");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yUnpackedCudaKey, this->frameSizeUnpacked);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "Failed to Allocate Memory for: yPackedCudaKey");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->yUnpackedCudaVideo, this->frameSizeUnpacked);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "Failed to Allocate Memory for: yPackedCudaVideo");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&this->cudaRGB, this->iWidth*this->iHeight*sizeof(uchar3));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "Failed to Allocate Memory for: yPackedCudaVideo");
		exit(-1);
	}

	std::cout<<"[Info]: Finished initializing cuda variables\n"<<std::endl;
}

bool Processor::toCuda(void* src, void* dst, long int size)
{
	this->cudaStatus = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	return this->cudaStatus != cudaSuccess;
}

void Processor::sendDataTo()
{
	// read video from the deck Link card and send it to cuda
	// retrive frame if there's one to retrieve
	this->cudaCleanup();
	this->cudaInit();
	this->deckLinkInput->WaitForFrames(this->iDelayFrames);
//	if(this->deckLinkInput->imagelistVideo.GetFrameCount()<1)
//	{
//		std::cout<<"No video"<<std::endl;
//		return;
//	}
	void* videoFrame = this->deckLinkInput->imagelistVideo.GetFrame(this->iDelayFrames < this->deckLinkInput->imagelistVideo.GetFrameCount());
	void* keyFrame = this->deckLinkInput->imagelistKey.GetFrame(true);
	void* fillFrame = this->deckLinkInput->imagelistFill.GetFrame(true);
	cudaError_t cudaStatus;

	if(this->toCuda((void*)videoFrame,(void*)this->yPackedCudaVideo, this->frameSizePacked))
	{
		fprintf(stderr, "[Error]: Failed to Copy Video to GPU\n[Error]: %s\n", cudaGetErrorString(this->cudaStatus));
		std::cerr<<"Exiting ..."<<std::endl;
		exit(-1);
	}

	if(this->toCuda(keyFrame, this->yPackedCudaKey, this->frameSizePacked))
	{
		fprintf(stderr, "[Error]: Failed to Copy Key to GPU\n[Error]: %s\n", cudaGetErrorString(this->cudaStatus));
		std::cout<<"Exiting ..."<<std::endl;
		exit(-1);
	}

	if(this->toCuda(fillFrame, this->yPackedCudaFill, this->frameSizePacked))
	{
		fprintf(stderr, "[Error]: Failed to Copy Fill to GPU\n[Error]: %s\n", cudaGetErrorString(this->cudaStatus));
		std::cout<<"Exiting ..."<<std::endl;
		exit(-1);
	}

	std::cout<<"[info]: Done sending data to cuda"<<std::endl;
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
	// Unpack yuv key from decklink and store it in yUnpackedCudaKey
	yuyvPackedToyuyvUnpacked <<<grid, block, 0, this->stream>>>(
				(uint4*)this->yPackedCudaKey,
				(uint4*)this->yUnpackedCudaKey,
				srcAlignedWidth,
				dstAlignedWidth,
				this->iHeight
			);
	// Unpack yuv fill from decklink and store it in yUnpackedCudaFill
	yuyvPackedToyuyvUnpacked <<<grid, block, 0, this->stream>>>(
				(uint4*)this->yPackedCudaFill,
				(uint4*)this->yUnpackedCudaFill,
				srcAlignedWidth,
				dstAlignedWidth,
				this->iHeight
			);

	this->cudaStatus = cudaDeviceSynchronize();
}

void Processor::snapshot(cv::cuda::GpuMat* RGBData)
{
	this->cudaReset();
	const int srcAlignedWidth = this->deckLinkInput->m_RowLength/SIZE_ULONG4_CUDA;
	const int dstAlignedWidth = this->iWidth/2;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(dstAlignedWidth, block.x), iDivUp(this->iHeight, block.y));

	yuyvUmPackedToRGB_lookup <<<grid, block , 0, this->stream>>> (
			(uint4 *)this->yUnpackedCudaVideo,
			this->cudaRGB,
			dstAlignedWidth,
			this->iWidth,
			this->iHeight,
			(uint4 *)this->yUnpackedCudaKey,
			nullptr // this variable is not used in the function
		);

	this->cudaStatus = cudaDeviceSynchronize();
	if(this->cudaStatus!=cudaSuccess)fprintf(stderr, "[Error]: Failed to syn with Device: yuvutoRGB");

	this->cudaStatus = cudaMemcpy(RGBData->data, (uchar*)this->cudaRGB, this->iWidth*this->iHeight*sizeof(uchar3), cudaMemcpyDeviceToDevice);

	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to copy memory to RGBData->data"<<std::endl;
	}

	std::cout<<"[info]: Finished copying snapshot\n";
}

void Processor::cudaReset()
{
	this->cudaStatus = cudaMemset(this->cudaRGB, 0, this->iHeight*this->iWidth*sizeof(uchar));
	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to reset cudaRGB"<<std::endl;
		std::cout<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)<<std::endl;
	}
}

void Processor::cudaCleanup()
{
	this->cudaStatus = cudaFree(this->yPackedCudaFill);
	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to free yPackedCudaFill"<<std::endl;
		std::cout<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)<<std::endl;
	}

	this->cudaStatus = cudaFree(this->yPackedCudaVideo);
	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to free yPackedCudaVideo"<<std::endl;
		std::cout<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)<<std::endl;
	}

	this->cudaStatus = cudaFree(this->yPackedCudaKey);
	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to free yPackedCudaKey"<<std::endl;
		std::cout<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)<<std::endl;
	}

	this->cudaStatus = cudaFree(this->yUnpackedCudaFill);
	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to free yUnpackedCudaFill"<<std::endl;
		std::cout<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)<<std::endl;
	}

	this->cudaStatus = cudaFree(this->yUnpackedCudaVideo);
	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to free yUnpackedCudaVideo"<<std::endl;
		std::cout<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)<<std::endl;
	}

	this->cudaStatus = cudaFree(this->yUnpackedCudaKey);
	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to free yUnpackedCudaKey"<<std::endl;
		std::cout<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)<<std::endl;
	}

	this->cudaStatus = cudaFree(this->cudaRGB);
	if(this->cudaStatus != cudaSuccess)
	{
		std::cout<<"[Error]: Failed to free cudaRGB"<<std::endl;
		std::cout<<"[Error]: "<<cudaGetErrorString(this->cudaStatus)<<std::endl;
	}
}

void Processor::run()
{
	this->sendDataTo();
	this->unpackYUV();
}
