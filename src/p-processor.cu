
#include <p-processor.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <InputLoopThrough.h>
#include <cuda_runtime.h>
#include <YUVUChroma.cuh>


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
	this->deckLinkInput.WaitForFrames(this->iDelayFrames);

	void* videoFrame = this->deckLinkInput.imagelistVideo.GetFrame(this->iDelayFrames < this->deckLinkInput.imagelistVideo.GetFrameCount());
	void* keyFrame = this->deckLinkInput.imagelistKey.GetFrame(true);
	void* fillFrame = this->deckLinkInput.imagelistFill.GetFrame(true);
	cudaError_t cudaStatus;

	if(this->toCuda((void*)videoFrame,(void*)this->yPackedCudaVideo, this->frameSizePacked))
	{
		fprintf(stderr, "[Error]: Failed to Copy Video to GPU\n[Error]: %s\n", cudaGetErrorString(this->cudaStatus));
		std::cout<<"Exiting ..."<<std::endl;
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
}


void Processor::unpackYUV()
{
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(this->deckLinkInput.m_RowLength/SIZE_ULONG4_CUDA, block.x), iDivUp(this->iHeight, block.y));
	const int srcAlignedWidth = this->deckLinkInput.m_RowLength/SIZE_ULONG4_CUDA;
	const int dstAlignedWidth = 1920/2;

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

void Processor::snapshot(cv::cuda::GpuMat& RGBData)
{
	const int srcAlignedWidth = this->deckLinkInput.m_RowLength/SIZE_ULONG4_CUDA;
	const int dstAlignedWidth = 1920/2;
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(dstAlignedWidth, block.x), iDivUp(1080, block.y));

	yuyvUmPackedToRGB_lookup <<<grid, block >>> (
			(uint4 *)this->yUnpackedCudaVideo,
			this->cudaRGB,
			dstAlignedWidth,
			this->iWidth,
			this->iHeight,
			(uint4 *)this->yUnpackedCudaKey,
			nullptr // this variable is not used in the function
		);
	RGBData.data = (uchar*)this->cudaRGB;
}
