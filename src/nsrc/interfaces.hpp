/*
 * interfaces.hpp
 *
 *  Created on: May 31, 2023
 *      Author: sipho-mancam
 */

#ifndef SRC_NSRC_INTERFACES_HPP_
#define SRC_NSRC_INTERFACES_HPP_

#include <p-processor.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <InputLoopThrough.h>
#include <cuda_runtime.h>
#include <YUVUChroma.cuh>
#include <stdio.h>
#include <opencv2/cudafilters.hpp>


class IPipeline
{
protected:

	uint4* video;
	uint4* fill;
	uint4* key;
	cudaStream_t stream;
	cudaError_t cudaStatus;
	long int frameSizePacked, frameSizeUnpacked;
	int iWidth, iHeight;
	std::mutex* mtx;

public:
	IPipeline();
	virtual ~IPipeline() = default;
	virtual void create() = 0;
	virtual void update() = 0;
	virtual void output() = 0;

	uint4* getVideo(){return this->video;}
	uint4* getFill(){ return this->fill;}
	uint4* getKey(){ return this->key;}
	std::mutex* getMutex(){return this->mtx;}
	long int getFrameSize(){return this->frameSizeUnpacked;}
	long int getPFrameSize(){return this->frameSizePacked;}

};

class IMask: public IPipeline
{

};











#endif /* SRC_NSRC_INTERFACES_HPP_ */
