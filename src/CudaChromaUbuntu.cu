#include <iostream>
#include <cstdlib>
#include <thread>
#include <chrono>
#include "cuda_runtime.h"
#include <npp.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/cudacodec.hpp>
#include "InputLoopThrough.h"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include "YUVUChroma.cuh"
#include "interfaces.hpp"
#include "yolo.hpp"
#include <ui.hpp>

#define MAX_PATH 260

extern void initPosUDPData();
extern void StartMonitor();
extern void ExitMonitor();
extern bool bGenGenlockStatus();

using namespace cv;
using namespace std;
using namespace cv::cuda;



int main()
{
	std::mutex mtxScreenCard;
	static int iIndex=0;
	static int iFrameIndex=0;
	StartMonitor();

	uchar *chrommaLookupBuffer, *chrommaMask;
	uchar2 *pVideo, *pKey, *pFill;
	uchar3* rgbVideo, *vSnapshot, *maskRGB;
	uint4 *video, *key, *fill, *aVideo, *snapShotV;

	float* inputData, *detectionsOutput, *maskOutput, *gpuBuffers[3];

	VideoIn decklink;

	Input *in = new Input(&decklink);
	/*************************************************************************************
	 * This is for memory alignment                                                      *
	 * It seems allocating device memory inside an object causes mis-alignment,           *
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

	// allocate yolo memory
	allocateMemory((void**)&inputData, kBatchSize * 3 * kInputH * kInputW * sizeof(float));
	allocateMemory((void**)&detectionsOutput, kBatchSize * kOutputSize1 * sizeof(float));
	allocateMemory((void**)&maskOutput, kBatchSize * kOutputSize2 * sizeof(float));

	allocateMemory((void**)&gpuBuffers[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float));
	allocateMemory((void**)&gpuBuffers[1], kBatchSize * kOutputSize1 * sizeof(float));
	allocateMemory((void**)&gpuBuffers[2], kBatchSize * kOutputSize2 * sizeof(float));


	in->load(pVideo, pKey, pFill);

	while(!in->isOutput())
		in->run();

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

	YoloMask *yolo  = new YoloMask(pp);
	yolo->load(inputData, detectionsOutput, maskOutput);
	yolo->load(gpuBuffers);

	Keyer *keyer = new Keyer(pp, cm->getMask());


	/******************************
	 * UI                         *
	 ******************************/
	WindowsContainer uiContainer;

	WindowI mainWindow(WINDOW_NAME_MAIN); // plays the video playback
	KeyingWindow keyingWindow(WINDOW_NAME_KEYING, in->getWidth(), in->getHeight()); // keying window
	SettingsWindow settings(WINDOW_NAME_SETTINGS); // settings
	WindowI maskPreview(WINDOW_NAME_MASK);
	WindowI outputWindow(WINDOW_NAME_OUTPUT);

	keyingWindow.enableMouse();
	keyingWindow.enableKeys();

	uiContainer.addWindow(&mainWindow);
	uiContainer.addWindow(&keyingWindow);
	uiContainer.addWindow(&settings);
	uiContainer.addWindow(&maskPreview);
	uiContainer.addWindow(&outputWindow);

	Pipeline* pipeline = new Pipeline(&uiContainer, &mtxScreenCard);

	pipeline->addPipelineObject(in, OBJECT_INPUT);
	pipeline->addPipelineObject(pp, OBJECT_PREPROCESSOR);
	pipeline->addPipelineObject(ss, OBJECT_SNAPSHOT);
	pipeline->addPipelineObject(lt, OBJECT_LOOKUP);
	pipeline->addPipelineObject(cm, OBJECT_CHROMMA_MASK);
	pipeline->addPipelineObject(yolo, OBJECT_YOLO_MASK);
	pipeline->addPipelineObject(keyer, OBJECT_KEYER);

	std::thread *processingThread = new std::thread(&startPipeline, pipeline);

	while(uiContainer.dispatchKey()!= WINDOW_EVENT_EXIT)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(40));
		uiContainer.dispatchEvent();
		uiContainer.updateWindows();
	}

	uiContainer.dispatchEvent();
	processingThread->join();

	cudaFree(chrommaLookupBuffer);
	cudaFree(chrommaMask);
	cudaFree(pVideo);
	cudaFree(pKey);
	cudaFree(pFill);
	cudaFree(rgbVideo);
	cudaFree(vSnapshot);
	cudaFree(maskRGB);
	cudaFree(video);
	cudaFree(key);
	cudaFree(fill);
	cudaFree(aVideo);
	cudaFree(snapShotV);
	cudaFree(inputData);
	cudaFree(detectionsOutput);
	cudaFree(maskOutput);
	cudaFree(gpuBuffers[0]);
	cudaFree(gpuBuffers[1]);
	cudaFree(gpuBuffers[2]);

	cudaDeviceReset();


	return 0;
}
