/*
 ============================================================================
 Name        : CudaChromaUbuntu.cu
 Author      : Jurie Vosloo
 Version     :
 Copyright   : dont know
 Description : CUDA compute reciprocals
 ============================================================================

 This is the programs main entry point.
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "cuda_runtime.h"
#include <npp.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <opencv2/cudacodec.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include "InputLoopThrough.h"
#include "CameraUDP.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <condition_variable>
#include <list>
#include <opencv2/opencv.hpp>
#include <dirent.h> // for linux systems
#include <sys/stat.h> // for linux systems
#include <algorithm>    // std::sort
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <cuda.h>
#include <p-processor.hpp>
#include <math.h>
#include <X11/Xlib.h>
#include "YUVUChroma.cuh"
#include <iostream>       // std::cout
#include "PosisionUpdateUDP.h"

#define MAX_PATH 260

extern void initPosUDPData();
extern void StartMonitor();
extern void ExitMonitor();
extern bool bGenGenlockStatus();
extern bool TestDetectionsPTR(int mousex,int mousey);
extern void initCameraUDPData();
extern void InitResnet18();
extern void DestroyResnet18();
extern int Classify(cv::Mat img_size);
extern int InitYolov5();
extern void CameraZero();
extern float *GetSegmentedMask();
extern int writeframe(cv::Mat frame);


Mutex MouseMutex;
void initOpenCVWindows();
void InitSettingsWindows();

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;
using namespace cv::cuda;

bool bExite = false;
bool bClearOutPut =false;
int iExitCount = 0;
bool bBypass = false;
bool bExitWorkerThread = false;
bool bRecordingTrainingData = false;
bool bEnableClick = false;
bool bTrackReset =false;
std::mutex mtxScreenCard;           // mutex for critical section
#define AVG_CALC 25

float m_fNMS=0.45;;
int m_BlendPos = 480;

struct MouseData
{

	int iXDown = 0;
	int iYDown = 0;
	int iXUp = 0;
	int iYUp = 0;
	int iXDownDynamic = 0;
	int iYDownDynamic = 0;
	int iXUpDynamic = 0;
	int iYUpDynamic = 0;
	int x;
	int y;

	bool bHandleLDown = false;
	bool bHandleRDown = false;
	bool bHandleL = false;
	bool bHandleR = false;

};

MouseData MouseData1;
MouseData MouseData2;
MouseData MouseData3;


double4 calc_parabola_vertex(double x1, double y1, double x2, double y2, double x3, double y3)
{
	//http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
	double4 ret;
	double denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
	ret.x = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
	ret.y = (x3*x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom;
	ret.z = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;
	if (x1 == 0 && x3 == 1024)
		ret.w = 0;
	else
		ret.w = 1;

	return ret;

}

struct WindowSettings
{
	int m_iUV_Diam;
	int m_iLum_Diam;
	int m_iOuter_Diam ;
	int m_iErase_Diam ;
	int m_iErase_Lum_Diam ;

	int m_iErode;
	int m_iDilate ;
	int m_iLowerlimit ;
	int m_iUpperlimit ;

	int m_cunnyb;

	int m_cunnyt;
	double4 m_ParabolicFunc;

	public:
		WindowSettings()
		{
			m_iUV_Diam=4;
			m_iLum_Diam=2;
			m_iOuter_Diam=14;
			m_iErase_Diam=15;
			m_iErase_Lum_Diam=15;
			 m_cunnyb=125;
			 m_cunnyt=274;
			m_iErode=2;
			m_iDilate=1;


			//m_BlendPos = 0;
			m_iLowerlimit = 80;
			m_iUpperlimit=80;
		}


};
int iUpdateIndex = 0;
WindowSettings FourSettings[3];
//FourSettings[0].m_BlendPos=877;

void  CallThisMouse_Masks(int event, int x, int y, int flags, void* userdata)
{
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
	case EVENT_RBUTTONDOWN:
		bool *ptr = (bool *)userdata;
		*ptr = false;

		break;
	}


}

void  CallThisMouseUpDown(int event, int x, int y, int flags, void* userdata)
{
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		MouseData2.iXDown = x;
		MouseData2.iYDown = y;

		MouseData2.bHandleLDown = true;
		break;

	case EVENT_RBUTTONDOWN:

		MouseData2.iXDown = x;
		MouseData2.iYDown = y;
		MouseData2.bHandleRDown = true;
		break;

	case EVENT_LBUTTONUP:
		MouseData2.bHandleLDown = false;
		//	bHandleL = true;
		MouseData2.iXUp = x;
		MouseData2.iYUp = y;

		break;
	case EVENT_RBUTTONUP:
		MouseData2.bHandleRDown = false;
		//	bHandleR = true;
		MouseData2.iXUp = x;
		MouseData2.iYUp = y;

		break;


	case EVENT_MOUSEMOVE:
	{

		MouseData2.iXUpDynamic = x - 10;
		MouseData2.iYUpDynamic = y - 10;
		MouseData2.iXDownDynamic = x + 10;
		MouseData2.iYDownDynamic = y + 10;

	}

	break;
	}
}

void  MouseUV_FRAME_INFO(int event, int x, int y, int flags, void* userdata)
{
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		MouseData3.iXDown = x;
		MouseData3.iYDown = y;
		break;

	case EVENT_RBUTTONDOWN:
		MouseData3.iXDown = x;
		MouseData3.iYDown = y;
		break;

	case EVENT_LBUTTONUP:
		MouseData3.bHandleLDown = true;
		MouseData3.iXUp = x;
		MouseData3.iYUp = y;
		break;

	case EVENT_RBUTTONUP:
		MouseData3.bHandleRDown = true;
		MouseData3.iXUp = x;
		MouseData3.iYUp = y;
		break;

	case EVENT_MOUSEMOVE:
		MouseData3.iXUpDynamic = x - 10;
		MouseData3.iYUpDynamic = y - 10;
		MouseData3.iXDownDynamic = x + 10;
		MouseData3.iYDownDynamic = y + 10;
		break;
	}
}


int iRecsize = 4;
void  CallThisMouse(int event, int x, int y, int flags, void* userdata)
{
	MouseMutex.lock();
	Rect tt = getWindowImageRect("RGB Output");
	double x1 = double(x)/(double)(tt.width)  * 1920.0;//window correction
	double y1 = double(y)/(double)(tt.height) * 1080.0;//window correction

	switch (event)
	{
		case EVENT_MOUSEWHEEL ://!< positive and negative values mean forward and backward scrolling, respectively.
			if (flags > 0)
			{
				if(iRecsize>1)
				iRecsize--;
			}
			else
			{
				iRecsize++;
				if (iRecsize > 20)iRecsize = 20;
			}

			MouseData1.iXUpDynamic = x1 - iRecsize;
			MouseData1.iYUpDynamic = y1 - iRecsize + 4;
			MouseData1.iXDownDynamic = x1 + iRecsize;
			MouseData1.iYDownDynamic = y1 + iRecsize + 4;
			break;

		case EVENT_LBUTTONDOWN:
			TestDetectionsPTR(x1,y1);
			MouseData1.iXDown = x1;
			MouseData1.iYDown = y1;
			MouseData1.iXUpDynamic = x1 - iRecsize;
			MouseData1.iYUpDynamic = y1 - iRecsize + 4;
			MouseData1.iXDownDynamic = x1 + iRecsize;
			MouseData1.iYDownDynamic = y1 + iRecsize + 4;
			MouseData1.bHandleLDown = true;
			break;

		case EVENT_RBUTTONDOWN:
			MouseData1.iXDown = x1;
			MouseData1.iYDown = y1;
			MouseData1.bHandleRDown = true;
			break;

		case EVENT_LBUTTONUP:
			MouseData1.bHandleLDown = false;
			MouseData1.iXUp = x1;
			MouseData1.iYUp = y1;
			break;

		case EVENT_RBUTTONUP:
			MouseData1.bHandleRDown = false;
			MouseData1.iXUp = x1;
			MouseData1.iYUp = y1;
			break;

		case EVENT_MOUSEMOVE:
			MouseData1.iXUpDynamic = x1-iRecsize;
			MouseData1.iYUpDynamic = y1- iRecsize+4;
			MouseData1.iXDownDynamic = x1+ iRecsize;
			MouseData1.iYDownDynamic = y1+ iRecsize+4;
			MouseData1.x=x1;
			MouseData1.y=y1;
			break;
	}
	MouseMutex.unlock();
}

bool EveryFrame_L = false;
unsigned int iDelayFrames =1;
int iVrArCut = 795;// - 64
int iVrArCut0 = 300;
int iVrArCut1 = 200;
int iVrArCut2 = 100;
bool DisableParabolicKeying = false;
int iAVGCutOff;

void Blending(int pos, void* userdata)
{
	m_BlendPos = pos;
}

void FrameDelay(int pos, void* userdata)
{
	iDelayFrames = pos;
}

void Erode(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_iErode = pos;
}

void Dilate(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_iDilate = pos;
}




void UV_DIAMETER(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_iUV_Diam = pos;
}

void LUM_DIAM(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_iLum_Diam = pos;
}

void OUTER_DIAM(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_iOuter_Diam = pos;
}

void ERASE_UV_DIAMETER(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_iErase_Diam = pos;
}

void ERASE_LUM_DIAM(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_iErase_Lum_Diam = pos;
}

void CUNNY_TOP(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_cunnyt=pos;
}
void NMS(int pos, void* userdata)
{
	m_fNMS=float(pos/100.0);
}

void CUNNY_BOT(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_cunnyb=pos;

}
void LUM_CUT_BOT(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_iLowerlimit = pos;
	FourSettings[iUpdateIndex].m_ParabolicFunc = calc_parabola_vertex(FourSettings[iUpdateIndex].m_iLowerlimit, 0, 512, 1, 1024- FourSettings[iUpdateIndex].m_iUpperlimit, 0);
}

void LUM_CUT_TOP(int pos, void* userdata)
{
	FourSettings[iUpdateIndex].m_iUpperlimit = pos;
	FourSettings[iUpdateIndex].m_ParabolicFunc = calc_parabola_vertex(FourSettings[iUpdateIndex].m_iLowerlimit, 0, 512, 1, 1024 - FourSettings[iUpdateIndex].m_iUpperlimit, 0);
}

struct Traininginfo
{
	Traininginfo()
	{
		DataL.w = -1;
		DataL.x = -1;
		DataL.y = -1;
		DataL.z = -1;
		DataR.w = -1;
		DataR.x = -1;
		DataR.y = -1;
		DataR.z = -1;
	}
	float4 DataL;
	float4 DataR;
};


std::list<Traininginfo*> TrainingDataList;


void DumpTraingData()
{

	char filename[200];
	sprintf(filename, "d:\\TrainingData%d.try",(int) TrainingDataList.size());
	ofstream myfile(filename, ios::binary);

	while (TrainingDataList.size())
	{
		Traininginfo* ptr = TrainingDataList.back();
		TrainingDataList.pop_back();
		myfile.write((char*)ptr, sizeof(Traininginfo));
	}
}


bool bSmall = false;
bool bTakeMask = true;
int bTakeOutput = -1;
bool bDoPaintBack=false;

int iLastCheck = 0;
bool bSafeSnapshot = false;

bool bTaining = false;
bool bErase = false;

int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 1;  // FPS limit for sampling
unsigned int frameCount = 0;
const char *sSDKsample = "RGB Output";


struct ThreadData
{
	ThreadData()
	{
		bUpdateRGB_Preview = false;
		RGB_Output_Cuda=0;
		MouseData1=0;
	}
	cuda::GpuMat *RGB_Output_Cuda;
	bool bUpdateRGB_Preview;
	MouseData *MouseData1;
};

template<typename T>
std::string toString(const T &t) {
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

void SaveImageThread(cv::Mat RGBImageFull,int iIndex,std::string FileName)
{

	int iOffset=0;
	switch(iIndex)
	{

	case 0:
		iOffset=0;
		FileName=FileName+"_0.bmp";
			break;
	case 1:
		iOffset=427;
		FileName=FileName+"_1.bmp";
				break;

	case 2:
		iOffset=853;
		FileName=FileName+"_2.bmp";
			break;


	case 3:
		iOffset=1280;
		FileName=FileName+"_3.bmp";
			break;
	}
	cv::Rect roi(iOffset,0,640,1080/2);
	RGBImageFull.cols=1920*2;
	RGBImageFull.rows=1080/2;
	RGBImageFull.step=1920*2*3;

	cv::Mat croppedImage = RGBImageFull(roi);
	cv::imwrite(FileName,croppedImage);
}

ThreadData myThreadData;
char OutputRenderthreadStatus[MAX_PATH];


void DrawOutputThreadData(Mat *DrawingMat)
{
	cv::Rect r(50,1080-100,10,10);
	cv::rectangle(*DrawingMat,r,Scalar(255,255,255),2);
	std::string text = OutputRenderthreadStatus;
	Mat img=*DrawingMat;

	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale =2.0;
	int thickness = 2;

	int baseline=0;
	Size textSize = getTextSize(text, fontFace,
					fontScale, thickness, &baseline);
	baseline += thickness;
	// center the text
	Point textOrg(r.x+r.width,
	  r.y/*+r.height/2*/);


	putText(img, text, textOrg, fontFace, fontScale,
		Scalar::all(255), thickness, 8);
}



bool bAutoTrain=false;
//
void *OutputRenderthread(void *lpParam)//https://developer.nvidia.com/blog/this-ai-can-automatically-remove-the-background-from-a-photo/
{

	#ifdef PREVIEW_OUTPUTRENDER
		initOpenCVWindows();
		namedWindow("y_only",WINDOW_OPENGL);
		namedWindow("u_only",WINDOW_OPENGL);
		namedWindow("v_only",WINDOW_OPENGL);
		namedWindow("mask Erode dilate GaussianFilter",WINDOW_OPENGL);
		namedWindow("UI Mouse training Window",WINDOW_OPENGL);
		namedWindow("Yolo generated mask",WINDOW_OPENGL);
		namedWindow("Yolo generated mask HistoGram",WINDOW_OPENGL);
		namedWindow("Yolomask",WINDOW_OPENGL);

//		std::cout<<"I run here"<<std::endl;

		int yspace = 1;
		int xspace = 380;
		int index = 0 ;
		moveWindow("y_only",0,0);
		moveWindow("mask Erode dilate GaussianFilter",0*xspace,0*yspace);
		moveWindow("UI Mouse training Window",1*xspace,1*yspace);
		moveWindow("Yolo generated mask",2*xspace,2*yspace);
		moveWindow("u_only",3*xspace,3*yspace);
		moveWindow("v_only",4*xspace,4*yspace);

	#endif
//
	bool bPopFront=false;
	unsigned int iDelayFramesStore=1;
	long framecounter=0;

	double avg_duration[AVG_CALC];
	double avg=0.0;
	int iAvgIndex=0;
	ThreadData *ptrThreadData=(ThreadData *)lpParam;
	InitYolov5();

	unsigned int Max_duration=0;
	VideoIn decklink_video_in; // Input video

//	Processor p;
//	p.sendDataTo();

	while (decklink_video_in.m_sizeOfFrame== -1)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(40));
	}

	CudaChromaInit(decklink_video_in.m_iWidth, decklink_video_in.m_iHeight,decklink_video_in.m_sizeOfFrame, decklink_video_in.m_iFrameSizeUnpacked);
	cudaLookUpInit();
	PrepareYoloData(false,0.9);
	FourSettings[0].m_ParabolicFunc = calc_parabola_vertex(0, 0, 512, 1, 1024, 0);
	FourSettings[1].m_ParabolicFunc = calc_parabola_vertex(0, 0, 512, 1, 1024, 0);
	FourSettings[2].m_ParabolicFunc = calc_parabola_vertex(0, 0, 512, 1, 1024, 0);

	decklink_video_in.WaitForFrames(1);
	decklink_video_in.imagelistVideo.ClearAll(0);
	decklink_video_in.imagelistFill.ClearAll(0);
	decklink_video_in.imagelistKey.ClearAll(0);
	decklink_video_in.ImagelistOutput.ClearAll(1);


	while (!bExitWorkerThread)
	{

	//	std::this_thread::sleep_for(std::chrono::milliseconds(10));
		auto timer_wait_start = std::chrono::system_clock::now();
		if((iDelayFramesStore!=iDelayFrames)||bClearOutPut)
		{
			bClearOutPut=false;
			iDelayFramesStore=iDelayFrames;
			decklink_video_in.WaitForFrames(-1);
			decklink_video_in.ImagelistOutput.ClearAll(1);
			decklink_video_in.imagelistVideo.ClearAll(0);
			decklink_video_in.imagelistFill.ClearAll(0);
			decklink_video_in.imagelistKey.ClearAll(0);
		}
		else
		{
			decklink_video_in.WaitForFrames(iDelayFrames);
		}

		auto timer_start = std::chrono::system_clock::now();

		avg=0.0;

		for(int x=0; x<AVG_CALC; x++)
		{
			avg=avg+avg_duration[x];
		}

		avg=avg/AVG_CALC;

		snprintf(
					OutputRenderthreadStatus,
					sizeof(OutputRenderthreadStatus),
					"avg:%f,Genlocked:%s Video:%d Key:%d Fill:%d Delay:%d Output:%d \r",
					avg,
					bGenGenlockStatus() ? "Yes" : "No",
					(int) decklink_video_in.imagelistVideo.GetFrameCount(),
					(int)(int) decklink_video_in.imagelistKey.GetFrameCount(),
					(int)(int) decklink_video_in.imagelistFill.GetFrameCount(),
					iDelayFrames,
					(int)(int) decklink_video_in.ImagelistOutput.GetFrameCount()
				);

		framecounter++;

		unsigned int iBufferCount = decklink_video_in.imagelistVideo.GetFrameCount();

		if(iDelayFrames < iBufferCount)
		{
			bPopFront = true;
		}
		else
		{
			bPopFront = false;
		}

		void * ptr_BG_Video = decklink_video_in.imagelistVideo.GetFrame(bPopFront);
		void * ptr__FILL_Video = decklink_video_in.imagelistFill.GetFrame(true);
		void * ptr__KEY_Video = decklink_video_in.imagelistKey.GetFrame(true);
		mtxScreenCard.lock();

		CudaSetInputData(ptr_BG_Video, ptr__FILL_Video, ptr__KEY_Video, false);
		auto  timer_start_CudaSetInputData = std::chrono::system_clock::now();

		if(bPopFront)
		{
			free(ptr_BG_Video);
		}

		free(ptr__FILL_Video);
		free(ptr__KEY_Video);

		#ifdef PREVIEW_OUTPUTRENDER

			Launch_yuyv10PackedToyuyvUnpacked(
					decklink_video_in.m_RowLength,
					bTakeMask,
					decklink_video_in.m_iFrameSizeUnpacked,
					ptrThreadData->RGB_Output_Cuda,
					FourSettings[0].m_cunnyb,
					FourSettings[0].m_cunnyt,
					true
					);
		#else
			Launch_yuyv10PackedToyuyvUnpacked(
					decklink_video_in.m_RowLength,
					bTakeMask,
					decklink_video_in.m_iFrameSizeUnpacked,
					ptrThreadData->RGB_Output_Cuda,
					FourSettings[0].m_cunnyb,
					FourSettings[0].m_cunnyt,
					bAutoTrain
					);
		#endif
		auto timer_Launch_yuyv10PackedToyuyvUnpacked = std::chrono::system_clock::now();

		//auto startyolo = std::chrono::system_clock::now();
		if(bTakeOutput==-1)
			PrepareYoloData(bTakeMask,m_fNMS);
		else
			PrepareYoloData(true,m_fNMS);
		auto timer_endyolo = std::chrono::system_clock::now();

		if (bTakeMask)
		{
			bTakeMask = false;
			ptrThreadData->bUpdateRGB_Preview = true; // update output window
		}

		#ifdef PREVIEW_OUTPUTRENDER
			Launch_yuyv_Unpacked_GenerateMask(0, x,true);//
		`	Launch_yuyvDilateAndErode(FourSettings[x].m_iErode, FourSettings[x].m_iDilate, x);
		#else
			Launch_yuyv_Unpacked_GenerateMask(0, 0,bAutoTrain);//
			Launch_yuyvDilateAndErode(FourSettings[0].m_iErode, FourSettings[0].m_iDilate, 0);
			Launch_yuyv_Unpacked_GenerateMask_yolo_seg(0, 1,bAutoTrain,GetSegmentedMask());//
		#endif

		auto timer_Launch_yuyvDilateAndErode_Launch_yuyv_Unpacked_GenerateMask_yolo = std::chrono::system_clock::now();

		Launch_yuyv_Unpacked_UnpackedComBineData(&m_BlendPos, &m_BlendPos, &m_BlendPos, decklink_video_in.m_RowLength, &FourSettings[0].m_ParabolicFunc, &FourSettings[1].m_ParabolicFunc, &FourSettings[2].m_ParabolicFunc,bBypass, iVrArCut+ 64, iVrArCut0+64, iVrArCut1 + 64, iVrArCut2 + 64, bTakeOutput);

		auto timer_Launch_yuyv_Unpacked_UnpackedComBineData = std::chrono::system_clock::now();

		if (bTakeOutput!=-1)
		{
			ptrThreadData->bUpdateRGB_Preview = true;
		}

		void *yuvdata = malloc(decklink_video_in.m_sizeOfFrame);
		CudaGetOutputData(yuvdata);
		// This is where we send output to the output channel of decklink
		decklink_video_in.ImagelistOutput.AddFrame(yuvdata);

		auto timer_end = std::chrono::system_clock::now();
		mtxScreenCard.unlock();

		auto duration_now = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start).count();
		auto duration_yolo = std::chrono::duration_cast<std::chrono::milliseconds>(timer_endyolo - timer_Launch_yuyv10PackedToyuyvUnpacked).count();
		auto timer_start_wait_duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer_start-timer_wait_start ).count();//

		avg_duration[iAvgIndex++]=duration_now;
		if(iAvgIndex == AVG_CALC)
		{
			iAvgIndex=0;
		}

		if(Max_duration<duration_now)
		{
			Max_duration=duration_now;
		}


		#ifdef DISPLAY_I_TIMINGS
				if((framecounter==10))
				{
					auto duration_CudaSetInputData				=std::chrono::duration_cast<std::chrono::milliseconds>(timer_start-timer_start_CudaSetInputData).count();
					auto duration_yuyv10PackedToyuyvUnpacked	=std::chrono::duration_cast<std::chrono::microseconds>(timer_start_CudaSetInputData-timer_Launch_yuyv10PackedToyuyvUnpacked).count();
					auto duration_PrepareYoloData				=std::chrono::duration_cast<std::chrono::milliseconds>(timer_Launch_yuyv10PackedToyuyvUnpacked-timer_endyolo).count();
					auto duration_generatemask_erode_dilate		=std::chrono::duration_cast<std::chrono::milliseconds>(timer_endyolo-timer_Launch_yuyvDilateAndErode_Launch_yuyv_Unpacked_GenerateMask_yolo).count();
					auto duration_add_to_decklink				=std::chrono::duration_cast<std::chrono::milliseconds>(timer_Launch_yuyv_Unpacked_UnpackedComBineData-timer_end).count();
					std::cout <<"duration_CudaSetInputData"<<duration_CudaSetInputData<< "ms  "<<std::endl;
					std::cout <<"duration_yuyv10PackedToyuyvUnpacked"<<duration_yuyv10PackedToyuyvUnpacked<< "us  "<<std::endl;
					std::cout <<"duration_PrepareYoloData"<<duration_PrepareYoloData<< "ms  "<<std::endl;
					std::cout <<"duration_generatemask_erode_dilate"<<duration_generatemask_erode_dilate<< "ms  "<<std::endl;
					std::cout <<"duration_add_to_decklink"<<duration_add_to_decklink<< "ms  "<<std::endl;

					framecounter=0;
					std::cout <<"Max:"<<Max_duration<< "ms Now:"<< duration_now << "ms  " <<std::endl;
					Max_duration=0;
				}
		#endif
	}

	std::cout<<"[Launch]: YUYV 10-bit Packed To YUYV Unpacked"<<std::endl;
	CudaChromaFree();
	cudaLookUpFree();
	return 0;


}


void InitSettingsWindows()
{
	// TODO: All window names must be stored in strings and passed as variables
	namedWindow("Settings", WINDOW_NORMAL);
	createTrackbar("Blending", "Settings", 0, 2000, Blending, 0);
	createTrackbar("Delay", "Settings", 0, 30, FrameDelay, 0);
	createTrackbar("Erode", "Settings", 0, 20, Erode, 0);
	createTrackbar("Dialate", "Settings", 0, 20, Dilate, 0);
	createTrackbar("Outer Diam", "Settings", 0, 200, OUTER_DIAM, 0);
	createTrackbar("UV Diam", "Settings", 0, 50, UV_DIAMETER, 0);
	createTrackbar("Lum Depth", "Settings", 0, 50, LUM_DIAM, 0);
	createTrackbar("E UV", "Settings", 0, 50, ERASE_UV_DIAMETER, 0);
	createTrackbar("E Lum", "Settings", 0, 50, ERASE_LUM_DIAM, 0);
	createTrackbar("Key Bot", "Settings", 0, 300, LUM_CUT_BOT, 0);
	createTrackbar("Key Top", "Settings", 0, 300, LUM_CUT_TOP, 0);
	createTrackbar("NMS", "Settings", 0, 100, NMS, 0);
}


void UpdateSettingsWindow()
{
	setTrackbarPos("Blending", "Settings", m_BlendPos);
	setTrackbarPos("Delay", "Settings", 3);
	setTrackbarPos("UV Diam", "Settings", FourSettings[iUpdateIndex].m_iUV_Diam);
	setTrackbarPos("Outer Diam", "Settings", FourSettings[iUpdateIndex].m_iOuter_Diam);
	setTrackbarPos("Lum Depth", "Settings", FourSettings[iUpdateIndex].m_iLum_Diam);
	setTrackbarPos("E UV", "Settings", FourSettings[iUpdateIndex].m_iErase_Diam);
	setTrackbarPos("E Lum", "Settings", FourSettings[iUpdateIndex].m_iErase_Lum_Diam);
	setTrackbarPos("Erode", "Settings", FourSettings[iUpdateIndex].m_iErode);
	setTrackbarPos("Dialate", "Settings", FourSettings[iUpdateIndex].m_iDilate);
	setTrackbarPos("Key Bot", "Settings", FourSettings[iUpdateIndex].m_iLowerlimit);
	setTrackbarPos("Key Top", "Settings", FourSettings[iUpdateIndex].m_iUpperlimit);
	setTrackbarPos("NMS", "Settings", m_fNMS*100);
}

void on_opengl(void* param)
{
	glViewport(0, 0,  1366.0, 768.0);
	glMatrixMode(GL_PROJECTION);                // Select The Projection Matrix
	glLoadIdentity();                           // Reset The Projection Matrix
	gluOrtho2D(0, 1366.0, 768.0, 0);			// Calculate The Aspect Ratio Of The Window
	glMatrixMode(GL_MODELVIEW);                 // Select The Modelview Matrix
	glLoadIdentity();                           // Reset The Modelview Matrix
	glEnable(GL_BLEND);
	glShadeModel(GL_SMOOTH);                    // Enable Smooth Shading
	glClearDepth(1.0f);                         // Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);                    // Enables Depth Testing
	glDepthFunc(GL_ALWAYS);                     // The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);          // Really Nice Perspective
	glDisable(GL_LIGHTING);
	glEnable(GL_MULTISAMPLE_ARB);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColor4f(GLfloat(40.0 / 255.0), GLfloat(40.0 / 255.0), GLfloat(40.0 / 255.0), GLfloat(1.0));
	glBegin(GL_QUADS);
	glVertex2f(10, 10);
	glVertex2f(1910, 10);
	glVertex2f(1910, 1070);
	glVertex2f(10, 1070);
	glVertex2f(10, 10);
	glEnd();
	glColor4f(GLfloat(1), GLfloat(1), GLfloat(1), GLfloat(1.0));
	glBegin(GL_LINE_LOOP);
	glVertex2f(10, 10);
	glVertex2f(1910, 10);
	glVertex2f(1910, 1070);
	glVertex2f(10, 1070);
	glVertex2f(10, 10);
	glEnd();
	glFlush();
}

void initOpenCVWindows()
{
	InitSettingsWindows();
	UpdateSettingsWindow();
	std::string rgbOutputWin = "RGB Output";
	std::string frameInfoWin = "Frame Info";

	namedWindow(rgbOutputWin,  WINDOW_NORMAL);
	namedWindow(frameInfoWin, WINDOW_NORMAL);
	setMouseCallback(rgbOutputWin, CallThisMouse, 0);
	updateWindow(rgbOutputWin);
}

void UpdateTXTFile(char *Buffer)
{
	std::ofstream outfile(GetTXTFileName(), std::ofstream::binary| std::ofstream::app);
	outfile.write(Buffer, strlen(Buffer));
}

void UpdateLookupFromMouse()
{
	if (!bTaining && !bErase)
	{
		if (bEnableClick)
		{
			if (MouseData1.bHandleLDown)
			{
				if (!bSmall)
				{
					char buffer[MAX_PATH];
					sprintf(buffer, "1 %d %d %d %d %d\n", MouseData1.iXUpDynamic, MouseData1.iYUpDynamic,
							MouseData1.iXDownDynamic, MouseData1.iYDownDynamic,FourSettings[iUpdateIndex].m_iUV_Diam);

					UpdateTXTFile(buffer);
					mtxScreenCard.lock();

					Launch_UpdateLookupFrom_XY_Posision(MouseData1.iXUpDynamic, MouseData1.iYUpDynamic,
							MouseData1.iXDownDynamic, MouseData1.iYDownDynamic, FourSettings[iUpdateIndex].m_iUV_Diam,
							FourSettings[iUpdateIndex].m_iLum_Diam, FourSettings[iUpdateIndex].m_iOuter_Diam, 200,bDoPaintBack);

					mtxScreenCard.unlock();
				}
				else
				{
					char buffer[MAX_PATH];
					sprintf(buffer, "1 %d %d %d %d\n", MouseData1.iXUpDynamic, MouseData1.iYUpDynamic,
							MouseData1.iXDownDynamic, MouseData1.iYDownDynamic);

					UpdateTXTFile(buffer);
					mtxScreenCard.lock();

					Launch_UpdateLookupFrom_XY_Posision(MouseData1.iXUpDynamic, MouseData1.iYUpDynamic,
							MouseData1.iXDownDynamic, MouseData1.iYDownDynamic, 1, 10, 5, 200,bDoPaintBack);

					mtxScreenCard.unlock();
				}
				return;
			}
		}

		if (bEnableClick)
		{
			if (MouseData1.bHandleRDown)
			{
				char buffer[MAX_PATH];
				sprintf(buffer, "0 %d %d %d %d\n", MouseData1.iXUpDynamic, MouseData1.iYUpDynamic, MouseData1.iXDownDynamic, MouseData1.iYDownDynamic);
				UpdateTXTFile(buffer);
				mtxScreenCard.lock();
				Launch_UpdateLookupFrom_XY_Posision_Erase(MouseData1.iXUpDynamic, MouseData1.iYUpDynamic, MouseData1.iXDownDynamic, MouseData1.iYDownDynamic, FourSettings[iUpdateIndex].m_iErase_Diam, FourSettings[iUpdateIndex].m_iErase_Lum_Diam,bDoPaintBack);
				mtxScreenCard.unlock();
			}
		}
	}
}

#define VK_LCONTROL 0
#define VK_F1 190
#define VK_F2 191
#define VK_F3 192
#define VK_F4 193
#define VK_F5 194

#define VK_F10 199
int iKey;

void UpdateKeyState()
{
	iKey=waitKey(10);
}


bool GetAsyncKeyState(int checkpressed)
{
	if(checkpressed==iKey)
	{
		iKey=-1;
		return true;
	}
	else
	{
		return false;
	}
}


void DrawMouseText(Mat *DrawingMat,string text,cv::Point r)
{
	char buffer[MAX_PATH];
	sprintf(buffer, "%d %d\n", MouseData1.x,MouseData1.y);
	Mat img=*DrawingMat;
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale =1.0;
	int thickness = 1;
	int baseline=0;

	Point textOrg(r.x,r.y);
	putText(img, buffer, textOrg, fontFace, fontScale,
			Scalar::all(255), thickness, 8);
}


int main()
{
	static int iIndex=0;
	static int iFrameIndex=0;
	StartMonitor();

	if (1)
	{
		#ifndef PREVIEW_OUTPUTRENDER
			initCameraUDPData();
			initOpenCVWindows();

			//InitVizSocket();
		#endif

		cuda::GpuMat RGB_Output_Cuda;
		RGB_Output_Cuda.create(1080, 1920, CV_8UC3); // fullHD image mat
		RGB_Output_Cuda.step = 5760; // step between the pixels -> allocates 3 bytes extra for every pixel

		cuda::GpuMat RGB_FrameInfo_Cuda;
		RGB_FrameInfo_Cuda.create(1024, 1024, CV_8UC3);
		RGB_FrameInfo_Cuda.step = 1024 * 3;

		cuda::GpuMat RGB_FrameInfo_Cuda_FullUpdate;
		RGB_FrameInfo_Cuda_FullUpdate.create(1024, 1024, CV_8UC3);
		RGB_FrameInfo_Cuda_FullUpdate.step = 1024 * 3;

		myThreadData.RGB_Output_Cuda = &RGB_Output_Cuda;
		Mat MASK_L(1080, 1920, CV_8UC1, Scalar(0));
		myThreadData.MouseData1 = &MouseData1;

		pthread_t threads;
		int rc;
		// send output renderer to a separate thread
		rc = pthread_create(&threads, NULL, OutputRenderthread, (void *) &myThreadData);

		if (rc)
		{
			cout << "[Error]: Unable to create Output Renderer thread! : " << rc << endl;
			exit(-1);
		}

		initPosUDPData();
		Mat RGB__Draw;
		Mat RGB_Output;
		Mat RGB_saving;
		bool bstart = false;
		bool bCapture=false;
		static unsigned int bFrameTimer = 0;

		#ifdef PREVIEW_OUTPUTRENDER
			while (1)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			};
		#endif

		unsigned long UI_Frame_Counter=0;

		while (1)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(40));
			UI_Frame_Counter++;

			if (GetAsyncKeyState('g'))
			{
				cudaLookUpFullKey();
			}

			if (GetAsyncKeyState('r'))//||(UI_Frame_Counter%10)==0
			{
				cudaLookUpReset(0);
			}

			if (GetAsyncKeyState('f'))//||(UI_Frame_Counter%10)==0
			{
				cudaLookUpReset(1);
			}

			if (GetAsyncKeyState('o'))
			{
				bClearOutPut=true;
			}

			if (GetAsyncKeyState('b'))
			{
				bBypass = !bBypass;
			}

			if (GetAsyncKeyState('h'))
			{
				printf("\n\n\n\n\n\r");
				printf("Chroma Software Usage: \n\t[h]\t->\tShows this help message.\n\n");
				printf("['q']\t->\tchroma snapshot update \n");
				printf("['a']\t->\tPaintItBack snapshot update \n");
				printf("['r']\t->\treset chroma lookup table\n");
				printf("['f']\t->\treset PaintItBack lookup table\n");
				printf("['o']\t->\tto clear output buffer frame list\n");
				printf("['i']\t->\tdisplay frame info\n\n");
				printf("[ctl+'l']\t->\tload settings\n");
				printf("[ctl+'s']\t->\tsave settings\n");
				printf("\n\n\n\r");
			}

			bFrameTimer++;

			if (GetAsyncKeyState('w'))
			{
				bTakeOutput = 0;
				bAutoTrain=true;

			}
			else
			{
				bAutoTrain=false;
			}

			if (GetAsyncKeyState('z'))
			{
				bTrackReset=true;
			}
			if (GetAsyncKeyState('c'))
			{
				bCapture=true;
			}else
			{
				bCapture=false;
			}

			if (GetAsyncKeyState('q'))
			{
				SetOnAirLookup(0);
				iUpdateIndex = 0;
				UpdateSettingsWindow();

				iLastCheck = 0;
				bTakeOutput = -1;
				bDoPaintBack=false;

				mtxScreenCard.lock();
				bTakeMask = true;
				mtxScreenCard.unlock();

				setWindowTitle("RGB Output", "Chroma");
				setWindowTitle("Settings","Settings Chroma");
			}

			if (GetAsyncKeyState('a'))
			{
				SetOnAirLookup(1);
				iUpdateIndex = 1;
				UpdateSettingsWindow();
				iLastCheck = 0;
				bTakeOutput = -1;
				bDoPaintBack=true;
				mtxScreenCard.lock();
				bTakeMask = true;
				mtxScreenCard.unlock();

				setWindowTitle("RGB Output", "PaintItBack");
				setWindowTitle("Settings","PaintItBack");
			}

			if (GetAsyncKeyState(VK_F1))
			{
				bTakeOutput = 0;
			}

			if (GetAsyncKeyState(VK_F2))
			{
				bTakeOutput = 1;
			}

			if (GetAsyncKeyState(VK_F3))
			{
				bTakeOutput = 2;
			}
			if (GetAsyncKeyState(VK_F4))
			{
				bTakeOutput = 3;
			}
			if (GetAsyncKeyState(VK_F5))
			{
				bTakeOutput = 4;
			}

			if (GetAsyncKeyState(VK_F10))
			{
				CameraZero();
			}

			if (GetAsyncKeyState('s'))
			{
				std::string FileAndPathName;
				std::time_t result = std::time(nullptr);
				std::string  FileName= toString(result);
				FileAndPathName="/home/jurie/Pictures/yolov5_soccer_training/"+FileName; // User relative Paths
				std::thread t1(SaveImageThread,RGB_saving.clone(),iIndex++,FileAndPathName);
				t1.join();
				if(iIndex==4)
				{
					iIndex=0;
				}

			}

			if (myThreadData.bUpdateRGB_Preview)// this is updated in the outputRenderer Thread
			{
				myThreadData.bUpdateRGB_Preview = false;
				mtxScreenCard.lock();

				RGB_Output_Cuda.download(RGB_Output);
				RGB_saving=RGB_Output.clone();
				iFrameIndex++;

				if(0) // What's this?
				if(iFrameIndex==50)
				{
					iFrameIndex=0;
					std::string FileAndPathName;
					std::time_t result = std::time(nullptr);
					std::string  FileName = toString(result);
					FileAndPathName="/home/jurie/Pictures/yolov5_soccer_training/"+FileName; // user relative paths

					std::thread t1(SaveImageThread,RGB_Output.clone(),iIndex++,FileAndPathName);
					t1.join();
					if(iIndex==4)
					{
						iIndex=0;
					}
				}

				DrawSnapShotDetections_clean(&RGB_Output,bTrackReset);

				if(bCapture)
				{
					writeframe(RGB_Output.clone()); // save frame
				}

				bTrackReset=false;
				DrawCameraData(&RGB_Output);
				bstart = true;
				mtxScreenCard.unlock();
			}
			if (bstart)
			{
				RGB__Draw = RGB_Output.clone();
				cv::Rect myROI(
							MouseData1.iXUpDynamic, MouseData1.iYUpDynamic,
							MouseData1.iXDownDynamic - MouseData1.iXUpDynamic,
							MouseData1.iYDownDynamic - MouseData1.iYUpDynamic
							);
				if((0 <= myROI.x && 0 <= myROI.width && myROI.x + myROI.width <= RGB__Draw.cols &&
					0 <= myROI.y && 0 <= myROI.height && myROI.y + myROI.height <= RGB__Draw.rows)){

					MouseMutex.lock();
					Mat RGB__Draw_Small = RGB__Draw(myROI);
					Mat RGB__Draw_SmallEnlarge;
					Size ssize = RGB__Draw_Small.size();
					if (!ssize.empty())
					{
						cv::resize(RGB__Draw_Small, RGB__Draw_SmallEnlarge,
								Size((MouseData1.iXDownDynamic - MouseData1.iXUpDynamic) * 25,
								(MouseData1.iYDownDynamic - MouseData1.iYUpDynamic) * 25), 0, 0, INTER_NEAREST);

						RGB__Draw_SmallEnlarge.copyTo(RGB__Draw.rowRange(0, RGB__Draw_SmallEnlarge.rows).colRange(0, RGB__Draw_SmallEnlarge.cols));
					}
					MouseMutex.unlock();
					bEnableClick = true;
				}
				else
				{
					bEnableClick = false;
				}

				rectangle(RGB__Draw, Point(MouseData1.iXUpDynamic, MouseData1.iYUpDynamic), Point(MouseData1.iXDownDynamic, MouseData1.iYDownDynamic), Scalar(255, 255, 255), 1, 8, 0);
				circle(RGB__Draw,Point(MouseData1.x,MouseData1.y),20,Scalar(255,255,255),3);\
				DrawMouseText(&RGB__Draw,"Hello World",cv::Point(50,50));
				DrawOutputThreadData(&RGB__Draw);
				imshow("RGB Output", RGB__Draw);
				RGB__Draw.release();
			}

			if(0)
			{
				if ((bFrameTimer%20)==0)
				{
					Launch_Frame_Info(&RGB_FrameInfo_Cuda);
					Mat prev;
					RGB_FrameInfo_Cuda.download(prev);
					imshow("Frame Info", prev);
				}
			}

			if (GetAsyncKeyState('i'))
			{
				Launch_Frame_Info(&RGB_FrameInfo_Cuda);
//					imshow("Frame Info", RGB_FrameInfo_Cuda);
			}

			UpdateLookupFromMouse();
			UpdateKeyState();

			if (GetAsyncKeyState(27))//"Esc"
			{
				// Do some clean up and free memory, c++ garbage collector doesn't clean up some things.
				bExitWorkerThread = true;
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				EndLoop();
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				std::cout << "Exit" << std::endl;
				break;
			}
			if (bExite)
			{
				iExitCount++;
				if (iExitCount == 100)
				{
					bExite = false;
					printf("Exit process canceled Exit process canceledExit process canceledExit process canceled\n\r");
				}
			}

			if (GetAsyncKeyState('y'))
			{
				if (bExite)
				{
					bExitWorkerThread = true;
					std::this_thread::sleep_for(std::chrono::milliseconds(1000));
					break;
				}

				if (GetAsyncKeyState('n'))
				{
					bExite = false;
				}
			};
		}
	}

	cudaError_t cudaStatus;
	 ExitMonitor();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	std::cout << "End Cuda" << std::endl;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}
