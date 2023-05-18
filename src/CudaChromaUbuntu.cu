/*
 ============================================================================
 Name        : CudaChromaUbuntu.cu
 Author      : Jurie Vosloo
 Version     :
 Copyright   : dont know
 Description : CUDA compute reciprocals
 ============================================================================
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
#include <math.h>
#include <X11/Xlib.h>
#include "YUVUChroma.cuh"
#include <iostream>       // std::cout
#include "PosisionUpdateUDP.h"

//#include <C:\Users\jurie\Documents\opencv\sources\modules\cudafilters\include\opencv2/cudafilters.hpp>

//#include "DeckLinkAPIVersion.h"



//#define PREVIEW_OUTPUTRENDER
//#define DISPLAY_I_TIMINGS


#define MAX_PATH 260


extern void initPosUDPData();
extern void StartMonitor();
extern void ExitMonitor();
extern bool bGenGenlockStatus();
// Utilities and timing functions
extern bool TestDetectionsPTR(int mousex,int mousey);
extern void initCameraUDPData();
extern void InitResnet18();
extern void DestroyResnet18();
extern int Classify(cv::Mat img_size);
extern int InitYolov5();
extern void CameraZero();
extern float *GetSegmentedMask();
//https://arxiv.org/pdf/1912.05445.pdf
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
		//	bHandleL = true;
		MouseData3.iXUp = x;
		MouseData3.iYUp = y;

		break;
	case EVENT_RBUTTONUP:
		MouseData3.bHandleRDown = true;
		//	bHandleR = true;
		MouseData3.iXUp = x;
		MouseData3.iYUp = y;

		break;


	case EVENT_MOUSEMOVE:
		//	if (bHandleRDown || bHandleLDown)
	{

		MouseData3.iXUpDynamic = x - 10;
		MouseData3.iYUpDynamic = y - 10;
		MouseData3.iXDownDynamic = x + 10;
		MouseData3.iYDownDynamic = y + 10;

	}

	break;
	}



}
 int iRecsize = 4;
void  CallThisMouse(int event, int x, int y, int flags, void* userdata)
{
	//Rect t= getWindowImageRect("RGB Output");


	MouseMutex.lock();


	Rect tt = getWindowImageRect("RGB Output");
//	std::cout<<"I execute"<<std::endl;
	//printf("\n\r%d %d %d %d %d %d\n", tt.x, tt.y, tt.width, tt.height,x,y);
	double x1 = double(x)/(double)(tt.width)  * 1920.0;//window correction
	double y1 = double(y)/(double)(tt.height) * 1080.0;//window correction
//	printf("%d %d %d %d %f %f\n\r", tt.x, tt.y, tt.width, tt.height, x1, y1);

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

			if (iRecsize > 20)
				iRecsize = 20;
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
		//std::cout << MouseData1.iXUpDynamic;
		 break;

	case EVENT_RBUTTONDOWN:

		MouseData1.iXDown = x1;
		MouseData1.iYDown = y1;
		MouseData1.bHandleRDown = true;
		break;

	case EVENT_LBUTTONUP:
		MouseData1.bHandleLDown = false;
	//	bHandleL = true;
		MouseData1.iXUp = x1;
		MouseData1.iYUp = y1;

		break;
	case EVENT_RBUTTONUP:
		MouseData1.bHandleRDown = false;
	//	bHandleR = true;
		MouseData1.iXUp = x1;
		MouseData1.iYUp = y1;

		break;


	case EVENT_MOUSEMOVE:

	//	if (bHandleRDown || bHandleLDown)
		{
			MouseData1.iXUpDynamic = x1-iRecsize;
			MouseData1.iYUpDynamic = y1- iRecsize+4;
			MouseData1.iXDownDynamic = x1+ iRecsize;
			MouseData1.iYDownDynamic = y1+ iRecsize+4;

		}

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
	VideoIn decklink_video_in;
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


//	while (!bExitWorkerThread)
//			{
//
//					//std::this_thread::sleep_for(std::chrono::milliseconds(1));
//					decklink_video_in.WaitForFrames();
//					void * ptr_BG_Video = decklink_video_in.imagelistVideo.GetFrame(true);
//					void * ptr__FILL_Video = decklink_video_in.imagelistFill.GetFrame(true);
//					void * ptr__KEY_Video = decklink_video_in.imagelistKey.GetFrame(true);
//					free(ptr_BG_Video);
//					free(ptr__FILL_Video);
//					free(ptr__KEY_Video);
//			}

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
		}else
			decklink_video_in.WaitForFrames(iDelayFrames);
			auto timer_start = std::chrono::system_clock::now();
			avg=0.0;
			for(int x=0;x<AVG_CALC;x++)
				avg=avg+avg_duration[x];
			avg=avg/AVG_CALC;
			snprintf(OutputRenderthreadStatus,sizeof(OutputRenderthreadStatus),"avg:%f,Genlocked:%s Video:%d Key:%d Fill:%d Delay:%d Output:%d \r",avg, bGenGenlockStatus() ? "Yes" : "No",(int) decklink_video_in.imagelistVideo.GetFrameCount(),  (int)(int) decklink_video_in.imagelistKey.GetFrameCount(), (int)(int) decklink_video_in.imagelistFill.GetFrameCount(), iDelayFrames, (int)(int) decklink_video_in.ImagelistOutput.GetFrameCount());
			framecounter++;
			unsigned int iBufferCount=decklink_video_in.imagelistVideo.GetFrameCount() ;
			if(iDelayFrames<iBufferCount)
			{
				bPopFront=true;
			}
			else
			{
				bPopFront=false;
			}
			void * ptr_BG_Video = decklink_video_in.imagelistVideo.GetFrame(bPopFront);
			void * ptr__FILL_Video = decklink_video_in.imagelistFill.GetFrame(true);
			void * ptr__KEY_Video = decklink_video_in.imagelistKey.GetFrame(true);
			mtxScreenCard.lock();

			CudaSetInputData(ptr_BG_Video,ptr__FILL_Video,ptr__KEY_Video,false);
			auto  timer_start_CudaSetInputData = std::chrono::system_clock::now();
			if(bPopFront)
				free(ptr_BG_Video);
			free(ptr__FILL_Video);
			free(ptr__KEY_Video);

#ifdef PREVIEW_OUTPUTRENDER
			Launch_yuyv10PackedToyuyvUnpacked(decklink_video_in.m_RowLength,bTakeMask,decklink_video_in.m_iFrameSizeUnpacked,ptrThreadData->RGB_Output_Cuda,FourSettings[0].m_cunnyb,FourSettings[0].m_cunnyt,true);
#else
			Launch_yuyv10PackedToyuyvUnpacked(decklink_video_in.m_RowLength,bTakeMask,decklink_video_in.m_iFrameSizeUnpacked,ptrThreadData->RGB_Output_Cuda,FourSettings[0].m_cunnyb,FourSettings[0].m_cunnyt,bAutoTrain);
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
			ptrThreadData->bUpdateRGB_Preview = true;
		}

		#ifdef PREVIEW_OUTPUTRENDER
			Launch_yuyv_Unpacked_GenerateMask(0, x,true);//
		`	Launch_yuyvDilateAndErode(FourSettings[x].m_iErode, FourSettings[x].m_iDilate, x);
		#else
			Launch_yuyv_Unpacked_GenerateMask(0, 0,bAutoTrain);//
			Launch_yuyvDilateAndErode(FourSettings[0].m_iErode, FourSettings[0].m_iDilate, 0);
			Launch_yuyv_Unpacked_GenerateMask_yolo_seg(0, 1,bAutoTrain,GetSegmentedMask());//
			//Launch_yuyvDilateAndErode(FourSettings[1].m_iErode, FourSettings[1].m_iDilate, 1);

		#endif

		auto timer_Launch_yuyvDilateAndErode_Launch_yuyv_Unpacked_GenerateMask_yolo = std::chrono::system_clock::now();
		Launch_yuyv_Unpacked_UnpackedComBineData(&m_BlendPos, &m_BlendPos, &m_BlendPos, decklink_video_in.m_RowLength, &FourSettings[0].m_ParabolicFunc, &FourSettings[1].m_ParabolicFunc, &FourSettings[2].m_ParabolicFunc,bBypass, iVrArCut+ 64, iVrArCut0+64, iVrArCut1 + 64, iVrArCut2 + 64, bTakeOutput);
		auto timer_Launch_yuyv_Unpacked_UnpackedComBineData = std::chrono::system_clock::now();
		if (bTakeOutput!=-1)
		{
			ptrThreadData->bUpdateRGB_Preview = true;
			//bTakeOutput = -1;
		}


		void *yuvdata = malloc(decklink_video_in.m_sizeOfFrame);
		CudaGetOutputData(yuvdata);
		decklink_video_in.ImagelistOutput.AddFrame(yuvdata);
		auto timer_end = std::chrono::system_clock::now();
		mtxScreenCard.unlock();
		auto duration_now=std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start).count();
	//	auto duration_now=std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start).count();
		auto duration_yolo=std::chrono::duration_cast<std::chrono::milliseconds>(timer_endyolo - timer_Launch_yuyv10PackedToyuyvUnpacked).count();

		auto timer_start_wait_duration=std::chrono::duration_cast<std::chrono::milliseconds>(timer_start-timer_wait_start ).count();//
		avg_duration[iAvgIndex++]=duration_now;
		if(iAvgIndex==AVG_CALC)
			iAvgIndex=0;
		if(Max_duration<duration_now)
			Max_duration=duration_now;

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
		std::cout <<"Launch_yuyv10PackedToyuyvUnpacked"<<std::endl;
		CudaChromaFree();
		cudaLookUpFree();
		return 0;


}


void ButtonCallback_KeyTopAndBottom(int state, void* userdata)
{





}




void InitSettingsWindows()
{
	namedWindow("Settings", WINDOW_NORMAL);
	//createTrackbar("Blending", "Settings", 0, 877, Blending, 0);
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

//	createTrackbar("AR/VR Cut", "Settings", &iVrArCut, (877),0, 0);
//	createTrackbar("VR Cut 1", "Settings", &iVrArCut0, (877), 0, 0);
//	createTrackbar("VR Cut 2", "Settings", &iVrArCut1, (877), 0, 0);
//	createTrackbar("VR Cut 3", "Settings", &iVrArCut2, (877), 0, 0);
//	setTrackbarPos("AR/VR Cut", "Settings", 877);

//	createTrackbar("canny L", "Settings", 0, 900, CUNNY_BOT, 0);
//	createTrackbar("canny T", "Settings", 0, 900, CUNNY_TOP, 0);
	createTrackbar("NMS", "Settings", 0, 100, NMS, 0);
	//createButton("Enable Key Top And Bottom,", ButtonCallback_KeyTopAndBottom, 0, QT_CHECKBOX, true);
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
	//setTrackbarPos("canny L", "Settings", FourSettings[iUpdateIndex].m_cunnyb);
	//setTrackbarPos("canny T", "Settings", FourSettings[iUpdateIndex].m_cunnyt);
	setTrackbarPos("NMS", "Settings", m_fNMS*100);

}

void on_opengl(void* param)
{

	glViewport(0, 0,  1366.0, 768.0);
	glMatrixMode(GL_PROJECTION);                        // Select The Projection Matrix
	glLoadIdentity();                           // Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	//	if(m_iWindowX>m_iWindowY)
	gluOrtho2D(0, 1366.0, 768.0, 0);

	//	gluPerspective(45.0f,(GLfloat)m_iWindowX/(GLfloat)m_iWindowY,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);                     // Select The Modelview Matrix
	glLoadIdentity();                           // Reset The Modelview Matrix
	/*glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho (0,1920,0,1080,-1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();*/
	//	glEnable(GL_TEXTURE_2D);                        // Enable Texture Mapping ( NEW )
	glEnable(GL_BLEND);
	glShadeModel(GL_SMOOTH);                        // Enable Smooth Shading
	glClearDepth(1.0f);                         // Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);                        // Enables Depth Testing
	glDepthFunc(GL_ALWAYS);                         // The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);          // Really Nice Perspective
	glDisable(GL_LIGHTING);
	glEnable(GL_MULTISAMPLE_ARB);
	//	glClear(
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

//	std::cout<<"Finished Setting the window"<<std::endl;


//	 glClear(GL_COLOR_BUFFER_BIT);
//
//	  // Draw a white grid "floor" for the tetrahedron to sit on.
//	  glColor3f(1.0, 1.0, 1.0);
//	  glBegin(GL_LINES);
//	  for (GLfloat i = -2.5; i <= 2.5; i += 0.25) {
//	    glVertex3f(i, 0, 2.5); glVertex3f(i, 0, -2.5);
//	    glVertex3f(2.5, 0, i); glVertex3f(-2.5, 0, i);
//	  }
//	  glEnd();
//
//	  // Draw the tetrahedron.  It is a four sided figure, so when defining it
//	  // with a triangle strip we have to repeat the last two vertices.
//	  glBegin(GL_TRIANGLE_STRIP);
//	    glColor3f(1, 1, 1); glVertex3f(0, 2, 0);
//	    glColor3f(1, 0, 0); glVertex3f(-1, 0, 1);
//	    glColor3f(0, 1, 0); glVertex3f(1, 0, 1);
//	    glColor3f(0, 0, 1); glVertex3f(0, 0, -1.4);
//	    glColor3f(1, 1, 1); glVertex3f(0, 2, 0);
//	    glColor3f(1, 0, 0); glVertex3f(-1, 0, 1);
//	  glEnd();
//
//	  glFlush();

}

void initOpenCVWindows()
{

	InitSettingsWindows();
	UpdateSettingsWindow();
	std::string rgbOutputWin = "RGB Output";
	std::string frameInfoWin = "Frame Info";

	namedWindow(rgbOutputWin,  WINDOW_NORMAL);
	//namedWindow("RGB Output1",  WINDOW_OPENGL);
	namedWindow(frameInfoWin, WINDOW_NORMAL);
//	namedWindow("Soccer Frame Info", WINDOW_OPENGL);
	setMouseCallback(rgbOutputWin, CallThisMouse, 0);
//	setOpenGlDrawCallback(rgbOutputWin, on_opengl, 0);
	updateWindow(rgbOutputWin);

//	std::cout<<"I run"<<std::endl;

}

void UpdateTXTFile(char *Buffer)
{
	std::ofstream outfile(GetTXTFileName(), std::ofstream::binary| std::ofstream::app);
	outfile.write(Buffer, strlen(Buffer));
}

void UpdateLookupFromMouse()
{

	if (bTaining || bErase)
	{
	/*	MASK->setTo(0);

		if (bTaining)
			rectangle(*MASK, Rect(Point(MouseData1.iXDown, MouseData1.iYDown), Point(MouseData1.iXUp, MouseData1.iYUp)), Scalar(255), CV_FILLED);
		if (bErase)
			rectangle(*MASK, Rect(Point(MouseData1.iXDown, MouseData1.iYDown), Point(MouseData1.iXUp, MouseData1.iYUp)), Scalar(128), CV_FILLED);

		return true;*/
	}
	else
	{

		if (bEnableClick)
		if (MouseData1.bHandleLDown)
		{

			//MASK->setTo(0);
			//rectangle(*MASK, Rect(Point(MouseData1.iXDownDynamic, MouseData1.iYDownDynamic), Point(MouseData1.iXUpDynamic, MouseData1.iYUpDynamic)), Scalar(255), CV_FILLED);

			if (!bSmall)
			{
				char buffer[MAX_PATH];
				sprintf(buffer, "1 %d %d %d %d %d\n", MouseData1.iXUpDynamic, MouseData1.iYUpDynamic, MouseData1.iXDownDynamic, MouseData1.iYDownDynamic,FourSettings[iUpdateIndex].m_iUV_Diam);
				UpdateTXTFile(buffer);
				//std::cout << buffer << std::endl;
				mtxScreenCard.lock();
				Launch_UpdateLookupFrom_XY_Posision(MouseData1.iXUpDynamic, MouseData1.iYUpDynamic, MouseData1.iXDownDynamic, MouseData1.iYDownDynamic, FourSettings[iUpdateIndex].m_iUV_Diam, FourSettings[iUpdateIndex].m_iLum_Diam, FourSettings[iUpdateIndex].m_iOuter_Diam, 200,bDoPaintBack);
				mtxScreenCard.unlock();

			}
			else
			{
				char buffer[MAX_PATH];
				sprintf(buffer, "1 %d %d %d %d\n", MouseData1.iXUpDynamic, MouseData1.iYUpDynamic, MouseData1.iXDownDynamic, MouseData1.iYDownDynamic);
				UpdateTXTFile(buffer);
				mtxScreenCard.lock();
				Launch_UpdateLookupFrom_XY_Posision(MouseData1.iXUpDynamic, MouseData1.iYUpDynamic, MouseData1.iXDownDynamic, MouseData1.iYDownDynamic, 1, 10, 5, 200,bDoPaintBack);
				mtxScreenCard.unlock();
			}
			return;
		}
		if (bEnableClick)
		if (MouseData1.bHandleRDown)
		{
			char buffer[MAX_PATH];
			sprintf(buffer, "0 %d %d %d %d\n", MouseData1.iXUpDynamic, MouseData1.iYUpDynamic, MouseData1.iXDownDynamic, MouseData1.iYDownDynamic);
			UpdateTXTFile(buffer);
			mtxScreenCard.lock();
			Launch_UpdateLookupFrom_XY_Posision_Erase(MouseData1.iXUpDynamic, MouseData1.iYUpDynamic, MouseData1.iXDownDynamic, MouseData1.iYDownDynamic, FourSettings[iUpdateIndex].m_iErase_Diam, FourSettings[iUpdateIndex].m_iErase_Lum_Diam,bDoPaintBack);
			mtxScreenCard.unlock();
		//	return;
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
//	while(waitKey(1)!=-1);
//	if(iKey!=-1)
	//	std::cout << iKey << std ::  endl;
//	std::this_thread::sleep_for(std::chrono::milliseconds(1));
}


bool GetAsyncKeyState(int checkpressed)
{
	if(checkpressed==iKey)
	{

		iKey=-1;
		return true;
	}
	else
		return false;

	return false;
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
//	Size textSize = getTextSize(text, fontFace,
//								fontScale, thickness, &baseline);
//	baseline += thickness;
	// center the text
	Point textOrg(r.x,r.y);


	putText(img, buffer, textOrg, fontFace, fontScale,
			Scalar::all(255), thickness, 8);
}





int main()
{



//	InitResnet18();

	//Classify(imread("/home/jurie/Pictures/resnet_test/1646501484_1_6.bmp"));

	static int iIndex=0;
	static int iFrameIndex=0;
	//DestroyResnet18();


	StartMonitor();

	if (1)
	{

#ifndef PREVIEW_OUTPUTRENDER

		initCameraUDPData();
		initOpenCVWindows();
		//InitVizSocket();
#endif

		cuda::GpuMat RGB_Output_Cuda;
		RGB_Output_Cuda.create(1080, 1920, CV_8UC3);
		RGB_Output_Cuda.step = 5760;

		cuda::GpuMat RGB_FrameInfo_Cuda;
		RGB_FrameInfo_Cuda.create(1024, 1024, CV_8UC3);
		RGB_FrameInfo_Cuda.step = 1024 * 3;

		cuda::GpuMat RGB_FrameInfo_Cuda_FullUpdate;
		RGB_FrameInfo_Cuda_FullUpdate.create(1024, 1024, CV_8UC3);
		RGB_FrameInfo_Cuda_FullUpdate.step = 1024 * 3;


		myThreadData.RGB_Output_Cuda = &RGB_Output_Cuda;
		Mat		MASK_L(1080, 1920, CV_8UC1, Scalar(0));
		myThreadData.MouseData1 = &MouseData1;
	//	imshow("RGB Output", RGB_Output_Cuda);


		pthread_t threads;
		int rc;
		rc = pthread_create(&threads, NULL, OutputRenderthread, (void *) &myThreadData);

		if (rc)
		{
			cout << "Error:unable to create thread," << rc << endl;
			exit(-1);
		}



		initPosUDPData();
		Mat RGB__Draw;
		Mat RGB_Output;
		Mat RGB_saving;
	//	std::vector<Yolo_Big::Detection> DetectionS;

		bool bstart = false;
		bool bCapture=false;

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
//			if (GetAsyncKeyState('s'))
//			{
//				//if (GetAsyncKeyState(VK_LCONTROL))
//				{
//					std::ofstream outfile("settings.crm", std::ofstream::binary);
//					if (outfile.good())
//					{
//						outfile.seekp(ios_base::beg);
//						outfile.write((char *)&FourSettings, sizeof(WindowSettings) * 3);
//						outfile.close();
//						printf("settings saved to settings.crm\n");
//
//					}
//					else
//					{
//						printf("could not load settings\n");
//					}
//				}
//			}
//
//
//			if (GetAsyncKeyState('l'))
//			{
//				//if (GetAsyncKeyState(VK_LCONTROL))
//				{
//					std::ifstream  infile("settings.crm", std::ofstream::binary);
//					if (infile.good())
//					{
//						infile.seekg(ios_base::beg);
//						infile.read((char *)&FourSettings, sizeof(WindowSettings) * 3);
//						infile.close();
//						printf("settings loaded to settings.crm\n");
//					}	else
//					{
//						printf("could not save settings\n");
//					}
//				}
//
//				UpdateSettingsWindow();
//			//	while (GetAsyncKeyState('L'));
//			}


			if (GetAsyncKeyState('g'))
			{
				cudaLookUpFullKey();
		//		while (GetAsyncKeyState('F'));
			}


			if (GetAsyncKeyState('r'))//||(UI_Frame_Counter%10)==0
			{
				cudaLookUpReset(0);
			//	while (GetAsyncKeyState('R'));
			}

			if (GetAsyncKeyState('f'))//||(UI_Frame_Counter%10)==0
			{
				cudaLookUpReset(1);
			//	while (GetAsyncKeyState('R'));
			}

//			if (GetAsyncKeyState('1'))
//			{
//				iUpdateIndex = 0;
//				UpdateSettingsWindow();
//				SetOnAirLookup(0);
//				while (GetAsyncKeyState('1'));
//
//				setWindowTitle("Settings","Settings 1");
//				setWindowTitle("RGB Output", "Settings 1");
//			}
//			if (GetAsyncKeyState('2'))
//			{
//				iUpdateIndex = 1;
//				UpdateSettingsWindow();
//				SetOnAirLookup(1);
//				while (GetAsyncKeyState('2'));
//
//				setWindowTitle("Settings", "Settings 2");
//				setWindowTitle("RGB Output", "Settings 2");
//			}
//			if (GetAsyncKeyState('3'))
//			{
//				iUpdateIndex = 2;
//				UpdateSettingsWindow();
//				SetOnAirLookup(2);
//				while (GetAsyncKeyState('3'));
//
//				setWindowTitle("Settings", "Settings 3");
//				setWindowTitle("RGB Output", "Settings 3");
//			}
			if (GetAsyncKeyState('P'))
			{
				//InKey->ClearAll();
				//InFill->ClearAll();
			//	while (GetAsyncKeyState('P'));
			}


			if (GetAsyncKeyState('o'))
			{
				bClearOutPut=true;
				//InPut->ClearOutPut();
			//	while (GetAsyncKeyState('O'));
			}


			if (GetAsyncKeyState('b'))
			{
				bBypass = !bBypass;
			//	while (GetAsyncKeyState('B'));
			}

			if (GetAsyncKeyState('h'))
			{
				for(int x=0;x<10;x++)
					printf("\n\r");
				//printf("'1','2','3' � select user updateble lookup tables\n\r");
				printf("'q' chroma snapshot update \n\r");
				printf("'a' PaintItBack snapshot update \n\r");
				printf("'r' reset chroma lookup table\n\r");
				printf("'f' reset PaintItBack lookup table\n\r");
				printf("'o' to clear output buffer frame list\n\r");
				printf("'i' display frame info\n\r");
				printf("ctl+'l' load settings\n\r");
				printf("ctl+'s' save settings\n\r");

				for (int x = 0; x < 10; x++)
					printf("\n\r");
			//    while (GetAsyncKeyState('H'));
			}



//			if (GetAsyncKeyState('A'))
//			{
//				bSmall = true;
//			}else
//				bSmall = false;
			static unsigned int bFrameTimer = 0;
			bFrameTimer++;


			if (GetAsyncKeyState('w'))
			{

				bTakeOutput = 0;
				bAutoTrain=true;

			}else
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
			//	if (!GetAsyncKeyState(VK_LCONTROL))
			//		bSafeSnapshot = true;
				mtxScreenCard.lock();
				bTakeMask = true;
			//	std::cout << "Snap Shot" << std::endl;

				mtxScreenCard.unlock();
				setWindowTitle("RGB Output", "Chroma");
				setWindowTitle("Settings","Settings Chroma");
			//	while (GetAsyncKeyState('Q'));
			}


			if (GetAsyncKeyState('a'))
			{

				SetOnAirLookup(1);
				iUpdateIndex = 1;
				UpdateSettingsWindow();
				iLastCheck = 0;
				bTakeOutput = -1;
				bDoPaintBack=true;
			//	if (!GetAsyncKeyState(VK_LCONTROL))
			//		bSafeSnapshot = true;
				mtxScreenCard.lock();
				bTakeMask = true;
				//std::cout << "Snap Shot" << std::endl;
				mtxScreenCard.unlock();
				setWindowTitle("RGB Output", "PaintItBack");
				setWindowTitle("Settings","PaintItBack");
			//	while (GetAsyncKeyState('Q'));
			}


			if (GetAsyncKeyState(VK_F1))
			{
				bTakeOutput = 0;
			//	while (GetAsyncKeyState(VK_F1));
			}

			if (GetAsyncKeyState(VK_F2))
			{
				bTakeOutput = 1;
			//	while (GetAsyncKeyState(VK_F2));
			}

			if (GetAsyncKeyState(VK_F3))
			{
				bTakeOutput = 2;
		//		while (GetAsyncKeyState(VK_F3));
			}
			if (GetAsyncKeyState(VK_F4))
			{
				bTakeOutput = 3;
				//		while (GetAsyncKeyState(VK_F3));
			}
			if (GetAsyncKeyState(VK_F5))
			{
				bTakeOutput = 4;
				//		while (GetAsyncKeyState(VK_F3));
			}

			if (GetAsyncKeyState(VK_F10))
				CameraZero();

			if (GetAsyncKeyState('s'))
			{
				std::string FileAndPathName;
				std::time_t result = std::time(nullptr);
				std::string  FileName= toString(result);
				FileAndPathName="/home/jurie/Pictures/yolov5_soccer_training/"+FileName;
				std::thread t1(SaveImageThread,RGB_saving.clone(),iIndex++,FileAndPathName);
				t1.join();
				if(iIndex==4)
					iIndex=0;
			}
//				iLastCheck = 1;
//				if (!GetAsyncKeyState(VK_LCONTROL))
//					bSafeSnapshot = true;
//
//			//	while (GetAsyncKeyState('W'));
//			}
//		//	Rect2d r = selectROI(im);

			if (myThreadData.bUpdateRGB_Preview)
			{
				myThreadData.bUpdateRGB_Preview = false;
			//	std::cout << "waiting" << std::endl;
				mtxScreenCard.lock();


				RGB_Output_Cuda.download(RGB_Output);
				RGB_saving=RGB_Output.clone();
				iFrameIndex++;
			//	std::cout << iFrameIndex << std::endl;
				if(0)
				if(iFrameIndex==50)
				{
					iFrameIndex=0;
					std::string FileAndPathName;
					std::time_t result = std::time(nullptr);
					std::string  FileName= toString(result);
					FileAndPathName="/home/jurie/Pictures/yolov5_soccer_training/"+FileName;
					std::thread t1(SaveImageThread,RGB_Output.clone(),iIndex++,FileAndPathName);
					t1.join();
					if(iIndex==4)
						iIndex=0;
				}
				DrawSnapShotDetections_clean(&RGB_Output,bTrackReset);
				//imwrite("/home/jurie/Documents/textures/from_camera.bmp",RGB_Output);
				//RGB_Output.cols=1920*2;
				//RGB_Output.rows=1080/2;

			//	DrawSnapShotDetectionsPTR(&RGB_Output,bTrackReset);
				if(bCapture)
					writeframe(RGB_Output.clone());
				bTrackReset=false;
				DrawCameraData(&RGB_Output);
				//RGB_Output.cols=1920;
				//RGB_Output.rows=1080;
				//writeframe(RGB_Output);
				bstart = true;
			//	bTakeOutput = 0;

			//	bTakeMask = true;
				mtxScreenCard.unlock();
			}
			if (bstart)
			{
				RGB__Draw = RGB_Output.clone();
				cv::Rect myROI(MouseData1.iXUpDynamic, MouseData1.iYUpDynamic, MouseData1.iXDownDynamic - MouseData1.iXUpDynamic, MouseData1.iYDownDynamic - MouseData1.iYUpDynamic);
				if((0 <= myROI.x && 0 <= myROI.width && myROI.x + myROI.width <= RGB__Draw.cols &&
					0 <= myROI.y && 0 <= myROI.height && myROI.y + myROI.height <= RGB__Draw.rows))
				{
					MouseMutex.lock();
					Mat RGB__Draw_Small = RGB__Draw(myROI);
					Mat RGB__Draw_SmallEnlarge;
					Size ssize = RGB__Draw_Small.size();
					if (!ssize.empty())
					{
						cv::resize(RGB__Draw_Small, RGB__Draw_SmallEnlarge, Size((MouseData1.iXDownDynamic - MouseData1.iXUpDynamic) * 25, (MouseData1.iYDownDynamic - MouseData1.iYUpDynamic) * 25), 0, 0, INTER_NEAREST);
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
				circle(RGB__Draw,Point(MouseData1.x,MouseData1.y),20,Scalar(255,255,255),3);
			//	RGB__Draw.cols=1920*2;
			//	RGB__Draw.rows=1080/2;


				DrawMouseText(&RGB__Draw,"Hello World",cv::Point(50,50));
				DrawOutputThreadData(&RGB__Draw);
				imshow("RGB Output", RGB__Draw);
				RGB__Draw.release();
			}

			if(0)
			if ((bFrameTimer%20)==0)
			{
				Launch_Frame_Info(&RGB_FrameInfo_Cuda);
				imshow("Frame Info", RGB_FrameInfo_Cuda);
			}

			if (GetAsyncKeyState('i'))
			{
				Launch_Frame_Info(&RGB_FrameInfo_Cuda);
				imshow("Frame Info", RGB_FrameInfo_Cuda);
			//	while (GetAsyncKeyState('I'));
			}

			UpdateLookupFromMouse();
			UpdateKeyState();


//			if (iKey == 49)
//			{
//				bTakeOutput = 1;
//
//
//			}
			if (GetAsyncKeyState(27))//"Esc"
			{
				// Do some clean up and free memory, c++ garbage collector doesn't clean up some things.
//				delete RGB__Draw.data;
				bExitWorkerThread = true;
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				EndLoop();
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				std::cout << "Exit" << std::endl;
				break;
//				bExite = true;
//				printf("Press 'Y' to exit or 'n' to cancel\n\r");
//				iExitCount = 0;
			}
			if (bExite)
			{
				iExitCount++;
			//	printf("%d\n\r", iExitCount);
				if (iExitCount == 100)
				{
					bExite = false;
					printf("Exit process canceled Exit process canceledExit process canceledExit process canceled\n\r");
				}
			}

			if (GetAsyncKeyState('y'))
				if (bExite)
				{
					//ExitMonitor();
					bExitWorkerThread = true;

					std::this_thread::sleep_for(std::chrono::milliseconds(1000));
					break;

				}

			if (GetAsyncKeyState('n'))
				//if (bExite)
					bExite = false;
		};


	}/*else
	ReadFileFast(true);*/
//	cudaProfilerStop();
	cudaError_t cudaStatus;
	 ExitMonitor();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	std::cout << "End Cuda" << std::endl;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}


	return 0;
}

// Helper function for using CUDA to add vectors in parallel.




