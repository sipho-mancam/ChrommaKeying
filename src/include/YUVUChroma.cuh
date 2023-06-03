


#ifndef YUVUCHROMA_H
#define YUVUCHROMA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
#include <list>


#include "opencv2/cudaimgproc.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "yololayer.h"
//#include "../cuda/cudaUtility.h"
//#include "../cuda/cudaYUV.h"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;
using namespace cv::cuda;



bool CudaSetInputData(void *mydataIN, void *mydataIN1, void *mydataIN2,bool SafeSnapShot);
bool CudaGetOutputData(void *mydataOut);
bool CudaChromaInit(int iWidth, int iHeight, int iFrameSizeYUV10Bit,int iFrameSizeUnpacked);
bool CudaChromaFree();
//bool ConvertToRGB(void *RGBData, unsigned char blend);

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
	std::string windowName;

};

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
	int m_BlendPos;
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
			m_BlendPos = 0;
			m_iLowerlimit = 80;
			m_iUpperlimit=80;
		}


};
//void Launch_yuyvToRgba10(int RowLength, void *RGBData, int iBlendPos, void *Mask, bool bDownloadRGB);
__global__ void yuyvUnpackedToRGB(uint4* ,  uchar3*, int , int , int ,  uint4*);
__global__ void yuyvPackedToyuyvUnpacked(uint4* , uint4 *,int , int , int );
__global__ void yuyvUnPackedToPlanarRGB_Split(uint4* , uint8_t *,uint8_t *, uint8_t *, uint8_t *, uint32_t , int ,int , int );
__global__ void yuyvUnpackedComBineDataThreeLookups(uint4* , uint4* , uint4* , int , int , int , int , uchar *, uchar *, uchar *, int , int , int , double4 , double4 , double4 , unsigned long int , unsigned long int , unsigned long int , unsigned long int );
__global__ void yuyvUnPackedToPlanarRGB_Split(uint4*, uint8_t* ,uint8_t*, uint8_t*, uint8_t*,	uint32_t ,int,int , int);
__global__ void yuyvUmPackedToRGB_lookup(uint4* , uchar3* , int , int , int , uint4* , uchar* );
__global__ void yuyv_Unpacked_GenerateMask(uint4* , uchar*, uchar* , int , int, int , int ,int);
__global__ void UpdateLookupFrom_XY_Posision_Diffrent_Scaling(uint4* , uchar* , int , int , int , int ,int , int ,float ,int );//


void PrepareYoloData(bool ,float );
void SetOnAirLookup(int );
bool Checkifnotoverplayer(cv::Rect );
void DrawSnapShotDetections(Mat *,bool );
void DrawSnapShotDetections_clean(Mat *,bool );
void DrawSnapShotDetectionsPTR(Mat *,bool );
void Launch_yuyv10PackedToyuyvUnpacked(int ,bool ,int , cuda::GpuMat *,int ,int ,bool );
void Launch_yuyv_Unpacked_GenerateMask(int ,int ,bool );
void Launch_yuyv_Unpacked_GenerateMask_yolo(int ,int ,bool );
void Launch_yuyv_Unpacked_GenerateMask_yolo_seg(int ,int ,bool ,float *);

void CreateYoloMask();
void Launch_yuyv_Unpacked_UnpackedComBineData(int *, int *, int *, int , double4 *, double4 *, double4 *, bool , unsigned long int , unsigned long int , unsigned long int , unsigned long int ,int );
void Launch_yuyvDilateAndErode(int , int , int );
void Launch_yuyv_Unpacked_ClearMask(int );
void Launch_UpdateLookupFrom_XY_Posision(int , int , int , int , int , int , int ,int ,bool );
void Launch_UpdateLookupFrom_XY_Posision_Erase(int , int , int , int , int , int ,bool );
void Launch_Frame_Info(cuda::GpuMat *);
void Launch_Frame_Info_SccoerBall(cuda::GpuMat *);
void Launch_UpdateLookup_Test();
void UpdateLookup(int iStartX, int , int , int ,bool );

	

cudaError_t cudaLookUpInit();
cudaError_t cudaLookUpFree();


int cudaDumpLookUp();
int cudaLoadLookUp(); 
cudaError_t cudaLookUpReset();
cudaError_t cudaLookUpReset(int iLookUp);
cudaError_t cudaLookUpFullKey();

bool DownloadMaskA(cv::Mat *mydataOut);
bool DownloadMaskB(cv::Mat *mydataOut);
char *GetTXTFileName();


struct TrackedObj
{
public:
	TrackedObj()
	{
		iActive=1;
		bSendToViz=false;
		iFrameCounter=0;
		iObjID=-1;
		iClasified=-1;
	//	std::cout << "TrackedOBJ Created" << std::endl;
	}
	~TrackedObj()
	{


	//	std::cout << "TrackedOBJ deleted" << std::endl;


	}

	long iObjID;
	unsigned int iFrameCounter;
	int iActive;
	bool bSendToViz;
	int iClasified;
	string Label;
	cv::Scalar Color;
	Detection m_Detection;

};




#endif
