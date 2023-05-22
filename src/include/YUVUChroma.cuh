


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


//void Launch_yuyvToRgba10(int RowLength, void *RGBData, int iBlendPos, void *Mask, bool bDownloadRGB);

__global__ void yuyvPackedToyuyvUnpacked(uint4* src_Video, uint4 *dst_video_all,int srcAlignedWidth, int dstAlignedWidth, int height);
__global__ void yuyvUnPackedToPlanarRGB_Split(uint4* src_Unapc, uint8_t *dpRgbA,uint8_t *dpRgbB, uint8_t *dpRgbC, uint8_t *dpRgbD,	uint32_t dstPlanePitchDst/*640 *sizeof(float)*/, int srcAlignedWidth,		int height, int dstHeight);
__global__ void yuyvUnpackedComBineDataThreeLookups(uint4* src_Video_Unapc, uint4* src__Fill_Unapc, uint4* src__Key_Unapc, int width, int height, int srcAlignedWidth, int dstAlignedWidth, uchar *maskUpload0, uchar *maskUpload1, uchar *maskUpload2, int iBlendPos0, int iBlendPos1, int iBlendPos2, double4 Parabolic0, double4 Parabolic1, double4 Parabolic2, unsigned long int iCutOff, unsigned long int iCutOff0, unsigned long int iCutOff1, unsigned long int iCutOff2);
__global__ void yuyvUnPackedToPlanarRGB_Split(uint4* src_Unapc, uint8_t *dpRgbA,uint8_t *dpRgbB, uint8_t *dpRgbC, uint8_t *dpRgbD,	uint32_t dstPlanePitchDst/*640 *sizeof(float)*/, int srcAlignedWidth,		int height, int dstHeight);
__global__ void yuyvUmPackedToRGB_lookup(uint4* , uchar3* , int , int , int , uint4* , uchar* );

void PrepareYoloData(bool bSnapShot,float fnms);
void SetOnAirLookup(int iLookup);
bool Checkifnotoverplayer(cv::Rect TestRect);
void DrawSnapShotDetections(Mat *DrawingMat,bool bTrackReset);
void DrawSnapShotDetections_clean(Mat *DrawingMat,bool bTrackReset);
void DrawSnapShotDetectionsPTR(Mat *DrawingMat,bool bTrackReset);
void Launch_yuyv10PackedToyuyvUnpacked(int RowLength,bool bSnapShot,int iFrameSizeUnpacked, cuda::GpuMat *RGB_Output_Cuda,int iBot,int iTop,bool bAutoTrain);
void Launch_yuyv_Unpacked_GenerateMask(int iAvgCutOff,int iUse,bool bAutoTrain);
void Launch_yuyv_Unpacked_GenerateMask_yolo(int iAvgCutOff,int iUse,bool bAutoTrain);
void Launch_yuyv_Unpacked_GenerateMask_yolo_seg(int iAvgCutOff,int iUse,bool bAutoTrain,float *segmented_mask);

void CreateYoloMask();
//void Launch_yuyv_Unpacked_UnpackedComBineData(int iBlendPos, int RowLength, double3 Parabolic,bool Bypass,bool m_DisableParabolicKeying, unsigned long int iCutOff);
void Launch_yuyv_Unpacked_UnpackedComBineData(int *iBlendPos0, int *iBlendPos1, int *iBlendPos2, int RowLength, double4 *Parabolic0, double4 *Parabolic1, double4 *Parabolic2, bool Bypass, unsigned long int iCutOff, unsigned long int iCutOff0, unsigned long int iCutOff1, unsigned long int iCutOff2,int bOutPutSnap);
void Launch_yuyvDilateAndErode(int iDilate, int iErode, int iUse);
void Launch_yuyv_Unpacked_ClearMask(int iUse);
void Launch_UpdateLookupFrom_XY_Posision(int istartX, int iStartY, int iEndX, int iEndY, int iUV_Diameter, int iLum_Diameter, int iOuter_Diameter,int iMaxKeyVal,bool bPaintItBack);
void Launch_UpdateLookupFrom_XY_Posision_Erase(int istartX, int iStartY, int iEndX, int iEndY, int iErase_Diameter, int iErase_Lum_Diameter,bool bPaintItBack);
void Launch_Frame_Info(cuda::GpuMat *RGB_FrameInfo);
void Launch_Frame_Info_SccoerBall(cuda::GpuMat *RGB_FrameInfo);
void Launch_UpdateLookup_Test();
void UpdateLookup(int iStartX, int iEndX, int iSartY, int iEndY,bool bTrain);

	

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
