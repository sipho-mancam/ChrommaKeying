#include "YUVUChroma.cuh"

#include <thread>         // std::thread
#include "npp.h"
#include "yololayer.h"
#include <map>
#include <iostream>
#include <string>
//#include <string_view>
#include <iostream>   // std::cout
#include <string>     // std::string, std::to_string
#include <sstream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudaoptflow.hpp>
#include "PosisionUpdateUDP.h"
//#define PREVIEW_OUTPUTRENDER

using namespace std;

inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }
#define SIZE_ULONG4_CUDA 16

uchar2* YUV_Upload_Video_YUV = 0;
uchar2* YUV_Upload_Key = 0;/*KEY*/
uchar2* YUV_Upload_Fill = 0;/*FILL*/
extern float *GetSegmentedMaskSnapshot();
extern std::vector<Detection> doInference_YoloV5(void *remote_buffers,float fnms,bool bSnaphot);;
std::vector<Detection> Yolov5Detection;
std::vector<Detection> SnapYolov5Detection;




uint4* YUV_Unpacked_Video = 0;
uint4* YUV_Unpacked_Key = 0;
uint4* YUV_Unpacked_Fill = 0;


uchar* Y_Unpacked_TraingMask2;


uint16_t* Y_Unpacked_Video = 0;
uint16_t* U_Unpacked_Video = 0;
uint16_t* V_Unpacked_Video = 0;
GpuMat GpuImg0;
GpuMat GpuImg1;
bool bDoOpticalFlow=true;
extern void SendVizSocket(int x,int y);
extern void initPosUDPData();
extern int Classify(cv::Mat img_size);


static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}


#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
void* m_RGBScaledFramePlanarDetectorptrs[8] =
{ 0, 0, 0, 0,0, 0, 0, 0 };
void* m_RGBScaledFramePlanarDetector;

uint4* YUV_Unpacked_Video_Duplicate = 0;//duplicate of yuv unpacked data
uint4* YUV_Unpacked_Video_SnapShot = 0;
uint4* YUV_Unpacked_Key_SnapShot = 0;

uint4* FrameColorData_Unpacked = 0;
uint4* LookUpColorDataOneDimention_Unpacked = 0;
ulong4* UploadChromaData_Unpacked_SnapShot = 0;

uchar* YoloGeneratedMask ;
uchar* ChromaGeneratedMask[3] ;
uchar* ptrChromaGeneratedMask = 0;
uchar* MaskRefineScratch = 0;

uchar3* DownloadRGBData = 0;

#define MAX_PATH 260
uchar3* DownloadRGBData_Frame_Info = 0;
uchar3* DownloadRGBData_Frame_InfoFull = 0;


int m_iWidth = -1;
int m_iHeight = -1;
long m_lFrameSizeYUV10Bit = -1;

char mySaveTxtPath[MAX_PATH];
#define MAX_LOOK_UP 3
#define BATCH_SIZE 8
uchar* LookUpDataArry[MAX_LOOK_UP];
uchar* ptrLookUpData;
uchar* ptrLookUpDataGenralUse;
int iOnAirLookup = 0;

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;
using namespace cv::cuda;




struct Distances
{
	Detection m_Person01;
	Detection m_Person02;


	void SetCenter(Detection Ball)
	{
		m_Person02=Ball;
		angleDivide=GetAngle(1920/2,1080/2,Ball.bbox[0],Ball.bbox[1]);
		angleToDegrees = atan(angleDivide);
		m_distance=distance(1920/2,1080/2,Ball.bbox[0],Ball.bbox[1]);


	}

	bool CheckCondition(Detection Person01,Detection Person02,int sizecondition)
	{

		if(abs(m_Person02.bbox[0]-m_Person01.bbox[0])>sizecondition||abs(m_Person02.bbox[1]-m_Person01.bbox[1])>sizecondition)
			return false;
		return true;


	}
	void Set(Detection Person01,Detection Person02)
	{
		m_Person01=Person01;
		m_Person02=Person02;
		volume1=m_Person01.bbox[2]*m_Person01.bbox[3];
		volume2=m_Person02.bbox[2]*m_Person02.bbox[3];
		volume_diff=abs(volume1-volume2)/((volume1+volume2)/2);

		angleDivide=GetAngle(m_Person02.bbox[0],m_Person02.bbox[1],m_Person01.bbox[0],m_Person01.bbox[1]);
		angleToDegrees = atan(angleDivide);
		m_distance=distance(m_Person02.bbox[0]+m_Person02.bbox[2]/2,m_Person02.bbox[1]+m_Person02.bbox[3]/2,m_Person01.bbox[0]+m_Person01.bbox[2]/2,m_Person01.bbox[1]+m_Person01.bbox[3]/2);



	}
	void Draw(cv::Mat *img)
	{
		cv::line(*img, cv::Point(m_Person01.bbox[0],m_Person01.bbox[1]*2),cv::Point(m_Person02.bbox[0],m_Person02.bbox[1]*2), Scalar(255,255,255), 1);
		//imshow("HistoryKeep",image_keep);
	}
		double m_distance;
		double angleDivide;
		double  angleToDegrees ;
		double volume1;
		double volume2;
		double volume_diff;
		double GetAngle(double x1, double y1,double y2,double x2 ){
		double dot = x1*x2 + y1*y2      ;//# dot product between [x1, y1] and [x2, y2]
		double det = x1*y2 - y1*x2      ;//# determinant
		double angle = atan2(det, dot) ; //# atan2(y, x) or atan2(sin, cos)
		return angle ;
	}
	float distance(float x1, float y1, float x2, float y2)
	{
	    // Calculating distance
	    return sqrt(pow(x2 - x1, 2) +
	                pow(y2 - y1, 2) * 1.0);
	}

};

 

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

//https://developer.apple.com/library/archive/technotes/tn2162/_index.html#v210
//__global__ void yuyvUnpackedComBineData(ulong4* src_Video_Unapc, ulong4* src__Fill_Unapc, ulong4* src__Key_Unapc, int width, int height, int srcAlignedWidth, int dstAlignedWidth, uchar *maskUpload, int iBlendPos, unsigned long int iCutOff);
__global__ void yuyvUnpackedComBineData(uint4* src_Video_Unapc, uint4* src__Fill_Unapc, uint4* src__Key_Unapc, int width, int height, int srcAlignedWidth, int dstAlignedWidth, uchar *maskUpload, int iBlendPos, double3 Parabolic, unsigned long int iCutOff);

__global__ void yuyvUnpackedComBineDataChromaBipass(uint4* src_Video_Unapc, uint4* src__Fill_Unapc, uint4* src__Key_Unapc, int width, int height, int srcAlignedWidth);
#define CUDA_LOOKUP_SIZE   1073741824// 134217728 1024*1024*1024



void SetOnAirLookup(int iLookup)
{
	iOnAirLookup = iLookup;
	//printf("LookUp %d\n\r", iOnAirLookup);
}

cudaError_t cudaLookUpFree()
{
	cudaError_t cudaStatus;

	for (int x = 0; x < MAX_LOOK_UP; x++)
	{
		cudaStatus = cudaFree(LookUpDataArry[x]);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudamemory release failed!");
			return cudaStatus;
		}
	}
	//LookUpData = LookUpDataArry[0];
	return cudaStatus;
}


cudaError_t cudaLookReset()
{
cudaError_t cudaStatus;

	for (int x = 0; x < MAX_LOOK_UP; x++)
	{

		cudaStatus = cudaMemset(LookUpDataArry[x], 0, CUDA_LOOKUP_SIZE);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return cudaStatus;
		}
	}

	return cudaStatus;
}


cudaError_t cudaLookUpInit()
{ 
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&MaskRefineScratch, (1920 * 1080) * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	 
	for (int x = 0; x < MAX_LOOK_UP; x++)
	{
		cudaStatus = cudaMalloc((void**)&LookUpDataArry[x], CUDA_LOOKUP_SIZE);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return cudaStatus;
		}
		cudaStatus = cudaMemset(LookUpDataArry[x], 0, CUDA_LOOKUP_SIZE);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return cudaStatus;
		}	
	}
	ptrLookUpData = LookUpDataArry[0];
	return cudaStatus;
}


cudaError_t cudaLookUpFullKey()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemset(ptrLookUpData, 255, CUDA_LOOKUP_SIZE);//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMemset(LookUpColorDataOneDimention_Unpacked, 255, (1024 / 2 * sizeof(uint4) * 1024));//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return  cudaStatus;
	}

	printf("\n\rLookup Data Reset\n\r");
	return cudaStatus;
}


cudaError_t cudaLookUpReset(int iLookUp)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemset(LookUpDataArry[iLookUp], 0, CUDA_LOOKUP_SIZE);//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMemset(LookUpColorDataOneDimention_Unpacked, 0, (1024 / 2 * sizeof(uint4) * 1024));//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
 		return  cudaStatus;
	}

	printf("\n\rLookup Data Reset\n\r");
	return cudaStatus;
}


cudaError_t cudaLookUpReset()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemset(ptrLookUpData, 0, CUDA_LOOKUP_SIZE);//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMemset(LookUpColorDataOneDimention_Unpacked, 0, (1024 / 2 * sizeof(uint4) * 1024));//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
 		return  cudaStatus;
	}

	printf("\n\rLookup Data Reset\n\r");
	return cudaStatus;
}

inline __device__ double GetBitPos3(double3 pos)
{
	return (pos.x* 1048576.0) + (pos.y*1024.0) + pos.z;
}


inline __device__ double GetBitPos(double4 pos)
{
	return (((int)pos.x) >>2) + (((int)pos.y) >> 2) / 2.0 * 524288 + (((int)pos.z )) * (1024.0) + (((int)pos.w));
}

inline __device__  bool GetBit(double4 pos, uchar* LookupTable)
{
	//	float bitpos = (int)pos.x * 1046529 + (int)pos.y * 1023.0 + (int)pos.z;for 10bit
	double bitpos = GetBitPos(pos);///* (((int)pos.x) >> 2) * (16581375.0) + (((int)pos.y) >> 2)*(65025.0) +*/ (((int)pos.x) >> 1) + (((int)pos.y) >> 1) / 2.0 * 261120 + ((pos.z >> 2)) * (1024.0) + ((pos.w));

	double GetByteDouble = bitpos / 8.0;
	int64 GetByteInt = (int64)GetByteDouble;
	int GetBitPos = (GetByteDouble - (double)GetByteInt) * 8.0;
	unsigned char *llVal = LookupTable + GetByteInt;
	unsigned char vals = 1 << GetBitPos;
	
	unsigned char ret = (*llVal & vals);
	return ret;
}

inline __device__  uchar GetBit3(double pos, uchar* LookupTable)
{
	if (pos < 0)
		return 255;

	if (pos > CUDA_LOOKUP_SIZE)
		return 255;

	int64 GetByteInt = (int64)pos;
	unsigned char *llVal = LookupTable + GetByteInt;
	uchar valueof = *llVal;
	return valueof;
}

inline __device__ void SetBit3(double pos, uchar* LookupTable,uchar value)
{
	if (0 > pos)
		return;

	if (CUDA_LOOKUP_SIZE < pos)
		return ;

	int64 GetByteInt = (int64)pos;
	uchar *llVal = LookupTable + GetByteInt;
	*llVal= value;
}
inline __device__ void ClearBit3(double pos, uchar* LookupTable)
{
	if (0 > pos)
		return;
	if (CUDA_LOOKUP_SIZE < pos)
		return;
	int64 GetByteInt = (int64)pos;
	uchar *llVal = LookupTable + GetByteInt;
	*llVal = 0;
}




inline __device__ void SetBit(double4 pos, uchar* LookupTable)
{
	double bitpos = GetBitPos(pos); 
//	if (CUDA_LOOKUP_SIZE < bitpos)
//		return;
	double GetByteDouble = bitpos /8.0;
	int64 GetByteInt = (int64)GetByteDouble;

	int myval = (GetByteDouble - (double)GetByteInt)*8.0;
	unsigned char *llVal = LookupTable + GetByteInt;
//	printf("%d %d %d %d %f %f  %d %lld\n", (int)pos.x, (int)pos.y, (int)pos.z, (int)pos.w, bitpos, GetByteDouble, myval, GetByteInt);
	*llVal |= 1UL << myval;
}

inline __device__ void ClearBit(double4 pos, uchar* LookupTable)
{
	double bitpos = GetBitPos(pos); 
	double GetByteDouble = bitpos / 8.0;
	int64 GetByteInt = (int64)GetByteDouble;

	int myval = (GetByteDouble - (double)GetByteInt)*8.0;
	unsigned char *llVal = LookupTable + GetByteInt;
	*llVal &= ~(1UL << myval);
}

__device__  float4 Sum(uchar4 *pos, uchar4 *pos1)
{
	float4 val;
	val.w = pos->w + pos->w;
	val.x = pos->x + pos->x;
	val.y = pos->y + pos->y;
	val.z = pos->z + pos->z;
	return val;

}
__device__  float4 Sumfloat(float4 *pos, float4 *pos1)
{
	float4 val;
	val.w = pos->w + pos->w;
	val.x = pos->x + pos->x;
	val.y = pos->y + pos->y;
	val.z = pos->z + pos->z;
	return val;

}





bool CudaChromaInit(int iWidth, int iHeight, int iFrameSizeYUV10Bit, int iFrameSizeUnpacked/*ulong*/)
{


	m_iWidth = iWidth;
	m_iHeight = iHeight;
	m_lFrameSizeYUV10Bit = iFrameSizeYUV10Bit;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&YUV_Upload_Key, iFrameSizeYUV10Bit);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Here");
		return false;
	}

	cudaStatus = cudaMalloc((void**)&YUV_Upload_Fill, iFrameSizeYUV10Bit);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}

	cudaStatus = cudaMalloc((void**)&V_Unpacked_Video, 960*1080*sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}




	cudaStatus = cudaMalloc((void**)&Y_Unpacked_TraingMask2, 1920*1080*sizeof(uchar));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				return false;
			}
	cudaStatus = cudaMalloc((void**)&Y_Unpacked_Video, 1920*1080*sizeof(uint16_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return false;
		}


		cudaStatus = cudaMalloc((void**)&U_Unpacked_Video, 960*1080*sizeof(uint16_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return false;
		}






	cudaStatus = cudaMalloc((void**)&YUV_Unpacked_Video, iFrameSizeUnpacked);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}

	cudaStatus = cudaMalloc((void**)&UploadChromaData_Unpacked_SnapShot, iFrameSizeUnpacked);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}


	cudaStatus = cudaMalloc((void**)&YUV_Unpacked_Video_SnapShot, iFrameSizeUnpacked);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}

	cudaStatus = cudaMalloc((void**)&YUV_Unpacked_Video_Duplicate, iFrameSizeUnpacked);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}



	cudaStatus = cudaMalloc((void**)&YUV_Unpacked_Key_SnapShot, iFrameSizeUnpacked);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}




	cudaStatus = cudaMalloc((void**)&FrameColorData_Unpacked, (1024 / 2 * sizeof(uint4) * 1024));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}
	cudaStatus = cudaMalloc((void**)&LookUpColorDataOneDimention_Unpacked, (1024 / 2 * sizeof(uint4) * 1024));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}


	cudaStatus = cudaMemset(LookUpColorDataOneDimention_Unpacked, 0, (1024 / 2 * sizeof(uint4) * 1024));//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}

	cudaStatus = cudaMalloc((void**)&YUV_Unpacked_Key, iFrameSizeUnpacked);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}
	cudaStatus = cudaMalloc((void**)&YUV_Unpacked_Fill, iFrameSizeUnpacked);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}

	cudaStatus = cudaMalloc((void**)&YUV_Upload_Video_YUV, iFrameSizeYUV10Bit);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}


	for (int x = 0; x < 3; x++)
	{
		cudaStatus = cudaMalloc((void**)&ChromaGeneratedMask[x], (m_iWidth * (m_iHeight)) * sizeof(uchar));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return false;
		}
	}

	cudaStatus = cudaMalloc((void**)&YoloGeneratedMask, (m_iWidth * (m_iHeight)) * sizeof(uchar));
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMalloc failed!");
					return false;
				}

	cudaStatus = cudaMalloc((void**)&MaskRefineScratch, (m_iWidth * (m_iHeight)) * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return false;
	}


	cudaStatus = cudaMalloc((void**)&DownloadRGBData, (m_iWidth * (m_iHeight)) * sizeof(uchar3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return true;
	}


	cudaStatus = cudaMalloc((void**)&DownloadRGBData_Frame_Info, 1024 * 1024 * sizeof(uchar3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return true;
	}


	cudaStatus = cudaMalloc((void**)&DownloadRGBData_Frame_InfoFull, 1024 * 1024 * sizeof(uchar3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return true;
	}



	CUDA_CHECK_RETURN(cudaMalloc((void **)&m_RGBScaledFramePlanarDetector,BATCH_SIZE*(640 * (640)) * 3 * sizeof(float)));

	for (int x = 0; x < BATCH_SIZE; x++)
		m_RGBScaledFramePlanarDetectorptrs[x] = (void*)
				m_RGBScaledFramePlanarDetector
						+ x * (640 * (640)) * 3 * sizeof(float);

	return true;

}




bool CudaChromaFree()
{
	
	cudaError_t cudaStatus;
	
	
	
	cudaStatus = cudaFree(Y_Unpacked_TraingMask2);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaFree failed YUV_Upload_Key!");
				return false;
			}
	cudaStatus = cudaFree(Y_Unpacked_Video);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaFree failed YUV_Upload_Key!");
			return false;
		}
		cudaStatus = cudaFree(U_Unpacked_Video);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaFree failed YUV_Upload_Key!");
				return false;
			}
			cudaStatus = cudaFree(V_Unpacked_Video);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaFree failed YUV_Upload_Key!");
					return false;
				}


	cudaStatus = cudaFree(YUV_Upload_Key);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed YUV_Upload_Key!");
		return false;
	}
	
	cudaStatus = cudaFree(YUV_Upload_Fill);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed YUV_Upload_Fill!");
		return false;
	}

	

	cudaStatus = cudaFree(YUV_Upload_Video_YUV);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed YUV_Unpacked_Video!");
		return false;
	}

	cudaStatus = cudaFree(UploadChromaData_Unpacked_SnapShot);
	if (cudaStatus != cudaSuccess) { 
		fprintf(stderr, "cudaFree failed! UploadChromaData_Unpacked_SnapShot");
		return false;
	}


	cudaStatus = cudaFree(YUV_Unpacked_Video_SnapShot);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed YUV_Unpacked_Video_SnapShot!");
		return false;
	}

	cudaStatus = cudaFree(YUV_Unpacked_Video_Duplicate);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed YUV_Unpacked_Video_Duplicate!");
		return false;
	}

	

	cudaStatus = cudaFree(YUV_Unpacked_Key_SnapShot);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed YUV_Unpacked_Key_SnapShot!");
		return false;
	}




	cudaStatus = cudaFree(FrameColorData_Unpacked);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed FrameColorData_Unpacked!");
		return false;
	}
	cudaStatus = cudaFree(LookUpColorDataOneDimention_Unpacked);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed LookUpColorDataOneDimention_Unpacked!");
		return false;
	}
	
	
	

	//
	//


	


	cudaStatus = cudaFree(YUV_Unpacked_Key);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed YUV_Unpacked_Key!");
		return false;
	}
	cudaStatus = cudaFree(YUV_Unpacked_Fill);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed YUV_Unpacked_Fill!");
		return false;
	}


	for (int x = 0; x < 3; x++)
	{
		cudaStatus = cudaFree(ChromaGeneratedMask[x]);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaFree failed ChromaGeneratedMask!");
			return false;
		}
	}

	cudaStatus = cudaFree(YoloGeneratedMask);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaFree failed ChromaGeneratedMask!");
				return false;
			}



	cudaStatus = cudaFree(MaskRefineScratch);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed MaskRefine!");
		return false;
	}
	cudaStatus = cudaFree(DownloadRGBData);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed DownloadRGBData!");
		return true;
	}
	cudaStatus = cudaFree(DownloadRGBData_Frame_Info);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed DownloadRGBData_Frame_Info!");
		return true;
	}


	cudaStatus = cudaFree(DownloadRGBData_Frame_InfoFull);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed DownloadRGBData_Frame_InfoFull!");
		return true;
	}

	


	return true;

}

bool CudaGetOutputData(void *mydataOut)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(mydataOut, YUV_Upload_Video_YUV, m_lFrameSizeYUV10Bit, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");		
		return false;
	}
	return true;
}


char *GetTXTFileName()
{
	return mySaveTxtPath;
}
/*
void WINAPI SaveToDiskthread(LPVOID lpParam)
{

	char *myData =(char *) lpParam;
	SYSTEMTIME		t;
	LPSYSTEMTIME	systemtimeformat(&t);
	GetLocalTime(systemtimeformat);
	char mySaveDataPath[MAX_PATH];
	sprintf(mySaveTxtPath, "c:\\ProgramData\\ChromaStills\\%d%d%d_%d%d_%d.txt", systemtimeformat->wYear, systemtimeformat->wMonth, systemtimeformat->wDay, systemtimeformat->wHour, systemtimeformat->wMinute, systemtimeformat->wSecond * 1000 + systemtimeformat->wMilliseconds);
	sprintf(mySaveDataPath, "c:\\ProgramData\\ChromaStills\\%d%d%d_%d%d_%d.raw", systemtimeformat->wYear, systemtimeformat->wMonth, systemtimeformat->wDay, systemtimeformat->wHour, systemtimeformat->wMinute, systemtimeformat->wSecond * 1000 + systemtimeformat->wMilliseconds);
	std::ofstream outfile(mySaveDataPath, std::ofstream::binary);
	outfile.write(myData, m_lFrameSizeYUV10Bit * 3);
	free(myData);

}*/

bool CudaSetInputData(void *mydataIN, void *mydataIN1, void *mydataIN2, bool SafeSnapShot)
{
	if (SafeSnapShot)
	{
		unsigned char *mydataSave = (unsigned char*)malloc(m_lFrameSizeYUV10Bit*3);
		memcpy(mydataSave, mydataIN, m_lFrameSizeYUV10Bit);
		memcpy(mydataSave+ m_lFrameSizeYUV10Bit, mydataIN1, m_lFrameSizeYUV10Bit);
		memcpy(mydataSave+ m_lFrameSizeYUV10Bit+ m_lFrameSizeYUV10Bit, mydataIN2, m_lFrameSizeYUV10Bit);


		//void *myData ;


	/*	DWORD gdwMyThreadId;
		HANDLE tt = CreateThread(NULL,							// default security attributes 
			0,								// use default stack size  
			(LPTHREAD_START_ROUTINE)SaveToDiskthread,	// thread function 
			mydataSave,							// argument to thread function 
			0,								// use default creation flags 
			&gdwMyThreadId);	// returns the thread identifier*/
	}
	ptrLookUpData = LookUpDataArry[iOnAirLookup];

	cudaError_t cudaStatus;
	
	cudaStatus = cudaMemcpy(YUV_Upload_Video_YUV, mydataIN, m_lFrameSizeYUV10Bit, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		
		return false;
	}

	cudaStatus = cudaMemcpy(YUV_Upload_Key, mydataIN1, m_lFrameSizeYUV10Bit, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		
		return false;
	}

	cudaStatus = cudaMemcpy(YUV_Upload_Fill, mydataIN2, m_lFrameSizeYUV10Bit, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		
		return false;
	}
	return true;
}


inline __device__ __host__ double clamp(double f, double a, double b)
{
	return (double)fmaxf(a, fminf(f, b));
}

 
inline __device__ __host__ void calculateBlend(double* dst, double* fill, double* key, double* dst2,double dBlendPos)
{
	double val = *key-64;
	val = val*dBlendPos;
	*dst2 = (*fill * val +  (*dst * (876.0 - val))) / 876.0;

}

inline __device__ __host__ void calculateBlend(unsigned int * dst, unsigned int * fill, unsigned int * key, unsigned int * dst2, double dBlendPos)
{

	double dubDst, dubfill, duBkey;
	dubDst = *dst;
	dubfill = *fill;
	duBkey = *key;
	double val = 1.0 - ((duBkey - 64) / 876.0)*dBlendPos;
	*dst2 = val * (dubDst - dubfill) + dubfill;

}

inline __device__ __host__ void calculateBlendFullKey(unsigned int * dst, unsigned int * fill, unsigned int * key, unsigned int * dst2)
{
	double dubDst, dubfill, duBkey;
	dubDst = *dst;
	dubfill = *fill;
	duBkey = *key;
	double val = 1.0-(duBkey - 64.0) / 876.0;
	*dst2 = val * (dubDst - dubfill) + dubfill;
}


__global__ void yuyvUnPackedToyuyvpacked(uint4* packed_Video, uint4 *unpacked_video, int srcAlignedWidth, int dstAlignedWidth, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;


	uint4 *macroPx;
	macroPx = &packed_Video[y * srcAlignedWidth + x];
	double Cr0;
	double Y0;
	double Cb0;

	double Y2;
	double Cb2;
	double Y1;

	double Cb4;
	double Y3;
	double Cr2;

	double Y5;
	double Cr4;
	double Y4;

	Cr0 = unpacked_video[y * dstAlignedWidth + (x * 3) + 0].x;
	Y0 = unpacked_video[y * dstAlignedWidth + (x * 3) + 0].y;
	Cb0 = unpacked_video[y * dstAlignedWidth + (x * 3) + 0].z;
	Y1 = unpacked_video[y * dstAlignedWidth + (x * 3) + 0].w;

	Cb2 = unpacked_video[y * dstAlignedWidth + (x * 3) + 1].z;;
	Y2 = unpacked_video[y * dstAlignedWidth + (x * 3) + 1].y;
	Cb4 = unpacked_video[y * dstAlignedWidth + (x * 3) + 2].z;
	Y3 = unpacked_video[y * dstAlignedWidth + (x * 3) + 1].w;

	Cr2 = unpacked_video[y * dstAlignedWidth + (x * 3) + 1].x;
	Y4 =unpacked_video[y * dstAlignedWidth + (x * 3) + 2].y;
	Cr4 = unpacked_video[y * dstAlignedWidth + (x * 3) + 2].x;
	Y5= unpacked_video[y * dstAlignedWidth + (x * 3) + 2].w;


	macroPx->x = ((unsigned int)Cr0 << 20) + ((unsigned int)Y0 << 10) + (unsigned int)Cb0;
	macroPx->y = macroPx->y & 0x3ffffc00;
	macroPx->y = macroPx->y | (unsigned int)Y1;

	macroPx->y = macroPx->y & 0x3ff;
	macroPx->y = macroPx->y | ((unsigned int)Y2 << 20) | ((unsigned int)Cb2 << 10);
	macroPx->z = macroPx->z & 0x3ff00000;
	macroPx->z = macroPx->z | (((unsigned int)Y3 << 10) | (unsigned int)Cr2);

	macroPx->z = macroPx->z & 0xfffff;
	macroPx->z = macroPx->z | ((unsigned int)Cb4 << 20);
	macroPx->w = ((long)Y5 << 20) + ((unsigned int)Cr4 << 10) + (unsigned int)Y4;
	
}

__global__ void yuyvUnPackedToRGB_Plain(uint4* src_Unapc, uchar3* dst, int srcAlignedWidth, int dstAlignedWidth, int height, uint4 * LookUpColorDataOneDimention_Unpacked1)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	uint4 *macroPx;
	uint4 *macroPxLookup;
	macroPx = &src_Unapc[y * srcAlignedWidth + x];
	macroPxLookup = &LookUpColorDataOneDimention_Unpacked1[y * srcAlignedWidth + x];
	
	if (macroPx->w != 0 && macroPx->x != 0 && macroPx->y != 0 && macroPx->z != 0)
	{
		double3 px_0 = make_double3(clamp(macroPx->y + 1.540f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(macroPx->y - 0.459f * (macroPx->x - 512.0) - 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(macroPx->y + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

		double3	px_1 = make_double3(clamp(macroPx->w + 1.540f *(macroPx->z - 512.0), 0.0, 1023.0),
			clamp(macroPx->w - 0.459f * (macroPx->x - 512.0) - 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(macroPx->w + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));
		
		dst[y * dstAlignedWidth + (x * 2) + 0].x = clamp(px_0.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 0].y = clamp(px_0.y / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 0].z = clamp(px_0.z / 1024.0*255.0, 0.0f, 255.0f);

		dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_1.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_1.y / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_1.z / 1024.0*255.0, 0.0f, 255.0f);
	}
	else if (macroPxLookup->x != 0 && macroPxLookup->z != 0)
	{

		double3 px_0 = make_double3(clamp(macroPxLookup->y + 1.540f * (macroPxLookup->z - 512.0), 0.0, 1023.0),
			clamp(macroPxLookup->y - 0.459f * (macroPxLookup->x - 512.0) - 0.183f * (macroPxLookup->z - 512.0), 0.0, 1023.0),
			clamp(macroPxLookup->y + 1.816f * (macroPxLookup->x - 512.0), 0.0, 1023.0));

		double3	px_1 = make_double3(clamp(macroPxLookup->w + 1.540f *(macroPxLookup->z - 512.0), 0.0, 1023.0),
			clamp(macroPxLookup->w - 0.459f * (macroPxLookup->x - 512.0) - 0.183f * (macroPxLookup->z - 512.0), 0.0, 1023.0),
			clamp(macroPxLookup->w + 1.816f * (macroPxLookup->x - 512.0), 0.0, 1023.0));

		dst[y * dstAlignedWidth + (x * 2) + 0].x = clamp(px_0.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 0].y = clamp(px_0.y / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 0].z = clamp(px_0.z / 1024.0*255.0, 0.0f, 255.0f);

		dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_1.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_1.y / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_1.z / 1024.0*255.0, 0.0f, 255.0f);


	}
	else
	{

		dst[y * dstAlignedWidth + (x * 2) + 0].x = 0;
		dst[y * dstAlignedWidth + (x * 2) + 0].y = 0;
		dst[y * dstAlignedWidth + (x * 2) + 0].z = 0;

		dst[y * dstAlignedWidth + (x * 2) + 1].x = 0;
		dst[y * dstAlignedWidth + (x * 2) + 1].y = 0;
		dst[y * dstAlignedWidth + (x * 2) + 1].z = 0;

	}

}


__global__ void MaskToRGB(uchar *maskDownload0, uchar *maskDownload1, uchar *maskDownload2,float* yolomaskdata, uchar3* dst, int srcAlignedWidth, int dstAlignedWidth, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;
	
	dst[y * dstAlignedWidth + (x * 2) + 0].x = maskDownload0[y * dstAlignedWidth + (x * 2) + 0];
	dst[y * dstAlignedWidth + (x * 2) + 0].y = maskDownload1[y * dstAlignedWidth + (x * 2) + 0];
	dst[y * dstAlignedWidth + (x * 2) + 0].z = maskDownload2[y * dstAlignedWidth + (x * 2) + 0];

	dst[y * dstAlignedWidth + (x * 2) + 1].x = maskDownload0[y * dstAlignedWidth + (x * 2) + 0];
	dst[y * dstAlignedWidth + (x * 2) + 1].y = maskDownload1[y * dstAlignedWidth + (x * 2) + 0];
	dst[y * dstAlignedWidth + (x * 2) + 1].z = maskDownload2[y * dstAlignedWidth + (x * 2) + 0];
}

__global__ void Msk2RGB(uchar *maskDownload0, uchar *maskDownload1, uchar *maskDownload2, uchar3* dst, int srcAlignedWidth, int dstAlignedWidth, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	dst[y * dstAlignedWidth + (x * 2) + 0].x = maskDownload0[y * dstAlignedWidth + (x * 2) + 0];
	dst[y * dstAlignedWidth + (x * 2) + 0].y = maskDownload1[y * dstAlignedWidth + (x * 2) + 0];
	dst[y * dstAlignedWidth + (x * 2) + 0].z = maskDownload2[y * dstAlignedWidth + (x * 2) + 0];

	dst[y * dstAlignedWidth + (x * 2) + 1].x = maskDownload0[y * dstAlignedWidth + (x * 2) + 0];
	dst[y * dstAlignedWidth + (x * 2) + 1].y = maskDownload1[y * dstAlignedWidth + (x * 2) + 0];
	dst[y * dstAlignedWidth + (x * 2) + 1].z = maskDownload2[y * dstAlignedWidth + (x * 2) + 0];
}



__global__ void yuyvUnPackedToRGB(uint4* src_Unapc, uchar3* dst,int srcAlignedWidth, int dstAlignedWidth, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
	return;

	uint4 *macroPx;
	macroPx = &src_Unapc[y * srcAlignedWidth + x];

	double3 px_0 = make_double3(
			clamp(macroPx->y + 1.540f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(
					macroPx->y - 0.459f * (macroPx->x - 512.0)
					- 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(macroPx->y + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

	double3 px_1 = make_double3(
			clamp(macroPx->w + 1.540f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(
					macroPx->w - 0.459f * (macroPx->x - 512.0)
					- 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(macroPx->w + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

	dst[y * dstAlignedWidth + (x * 2) + 0].x = clamp(px_0.x / 1024.0 * 255.0,
			0.0f, 255.0f);
	dst[y * dstAlignedWidth + (x * 2) + 0].y = clamp(px_0.y / 1024.0 * 255.0,
			0.0f, 255.0f);
	dst[y * dstAlignedWidth + (x * 2) + 0].z = clamp(px_0.z / 1024.0 * 255.0,
			0.0f, 255.0f);

	dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_1.x / 1024.0 * 255.0,
			0.0f, 255.0f);
	dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_1.y / 1024.0 * 255.0,
			0.0f, 255.0f);
	dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_1.z / 1024.0 * 255.0,
			0.0f, 255.0f);

}





__global__ void yuyvUmPackedToRGB_lookup(uint4* src_Unapc, uchar3* dst, int srcAlignedWidth, int dstAlignedWidth, int height, uint4* src__Key_Unapc, uchar* LookupTable)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;


	uint4 *macroPxKey;
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];


	uint4 *macroPx;
	macroPx = &src_Unapc[y * srcAlignedWidth + x];

	double3 px_0 = make_double3(clamp(macroPx->y + 1.540f * (macroPx->z - 512.0), 0.0, 1023.0),
		clamp(macroPx->y - 0.459f * (macroPx->x - 512.0) - 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
		clamp(macroPx->y + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

	double3	px_1 = make_double3(clamp(macroPx->w + 1.540f *(macroPx->z - 512.0), 0.0, 1023.0),
		clamp(macroPx->w - 0.459f * (macroPx->x - 512.0) - 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
		clamp(macroPx->w + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));





	if (macroPxKey->y < 65.0 && macroPxKey->w < 65.0)
	{
		dst[y * dstAlignedWidth + (x * 2) + 0].x = clamp(px_0.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 0].y = clamp(px_0.y / 1024.0*255.0, 0.0f, 255.0f) / 1.5;
		dst[y * dstAlignedWidth + (x * 2) + 0].z = clamp(px_0.z / 1024.0*255.0, 0.0f, 255.0f);

		dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_1.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_1.y / 1024.0*255.0, 0.0f, 255.0f) / 1.5;
		dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_1.z / 1024.0*255.0, 0.0f, 255.0f);
	}
	else
	{
		if (GetBit3(GetBitPos3(make_double3(macroPx->x, macroPx->z, macroPx->w)), LookupTable))
		{

			dst[y * dstAlignedWidth + (x * 2) + 0].x = clamp(px_0.x / 1024.0*255.0, 0.0f, 255.0f)/3;
			dst[y * dstAlignedWidth + (x * 2) + 0].y = clamp(px_0.y / 1024.0*255.0, 0.0f, 255.0f)/3;
			dst[y * dstAlignedWidth + (x * 2) + 0].z = clamp(px_0.z / 1024.0*255.0, 0.0f, 255.0f);

			dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_0.x / 1024.0*255.0, 0.0f, 255.0f)/3;
			dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_0.y / 1024.0*255.0, 0.0f, 255.0f)/3;
			dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_0.z / 1024.0*255.0, 0.0f, 255.0f);
		}
		else
		{
			dst[y * dstAlignedWidth + (x * 2) + 0].x = clamp(px_0.x / 1024.0*255.0, 0.0f, 255.0f);
			dst[y * dstAlignedWidth + (x * 2) + 0].y = clamp(px_0.y / 1024.0*255.0, 0.0f, 255.0f);
			dst[y * dstAlignedWidth + (x * 2) + 0].z = clamp(px_0.z / 1024.0*255.0, 0.0f, 255.0f);

			dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_0.x / 1024.0*255.0, 0.0f, 255.0f);
			dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_0.y / 1024.0*255.0, 0.0f, 255.0f);
			dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_0.z / 1024.0*255.0, 0.0f, 255.0f);

		}
		if (GetBit3(GetBitPos3(make_double3(macroPx->x, macroPx->z, macroPx->y)), LookupTable))
		{
			dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_1.x / 1024.0*255.0, 0.0f, 255.0f)/3;
			dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_1.y / 1024.0*255.0, 0.0f, 255.0f)/3;
			dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_1.z / 1024.0*255.0, 0.0f, 255.0f);
		}
		else
		{
			dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_1.x / 1024.0*255.0, 0.0f, 255.0f);
			dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_1.y / 1024.0*255.0, 0.0f, 255.0f);
			dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_1.z / 1024.0*255.0, 0.0f, 255.0f);
		}
	}
}

__global__ void yuyvUnpackedToRGB(uint4* src_Unapc,  uchar3* dst, int srcAlignedWidth, int dstAlignedWidth, int height,  uint4* src__Key_Unapc)
{

//	printf("I execute\n");
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;


	uint4 *macroPxKey;
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];


	uint4 *macroPx;
	macroPx = &src_Unapc[y * srcAlignedWidth + x];

	double3 px_0 = make_double3(clamp(macroPx->y + 1.540f * (macroPx->z - 512.0), 0.0, 1023.0),
				clamp(macroPx->y - 0.459f * (macroPx->x - 512.0) - 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
				clamp(macroPx->y + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

	double3	px_1 = make_double3(clamp(macroPx->w + 1.540f *(macroPx->z - 512.0), 0.0, 1023.0),
				clamp(macroPx->w - 0.459f * (macroPx->x - 512.0) - 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
				clamp(macroPx->w + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

//	printf("%lu %lu %lu %lu\n",macroPx->w, macroPx->x,macroPx->y, macroPx->z);
		

	if (macroPxKey->y < 65.0 && macroPxKey->w < 65.0)
	{
		dst[y * dstAlignedWidth + (x * 2) + 0].x = clamp(px_0.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 0].y = clamp(px_0.y / 1024.0*255.0, 0.0f, 255.0f)/3;
		dst[y * dstAlignedWidth + (x * 2) + 0].z = clamp(px_0.z / 1024.0*255.0, 0.0f, 255.0f);

		dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_1.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_1.y / 1024.0*255.0, 0.0f, 255.0f)/3;
		dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_1.z / 1024.0*255.0, 0.0f, 255.0f);
	}
	else
	{
		dst[y * dstAlignedWidth + (x * 2) + 0].x = clamp(px_0.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 0].y = clamp(px_0.y / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 0].z = clamp(px_0.z / 1024.0*255.0, 0.0f, 255.0f);

		dst[y * dstAlignedWidth + (x * 2) + 1].x = clamp(px_1.x / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 1].y = clamp(px_1.y / 1024.0*255.0, 0.0f, 255.0f);
		dst[y * dstAlignedWidth + (x * 2) + 1].z = clamp(px_1.z / 1024.0*255.0, 0.0f, 255.0f);
	}
}

__global__ void yuyvUnpackedMaskDilation(uchar *maskDownload, uchar *maskRefine, int width, int height, int srcAlignedWidth, int dstAlignedWidth)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= (srcAlignedWidth-3) || y >= (height-2))
		return;
	if (x <= (3) || y <= ( 2))
		return;

	if (maskDownload[y * dstAlignedWidth + (x)] != 0)
		return;
	int iCount = 0;
	double Avarage = 0;
	for (int ya = -1; ya < 2; ya ++)//for (int ya = -2; ya < 4; ya = ya + 2)//this bug cuased the program to work
		for (int xa = -1; xa < 2; xa++)
			if (ya != 0 && xa != 0)
				if (maskDownload[(y - ya) * dstAlignedWidth + (x) + xa] != 0)
				{
					Avarage = Avarage + maskDownload[(y - ya) * dstAlignedWidth + (x)+xa];
					iCount++;
				}
				
		if(iCount)
		{
		//	maskRefine[y * dstAlignedWidth + (x)] = Avarage/ iCount;

			for (int ya = -1; ya < 2; ya ++)//for (int ya = -2; ya < 4; ya = ya + 2)//this bug cuased the program to work
					for (int xa = -1; xa < 2; xa++)
					//	if (ya != 0 && xa != 0)
							maskRefine[(y+ya) * dstAlignedWidth + (x+xa)]= 255;



		}

	 
}

__global__ void yuyvUnpackedMaskDilationCombine(uchar *maskDownload, uchar *maskRefine, int width, int height, int srcAlignedWidth, int dstAlignedWidth)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;



	if (maskRefine[y * dstAlignedWidth + (x)] != 0)
		maskDownload[y * dstAlignedWidth + (x)] = 255;




}




__global__ void yuyvUnpackedMaskErosion(uchar *maskDownload, uchar *maskRefine, int width, int height, int srcAlignedWidth, int dstAlignedWidth)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= (srcAlignedWidth-2) || y >= (height-4))
		return;

	if (x <= 2 || y <= 4)
		return;

	if (maskDownload[y * dstAlignedWidth + (x)] == 0)
		return;
	int i = 0;
 	for (int ya = -1; ya < 2;  ya ++)
		for (int xa = -1; xa < 2; xa++)
			if (ya != 0 && xa != 0)
				if (maskDownload[(y - ya) * dstAlignedWidth + (x) + xa] == 0)
				{
					i++;
				}

	if(i)
	 maskRefine[y * dstAlignedWidth + (x)] = 255;

	
}

__global__ void yuyvUnpackedMaskErosionCombine(uchar *maskDownload, uchar *maskRefine, int width, int height, int srcAlignedWidth, int dstAlignedWidth)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	if (maskRefine[y * dstAlignedWidth + (x)] == 255)
	{

		 	for (int ya = -1; ya < 2;  ya ++)
				for (int xa = -1; xa < 2; xa++)
					maskDownload[y * dstAlignedWidth + (x)] = 0;
	}
//	else
//		maskDownload[y * dstAlignedWidth + (x)] = 255;

}




inline __device__ void UpdateOneDimetionLookup(int x, int z , uint4 *LookUpColorDataOneDimention_Unpacked1)
{
	uint4 *macroFrameData;
	macroFrameData = &LookUpColorDataOneDimention_Unpacked1[z * 512 + x / 2];
	macroFrameData->x = x;
	macroFrameData->y = 512;
	macroFrameData->z = z;
	macroFrameData->w = 512;
}

inline __device__ void ClearOneDimetionLookup(int x, int z, uint4 *LookUpColorDataOneDimention_Unpacked1)
{
	uint4 *macroFrameData;
	macroFrameData = &LookUpColorDataOneDimention_Unpacked1[z * 512 + x / 2];
	macroFrameData->x = 0;
	macroFrameData->y = 0;
	macroFrameData->z = 0;
	macroFrameData->w = 0;
}






__global__ void yuyvUnpackedGetFrameInfo(ulong4* src_Video_Unapc, ulong4* src__Key_Unapc, ulong1* src_Lum, int width, int height, int srcAlignedWidth,int LumAlliagn)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	ulong4 *macroPxVideo;



	ulong4 *macroPxKey;
	
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];
	
	if (macroPxKey->y < 65.0 && macroPxKey->w < 65.0)
	{
		//printf("ret\n")
		return;
	}
	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	ulong1 *macroPxLum;
	macroPxLum = &src_Lum[y * LumAlliagn + x];
	macroPxLum->x = (macroPxVideo->y + macroPxVideo->w) / 2.0;
	__syncthreads();
}



__global__ void yuyvUnpackedCreateFrameInfo(uint4* src_Video_Unapc, uint4* src__Key_Unapc, uint4* FrameUpload, int width, int height, int srcAlignedWidth, uint4 * LookUpColorDataOneDimention_Unpacked1)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	uint4 *macroPxVideo;
	uint4 *macroFrameData;
	uint4 *macroFrameLookup;


	uint4 *macroPxKey;
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];
	if (macroPxKey->y < 65.0 && macroPxKey->w < 65.0)
	{
		//printf("ret\n")
		return;
	}


	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];


	macroFrameLookup =&LookUpColorDataOneDimention_Unpacked1[macroPxVideo->z * 512 + macroPxVideo->x / 2];

	macroFrameData = &FrameUpload[macroPxVideo->z * 512 + macroPxVideo->x/2];
	macroFrameData->x = macroPxVideo->x;
	if(macroFrameLookup->x==0)
		macroFrameData->y = 512;
	else
		macroFrameData->y = 1000;
	macroFrameData->z = macroPxVideo->z;
	if (macroFrameLookup->x == 0)
		macroFrameData->w =512;
	else
		macroFrameData->w = 1000;
}


__global__ void yuyv_Unpacked_GenerateMaskYolo(uint4* src_Video_Unapc,uint4* src__Key_Unapc, uchar *CannyEdges, uchar *YoloPlayerMask, uchar *TraingMask, uchar *maskDownload, uchar* LookupTable, int width, int height, int srcAlignedWidth, int dstAlignedWidth,int iAvgCutOff,bool bAutoTrain)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;


	uint4 *macroPxVideo;
	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];

	uint4 *macroPxKey;
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];

	if (macroPxKey->y < 65 && macroPxKey->w < 65)
	{
		return;
	}

	uint4 macroPxVideoReal;

	macroPxVideoReal = *macroPxVideo;
	double3 val1 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.y);
	double bitpos1 = GetBitPos3(val1);
	double3 val2 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.w);
	double bitpos2 = GetBitPos3(val2);




	int pos=y * dstAlignedWidth + (x * 2) ;
	if(YoloPlayerMask[pos+ 0]==255)
		maskDownload[pos + 0] =  GetBit3(bitpos1, LookupTable);
	else
	{
		//if(CannyEdges[pos]==0)
		{
		//	if(CannyEdges[pos]==0)
				maskDownload[pos + 0]=200;
		//	else
		//		maskDownload[pos + 0]=0;
			if(!GetBit3(bitpos1, LookupTable))
			{
					TraingMask[pos+0]=250;
					if(bAutoTrain&&CannyEdges[pos]==0)
					{
					//	while (!GetBit3(bitpos1, LookupTable))
						{
							for(int lum=-7;lum<7;lum++)
								for(int cy=-2;cy<2;cy++)
									for(int cx=-2;cx<2;cx++)
							{
								double3 val1 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z+cy, macroPxVideoReal.y+lum);
								double bitpos1 = GetBitPos3(val1);
								while (GetBit3(bitpos1, LookupTable)<200)
									SetBit3(bitpos1, LookupTable,200);
							}

						}
					}
			}
		}
	}

	if(YoloPlayerMask[pos+ 1]==255)
		maskDownload[pos + 1] = GetBit3(bitpos2, LookupTable);
	else
	{


	//	if(CannyEdges[pos+1]==0)
		{
		//	if(CannyEdges[pos+1]==0)
				maskDownload[pos + 1] = 200;
		//	elseb
		//		maskDownload[pos + 1] = 0;
		if(!GetBit3(bitpos2, LookupTable))
		{
			TraingMask[pos+1]=255;
			if(bAutoTrain&&CannyEdges[pos+1]==0)

			{

				for(int lum=-7;lum<7;lum++)
					for(int cy=-2;cy<2;cy++)
						for(int cx=-2;cx<2;cx++)
				{
				double3 val2 = make_double3(macroPxVideoReal.x+cx, macroPxVideoReal.z+cy, macroPxVideoReal.w+lum);
					double bitpos2 = GetBitPos3(val2);
					while (GetBit3(bitpos2, LookupTable)<200)
									SetBit3(bitpos2, LookupTable,200);
				}
			}

		}
		}
	}

}



__global__ void yuyv_Unpacked_GenerateMask(uint4* src_Video_Unapc, uchar *maskDownload, uchar* LookupTable, int width, int height, int srcAlignedWidth, int dstAlignedWidth,int iAvgCutOff)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	uint4 *macroPxVideo;
	uint4 macroPxVideoReal;

//	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];

	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	macroPxVideoReal = *macroPxVideo;

	double3 val1 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.y);
	double bitpos1 = GetBitPos3(val1);
	double3 val2 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.w);
	double bitpos2 = GetBitPos3(val2);
	maskDownload[y * dstAlignedWidth + (x * 2) + 0] = GetBit3(bitpos1,LookupTable);
	maskDownload[y * dstAlignedWidth + (x * 2) + 1] = GetBit3(bitpos2, LookupTable);
}
__global__ void yuyv_Unpacked_GenerateMask_yolo(uint4* src_Video_Unapc, uchar *maskDownload, uchar* LookupTable, int width, int height, int srcAlignedWidth, int dstAlignedWidth,int iAvgCutOff,uchar *yolomask)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	uint4 *macroPxVideo;
	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	uint4 macroPxVideoReal;
	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	macroPxVideoReal = *macroPxVideo;
	double3 val1 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.y);
	double bitpos1 = GetBitPos3(val1);
	double3 val2 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.w);
	double bitpos2 = GetBitPos3(val2);
	if(yolomask[y * dstAlignedWidth + (x * 2) + 0]!=0)
		maskDownload[y * dstAlignedWidth + (x * 2) + 0] = GetBit3(bitpos1, LookupTable);
	if(yolomask[y * dstAlignedWidth + (x * 2) + 1]!=0)
		maskDownload[y * dstAlignedWidth + (x * 2) + 1] = GetBit3(bitpos2, LookupTable);
}



__global__ void yuyv_Unpacked_GenerateMask_yolo_segmented(uint4* src_Video_Unapc, uchar *maskDownload, uchar* LookupTable, int width, int height, int srcAlignedWidth, int dstAlignedWidth,int iAvgCutOff,float *yolomask)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	uint4 *macroPxVideo;
	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	uint4 macroPxVideoReal;
	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	macroPxVideoReal = *macroPxVideo;
	double3 val1 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.y);
	double bitpos1 = GetBitPos3(val1);
	double3 val2 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.w);
	double bitpos2 = GetBitPos3(val2);

	//float *maskDownload_val = &maskDownload0[x + y * 960+iOffset];


//	if(yolomask[y * dstAlignedWidth + (x * 2) + 0]!=0)
//		maskDownload[y * dstAlignedWidth + (x * 2) + 0] = GetBit3(bitpos1, LookupTable);
//	if(yolomask[y * dstAlignedWidth + (x * 2) + 1]!=0)
//		maskDownload[y * dstAlignedWidth + (x * 2) + 1] = GetBit3(bitpos2, LookupTable);


	if(yolomask[y * dstAlignedWidth + (x * 2) + 0]!=0)
		maskDownload[y * dstAlignedWidth + (x * 2) + 0] = 255;
	if(yolomask[y * dstAlignedWidth + (x * 2) + 1]!=0)
		maskDownload[y * dstAlignedWidth + (x * 2) + 1] = 255;
}




void Launch_yuyv_Unpacked_UnpackedComBineData(int *iBlendPos0, int *iBlendPos1, int *iBlendPos2, int RowLength, double4 *Parabolic0, double4 *Parabolic1, double4 *Parabolic2, bool Bypass, unsigned long int iCutOff, unsigned long int iCutOff0, unsigned long int iCutOff1, unsigned long int iCutOff2, int bOutPutSnap)
{
	
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(RowLength / 16, block.x), iDivUp(1080, block.y));
	const int srcAlignedWidth = RowLength / SIZE_ULONG4_CUDA;
	const int dstAlignedWidthUnpackedData = (1920 / 2);

	cudaError_t cudaStatus;
	const int dstAlignedWidthUnpackedData1 = (1920 / 2);
	const dim3 blockRUN(16, 16);
	const dim3 gridRun(iDivUp(dstAlignedWidthUnpackedData1, blockRUN.x), iDivUp(1080, blockRUN.y));
	const dim3 gridFull(iDivUp(1920, blockRUN.x), iDivUp(1080, blockRUN.y));
	const int dstAlignedWidthMask = 1920;
	if (!Bypass)
	{

		yuyvUnpackedComBineDataThreeLookups << <gridRun, blockRUN >> > (YUV_Unpacked_Video, YUV_Unpacked_Fill, YUV_Unpacked_Key, 1920, 1080, dstAlignedWidthUnpackedData1, dstAlignedWidthMask, ChromaGeneratedMask[0], ChromaGeneratedMask[1], ChromaGeneratedMask[2], *iBlendPos0, *iBlendPos1, *iBlendPos2, *Parabolic0, *Parabolic1, *Parabolic2, iCutOff, iCutOff0, iCutOff1, iCutOff2);
	}
	else
		yuyvUnpackedComBineDataChromaBipass <<<gridRun, blockRUN >> > (YUV_Unpacked_Video, YUV_Unpacked_Fill, YUV_Unpacked_Key, 1920, 1080, dstAlignedWidthUnpackedData1);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		printf("%s", stderr);
		return;
	}
	//__global__ void yuyvUnPackedToyuyvpacked(ulong4* packed_Video, ulong4 *unpacked_video, int srcAlignedWidth, int dstAlignedWidth, int height)
	yuyvUnPackedToyuyvpacked << <grid, block >> > ((uint4*)YUV_Upload_Video_YUV, YUV_Unpacked_Video, srcAlignedWidth, dstAlignedWidthUnpackedData, 1080);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}



	if (bOutPutSnap != -1)
	{
		const dim3 blockRGB(16, 16);
		const dim3 gridRGB(iDivUp(dstAlignedWidthUnpackedData, blockRGB.x), iDivUp(1080, blockRGB.y));
		const int dstAlignedWidthRGB = 1920;
	//	float *
		switch (bOutPutSnap)
		{
		case 0:
			yuyvUnPackedToRGB << <gridRGB, blockRGB >> > ((uint4 *)YUV_Unpacked_Video, DownloadRGBData, dstAlignedWidthUnpackedData, dstAlignedWidthRGB, 1080);
			break;

		case 1:
			yuyvUnPackedToRGB << <gridRGB, blockRGB >> > ((uint4 *)YUV_Unpacked_Fill, DownloadRGBData, dstAlignedWidthUnpackedData, dstAlignedWidthRGB, 1080);
			break;
		case 2:
			yuyvUnPackedToRGB << <gridRGB, blockRGB >> > ((uint4 *)YUV_Unpacked_Key, DownloadRGBData, dstAlignedWidthUnpackedData, dstAlignedWidthRGB, 1080);
			break;
		case 3:
//			MaskToRGB << <gridRGB, blockRGB >> > (ChromaGeneratedMask[0], ChromaGeneratedMask[1], ChromaGeneratedMask[2],GetSegmentedMaskSnapshot(), DownloadRGBData, dstAlignedWidthUnpackedData, dstAlignedWidthRGB, 1080);
			break;
		case 4:
//			MaskToRGB << <gridRGB, blockRGB >> > (ChromaGeneratedMask[0], ChromaGeneratedMask[1], ChromaGeneratedMask[2],GetSegmentedMaskSnapshot(), DownloadRGBData, dstAlignedWidthUnpackedData, dstAlignedWidthRGB, 1080);
			break;

		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return;
		}
	}

}





void Launch_yuyv_Unpacked_UnpackedComBineData(int iBlendPos, int RowLength,double3 Parabolic,bool Bypass, bool m_DisableParabolicKeying, unsigned long int iCutOff)
{


	const dim3 block(16, 16);
	const dim3 grid(iDivUp(RowLength / 16, block.x), iDivUp(1080, block.y));
	const int srcAlignedWidth = RowLength / sizeof(ulong4);
	const int dstAlignedWidthUnpackedData = (1920 / 2);

	cudaError_t cudaStatus;
	const int dstAlignedWidthUnpackedData1 = (1920 / 2);
	const dim3 blockRUN(32, 32);
	const dim3 gridRun(iDivUp(dstAlignedWidthUnpackedData1, blockRUN.x), iDivUp(1080, blockRUN.y));
	const dim3 gridFull(iDivUp(1920, blockRUN.x), iDivUp(1080, blockRUN.y));
	const int dstAlignedWidthMask = 1920;
	if (!Bypass)
	{
		yuyvUnpackedComBineData << <gridRun, blockRUN >> > (YUV_Unpacked_Video, YUV_Unpacked_Fill, YUV_Unpacked_Key, 1920, 1080, dstAlignedWidthUnpackedData1, dstAlignedWidthMask, ChromaGeneratedMask[0], iBlendPos, Parabolic, iCutOff);

	}
	else
	  yuyvUnpackedComBineDataChromaBipass << <gridRun, blockRUN >> > (YUV_Unpacked_Video, YUV_Unpacked_Fill, YUV_Unpacked_Key, 1920, 1080, dstAlignedWidthUnpackedData1);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		printf("%s",stderr);
		return;
	}
	//__global__ void yuyvUnPackedToyuyvpacked(ulong4* packed_Video, ulong4 *unpacked_video, int srcAlignedWidth, int dstAlignedWidth, int height)
	yuyvUnPackedToyuyvpacked << <grid, block >> > ((uint4*)YUV_Upload_Video_YUV, YUV_Unpacked_Video, srcAlignedWidth, dstAlignedWidthUnpackedData, 1080);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}


}

void Launch_yuyv_Unpacked_ClearMask(int iUse)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemset(ChromaGeneratedMask[iUse], 0, (m_iWidth * (m_iHeight)) * sizeof(uchar));//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

}



__global__ void GenerateMask(uint4* src_Video_Unapc, uchar *maskDownload, uchar* LookupTable, int width, int height, int srcAlignedWidth, int dstAlignedWidth)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	maskDownload[y * dstAlignedWidth + (x * 2) + 0] = 0;
	maskDownload[y * dstAlignedWidth + (x * 2) + 1] = 0;
}




cv::Rect get_rect2( float bbox[4]) {

    return cv::Rect(bbox[0]-bbox[2]/2,bbox[1]*2-bbox[3],bbox[2],bbox[3]*2);
}

inline cv::Rect get_rect_normal_size_widow( float bbox[4],int iOffset) {

    return cv::Rect(bbox[0]-bbox[2]/2-iOffset,bbox[1]-(bbox[3])/2-iOffset,bbox[2]+iOffset*2,bbox[3]+iOffset*2);
}
inline cv::Rect get_rect_normal_half_widow( float bbox[4],int iOffset) {

    return cv::Rect(bbox[0]/2-bbox[2]/4-iOffset,bbox[1]-(bbox[3])/2-iOffset,bbox[2]/2+iOffset*2,bbox[3]+iOffset*2);
}
//std::vector<Detection> dlist[2];
//int iActive=0;
//int iCheckThis=1;




std::vector <TrackedObj> TrackedObjectsLive;
std::vector <TrackedObj> TrackedObjectsNext;

std::vector <TrackedObj*> TrackedObjectsLivePtr;
std::vector <TrackedObj*> TrackedObjectsNextPtr;



//std::vector<Distances> distances_between_players_players;


int iHitTestindex=0;


bool TestDetectionsPTR(int mousex,int mousey)
{

	int iIndexToSet=-1;
	int iIndex=0;

	for( TrackedObj *obj:TrackedObjectsNextPtr)
	{

		if(  (   (obj->m_Detection.bbox[0]-obj->m_Detection.bbox[2]/2) <mousex )&&mousex<(obj->m_Detection.bbox[0]+obj->m_Detection.bbox[2]/2))
		{
			std::cout << obj->m_Detection.bbox[1] << std::endl;
			if(  (   (obj->m_Detection.bbox[1]*2-obj->m_Detection.bbox[3]) <mousey)&&mousey<(obj->m_Detection.bbox[1]*2+obj->m_Detection.bbox[3]))
			{

				obj->bSendToViz=true;
				iIndexToSet=iIndex;
				std::cout <<  "Mouse:" << mousex<< ","<< mousey<<   std::endl;
			}
		}
		iIndex++;
	}
	if(iIndexToSet!=-1)
	{
		std::cout << "hittest:" << iHitTestindex++ << std::endl;
		TrackedObjectsNextPtr[iIndexToSet]->bSendToViz=true;

	}

	return false;
}




bool TestDetections(int mousex,int mousey)
{

	int iIndexToSet=-1;
	int iIndex=0;

	for( TrackedObj f:TrackedObjectsNext)
	{
		if(((f.m_Detection.bbox[0]-f.m_Detection.bbox[2]/2) <mousex)&&mousex<(f.m_Detection.bbox[0]+f.m_Detection.bbox[2]/2))
		{
			std::cout << f.m_Detection.bbox[1] << std::endl;
			if(  (   (f.m_Detection.bbox[1]*2-f.m_Detection.bbox[3]) <mousey)&&mousey<(f.m_Detection.bbox[1]*2+f.m_Detection.bbox[3]))
			{

				f.bSendToViz=true;
				iIndexToSet=iIndex;
				std::cout <<  "Mouse:" << mousex<< ","<< mousey<<   std::endl;
			//	return true;

			}
		}
		iIndex++;
	}
	if(iIndexToSet!=-1)
	{
		std::cout << "hittest:" << iHitTestindex++ << std::endl;
		TrackedObjectsNext[iIndexToSet].bSendToViz=true;

	}

	return false;
}

//sudo mkdir /mnt/ramdisk
//sudo mount -t tmpfs -o rw,size=2G tmpfs /mnt/ramdisk

long iId=0;
TrackedObj *LinkDetectionsPTR(Detection k0,int iFrameCounter,Mat *DrawingMat)
{
	TrackedObj *returnObj=0;
	std::vector<Distances> distances_between_players_players;
	double fShortestDist=10000.0;
	TrackedObj *keepTrackedObj;
	Distances temp_keep_Distances;

	if((k0.bbox[2]/k0.bbox[3])<0.4)
	{

		return returnObj;
	}



	//std::iterator<TrackedObj> position_keep;


	int iRemoveIndex=-1;
	int index=0;
	for( TrackedObj *f:TrackedObjectsLivePtr)
	{
		Distances temp;

	//	if(temp.CheckCondition(k0,f.m_Detection,50))
		{

			temp.Set(k0,f->m_Detection);


			distances_between_players_players.push_back(temp);
		//temp.Draw(DrawingMat);
			double volume_ratio;
			if(temp.volume1<temp.volume2)
				volume_ratio=temp.volume1/temp.volume2;
			else
				volume_ratio=temp.volume2/temp.volume1;


			if(fShortestDist>temp.m_distance&&volume_ratio>0.60)
			{
				//std::cout << volume_ratio <<"vol ratio"<<std::endl;
				//std::cout << abs(temp.volume1-temp.volume2)<< std::endl;
				fShortestDist=temp.m_distance;
				keepTrackedObj=f;
				//position_keep=f;
				temp_keep_Distances=temp;
				iRemoveIndex=index;

			}
		}
		index++;
	}





	if(iRemoveIndex!=-1&&fShortestDist<50.0)
	{
	//	std::cout << fShortestDist << std::endl;
		TrackedObjectsLivePtr.erase(TrackedObjectsLivePtr.begin()+iRemoveIndex);
	}


	if(fShortestDist<50.0/*&&temp_keep.volume_diff<0.40*/)
	{
		temp_keep_Distances.Draw(DrawingMat);
		//NewObject.iObjID=keepTrackedObj.iObjID;
		//NewObject.Color=keepTrackedObj.Color;
		keepTrackedObj->iFrameCounter=keepTrackedObj->iFrameCounter+1;




		//iId=keep.iObjID;
		returnObj=keepTrackedObj;
		returnObj->m_Detection=k0;
	//	std::cout <<  keep.m_Detection.bbox[0] <<" ID:"<< keep.iObjID << "d"<<fShortestDist <<std::endl;
		char label[260];
		//sprintf(label,"%0.0f %d %f %f",keep.m_Detection.bbox[0],keep.iObjID,fShortestDist,temp_keep.volume_diff);
		//sprintf(label,"%d %f %f",keep.iObjID,fShortestDist,temp_keep.volume_diff);
		//std::cout << "Box Rasio:"<<k0.bbox[2]/k0.bbox[3] << std::endl;
		sprintf(label,"%d %d %f",keepTrackedObj->iObjID,keepTrackedObj->iFrameCounter,(k0.bbox[2]/k0.bbox[3]));
	//	std::cout <<  label << std::endl;
		returnObj->Label=label;
	//	std::cout <<  keep.Label << std::endl;
	}else
	{	//test to meet criteria
	//	NewObject.m_Detection

		returnObj=new TrackedObj;

		returnObj->m_Detection=k0;
		returnObj->iObjID=iId;
		iId++;
	//	std::cout <<  iId << std::endl;
		returnObj->Color=Scalar(rand()%255,rand()%255,rand()%255);

		char label[260];
	//	sprintf(label,"%0.0f %d %f %f",keep.m_Detection.bbox[0],keep.iObjID,fShortestDist,temp_keep.volume_diff);
		sprintf(label,"New New New New New %d %f ",returnObj->iObjID,(k0.bbox[2]/k0.bbox[3]));
		//std::cout <<  label << std::endl;
		returnObj->Label=label;
	}
	TrackedObjectsNextPtr.push_back(returnObj);
	return returnObj;



}



TrackedObj LinkDetections(Detection k0,int iFrameCounter,Mat *DrawingMat)
{
	TrackedObj returnObj;
	std::vector<Distances> distances_between_players_players;
	double fShortestDist=10000.0;
	TrackedObj keep;
	Distances temp_keep;

	if((k0.bbox[2]/k0.bbox[3])<0.4)
	{
		returnObj.iActive=0;
		return returnObj;
	}



	//std::iterator<TrackedObj> position_keep;


	int iRemoveIndex=-1;
	int index=0;
	for( TrackedObj f:TrackedObjectsLive)
	{
		Distances temp;

	//	if(temp.CheckCondition(k0,f.m_Detection,50))
		{

			temp.Set(k0,f.m_Detection);


			distances_between_players_players.push_back(temp);
		//temp.Draw(DrawingMat);
			double volume_ratio;
			if(temp.volume1<temp.volume2)
				volume_ratio=temp.volume1/temp.volume2;
			else
				volume_ratio=temp.volume2/temp.volume1;


			if(fShortestDist>temp.m_distance&&volume_ratio>0.60)
			{
				//std::cout << volume_ratio <<"vol ratio"<<std::endl;
				//std::cout << abs(temp.volume1-temp.volume2)<< std::endl;
				fShortestDist=temp.m_distance;
				keep=f;
				//position_keep=f;
				temp_keep=temp;
				iRemoveIndex=index;

			}
		}
		index++;
	}

	TrackedObj NewObject;
//	NewObject.iFrameNumber=iFrameCounter;
	NewObject.m_Detection=k0;
	NewObject.bSendToViz=keep.bSendToViz;



	if(iRemoveIndex!=-1&&fShortestDist<50.0)
	{
	//	std::cout << fShortestDist << std::endl;
		TrackedObjectsLive.erase(TrackedObjectsLive.begin()+iRemoveIndex);
	}


	if(fShortestDist<50.0/*&&temp_keep.volume_diff<0.40*/)
	{
		temp_keep.Draw(DrawingMat);
		NewObject.iObjID=keep.iObjID;
		NewObject.Color=keep.Color;
		NewObject.iFrameCounter=keep.iFrameCounter+1;



		//iId=keep.iObjID;
		returnObj=keep;
	//	std::cout <<  keep.m_Detection.bbox[0] <<" ID:"<< keep.iObjID << "d"<<fShortestDist <<std::endl;
		char label[260];
		//sprintf(label,"%0.0f %d %f %f",keep.m_Detection.bbox[0],keep.iObjID,fShortestDist,temp_keep.volume_diff);
		//sprintf(label,"%d %f %f",keep.iObjID,fShortestDist,temp_keep.volume_diff);
		//std::cout << "Box Rasio:"<<k0.bbox[2]/k0.bbox[3] << std::endl;
		sprintf(label,"%d %d %f",keep.iObjID,NewObject.iFrameCounter,(k0.bbox[2]/k0.bbox[3]));
	//	std::cout <<  label << std::endl;
		returnObj.Label=label;
	//	std::cout <<  keep.Label << std::endl;
	}else
	{	//test to meet criteria
	//	NewObject.m_Detection
		NewObject.iObjID=iId;
		iId++;
	//	std::cout <<  iId << std::endl;
		NewObject.Color=Scalar(rand()%255,rand()%255,rand()%255);
		returnObj=NewObject;
		char label[260];
	//	sprintf(label,"%0.0f %d %f %f",keep.m_Detection.bbox[0],keep.iObjID,fShortestDist,temp_keep.volume_diff);
		sprintf(label,"New New New New New %d %f ",keep.iObjID,(k0.bbox[2]/k0.bbox[3]));
		//std::cout <<  label << std::endl;
		returnObj.Label=label;
	}
	TrackedObjectsNext.push_back(NewObject);
	return returnObj;
}

void ResetDetectionsPTR(int iFrameCounter,bool bTrackReset)
{
//	FrameInfo

	for(int i=0; i < TrackedObjectsLivePtr.size(); i++){
		std::cout <<  TrackedObjectsLivePtr[i]->m_Detection.bbox[0] << " " <<TrackedObjectsLivePtr[i]->iObjID <<" "<< TrackedObjectsLivePtr[i]->iFrameCounter << std::endl;
		delete(TrackedObjectsLivePtr[i]);

		}
	std::cout <<  "before clear objcount" << TrackedObjectsLivePtr.size() << std::endl;
	TrackedObjectsLivePtr.clear();
	std::cout <<  "after clear objcount" << TrackedObjectsLivePtr.size() << std::endl;
	TrackedObjectsLivePtr=TrackedObjectsNextPtr;
	TrackedObjectsNextPtr.clear();

	if(bTrackReset)
	for(int i=0; i < TrackedObjectsLivePtr.size(); i++){
		TrackedObjectsLivePtr[i]->bSendToViz=false;
	}
	std::cout <<  "objcount" << TrackedObjectsLivePtr.size() << std::endl;
}





void ResetDetections(int iFrameCounter,bool bTrackReset)
{
//	FrameInfo

	for(int i=0; i < TrackedObjectsLive.size(); i++){
		std::cout <<  TrackedObjectsLive[i].m_Detection.bbox[0] << " " <<TrackedObjectsLive[i].iObjID <<" "<< TrackedObjectsLive[i].iFrameCounter << std::endl;

		}

	TrackedObjectsLive.clear();
//	std::cout <<  "after clear objcount" << TrackedObjectsLive.size() << std::endl;
	TrackedObjectsLive=TrackedObjectsNext;
	TrackedObjectsNext.clear();

	if(bTrackReset)
	for(int i=0; i < TrackedObjectsLive.size(); i++){
		TrackedObjectsLive[i].bSendToViz=false;
	}


//	std::cout <<  "objcount" << TrackedObjectsLive.size() << std::endl;

}


void ClassifyDetection(TrackedObj  *linked)
{

	YUV_Unpacked_Video;
	linked->m_Detection.bbox[0];

//	cuda::sum;
}










void DrawSnapShotDetectionsPTR(Mat *DrawingMat,bool bTrackReset)
{
	static long iFrameCounter=0;
	//std::vector<Detection> SnapYolov5Detection;
	ResetDetectionsPTR(iFrameCounter,bTrackReset);
	SendSocketResetFrame();
	//log to mysql backprojection x y and camera matrix
	for(Detection k:SnapYolov5Detection)
		{
			cv::Rect r = get_rect2(k.bbox);
			if(r.x>1920)
				continue;
			TrackedObj  *linkedptr=LinkDetectionsPTR(k,iFrameCounter,DrawingMat);
			if(linkedptr==0)
			{
				continue;
			}

			if(linkedptr->iActive==1)
			{
			//	if(int(k.class_id)==0)
					SendSocketUpdatePos(linkedptr->iObjID,linkedptr->Color[0],linkedptr->Color[1],linkedptr->Color[2],linkedptr->m_Detection.bbox[0]+linkedptr->m_Detection.bbox[2]/2,linkedptr->m_Detection.bbox[1]*2+linkedptr->m_Detection.bbox[3]*2);
				if(linkedptr->bSendToViz)
				{
			//		SendSocketUpdatePos(linkedptr->iObjID,linkedptr->Color[0],linkedptr->Color[1],linkedptr->Color[2],linkedptr->m_Detection.bbox[0]+linkedptr->m_Detection.bbox[2]/2,linkedptr->m_Detection.bbox[1]*2+linkedptr->m_Detection.bbox[3]*2);
					SendVizSocket(r.x+r.width/2,r.y);
					cv::rectangle(*DrawingMat,r,Scalar(0,0,255),5);
				}
				std::cout << linkedptr->iFrameCounter << std::endl;
				if(linkedptr->iFrameCounter<50)
					cv::rectangle(*DrawingMat,r,cv::Scalar(0,0,0),1);
				else
				{
					if(linkedptr->iClasified==-1)
					{
						Mat myclassify=Mat(*DrawingMat,r);
//						linkedptr->iClasified=Classify(myclassify.clone());

						//ClassifyDetection(linked);

					}else
					{
						if(linkedptr->iClasified==1)
							cv::rectangle(*DrawingMat,r,cv::Scalar(0,0,0),7);


						cv::rectangle(*DrawingMat,r,linkedptr->Color,3);

					}

				}
				String text;
				Mat img=*DrawingMat;
				if(int(k.class_id)==1)
					 text = "B"+ linkedptr->Label ;
				else
					 text = "P"+ linkedptr->Label ;
				int fontFace = FONT_HERSHEY_PLAIN;

				if(linkedptr->iClasified!=-1)
					text=text+" "+std::to_string(linkedptr->iClasified);

				double fontScale =1.0;
				int thickness = 1;

				int baseline=0;
				Size textSize = getTextSize(text, fontFace,
											fontScale, thickness, &baseline);
				baseline += thickness;
				// center the text
				Point textOrg(r.x+r.width/2,
							  r.y/*+r.height/2*/);


				putText(img, text, textOrg, fontFace, fontScale,
						Scalar::all(255), thickness, 8);
				//std::cout<< "MyTime"<< std::endl;

			}

		}

	ResetDetectionsPTR(iFrameCounter,false);

	for(Detection k:SnapYolov5Detection)
			{
				cv::Rect r = get_rect2(k.bbox);
				if(r.x<1920)
					continue;



				k.bbox[0]=k.bbox[0]-1920;
				r = get_rect2(k.bbox);
				TrackedObj  *linkedptr=LinkDetectionsPTR(k,iFrameCounter,DrawingMat);
				if(linkedptr==0)
							{
								continue;

							}




				if(linkedptr->iActive==1)
				{

					if(linkedptr->iFrameCounter<50)
						cv::rectangle(*DrawingMat,r,cv::Scalar(0,0,0),1);
					else
						cv::rectangle(*DrawingMat,r,linkedptr->Color,3);
			//	cv::putText(*DrawingMat,"Player_ID",r.tl(),Scalar(255,255,255),2);

				String text;
				Mat img=*DrawingMat;
				if(int(k.class_id)==1)
					 text = "B"+ linkedptr->Label ;
				else
					 text = "P"+ linkedptr->Label ;
				int fontFace = FONT_HERSHEY_PLAIN;
				double fontScale =1.0;
				int thickness = 1;

				int baseline=0;
				Size textSize = getTextSize(text, fontFace,
				                            fontScale, thickness, &baseline);
				baseline += thickness;
				// center the text
				Point textOrg(r.x+r.width/2,
				              r.y/*+r.height/2*/);

				// then put the text itself
				putText(img, text, textOrg, fontFace, fontScale,
				        Scalar::all(255), thickness, 8);
				//std::cout<< "MyTime"<< std::endl;
				}


			}

}











void DrawSnapShotDetections(Mat *DrawingMat,bool bTrackReset)
{


	static long iFrameCounter=0;
	//std::vector<Detection> SnapYolov5Detection;
	ResetDetections(iFrameCounter,bTrackReset);
	SendSocketResetFrame();


	//log to mysql backprojection x y and camera matrix
	for(Detection k:SnapYolov5Detection)
		{
			cv::Rect r = get_rect2(k.bbox);
			if(r.x>1920)
				continue;
			TrackedObj  linked=LinkDetections(k,iFrameCounter,DrawingMat);

			if(linked.iActive==1)
			{

			//	if(int(k.class_id)==0)
					SendSocketUpdatePos(linked.iObjID,linked.Color[0],linked.Color[1],linked.Color[2],linked.m_Detection.bbox[0]+linked.m_Detection.bbox[2]/2,linked.m_Detection.bbox[1]*2+linked.m_Detection.bbox[3]*2);
				if(linked.bSendToViz)
				{
			//		SendSocketUpdatePos(linked.iObjID,linked.Color[0],linked.Color[1],linked.Color[2],linked.m_Detection.bbox[0]+linked.m_Detection.bbox[2]/2,linked.m_Detection.bbox[1]*2+linked.m_Detection.bbox[3]*2);
					SendVizSocket(r.x+r.width/2,r.y);
					cv::rectangle(*DrawingMat,r,Scalar(0,0,255),5);
				}
				std::cout << linked.iFrameCounter << std::endl;
				if(linked.iFrameCounter<50)
					cv::rectangle(*DrawingMat,r,cv::Scalar(0,0,0),1);
				else
				{
					if(linked.iClasified==-1)
					{
						//ClassifyDetection(linked);


					}
					cv::rectangle(*DrawingMat,r,linked.Color,3);

				}
				String text;
				Mat img=*DrawingMat;
				if(int(k.class_id)==1)
					 text = "B"+ linked.Label ;
				else
					 text = "P"+ linked.Label ;
				int fontFace = FONT_HERSHEY_PLAIN;
				double fontScale =1.0;
				int thickness = 1;

				int baseline=0;
				Size textSize = getTextSize(text, fontFace,
											fontScale, thickness, &baseline);
				baseline += thickness;
				// center the text
				Point textOrg(r.x+r.width/2,
							  r.y/*+r.height/2*/);


				putText(img, text, textOrg, fontFace, fontScale,
						Scalar::all(255), thickness, 8);
				//std::cout<< "MyTime"<< std::endl;

			}

		}

	ResetDetections(iFrameCounter,false);

	for(Detection k:SnapYolov5Detection)
			{
				cv::Rect r = get_rect2(k.bbox);
				if(r.x<1920)
					continue;



				k.bbox[0]=k.bbox[0]-1920;
				r = get_rect2(k.bbox);
				TrackedObj  linked=LinkDetections(k,iFrameCounter,DrawingMat);
				if(linked.iActive==1)
				{

					if(linked.iFrameCounter<50)
						cv::rectangle(*DrawingMat,r,cv::Scalar(0,0,0),1);
					else
						cv::rectangle(*DrawingMat,r,linked.Color,3);
			//	cv::putText(*DrawingMat,"Player_ID",r.tl(),Scalar(255,255,255),2);

				String text;
				Mat img=*DrawingMat;
				if(int(k.class_id)==1)
					 text = "B"+ linked.Label ;
				else
					 text = "P"+ linked.Label ;
				int fontFace = FONT_HERSHEY_PLAIN;
				double fontScale =1.0;
				int thickness = 1;

				int baseline=0;
				Size textSize = getTextSize(text, fontFace,
				                            fontScale, thickness, &baseline);
				baseline += thickness;
				// center the text
				Point textOrg(r.x+r.width/2,
				              r.y/*+r.height/2*/);

				// then put the text itself
				putText(img, text, textOrg, fontFace, fontScale,
				        Scalar::all(255), thickness, 8);
				//std::cout<< "MyTime"<< std::endl;
				}


			}

}





void DrawSnapShotDetections_clean(Mat *DrawingMat,bool bTrackReset)
{


	static long iFrameCounter = 0;

	//log to mysql backprojection x y and camera matrix
	for(Detection k: SnapYolov5Detection)
	{
		cv::Rect r = get_rect2(k.bbox);
		if(r.x>1920)
			continue;
		cv::rectangle(*DrawingMat,r,cv::Scalar(0,0,0),1);
	}

	for(Detection k: SnapYolov5Detection)
	{
		cv::Rect r = get_rect2(k.bbox);
		if(r.x<1920)
			continue;
		k.bbox[0]=k.bbox[0]-1920;
		r = get_rect2(k.bbox);
		cv::rectangle(*DrawingMat,r,cv::Scalar(0,0,0),1);
	}

}

bool Checkifnotoverplayer(cv::Rect TestRect)
{
	for(Detection k:SnapYolov5Detection)
		{
			cv::Rect r = get_rect_normal_size_widow(k.bbox,0);
			//cv::Rect r2 = get_rectHalf(k.bbox,0);
			//r2=r2+Point(10,10);
			cv::Rect InterSection=TestRect&r;
		//	std::cout <<InterSection<< " "<< r<< " "<< TestRect<< std::endl;
			if(InterSection.height!=0||InterSection.width!=0)
				return true;

		}
	return false;
}


void drawHistogram(cv::Mat& b_hist, cv::Mat *histImage,int histSize) {

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / (double)histSize);

 //   cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::normalize(b_hist, b_hist, 0, histImage->rows, cv::NORM_MINMAX, -1,
                  cv::Mat());


    for (int i = 1; i < histSize; i++) {
      cv::line(
          *histImage,
          cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
          cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
          cv::Scalar(255, 0, 0), 2, 8, 0);

    }

   // cv::namedWindow("calcHist Demo", cv::WINDOW_AUTOSIZE);
  //  cv::imshow("calcHist Demo", histImage);

}



void DrawMask(GpuMat *Matptr,Detection k)
{
		cv::Rect r = get_rect_normal_size_widow(k.bbox,2);
			//cv::Rect r_half = get_rect_normal_half_widow(k.bbox,2);

			if((r.height+r.y)>(1080/2))
			{
				r.height=(1080/2)-r.y;
			}
			try{
				//Matptr(r)->setTo(Scalar::all(255));
				GpuMat n=*Matptr;
				n(r).setTo(Scalar::all(255));
//				Cuda_MASK_HISTOGRAM(r_half).setTo(Scalar::all(255));
//				auto resu=cv::mean(umap,Cuda_MASK_HISTOGRAM);
//				auto resv=cv::mean(vmap,Cuda_MASK_HISTOGRAM);
//
//				std::cout << resu[0]<<" "<<resv[0] << std::endl;
//				cv::Point   t(resu[0],resv[0]);
//				cv::Scalar  sc(resu[0],resv[0],resu[0]);
//				circle(uv_pos_map,t,2,sc,1);
//
//				Cuda_MASK_HISTOGRAM.setTo(0);



			}catch(Exception c)
			{
			//	std::cout << c.err << std::endl;

			}



}


void CreateYoloMask()
{



}

void Launch_yuyv_Unpacked_GenerateMask(int iAvgCutOff, int iUse,bool bAutoTrain)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemset(ChromaGeneratedMask[iUse], 0, (m_iWidth * (m_iHeight)) * sizeof(uchar));//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	const int dstAlignedWidthUnpackedData1 = (1920 / 2);
	const dim3 blockRUN(16, 16);
	const dim3 gridRun(iDivUp(dstAlignedWidthUnpackedData1, blockRUN.x), iDivUp(1080, blockRUN.y));
	const dim3 gridFull(iDivUp(1920, blockRUN.x), iDivUp(1080, blockRUN.y));
	const int dstAlignedWidthMask = 1920;

	yuyv_Unpacked_GenerateMask << <gridRun, blockRUN >>> (YUV_Unpacked_Video, ChromaGeneratedMask[iUse], LookUpDataArry[iUse], 1920, 1080, dstAlignedWidthUnpackedData1, dstAlignedWidthMask, iAvgCutOff);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}

}


void Launch_yuyv_Unpacked_GenerateMask_yolo_seg(int iAvgCutOff, int iUse,bool bAutoTrain,float *segmented_mask)
{


	cudaError_t cudaStatus;
	cudaStatus = cudaMemset(ChromaGeneratedMask[iUse], 0, (m_iWidth * (m_iHeight)) * sizeof(uchar));//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}
	const int dstAlignedWidthUnpackedData1 = (1920 / 2);
	const dim3 blockRUN(16, 16);
	const dim3 gridRun(iDivUp(dstAlignedWidthUnpackedData1, blockRUN.x), iDivUp(1080, blockRUN.y));
	const dim3 gridFull(iDivUp(1920, blockRUN.x), iDivUp(1080, blockRUN.y));
	const int dstAlignedWidthMask = 1920;

	yuyv_Unpacked_GenerateMask_yolo_segmented << <gridRun, blockRUN >> > (YUV_Unpacked_Video, ChromaGeneratedMask[iUse], LookUpDataArry[iUse], 1920, 1080, dstAlignedWidthUnpackedData1, dstAlignedWidthMask, iAvgCutOff,segmented_mask);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}
}



void Launch_yuyv_Unpacked_GenerateMask_yolo(int iAvgCutOff, int iUse,bool bAutoTrain)
{

	auto startmask = std::chrono::system_clock::now();
	GpuMat Cuda_MASK;
	Cuda_MASK.create(1080/2, 1920*2, CV_8UC1);
	Cuda_MASK.step = 1920*2;
	Cuda_MASK.setTo(0);
	int index=0;
	for (Detection k:Yolov5Detection)
	{
		DrawMask(&Cuda_MASK,k);
	}
	auto endmask = std::chrono::system_clock::now();
	auto duration_mask=std::chrono::duration_cast<std::chrono::microseconds>(endmask - startmask).count();

#ifdef DISPLAY_I_TIMINGS
	std::cout <<"mask:"<<duration_mask<< "us  " <<std::endl;
#endif

	#ifdef PREVIEW_OUTPUTRENDER
	imshow("Yolo generated mask",Cuda_MASK );
	#endif


	cudaError_t cudaStatus;
	cudaStatus = cudaMemset(ChromaGeneratedMask[iUse], 0, (m_iWidth * (m_iHeight)) * sizeof(uchar));//4228250625
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}
	const int dstAlignedWidthUnpackedData1 = (1920 / 2);
	const dim3 blockRUN(16, 16);
	const dim3 gridRun(iDivUp(dstAlignedWidthUnpackedData1, blockRUN.x), iDivUp(1080, blockRUN.y));
	const dim3 gridFull(iDivUp(1920, blockRUN.x), iDivUp(1080, blockRUN.y));
	const int dstAlignedWidthMask = 1920;

	yuyv_Unpacked_GenerateMask_yolo << <gridRun, blockRUN >> > (YUV_Unpacked_Video, ChromaGeneratedMask[iUse], LookUpDataArry[iUse], 1920, 1080, dstAlignedWidthUnpackedData1, dstAlignedWidthMask, iAvgCutOff,Cuda_MASK.data);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}
}





__global__ void UpdateLookupFrom_XY_Posision(uint4* src_Video_Unapc, uchar* LookupTable, int srcAlignedWidth, int istartX, int iStartY, int iEndX, int iEndY, int iUV_Diameter, int iLum_Diameter)//
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (!((x >= istartX&&x <= iEndX) && (y >= iStartY&&y <= iEndY)))
		return;

	if (x >= srcAlignedWidth || y >= 1080)
		return;


	uint4 *macroPxVideo;
	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	
	
	for (int a = -iUV_Diameter; a < iUV_Diameter; a++)
		for (int b = -iUV_Diameter; b < iUV_Diameter; b++)
			for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
			{
				double bitpos1a = GetBitPos3(make_double3(macroPxVideo->x + a, macroPxVideo->z + b, macroPxVideo->y + c));
				while (!GetBit3(bitpos1a, LookupTable))
				{
					SetBit3(bitpos1a, LookupTable,1);
					//UpdateOneDimetionLookup(macroPxVideoReal.x + a, macroPxVideoReal.z + b, LookUpColorDataOneDimention_Unpacked1);
				}
			}



	for (int a = -iUV_Diameter; a < iUV_Diameter; a++)
		for (int b = -iUV_Diameter; b < iUV_Diameter; b++)
			for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
			{
				double bitpos1a = GetBitPos3(make_double3(macroPxVideo->x + a, macroPxVideo->z + b, macroPxVideo->w + c));
				while (!GetBit3(bitpos1a, LookupTable))
				{
					SetBit3(bitpos1a, LookupTable,1);
				//	UpdateOneDimetionLookup(macroPxVideoReal.x + a, macroPxVideoReal.z + b, LookUpColorDataOneDimention_Unpacked1);
				}
			}
}


__global__ void UpdateLookupFrom_XY_Posision_Diffrent(uint4* src_Video_Unapc, uint4* src_Key_Unapc, uchar* LookupTable,int PixelPosX, int PixelPosY,int srcAlignedWidth,int iUV_Diameter,int iLum_Diameter)//
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= iUV_Diameter || y >= iUV_Diameter)
		return;

	uint4 *macroPxVideo;
	
	macroPxVideo = &src_Video_Unapc[PixelPosY * srcAlignedWidth + PixelPosX];
	

	

	for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
	{

		double bitpos1a = GetBitPos3(make_double3(macroPxVideo->x + x- iUV_Diameter/2, macroPxVideo->z + y - iUV_Diameter / 2, macroPxVideo->y + c));
		while (GetBit3(bitpos1a, LookupTable))
		{
			ClearBit3(bitpos1a, LookupTable);
		}
	}

	
	for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
	{
		double bitpos1a = GetBitPos3(make_double3(macroPxVideo->x + x - iUV_Diameter / 2, macroPxVideo->z + y - iUV_Diameter / 2, macroPxVideo->w + c));
		while (GetBit3(bitpos1a, LookupTable))
		{
			ClearBit3(bitpos1a, LookupTable);
		}
	}
}



__global__ void UpdateLookupFrom_XY_Posision_Diffrent_Scaling(uint4* src_Video_Unapc, uchar* LookupTable, int PixelPosX, int PixelPosY, int srcAlignedWidth, int iOuter_Diameter,int iUV_Diameter, int iLum_Diameter,float dSScaling,int iMaxKeyVal)//
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= ((iOuter_Diameter + iUV_Diameter) ) || y >= ((iOuter_Diameter + iUV_Diameter)))
		return;


	uint4 *macroPxVideo;
	macroPxVideo = &src_Video_Unapc[PixelPosY * srcAlignedWidth + PixelPosX];

	float fCenter = (iOuter_Diameter + iUV_Diameter) / 2;
	float fx = x - fCenter;
	float fy = y - fCenter;
	float Distance = sqrtf(powf(fx, 2.0) + powf(fy, 2.0));
	if (Distance > fCenter)
		return;

	float KeyValue = iMaxKeyVal;
	if (Distance > (iUV_Diameter / 2))
	{
		float fScaleDistance = (iOuter_Diameter/2)-(Distance - (iUV_Diameter / 2));
		KeyValue = dSScaling*fScaleDistance;
	}


	uchar uKeyValue = KeyValue;
	if (uKeyValue == 0)
		return;

	int myFX = macroPxVideo->x + fx ;
	int myFy = macroPxVideo->z + fy ;

	for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
	{
	    double bitpos1a = GetBitPos3(make_double3(myFX, myFy, macroPxVideo->y+c ));

		while (GetBit3(bitpos1a, LookupTable)< uKeyValue)
		{

			SetBit3(bitpos1a, LookupTable, uKeyValue);
		}

	}
	for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
	{
		double bitpos1a = GetBitPos3(make_double3(myFX, myFy, macroPxVideo->w + c));
		while (GetBit3(bitpos1a, LookupTable) < uKeyValue)
		{

			SetBit3(bitpos1a, LookupTable, uKeyValue);
		}
	}

	macroPxVideo->w=0;
	macroPxVideo->x=0;
	macroPxVideo->y=0;
	macroPxVideo->z=0;
}




__global__ void UpdateLookupFrom_XY_Posision_Diffrent_Scaling_ORIGANAL(uint4* src_Video_Unapc, uchar* LookupTable, int PixelPosX, int PixelPosY, int srcAlignedWidth, int iOuter_Diameter,int iUV_Diameter, int iLum_Diameter,float dSScaling,int iMaxKeyVal,  int iOperation)//
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= ((iOuter_Diameter + iUV_Diameter) ) || y >= ((iOuter_Diameter + iUV_Diameter)))
		return;

	uint4 *macroPxVideo;
	macroPxVideo = &src_Video_Unapc[PixelPosY * srcAlignedWidth + PixelPosX];

	//printf("%d %d %d %d\n",macroPxVideo->x,macroPxVideo->z,macroPxVideo->y);


	float fCenter = (iOuter_Diameter + iUV_Diameter) / 2;
	float fx = x - fCenter;
	float fy = y - fCenter;
	float Distance = sqrtf(powf(fx, 2.0) + powf(fy, 2.0));
	if (Distance > fCenter)
		return;

	float KeyValue = iMaxKeyVal;
	if (Distance > (iUV_Diameter / 2))
	{
		float fScaleDistance = (iOuter_Diameter/2)-(Distance - (iUV_Diameter / 2));
		KeyValue = dSScaling*fScaleDistance;
	}


	uchar uKeyValue = KeyValue;
	if (uKeyValue == 0)
		return;

	int myFX = macroPxVideo->x + x - fCenter;
	int myFy = macroPxVideo->z + y - fCenter;

	// printf("%f\n",fCenter );
	for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
	{
	    double bitpos1a = GetBitPos3(make_double3(myFX, myFy, macroPxVideo->y+ c ));

		while (GetBit3(bitpos1a, LookupTable)< uKeyValue)
		{

			SetBit3(bitpos1a, LookupTable, uKeyValue);
		}
	}
	for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
	{
		double bitpos1a = GetBitPos3(make_double3(myFX, myFy, macroPxVideo->w + c));
		while (GetBit3(bitpos1a, LookupTable) < uKeyValue)
		{

			SetBit3(bitpos1a, LookupTable, uKeyValue);
		}
	}
	macroPxVideo->w=0;
	macroPxVideo->x=0;
	macroPxVideo->y=0;
	macroPxVideo->z=0;
}


void Launch_UpdateLookupFrom_XY_Posision_Erase(int istartX, int iStartY, int iEndX, int iEndY, int iErase_Diameter,int iErase_Lum_Diameter,bool bPaintItBack)
{
	cudaError_t cudaStatus;
	const dim3 blockRUN(16, 16);
	const dim3 gridRun(iDivUp(iErase_Diameter *2, blockRUN.x), iDivUp(iErase_Diameter *2, blockRUN.y));

	uchar* ptrLookUpDataToUse=LookUpDataArry[0];
		if(bPaintItBack)
			ptrLookUpDataToUse=LookUpDataArry[1];

	for(int x= (istartX/2);x<(iEndX/2);x++)
		for (int y = iStartY; y < iEndY; y++)
			{
				UpdateLookupFrom_XY_Posision_Diffrent << <gridRun, blockRUN >> > (YUV_Unpacked_Video_SnapShot, YUV_Unpacked_Key_SnapShot, ptrLookUpDataToUse,x,y, (1920 / 2), iErase_Diameter *2, iErase_Lum_Diameter);
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
				return;
			}
		}
}

 void Set_YUV_Unpacked_Video_SnapShot_ptr(uint4 *tmpYUV_Unpacked_Video_SnapShot)
 {
	 YUV_Unpacked_Video_SnapShot=tmpYUV_Unpacked_Video_SnapShot;
 }
 __global__ void UpdateLookupFrom_XY_Posision_Diffrent_Scaling_CudaLuanch(unsigned vectorSize,int4 *ptrTrainingList,int *ptrResultsList, uint4* src_Key_Unapc,int srcAlignedWidth)//
 {



	 unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;


	 	if (idx < vectorSize)
	 	{
	 		 ptrResultsList[idx]=0;
	 		int x=ptrTrainingList[idx].x/2;
	 		int y=ptrTrainingList[idx].y;
	 		int w=ptrTrainingList[idx].w;
	 		int z=ptrTrainingList[idx].z/2;

	 		uint4 macroPxVideo0 = src_Key_Unapc[y * srcAlignedWidth + x];
	 		uint4 macroPxVideo1 = src_Key_Unapc[y * srcAlignedWidth + z];
	 		uint4 macroPxVideo2 = src_Key_Unapc[(w) * srcAlignedWidth + x];
	 		uint4 macroPxVideo3 = src_Key_Unapc[(w) * srcAlignedWidth + z];

	 		if (macroPxVideo0.y > 65.0 || macroPxVideo0.w > 65.0)
	 			ptrResultsList[idx]=1;
			if (macroPxVideo1.y > 65.0 || macroPxVideo1.w > 65.0)
 				ptrResultsList[idx]=1;
			if (macroPxVideo2.y > 65.0 || macroPxVideo2.w > 65.0)
				ptrResultsList[idx]=1;
			if (macroPxVideo3.y > 65.0 || macroPxVideo3.w > 65.0)
				ptrResultsList[idx]=1;

			/*if(ptrResultsList[idx]==1)
				printf("train");
			else
				printf("not\n");*/

	 	}

 }

 void Launch_UpdateLookupFrom_XY_Posision_check(Mat *DrawResults,std::list<int4>& Rectangles, int iUV_Diameter, int iLum_Diameter, int iOuter_Diameter, int iMaxKeyVal, int iOperation)
 {
		cudaError_t cudaStatus;
	 int4 *ptrTrainingList;
	 int *ptrResultsList;
	 int size=Rectangles.size();
	 cudaMallocManaged(&ptrTrainingList,size*sizeof(int4));
	 cudaMallocManaged(&ptrResultsList,size*sizeof(int));
	 int index=0;
	for (int4 rect:Rectangles)
	{
		ptrTrainingList[index++]=rect;
	}

	const int dstAlignedWidthUnpackedData1 = (1920 / 2);
	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	UpdateLookupFrom_XY_Posision_Diffrent_Scaling_CudaLuanch<<<blockCount, BLOCK_SIZE>>> (size,ptrTrainingList,ptrResultsList, YUV_Unpacked_Key_SnapShot,dstAlignedWidthUnpackedData1);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}

 	for(int x=0;x<size;x++)
	{
		if(ptrResultsList[x])
		{
			cv::rectangle(*DrawResults,Point(ptrTrainingList[x].x,ptrTrainingList[x].y),Point(ptrTrainingList[x].z,ptrTrainingList[x].w),Scalar(255,255,255),1);
			//Launch_UpdateLookupFrom_XY_Posision(ptrTrainingList[x].x,ptrTrainingList[x].y,ptrTrainingList[x].z,ptrTrainingList[x].w,  iUV_Diameter,  iLum_Diameter,  iOuter_Diameter,  iMaxKeyVal,  iOperation);
		}else
		{
			//cv::rectangle(*DrawResults,Point(ptrTrainingList[x].x,ptrTrainingList[x].y),Point(ptrTrainingList[x].z,ptrTrainingList[x].w),Scalar(0,255,0),1);
		}
	}
 }
void Launch_UpdateLookupFrom_XY_Posision(int istartX, int iStartY, int iEndX, int iEndY, int iUV_Diameter, int iLum_Diameter, int iOuter_Diameter, int iMaxKeyVal,bool bPaintItBack)
{

	cudaError_t cudaStatus;
	const dim3 blockRUN(16, 16);
	float ScalingValue = (double)iMaxKeyVal / (double)iOuter_Diameter;
	const dim3 gridRun2(iDivUp((iOuter_Diameter+ iUV_Diameter) * 2, blockRUN.x), iDivUp((iOuter_Diameter + iUV_Diameter) * 2, blockRUN.y));
	uchar* ptrLookUpDataToUse=LookUpDataArry[0];
	if(bPaintItBack)
		ptrLookUpDataToUse=LookUpDataArry[1];


	for (int x = (istartX / 2); x<(iEndX / 2); x++)
		for (int y = iStartY; y < iEndY; y=y+2)
		{
			UpdateLookupFrom_XY_Posision_Diffrent_Scaling << <gridRun2, blockRUN >> > (YUV_Unpacked_Video_SnapShot, ptrLookUpDataToUse, x, y, (1920 / 2), iOuter_Diameter*2, iUV_Diameter * 2, iLum_Diameter, ScalingValue, iMaxKeyVal);
		}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}
}

__global__ void yuyvUnpackedGenerateMask(uint4* src_Video_Unapc,uint4* src__Key_Unapc, uchar *maskUpload, uchar *maskDownload, uchar* LookupTable, int width, int height, int srcAlignedWidth, int dstAlignedWidth,int bTraining, uint4 *LookUpColorDataOneDimention_Unpacked1, int iUV_Diameter, int iLum_Diameter)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	uint4 macroPxVideoReal;
	uint4 *macroPxVideo;
	uint4 *macroPxVideo1;
	uint4 *macroPxVideo2;
	uint4 *macroPxVideo3;
	uint4 *macroPxVideo4;
	//if (x > 1 && x < (srcAlignedWidth - 1)&& y>1 && y<(1080 - 1))
	if(0)
	{
		macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
		macroPxVideo1 = &src_Video_Unapc[y * srcAlignedWidth + (x - 1)];
		macroPxVideo2 = &src_Video_Unapc[y * srcAlignedWidth + (x + 1)];
		macroPxVideo3 = &src_Video_Unapc[(y - 2) * srcAlignedWidth + x];
		macroPxVideo4 = &src_Video_Unapc[(y + 2) * srcAlignedWidth + x];

		macroPxVideoReal.w = (macroPxVideo->w + macroPxVideo1->w + macroPxVideo2->w + macroPxVideo3->w + macroPxVideo4->w) / 5;
		macroPxVideoReal.x = (macroPxVideo->x + macroPxVideo1->x + macroPxVideo2->x + macroPxVideo3->x + macroPxVideo4->x) / 5;
		macroPxVideoReal.y = (macroPxVideo->y + macroPxVideo1->y + macroPxVideo2->y + macroPxVideo3->y + macroPxVideo4->y) / 5;
		macroPxVideoReal.z = (macroPxVideo->z + macroPxVideo1->z + macroPxVideo2->z + macroPxVideo3->z + macroPxVideo4->z) / 5;

	}
	else
	{
		macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
		macroPxVideoReal = *macroPxVideo;
	}

	





	uint4 *macroPxKey;
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];

	//FrameData[macroPxVideo->w]



	if (macroPxKey->y < 65.0 && macroPxKey->w < 65.0)
	{
		//printf("ret\n")
		return;
	}
	double3 val1 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.y);
	double bitpos1 = GetBitPos3(val1);
	double3 val2 = make_double3(macroPxVideoReal.x, macroPxVideoReal.z, macroPxVideoReal.w);
	double bitpos2 = GetBitPos3(val2);

	if (bTraining == 1)
	{
	
		if (maskUpload[y * dstAlignedWidth + (x * 2) + 0] == 255)
		{
			for (int a = -iUV_Diameter; a < iUV_Diameter; a++)
				for (int b = -iUV_Diameter; b < iUV_Diameter; b++)
					for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
					{
						double3 val1a = make_double3(macroPxVideoReal.x + a, macroPxVideoReal.z + b, macroPxVideoReal.y + c);
						double bitpos1a = GetBitPos3(val1a);
						while (!GetBit3(bitpos1a, LookupTable))
						{
							SetBit3(bitpos1a, LookupTable,1);
							UpdateOneDimetionLookup(macroPxVideoReal.x + a, macroPxVideoReal.z + b, LookUpColorDataOneDimention_Unpacked1);
						}
					}
			/*	for (int t = -UP_DOWN_SET; t < UP_DOWN_SET; t++)
			{
				double3 val1a = make_double3(macroPxVideo->x, macroPxVideo->z+t, macroPxVideo->y );
				double bitpos1a = GetBitPos3(val1a);
				while (!GetBit3(bitpos1a, LookupTable))
				{
					SetBit3(bitpos1a, LookupTable);
					UpdateOneDimetionLookup(macroPxVideo->x, macroPxVideo->z+t, LookUpColorDataOneDimention_Unpacked1);
				}
			}


			for (int t = -UP_DOWN_SET; t < UP_DOWN_SET; t++)
			{
				double3 val1a = make_double3(macroPxVideo->x+t, macroPxVideo->z, macroPxVideo->y);
				double bitpos1a = GetBitPos3(val1a);
				while (!GetBit3(bitpos1a, LookupTable))
				{
					SetBit3(bitpos1a, LookupTable);
					UpdateOneDimetionLookup(macroPxVideo->x+ t, macroPxVideo->z , LookUpColorDataOneDimention_Unpacked1);
				}
			}*/
		}
			
		if (maskUpload[y * dstAlignedWidth + (x * 2) + 1] == 255)
		{

			for (int a = -iUV_Diameter; a < iUV_Diameter; a++)
				for (int b = -iUV_Diameter; b < iUV_Diameter; b++)
					for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
					{
						double3 val1a = make_double3(macroPxVideoReal.x + a, macroPxVideoReal.z + b, macroPxVideoReal.w + c);
						double bitpos1a = GetBitPos3(val1a);
						while (!GetBit3(bitpos1a, LookupTable))
						{
							SetBit3(bitpos1a, LookupTable,1);
							UpdateOneDimetionLookup(macroPxVideoReal.x + a, macroPxVideoReal.z + b, LookUpColorDataOneDimention_Unpacked1);
						}
					}
		}

		if (maskUpload[y * dstAlignedWidth + (x * 2) + 0] == 128)
		{
			for (int a = -iUV_Diameter/2; a < iUV_Diameter/2; a++)
				for (int b = -iUV_Diameter/2; b < iUV_Diameter/2; b++)
					for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
					{
						double3 val1a = make_double3(macroPxVideoReal.x + a, macroPxVideoReal.z + b, macroPxVideoReal.y + c);
						double bitpos1a = GetBitPos3(val1a);
						while (GetBit3(bitpos1a, LookupTable))
						{
							ClearBit3(bitpos1a, LookupTable);
							ClearOneDimetionLookup(macroPxVideoReal.x + a, macroPxVideoReal.z + b, LookUpColorDataOneDimention_Unpacked1);
						}
					}

		}


			
		if (maskUpload[y * dstAlignedWidth + (x * 2) + 1] == 128)
		{

			for (int a = -iUV_Diameter / 2; a < iUV_Diameter / 2; a++)
				for (int b = -iUV_Diameter / 2; b < iUV_Diameter / 2; b++)
					for (int c = -iLum_Diameter; c < iLum_Diameter; c++)
					{
						double3 val1a = make_double3(macroPxVideoReal.x + a, macroPxVideoReal.z + b, macroPxVideoReal.w + c);
						double bitpos1a = GetBitPos3(val1a);
						while (GetBit3(bitpos1a, LookupTable))
						{
							ClearBit3(bitpos1a, LookupTable);
							ClearOneDimetionLookup(macroPxVideoReal.x + a, macroPxVideoReal.z + b, LookUpColorDataOneDimention_Unpacked1);
						}
					}


			}
	}
	
	 
	if(GetBit3(bitpos1,LookupTable))
		maskDownload[y * dstAlignedWidth + (x * 2) + 0]=255;
	if (GetBit3(bitpos2, LookupTable))
		maskDownload[y * dstAlignedWidth + (x * 2) + 1]=255;
	
}



__global__ void yuyvUnpackedComBineDataChromaBipass(uint4* src_Video_Unapc,uint4* src__Fill_Unapc,uint4* src__Key_Unapc, int width, int height, int srcAlignedWidth)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;


	uint4 *macroPxVideo;
	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	uint4 *macroPxFill;
	macroPxFill = &src__Fill_Unapc[y * srcAlignedWidth + x];
	uint4 *macroPxKey;
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];

	if (macroPxKey->y < 65 && macroPxKey->w < 65)
	{
		return;
	}

	calculateBlendFullKey(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w);
	calculateBlendFullKey(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->y, &macroPxVideo->x);
	calculateBlendFullKey(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y);
	calculateBlendFullKey(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->w, &macroPxVideo->z);

}








__global__ void yuyvUnpackedComBineData(uint4* src_Video_Unapc,uint4* src__Fill_Unapc,uint4* src__Key_Unapc, int width, int height, int srcAlignedWidth, int dstAlignedWidth, uchar *maskUpload, int iBlendPos, double3 Parabolic, unsigned long int iCutOff)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)return;

	uint4 *macroPxVideo;
	uint4 *macroPxFill;
	uint4 *macroPxKey;
	uint4 *macroPxKeyTop;
	uint4 *macroPxKeyBottom;
	uint4 *macroPxKeyLeft;
	uint4 *macroPxKeyRight;

	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	macroPxFill = &src__Fill_Unapc[y * srcAlignedWidth + x];
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];

	if (y > 0)
	{
		macroPxKeyTop = &src__Key_Unapc[(y - 1) * srcAlignedWidth + x];
		macroPxKeyLeft = &src__Key_Unapc[(y)* srcAlignedWidth + (x - 1)];
	}
	else
	{
		macroPxKeyTop = &src__Key_Unapc[(y)* srcAlignedWidth + x];
		macroPxKeyLeft = &src__Key_Unapc[(y)* srcAlignedWidth + x];

	}

	if (y < 1919)
	{
		macroPxKeyBottom = &src__Key_Unapc[(y + 1) * srcAlignedWidth + x];
		macroPxKeyRight = &src__Key_Unapc[(y  * srcAlignedWidth + x + 1)];
	}
	else
	{
		macroPxKeyBottom = &src__Key_Unapc[(y)* srcAlignedWidth + x];
		macroPxKeyRight = &src__Key_Unapc[(y)* srcAlignedWidth + x];
	}

	if (macroPxKey->y < 65 && macroPxKey->w < 65)
	{
		return;
	}

	double dBlendPos = iBlendPos / 876.0;

	if (macroPxKey->y > iCutOff || macroPxKey->w > iCutOff || macroPxKeyTop->y > iCutOff || macroPxKeyTop->w > iCutOff || macroPxKeyBottom->y > iCutOff || macroPxKeyBottom->w > iCutOff
		|| macroPxKeyLeft->y > iCutOff || macroPxKeyLeft->w > iCutOff || macroPxKeyRight->y > iCutOff || macroPxKeyRight->w > iCutOff)
	{
		calculateBlendFullKey(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w);
		calculateBlendFullKey(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->y, &macroPxVideo->x);
		calculateBlendFullKey(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y);
		calculateBlendFullKey(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->w, &macroPxVideo->z);
	}
	else if (maskUpload[y * dstAlignedWidth + (x * 2) + 0] != 0 && maskUpload[y * dstAlignedWidth + (x * 2) + 1] != 0)
	{
		dBlendPos = maskUpload[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
		double Lum = (macroPxVideo->y + macroPxVideo->w) / 2.0;
		double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
		if (CalculateLumKeyVal < 0)
			return;

		dBlendPos = dBlendPos * CalculateLumKeyVal;
		calculateBlend(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w, dBlendPos);
		calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->y, &macroPxVideo->x, dBlendPos);
		calculateBlend(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y, dBlendPos);
		calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);
	}
	else if (maskUpload[y * dstAlignedWidth + (x * 2) + 0] != 0)
	{
		dBlendPos = maskUpload[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;

		double Lum = (macroPxVideo->y);
		double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
		if (CalculateLumKeyVal < 0)
			return;

		dBlendPos = dBlendPos * CalculateLumKeyVal;
		calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->w, &macroPxVideo->x, dBlendPos);
		calculateBlend(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y, dBlendPos);
		calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);


	}
	else if (maskUpload[y * dstAlignedWidth + (x * 2) + 1] != 0)
	{
		dBlendPos = maskUpload[y * dstAlignedWidth + (x * 2) + 1] / 255.0 * dBlendPos;
		double Lum = (macroPxVideo->w);
		double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
		if (CalculateLumKeyVal < 0)
			return;

		dBlendPos = dBlendPos * CalculateLumKeyVal;
		calculateBlend(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w, dBlendPos);
		calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->x, &macroPxVideo->x, dBlendPos);
		calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);
	}
}



__global__ void yuyvUnpackedComBineDataThreeLookups(uint4* src_Video_Unapc,uint4* src__Fill_Unapc,uint4* src__Key_Unapc, int width, int height, int srcAlignedWidth, int dstAlignedWidth, uchar *maskUpload0, uchar *maskUpload1, uchar *maskUpload2, int iBlendPos0, int iBlendPos1, int iBlendPos2, double4 Parabolic0, double4 Parabolic1, double4 Parabolic2, unsigned long int iCutOff, unsigned long int iCutOff0, unsigned long int iCutOff1, unsigned long int iCutOff2)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	uint4 *macroPxVideo;
	uint4 *macroPxFill;
	uint4 *macroPxKey;

	uint4 *macroPxKeyTop;
	uint4 *macroPxKeyBottom;
	uint4 *macroPxKeyLeft;
	uint4 *macroPxKeyRight;

	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	macroPxFill = &src__Fill_Unapc[y * srcAlignedWidth + x];
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];




	if (y > 0)
	{
		macroPxKeyTop = &src__Key_Unapc[(y - 1) * srcAlignedWidth + x];
		macroPxKeyLeft = &src__Key_Unapc[(y)* srcAlignedWidth + (x - 1)];
	}
	else
	{
		macroPxKeyTop = &src__Key_Unapc[(y)* srcAlignedWidth + x];
		macroPxKeyLeft = &src__Key_Unapc[(y)* srcAlignedWidth + x];

	}

	if (y < 1919)
	{
		macroPxKeyBottom = &src__Key_Unapc[(y + 1) * srcAlignedWidth + x];
		macroPxKeyRight = &src__Key_Unapc[(y  * srcAlignedWidth + x + 1)];
	}
	else
	{
		macroPxKeyBottom = &src__Key_Unapc[(y)* srcAlignedWidth + x];
		macroPxKeyRight = &src__Key_Unapc[(y)* srcAlignedWidth + x];
	}

	double4 Parabolic = Parabolic0;
	double dBlendPos = iBlendPos0 / 876.0;

	if (maskUpload0[y * dstAlignedWidth + (x * 2) + 0] != 0 && maskUpload0[y * dstAlignedWidth + (x * 2) + 1] != 0)
	if (maskUpload1[y * dstAlignedWidth + (x * 2) + 0] == 0 || maskUpload1[y * dstAlignedWidth + (x * 2) + 1] == 0)
	{
		if (Parabolic.w)
		{
			dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
			double Lum = (macroPxVideo->y + macroPxVideo->w) / 2.0;
			double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
			if (CalculateLumKeyVal < 0)
				return;

			dBlendPos = dBlendPos * CalculateLumKeyVal;
		}
		else
		{
			dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
		}

		calculateBlend(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w, dBlendPos);
		calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->y, &macroPxVideo->x, dBlendPos);
		calculateBlend(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y, dBlendPos);
		calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);
	}
	else if (maskUpload0[y * dstAlignedWidth + (x * 2) + 0] != 0)
	{
		if (maskUpload1[y * dstAlignedWidth + (x * 2) + 0] == 0)
		{
				if (Parabolic.w)
				{
					dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;

					double Lum = (macroPxVideo->y);
					double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
					if (CalculateLumKeyVal < 0)
						return;

					dBlendPos = dBlendPos * CalculateLumKeyVal;
				}
				else
				{
					dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
				}

				calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->w, &macroPxVideo->x, dBlendPos);
				calculateBlend(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y, dBlendPos);
				calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);

			}
			else if (maskUpload0[y * dstAlignedWidth + (x * 2) + 1] != 0)
			{
				if (maskUpload1[y * dstAlignedWidth + (x * 2) + 1] == 0)
				{
					if (Parabolic.w)
					{
						dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 1] / 255.0 * dBlendPos;
						double Lum = (macroPxVideo->w);
						double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
						if (CalculateLumKeyVal < 0)
							return;

						dBlendPos = dBlendPos * CalculateLumKeyVal;
					}
					else
					{
						dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 1] / 255.0 * dBlendPos;

					}
					calculateBlend(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w, dBlendPos);
					calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->x, &macroPxVideo->x, dBlendPos);
					calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);
				}
			}
		}
}


__global__ void keyAndFill(uint4* src_Video_Unapc,uint4* src__Fill_Unapc,uint4* src__Key_Unapc, int width, int height, int srcAlignedWidth, int dstAlignedWidth, uchar *maskUpload0, int iBlendPos0, double4 Parabolic0)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;

	uint4 *macroPxVideo;
	uint4 *macroPxFill;
	uint4 *macroPxKey;

	uint4 *macroPxKeyTop;
	uint4 *macroPxKeyBottom;
	uint4 *macroPxKeyLeft;
	uint4 *macroPxKeyRight;

	double4 Parabolic = Parabolic0;
	double dBlendPos = iBlendPos0 / 876.0;

	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
	macroPxFill = &src__Fill_Unapc[y * srcAlignedWidth + x];
	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];

//	printf("%d %d %d %d\n", macroPxKey->w, macroPxKey->x, macroPxKey->y, macroPxKey->z);

	if(macroPxKey->w <= 64) return;

	if (y > 0)
	{
		macroPxKeyTop = &src__Key_Unapc[(y - 1) * srcAlignedWidth + x];
		macroPxKeyLeft = &src__Key_Unapc[(y)* srcAlignedWidth + (x - 1)];
	}
	else
	{
		macroPxKeyTop = &src__Key_Unapc[(y)* srcAlignedWidth + x];
		macroPxKeyLeft = &src__Key_Unapc[(y)* srcAlignedWidth + x];

	}

	if (y < width-1)
	{
		macroPxKeyBottom = &src__Key_Unapc[(y + 1) * srcAlignedWidth + x];
		macroPxKeyRight = &src__Key_Unapc[(y  * srcAlignedWidth + x + 1)];
	}
	else
	{
		macroPxKeyBottom = &src__Key_Unapc[(y)* srcAlignedWidth + x];
		macroPxKeyRight = &src__Key_Unapc[(y)* srcAlignedWidth + x];
	}

	if (maskUpload0[y * dstAlignedWidth + (x * 2) + 0] != 0 && maskUpload0[y * dstAlignedWidth + (x * 2) + 1] != 0)
	{
		if (Parabolic.w)
		{
			dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
			double Lum = (macroPxVideo->y + macroPxVideo->w) / 2.0;
			double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
			if (CalculateLumKeyVal < 0)
				return;

			dBlendPos = dBlendPos * CalculateLumKeyVal;
		}
		else
		{
			dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
		}

		calculateBlend(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w, dBlendPos);
		calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->y, &macroPxVideo->x, dBlendPos);
		calculateBlend(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y, dBlendPos);
		calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);
	}
	else if (maskUpload0[y * dstAlignedWidth + (x * 2) + 0] != 0)
	{
		if (Parabolic.w)
		{
			dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;

			double Lum = (macroPxVideo->y);
			double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
			if (CalculateLumKeyVal < 0)
				return;

			dBlendPos = dBlendPos * CalculateLumKeyVal;
		}
		else
		{
			dBlendPos = maskUpload0[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
		}

		calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->w, &macroPxVideo->x, dBlendPos);
		calculateBlend(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y, dBlendPos);
		calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);
	}
}


//
//
//__global__ void yuyvUnpackedComBineDataThreeLookups(uint4* src_Video_Unapc,uint4* src__Fill_Unapc,uint4* src__Key_Unapc, int width, int height, int srcAlignedWidth, int dstAlignedWidth, uchar *maskUpload0, uchar *maskUpload1, uchar *maskUpload2, int iBlendPos0, int iBlendPos1, int iBlendPos2, double4 Parabolic0, double4 Parabolic1, double4 Parabolic2, unsigned long int iCutOff, unsigned long int iCutOff0, unsigned long int iCutOff1, unsigned long int iCutOff2)
//{
//	const int x = blockIdx.x * blockDim.x + threadIdx.x;
//	const int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (x >= srcAlignedWidth || y >= height)
//		return;
//
//
//	uint4 *macroPxVideo;
//	macroPxVideo = &src_Video_Unapc[y * srcAlignedWidth + x];
//	uint4 *macroPxFill;
//	macroPxFill = &src__Fill_Unapc[y * srcAlignedWidth + x];
//	uint4 *macroPxKey;
//	macroPxKey = &src__Key_Unapc[y * srcAlignedWidth + x];
//
//	uint4 *macroPxKeyTop;
//	uint4 *macroPxKeyBottom;
//
//	uint4 *macroPxKeyLeft;
//	uint4 *macroPxKeyRight;
//
//
//	if (y > 0)
//	{
//		macroPxKeyTop = &src__Key_Unapc[(y - 1) * srcAlignedWidth + x];
//		macroPxKeyLeft = &src__Key_Unapc[(y)* srcAlignedWidth + (x - 1)];
//	}
//	else
//	{
//		macroPxKeyTop = &src__Key_Unapc[(y)* srcAlignedWidth + x];
//		macroPxKeyLeft = &src__Key_Unapc[(y)* srcAlignedWidth + x];
//
//	}
//
//	if (y < 1919)
//	{
//		macroPxKeyBottom = &src__Key_Unapc[(y + 1) * srcAlignedWidth + x];
//		macroPxKeyRight = &src__Key_Unapc[(y  * srcAlignedWidth + x + 1)];
//	}
//	else
//	{
//		macroPxKeyBottom = &src__Key_Unapc[(y)* srcAlignedWidth + x];
//		macroPxKeyRight = &src__Key_Unapc[(y)* srcAlignedWidth + x];
//	}
//
//
//
//
//
//	if (macroPxKey->y < 65 && macroPxKey->w < 65)
//	{
//		return;
//	}
//
//
//
//
//	if (macroPxKey->y > iCutOff || macroPxKey->w > iCutOff || macroPxKeyTop->y > iCutOff || macroPxKeyTop->w > iCutOff || macroPxKeyBottom->y > iCutOff || macroPxKeyBottom->w > iCutOff
//		|| macroPxKeyLeft->y > iCutOff || macroPxKeyLeft->w > iCutOff || macroPxKeyRight->y > iCutOff || macroPxKeyRight->w > iCutOff)
//	{
//		calculateBlendFullKey(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w);
//		calculateBlendFullKey(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->y, &macroPxVideo->x);
//		calculateBlendFullKey(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y);
//		calculateBlendFullKey(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->w, &macroPxVideo->z);
//	}
//	else
//	{
//
//		uchar *mask = maskUpload0;
//		double4 Parabolic = Parabolic0;
//		double dBlendPos = iBlendPos0 / 876.0;
//		if (macroPxKey->y > iCutOff0 || macroPxKey->w > iCutOff0 || macroPxKeyTop->y > iCutOff0 || macroPxKeyTop->w > iCutOff0 || macroPxKeyBottom->y > iCutOff0 || macroPxKeyBottom->w > iCutOff0
//			|| macroPxKeyLeft->y > iCutOff0 || macroPxKeyLeft->w > iCutOff0 || macroPxKeyRight->y > iCutOff0 || macroPxKeyRight->w > iCutOff0)
//		{
//
//			mask = maskUpload0;
//			Parabolic = Parabolic0;
//			dBlendPos = iBlendPos0 / 476.0;
//
//
//
//		}else
//			if (macroPxKey->y > iCutOff1 || macroPxKey->w > iCutOff1 || macroPxKeyTop->y > iCutOff1 || macroPxKeyTop->w > iCutOff1 || macroPxKeyBottom->y > iCutOff1 || macroPxKeyBottom->w > iCutOff1
//				|| macroPxKeyLeft->y > iCutOff1 || macroPxKeyLeft->w > iCutOff1 || macroPxKeyRight->y > iCutOff1 || macroPxKeyRight->w > iCutOff1)
//			{
//				mask = maskUpload1;
//				Parabolic = Parabolic1;
//				dBlendPos = iBlendPos1 / 476.0;
//			}
//			else
//			{
//				mask = maskUpload2;
//				Parabolic = Parabolic2;
//				dBlendPos = iBlendPos2 / 476.0;
//			}
//
//		if (mask[y * dstAlignedWidth + (x * 2) + 0] != 0 && mask[y * dstAlignedWidth + (x * 2) + 1] != 0)
//		{
//			if (Parabolic.w)
//			{
//				dBlendPos = mask[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
//				double Lum = (macroPxVideo->y + macroPxVideo->w) / 2.0;
//				double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
//				if (CalculateLumKeyVal < 0)
//					return;
//
//				dBlendPos = dBlendPos * CalculateLumKeyVal;
//			}
//			else
//			{
//				dBlendPos = mask[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
//			//	printf("%f", dBlendPos);
//			}
//
//			calculateBlend(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w, dBlendPos);
//			calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->y, &macroPxVideo->x, dBlendPos);
//			calculateBlend(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y, dBlendPos);
//			calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);
//		}
//		else
//			if (mask[y * dstAlignedWidth + (x * 2) + 0] != 0)
//			{
//				if (Parabolic.w)
//				{
//
//					dBlendPos = mask[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
//
//					double Lum = (macroPxVideo->y);
//					double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
//					if (CalculateLumKeyVal < 0)
//						return;
//
//					dBlendPos = dBlendPos * CalculateLumKeyVal;
//				}
//				else
//				{
//					dBlendPos = mask[y * dstAlignedWidth + (x * 2) + 0] / 255.0  * dBlendPos;
//				}
//
//				calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->w, &macroPxVideo->x, dBlendPos);
//				calculateBlend(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y, dBlendPos);
//				calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);
//
////				calculateBlendFullKey(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w);
////					calculateBlendFullKey(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->y, &macroPxVideo->x);
////					calculateBlendFullKey(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y);
////					calculateBlendFullKey(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->w, &macroPxVideo->z);
//
//			}
//			else
//				if (mask[y * dstAlignedWidth + (x * 2) + 1] != 0)
//				{
//					if (Parabolic.w)
//					{
//						dBlendPos = mask[y * dstAlignedWidth + (x * 2) + 1] / 255.0 * dBlendPos;
//						double Lum = (macroPxVideo->w);
//						double CalculateLumKeyVal = Parabolic.x*(Lum*Lum) + Parabolic.y*Lum + Parabolic.z;
//						if (CalculateLumKeyVal < 0)
//							return;
//
//						dBlendPos = dBlendPos * CalculateLumKeyVal;
//					}
//					else
//					{
//						dBlendPos = mask[y * dstAlignedWidth + (x * 2) + 1] / 255.0 * dBlendPos;
//
//					}
//					calculateBlend(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w, dBlendPos);
//					calculateBlend(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->x, &macroPxVideo->x, dBlendPos);
//					calculateBlend(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->z, &macroPxVideo->z, dBlendPos);
//
//
////					calculateBlendFullKey(&macroPxVideo->w, &macroPxFill->w, &macroPxKey->w, &macroPxVideo->w);
////						calculateBlendFullKey(&macroPxVideo->x, &macroPxFill->x, &macroPxKey->y, &macroPxVideo->x);
////						calculateBlendFullKey(&macroPxVideo->y, &macroPxFill->y, &macroPxKey->y, &macroPxVideo->y);
////						calculateBlendFullKey(&macroPxVideo->z, &macroPxFill->z, &macroPxKey->w, &macroPxVideo->z);
//				}
//}
//
//
//
//}




__global__ void UpdateLookupTable(uchar* LookupTable, int iStartX, int iEndX, int iStartY, int iEndY,uint4 * LookUpColorDataOneDimention_Unpacked1, bool bTrain)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (iStartX >=x || x >= iEndX || y <= iStartY || y >= iEndY)
		return;

	if (bTrain)
		UpdateOneDimetionLookup(x, y , LookUpColorDataOneDimention_Unpacked1);
	else
	{
		ClearOneDimetionLookup(x, y, LookUpColorDataOneDimention_Unpacked1);
	}
	for (int c = 0; c < 1024; c++)
	{
		double3 val1a = make_double3(x, y, c);
		double bitpos1a = GetBitPos3(val1a);
		if(bTrain)
			SetBit3(bitpos1a, LookupTable,1);
		else
			ClearBit3(bitpos1a, LookupTable);
	}


}

void UpdateLookup(int iStartX, int iEndX, int iSartY, int iEndY, bool bTrain)
{
	cudaError_t cudaStatus;
	const dim3 block(32, 32);
	const dim3 grid(iDivUp(1024 , block.x), iDivUp(1024, block.y));
	UpdateLookupTable << <grid, block >> > (ptrLookUpData, iStartX, iEndX, iSartY, iEndY, LookUpColorDataOneDimention_Unpacked, bTrain);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}
}


__global__ void yuyvPackedToyuyvUnpacked_test(uint4* src_Video, uint4 *dst_video_all,int srcAlignedWidth, int dstAlignedWidth, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
	return;
	uint4 *macroPx;
	macroPx = &src_Video[y * srcAlignedWidth + x];
	double Cr0;
	double Y0;
	double Cb0;

	double Y2;
	double Cb2;
	double Y1;

	double Cb4;
	double Y3;
	double Cr2;

	double Y5;
	double Cr4;
	double Y4;
	Cr0 = (macroPx->x >> 20);
	Y0 = ((macroPx->x & 0xffc00) >> 10);
	Cb0 = (macroPx->x & 0x3ff);
	Y2 = (macroPx->y >> 20);

	//if (x == 10 && y == 10)
	//	printf("%dx%d %d %0.0f %0.0f %0.0f %0.0f %d \n\r",sizeof(uint4),x,y,Cr0,Y0,Cb0,Y2,macroPx->x);

	Cb2 = ((macroPx->y & 0xffc00) >> 10);
	Y1 = (macroPx->y & 0x3ff);
	Cb4 = (macroPx->z >> 20);
	Y3 = ((macroPx->z & 0xffc00) >> 10);

	Cr2 = (macroPx->z & 0x3ff);
	Y5 = (macroPx->w >> 20);
	Cr4 = ((macroPx->w & 0xffc00) >> 10);
	Y4 = (macroPx->w & 0x3ff);

	/*	double y1 = (double)macroPx.w;
	 double v = (double)macroPx.x;
	 double y0 = (double)macroPx.y;
	 double u = (double)macroPx.z;*/

	dst_video_all[y * dstAlignedWidth + (x * 3) + 0] = make_uint4(512,800,512,800);// x y z w
	dst_video_all[y * dstAlignedWidth + (x * 3) + 1] = make_uint4(512,800,512,800);
	dst_video_all[y * dstAlignedWidth + (x * 3) + 2] = make_uint4(512,800,512,800);
}



__global__ void yuyvPackedToyuyvUnpacked(uint4* src_Video, uint4 *dst_video_all,int srcAlignedWidth, int dstAlignedWidth, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;
	uint4 *macroPx;
//	printf("%d\n",y * srcAlignedWidth + x);
	macroPx = &src_Video[y * srcAlignedWidth + x];


	double Cr0;
	double Y0;
	double Cb0;
	double Y1;

	double Y2;
	double Cb2;
	double Cr2;
	double Y3;

	double Cb4;
	double Y5;
	double Cr4;
	double Y4;


	Cr0 = (macroPx->x >> 20);
	Y0 = ((macroPx->x & 0xffc00) >> 10);
	Cb0 = (macroPx->x & 0x3ff);
	Y2 = (macroPx->y >> 20);

	//if (x == 10 && y == 10)
	//	printf("%dx%d %d %0.0f %0.0f %0.0f %0.0f %d \n\r",sizeof(uint4),x,y,Cr0,Y0,Cb0,Y2,macroPx->x);

	Cb2 = ((macroPx->y & 0xffc00) >> 10);
	Y1 = (macroPx->y & 0x3ff);
	Cb4 = (macroPx->z >> 20);
	Y3 = ((macroPx->z & 0xffc00) >> 10);

	Cr2 = (macroPx->z & 0x3ff);
	Y5 = (macroPx->w >> 20);
	Cr4 = ((macroPx->w & 0xffc00) >> 10);
	Y4 = (macroPx->w & 0x3ff);

//	printf("%d %d %d %d\n",macroPx->w, macroPx->x, macroPx->y, macroPx->z)

	dst_video_all[y * dstAlignedWidth + (x * 3) + 0] = make_uint4(Cr0, Y0, Cb0,
			Y1);// x y z w
	dst_video_all[y * dstAlignedWidth + (x * 3) + 1] = make_uint4(Cr2, Y2, Cb2,
			Y3);
	dst_video_all[y * dstAlignedWidth + (x * 3) + 2] = make_uint4(Cr4, Y4, Cb4,
			Y5);
}


__global__ void yuyvPackedToyuyvUnpackedPlanner(uint4* src_Video, uint16_t *dst_video_y, uint16_t *dst_video_u, uint16_t *dst_video_v,int srcAlignedWidth, int dstAlignedWidth_y,int dstAlignedWidth_uv, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
	return;
	uint4 *macroPx;
	macroPx = &src_Video[y * srcAlignedWidth + x];
	double Cr0;
	double Y0;
	double Cb0;

	double Y2;
	double Cb2;
	double Y1;

	double Cb4;
	double Y3;
	double Cr2;

	double Y5;
	double Cr4;
	double Y4;


	Cr0 = (macroPx->x >> 20);
	Y0 = ((macroPx->x & 0xffc00) >> 10);
	Cb0 = (macroPx->x & 0x3ff);
	Y2 = (macroPx->y >> 20);

	//if (x == 10 && y == 10)
	//	printf("%dx%d %d %0.0f %0.0f %0.0f %0.0f %d \n\r",sizeof(uint4),x,y,Cr0,Y0,Cb0,Y2,macroPx->x);

	Cb2 = ((macroPx->y & 0xffc00) >> 10);
	Y1 = (macroPx->y & 0x3ff);
	Cb4 = (macroPx->z >> 20);
	Y3 = ((macroPx->z & 0xffc00) >> 10);

	Cr2 = (macroPx->z & 0x3ff);
	Y5 = (macroPx->w >> 20);
	Cr4 = ((macroPx->w & 0xffc00) >> 10);
	Y4 = (macroPx->w & 0x3ff);





	auto i=y * dstAlignedWidth_y + (x * 6);
	dst_video_y[i + 0] = Y0;
	dst_video_y[i + 1] = Y1;// x y z w
	dst_video_y[i + 2] = Y2;// x y z w
	dst_video_y[i + 3] = Y3;// x y z w
	dst_video_y[i + 4] = Y4;// x y z w
	dst_video_y[i + 5] = Y5;// x y z w


	auto n=y * dstAlignedWidth_uv + (x * 3);
	dst_video_u[n + 0] = Cr0;
	dst_video_u[n + 1] = Cr2;// x y z w
	dst_video_u[n + 2] = Cr4;// x y z w


	dst_video_v[n + 0] = Cb0;
	dst_video_v[n + 1] = Cb2;// x y z w
	dst_video_v[n + 2] = Cb4;// x y z w


}








__global__ void yuyvPackedToyUnpacked(uint4* src_Video, uchar *dst_video_all,int srcAlignedWidth, int dstAlignedWidth, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
	return;
	uint4 *macroPx;
	macroPx = &src_Video[y * srcAlignedWidth + x];

	double Y0;
	double Y2;
	double Y1;
	double Y3;
	double Y5;
	double Y4;

	Y0 = ((macroPx->x & 0xffc00) >> 10);
	Y2 = (macroPx->y >> 20);
	Y1 = (macroPx->y & 0x3ff);
	Y3 = ((macroPx->z & 0xffc00) >> 10);
	Y5 = (macroPx->w >> 20);
	Y4 = (macroPx->w & 0x3ff);


	auto i=y * dstAlignedWidth + (x * 6);
	dst_video_all[i + 0] = Y0/1024.0*255.0;
	dst_video_all[i + 1] = Y1/1024.0*255.0;// x y z w
	dst_video_all[i + 2] = Y2/1024.0*255.0;// x y z w
	dst_video_all[i + 3] = Y3/1024.0*255.0;// x y z w
	dst_video_all[i + 4] = Y4/1024.0*255.0;// x y z w
	dst_video_all[i + 5] = Y5/1024.0*255.0;// x y z w

}


//void PrepareYoloData(bool bTakeMask,float fnms)
//{
//
//	const int dstAlignedWidthUnpackedData = (1920 / 2);
//	dim3 blockRGB(16, 16);
//	dim3 gridRGB_Split(iDivUp(dstAlignedWidthUnpackedData, blockRGB.x),iDivUp(1080 / 2, blockRGB.y));
//	//dim3 gridRGB(iDivUp(dstAlignedWidthUnpackedData, blockRGB.x),iDivUp(1080, blockRGB.y));
//
//	yuyvUnPackedToPlanarRGB_Split<<<gridRGB_Split, blockRGB>>> ((uint4*)YUV_Unpacked_Video, (uint8_t *)m_RGBScaledFramePlanarDetectorptrs[0], (uint8_t *)m_RGBScaledFramePlanarDetectorptrs[1], (uint8_t *)m_RGBScaledFramePlanarDetectorptrs[2], (uint8_t *)m_RGBScaledFramePlanarDetectorptrs[3], 640 * sizeof(float), dstAlignedWidthUnpackedData*2, 1080/2, 640);
//	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
//
//	yuyvUnPackedToPlanarRGB_Split<<<gridRGB_Split, blockRGB>>>((uint4*) (YUV_Unpacked_Video+dstAlignedWidthUnpackedData),(uint8_t *) m_RGBScaledFramePlanarDetectorptrs[4], (uint8_t *)m_RGBScaledFramePlanarDetectorptrs[5], (uint8_t *)m_RGBScaledFramePlanarDetectorptrs[6], (uint8_t *)m_RGBScaledFramePlanarDetectorptrs[7], 640 * sizeof(float), dstAlignedWidthUnpackedData*2, 1080/2, 640);
//	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
//	Yolov5Detection=doInference_YoloV5(m_RGBScaledFramePlanarDetector,fnms,bTakeMask);
//	if(bTakeMask)
//		SnapYolov5Detection=Yolov5Detection;
//
//}



void Launch_yuyv10PackedToyuyvUnpacked(int RowLength, bool bSnapShot, int iFrameSizeUnpacked, cuda::GpuMat *RGB_Output_Cuda, int iBot,int iTop,bool bAutoTrain)
{
	cudaError_t cudaStatus;
	//****************************************************************************************************************************************************
	const dim3 block(16, 16);
	const dim3 grid(iDivUp(RowLength / SIZE_ULONG4_CUDA, block.x), iDivUp(m_iHeight, block.y));
	const int srcAlignedWidth = RowLength / SIZE_ULONG4_CUDA;
	const int dstAlignedWidthUnpackedData = (1920 / 2);


	yuyvPackedToyuyvUnpacked << <grid, block >> > ((uint4*)YUV_Upload_Video_YUV, (uint4*)YUV_Unpacked_Video, srcAlignedWidth, dstAlignedWidthUnpackedData, 1080);

/*	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

		return;
	}*/

#ifdef PREVIEW_OUTPUTRENDER
//
	Mat test;
//	test_gpu.download(test);
	//test_gpu.download(test);
	//imshow("y_only",test_gpu);
	//imshow("u_only",test_u);
	//imshow("v_only",test_v);

	waitKey(1);
#endif

	yuyvPackedToyuyvUnpacked << <grid, block >> > ((uint4*)YUV_Upload_Key,(uint4*) YUV_Unpacked_Key, srcAlignedWidth, dstAlignedWidthUnpackedData, 1080);

/*	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

		return;
	}*/
	yuyvPackedToyuyvUnpacked << <grid, block >> > ((uint4*)YUV_Upload_Fill, (uint4*)YUV_Unpacked_Fill, srcAlignedWidth, dstAlignedWidthUnpackedData, 1080);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!:\n[Error]: %s",cudaStatus, cudaGetErrorString(cudaStatus));

		return;
	}

#ifdef PREVIEW_OUTPUTRENDER
	bSnapShot=true;
#endif


	if (1)//bSnapShot)
	{
		cudaStatus = cudaMemcpy(YUV_Unpacked_Video_SnapShot,YUV_Unpacked_Video,  iFrameSizeUnpacked, cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}
		cudaStatus = cudaMemcpy(YUV_Unpacked_Key_SnapShot,YUV_Unpacked_Key, iFrameSizeUnpacked, cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		const dim3 blockRGB(16, 16);
		const dim3 gridRGB(iDivUp(dstAlignedWidthUnpackedData, block.x), iDivUp(1080, block.y));
		const int dstAlignedWidthRGB = 1920;
		yuyvUmPackedToRGB_lookup <<<gridRGB, blockRGB>>> ((uint4 *)YUV_Unpacked_Video, DownloadRGBData, dstAlignedWidthUnpackedData, dstAlignedWidthRGB, 1080, (uint4 *)YUV_Unpacked_Key, ptrLookUpData);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return;
		}
	}
	RGB_Output_Cuda->data = (uchar*)DownloadRGBData;



#ifdef PREVIEW_OUTPUTRENDER
	imshow("UI Mouse training Window",	*RGB_Output_Cuda);

#endif
}


void Launch_yuyvDilateAndErode( int iDilate, int iErode, int iUse)//https://docs.nvidia.com/cuda/npp/group__image__morphological__operations.html
{

	cudaError_t cudaStatus;
	cudaStatus = cudaMemset(MaskRefineScratch, 0, (m_iWidth * (m_iHeight)) * sizeof(uchar));//4228250625

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}
	GpuMat test_gpu_smooth;
	GpuMat test_gpu1(1080/2,1920*2,CV_8UC1, ChromaGeneratedMask[iUse],Mat::CONTINUOUS_FLAG);
	test_gpu1.step=1920*2;



	int erode_dilate_pos=iErode;
	int max_iters=1;
	int n = erode_dilate_pos - max_iters;
	int an = iErode;

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(an*2+1, an*2+1), Point(an, an));
	Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter(MORPH_ERODE, test_gpu1.type(), element);
	erodeFilter->apply(test_gpu1, test_gpu_smooth);


	 erode_dilate_pos=-iErode;

	 n = erode_dilate_pos - max_iters;
	 an = iDilate;
	 element = getStructuringElement(MORPH_ELLIPSE, Size(an*2+1, an*2+1), Point(an, an));

	Ptr<cv::cuda::Filter> erodeFilter2 = cv::cuda::createMorphologyFilter(MORPH_DILATE, test_gpu1.type(), element);
	erodeFilter2->apply(test_gpu_smooth, test_gpu1);


	//	Ptr<cv::cuda::Filter> gaussianfilter = cv::cuda::createGaussianFilter( test_gpu1.type(), test_gpu1.type(),Size(3,3),6,3,BORDER_DEFAULT,-1);
	//	gaussianfilter->apply(test_gpu1,test_gpu_smooth);
	//	test_gpu_smooth.copyTo(test_gpu1);




#ifdef PREVIEW_OUTPUTRENDER

	Mat test,dest;
//	test_gpu_smooth.download(test);
//	  GaussianBlur( test, dest, Size( 3,3 ), 0, 0 );

		imshow("Yolomask",test_gpu1);

	imshow("mask Erode dilate GaussianFilter",	test_gpu_smooth);
	int i=0;
	//int i=waitKey(0);
	if(i==114)
	{
		cout << i <<"r - CudaLookupReset"<< endl;
		cudaLookReset();
	}
//	imwrite("/home/jurie/res4.bmp",test);
#endif
}


void Launch_Frame_Info(cuda::GpuMat *RGB_FrameInfo)
{

	cudaError_t cudaStatus;

	const int dstAlignedWidthUnpackedData = (1920 / 2);
	const int dstAlignedWidthUnpackedData1 = (1920 / 2);
	const dim3 blockRUN(16, 16);
	const dim3 gridRun(iDivUp(dstAlignedWidthUnpackedData1, blockRUN.x), iDivUp(1080, blockRUN.y));
	const dim3 gridFull(iDivUp(1920, blockRUN.x), iDivUp(1080, blockRUN.y));




		//******************************************Frame color data    ***********************************************************************


		cudaStatus = cudaMemset(FrameColorData_Unpacked, 0, (1024 / 2 * sizeof(uint4) * 1024));//4228250625
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
			return;
		}

		yuyvUnpackedCreateFrameInfo << <gridRun, blockRUN >> > (YUV_Unpacked_Video_SnapShot, YUV_Unpacked_Key_SnapShot, FrameColorData_Unpacked, 1920, 1080, dstAlignedWidthUnpackedData1, LookUpColorDataOneDimention_Unpacked);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return;
		}


		const dim3 blockRGB1(16, 16);
		const dim3 gridRGB1(iDivUp(dstAlignedWidthUnpackedData, blockRGB1.x), iDivUp(1080, blockRGB1.y));

		yuyvUnPackedToRGB_Plain << <gridRGB1, blockRGB1 >> > (FrameColorData_Unpacked, DownloadRGBData_Frame_Info, 512, 1024, 1024, LookUpColorDataOneDimention_Unpacked);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return;
		}


		RGB_FrameInfo->data = (uchar*)DownloadRGBData_Frame_Info;
}


int m_iFrameSize = -1;
long m_lFrameDiv4 = -1;

int cudaLoadLookUp()
{

	cudaError_t cudaStatus;
	void *TempMem = malloc(CUDA_LOOKUP_SIZE);


	std::ifstream infile("c:\\temp\\tempLookup.cud", std::ifstream::binary);

	infile.read((char*)TempMem, CUDA_LOOKUP_SIZE);
	infile.close();



	cudaStatus = cudaMemcpy(ptrLookUpData, TempMem, CUDA_LOOKUP_SIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;

	}
	free(TempMem);
	return cudaStatus;
}




int cudaDumpLookUp()
{
	cudaError_t cudaStatus;
	void *TempMem = malloc(CUDA_LOOKUP_SIZE);


	cudaStatus = cudaMemcpy(TempMem, ptrLookUpData, CUDA_LOOKUP_SIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;

	}
	std::ofstream outfile("c:\\temp\\tempLookup.cud", std::ofstream::binary);
	outfile.write((char*)TempMem, CUDA_LOOKUP_SIZE);
	outfile.close();
	free(TempMem);

	return cudaStatus;
}



__global__ void yuyvUnPackedToPlanarRGB_Split(uint4* src_Unapc, uint8_t *dpRgbA,uint8_t *dpRgbB, uint8_t *dpRgbC, uint8_t *dpRgbD,	uint32_t dstPlanePitchDst/*640 *sizeof(float)*/, int srcAlignedWidth,		int height, int dstHeight)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//memcpy(dpRgbA, src_Unapc, 1);

	if (x >= srcAlignedWidth || y >= height)
	return;

	uint4 *macroPx;
	macroPx = &src_Unapc[y * srcAlignedWidth + x];

	float3 px_0 = make_float3(
			clamp(macroPx->y + 1.540f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(
					macroPx->y - 0.459f * (macroPx->x - 512.0)
					- 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(macroPx->y + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

	float3 px_1 = make_float3(
			clamp(macroPx->w + 1.540f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(
					//memcpy(dpRgbA, src_Unapc, 1);

					macroPx->w - 0.459f * (macroPx->x - 512.0)
					- 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
			clamp(macroPx->w + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

	uint8_t *pDst;// = dpRgbA + x * sizeof(float2) + (y / 2) * dstPlanePitch;

	int iPos1 = 428;
	int iPos2 = 640;
	int iPos3 = 854;
	int iPos4 = 1068;
	int iPos5 = 1278;
	int iPos6 = 1494;
	int yOffset = y;

	//if ((y % 2) == 0)
	{
		if (x < (iPos1 / 2))
		{
			pDst = dpRgbA + x * sizeof(float2) + (yOffset) * dstPlanePitchDst;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};
		}
		else if (x < (iPos2 / 2))
		{
			pDst = dpRgbA + x * sizeof(float2) + (yOffset) * dstPlanePitchDst;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};

			pDst = dpRgbB + (x - (iPos1 / 2)) * sizeof(float2)
			+ (yOffset) * dstPlanePitchDst;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};

		}
		else if (x < (iPos3 / 2))
		{
			pDst = dpRgbB + (x - (iPos1 / 2)) * sizeof(float2)
			+ (yOffset) * dstPlanePitchDst;

			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};

		}
		else if (x < (iPos4 / 2))
		{
			pDst = dpRgbB + (x - (iPos1 / 2)) * sizeof(float2)
			+ (yOffset) * dstPlanePitchDst;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};

			pDst = dpRgbC + (x - (iPos3 / 2)) * sizeof(float2)
			+ (yOffset) * dstPlanePitchDst;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};

		}
		else if (x < iPos5 / 2)
		{
			pDst = dpRgbC + (x - iPos3 / 2) * sizeof(float2)
			+ (yOffset) * dstPlanePitchDst;

			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};
		}
		else if (x < iPos6 / 2)
		{
			pDst = dpRgbC + (x - iPos3 / 2) * sizeof(float2)
			+ (yOffset) * dstPlanePitchDst;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};

			pDst = dpRgbD + (x - iPos5 / 2) * sizeof(float2)
			+ (yOffset) * dstPlanePitchDst;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};

		}
		else
		{

			pDst = dpRgbD + (x - iPos5 / 2) * sizeof(float2)
			+ (yOffset) * dstPlanePitchDst;

			*(float2 *) pDst = float2
			{	__fdividef(px_0.x, 1024.0), __fdividef(px_1.z, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.y, 1024.0), __fdividef(px_1.y, 1024.0)};
			pDst += dstPlanePitchDst * dstHeight;
			*(float2 *) pDst = float2
			{	__fdividef(px_0.z, 1024.0), __fdividef(px_1.x, 1024.0)};
		}
	}
	/*else
	 {
	 return;
	 }*/
}
