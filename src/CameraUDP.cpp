#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <errno.h>
#include <iostream>
#include <arpa/inet.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <opencv2/opencv.hpp>
using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;
using namespace cv::cuda;
/* Server.c */

//int ZoomMax=0;
//int ZoomMin=650000;
//int FocusMax=0;
//int FocusMin=650000;

int ZoomMax=50171;
int ZoomMin=20;
int FocusMax=53893;
int FocusMin=1;


struct CameraData_Raw {
	unsigned char MessageType;
	unsigned char CameraID;
	unsigned char rotx[3];
	unsigned char roty[3];
	unsigned char rotz[3];
	unsigned char posx[3];
	unsigned char posy[3];
	unsigned char posz[3];
	unsigned char zoom[3];
	unsigned char focus[3];
	unsigned char user[2];
	unsigned char Checksum;
};


struct CameraData_Processed {
	CameraData_Processed ()
	{


		 rotx_zero=0;
		 roty_zero=0;
		 rotz_zero=0;



	}



int rotx;
int roty ;
int rotz ;

int posx ;
int posy ;
int posz ;


int zoom;
int focus;

double degx ;
double degy;
double degz;
double zoompersentage;
double focuspersentage;

int rotx_zero;
int roty_zero;
int rotz_zero;


};


int ConvertToInt32(unsigned char rotx[3])
{

	int returnval = 0;
	if ((rotx[0] & 128) == 0)
	{
		returnval = returnval | (rotx[0] << 16);
		returnval = returnval | (rotx[1] << 8);
		returnval = returnval | rotx[2];
	}
	else
	{
		returnval = returnval | (rotx[0] << 16);
		returnval = returnval | (rotx[1] << 8);
		returnval = returnval | rotx[2];
		returnval = returnval ^ 0xff000000;
	}
	return returnval;
}
CameraData_Raw raw_data;
CameraData_Processed processed_data;

std::list<CameraData_Raw> imagelist;



void *threadmain(void *lpParam);

void initCameraUDPData()

{



	pthread_t threads;
			int rc;
			rc = pthread_create(&threads, NULL, threadmain, 0);
			if (rc)
						{
							std::cout << "Error:unable to create thread," << rc << std::endl;
							exit(-1);
						}



}


void CameraZero()
{
	processed_data.rotx_zero=processed_data.rotx ;
	processed_data.roty_zero=processed_data.roty ;
	processed_data.rotz_zero=processed_data.rotz ;
	std::cout << "zero "<< std::endl;
}



void DrawCameraData(Mat *DrawingMat)
{
	cv::Rect r(1920/2,1080/2,10,10);
	cv::rectangle(*DrawingMat,r,Scalar(255,255,255),2);


	char buff[100];
	snprintf(buff, sizeof(buff), "%0.3f %0.3f %0.3f %0.1f %0.1f", processed_data.degx, processed_data.degy, processed_data.degz,processed_data.zoompersentage,processed_data.focuspersentage);
	std::string text = buff;

	Mat img=*DrawingMat;

	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale =1.0;
	int thickness = 1;

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


void *threadmain(void *lpParam)
//int threadmain( int argc, char *argv[])
{
   std::cout << "Starting..." << std::endl;
   int server_socket;
   struct sockaddr_in server_address, client_address;
    char buf[512];
   unsigned int clientLength;
   int checkCall, message;

   /*Create socket */
   server_socket=socket(AF_INET, SOCK_DGRAM,IPPROTO_UDP);
   if(server_socket == -1)
        perror("Error: socket failed");

   uint yes=1;
   setsockopt(server_socket,SOL_SOCKET,SO_REUSEADDR,(char*)&yes,sizeof(yes));

   bzero((char*) &server_address, sizeof(server_address));

   /*Fill in server's sockaddr_in*/
   server_address.sin_family=AF_INET;
   server_address.sin_addr.s_addr=htonl(INADDR_ANY);
   server_address.sin_port=htons(6302);

   /*Bind server socket and listen for incoming clients*/
   checkCall = bind(server_socket, (struct sockaddr *) &server_address, sizeof(struct sockaddr));
   if(checkCall == -1)
        perror("Error: bind call failed");

    int iCount=0;

   while(1)
   {

//	printf("SERVER: waiting for data from client\n");

	clientLength = sizeof(client_address);
	std::this_thread::sleep_for(std::chrono::milliseconds(1));
	message = recvfrom(server_socket, buf, 512, 0,
		  (struct sockaddr*) &client_address, &clientLength);
	if(message == -1)
        perror("Error: recvfrom call failed");



	if(message==29)
	{
		memcpy(&raw_data,buf,29);
		//imagelist.push_front(raw_data);
		processed_data.rotx = ConvertToInt32(raw_data.rotx);
		processed_data.roty = ConvertToInt32(raw_data.roty);
		processed_data.rotz = ConvertToInt32(raw_data.rotz);

		processed_data.posx = ConvertToInt32(raw_data.posx)/640.0;
		processed_data.posy = ConvertToInt32(raw_data.posy)/640.0;
		processed_data.posz = ConvertToInt32(raw_data.posz)/640.0;


		processed_data.zoom = ConvertToInt32(raw_data.zoom)- 524288;
		processed_data.focus = ConvertToInt32(raw_data.focus)- 524288;

		processed_data.degx = ((double(processed_data.rotx-processed_data.rotx_zero) / (double) 32768.0));
		processed_data.degy = ((double(processed_data.roty-processed_data.roty_zero) / (double) 32768.0));
		processed_data.degz = ((double(processed_data.rotz-processed_data.rotz_zero) / (double) 32768.0));



			if(ZoomMax<processed_data.zoom)
				ZoomMax=processed_data.zoom;

			if(ZoomMin>processed_data.zoom)
				ZoomMin=processed_data.zoom;

			processed_data.zoompersentage=(double(processed_data.zoom-ZoomMin)/double(ZoomMax-ZoomMin))*100.0;



			if(FocusMax<processed_data.focus)
						FocusMax=processed_data.focus;

					if(FocusMin>processed_data.focus)
						FocusMin=processed_data.focus;

			processed_data.focuspersentage=(double(processed_data.focus-FocusMin)/double(FocusMax-FocusMin))*100.0;


	//	std::cout << degx << " " << degy << " " << degz <<" "<< posx <<" " << posy <<" " << posz <<" " << zoom << " " << focus <<std::endl;
}









//	for(int x=0;x<message;x++)
//	{
//		if(((unsigned char)buf[x])==209)
//		{
//
//			printf("\n%d %d ",message,(unsigned char)buf[x]);
//			iCount=0;
//		}else
//		{
//			iCount++;
//			printf("%d ",(unsigned char)buf[x]);
//		}
//
//	}

	//printf("SERVER: read %d unsigned chars from IP %s(%s)\n", message,
	//	  inet_ntoa(client_address.sin_addr), buf);

//	if(!strcmp(buf,"quit"))
 //          break;

	//strcpy(buf,"ok");

//	message = sendto(server_socket, buf, strlen(buf)+1, 0,
//		  (struct sockaddr*) &client_address, sizeof(client_address));
	//if(message == -1)
 //       perror("Error: sendto call failed");

//	printf("SERVER: send completed\n");
   }

   checkCall = shutdown(server_socket,SHUT_RDWR);
   if(checkCall == -1)
        perror("Error: bind call failed");

}
