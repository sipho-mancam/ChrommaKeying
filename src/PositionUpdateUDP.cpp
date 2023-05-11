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


void *threadPosisionUpdate(void *lpParam);

void initPosUDPData()

{


	pthread_t threads;
			int rc;
			rc = pthread_create(&threads, NULL, threadPosisionUpdate, 0);
			if (rc)
						{
							std::cout << "Error:unable to create thread," << rc << std::endl;
							exit(-1);
						}



}
struct sockaddr_in server_address, client_address;
int server_socket;
void SendSocket(char *buffer);
void SendSocketUpdatePos(int ID,unsigned char r,unsigned char g,unsigned char b,int x,int y)
{
	char buffer[256];
	sprintf(buffer,"00,%d,%d,%d,%d,%d,%d",ID,r,g,b,x,y);
	//std::cout << buffer << std::endl;
	SendSocket(buffer);

}
void SendSocketResetFrame()
{
	char buffer[256];
	sprintf(buffer,"01,00");
	SendSocket(buffer);

}
int fd=-1;
struct sockaddr_in addr;
void SendSocket(char *buffer)
{




char *message = buffer;
   if(fd==-1)
   {

    if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        perror("socket");
        exit(1);
    }
    /* set up destination address */
    memset(&addr,0,sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port=htons(60000);
   }




	if (sendto(fd, message, strlen(message), 0,(struct sockaddr *) &addr, sizeof(addr)) < 0)
	{
		perror("sendto");
		exit(1);
	}

}



void *threadPosisionUpdate(void *lpParam)
//int threadmain( int argc, char *argv[])
{


	return 0;
   std::cout << "Starting..." << std::endl;


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
   server_address.sin_addr.s_addr=inet_addr("127.0.0.1");
   server_address.sin_port=htons(60000);

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
		//	std::cout << degx << " " <0000000000< degy << " " << degz <<" "<< posx <<" " << posy <<" " << posz <<" " << zoom << " " << focus <<std::endl;
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
