/*
 ============================================================================
 Name        : vizSocketConnection.cu
 Author      : Jurie Vosloo
 Version     :
 Copyright   : dont know
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


bool bInit=false;


void error(const char *msg)
{
    perror(msg);
    exit(0);
}



int sockfd,  n;
struct sockaddr_in serv_addr;
struct hostent *server;

char buffer[256];
int portno = 6100;



void InitVizSocket()
{
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0)
		error("ERROR opening socket");
	server = gethostbyname("10.0.0.12");
	if (server == NULL) {
		fprintf(stderr,"ERROR, no such host\n");
		exit(0);
	}
	bzero((char *) &serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	bcopy((char *)server->h_addr,
		 (char *)&serv_addr.sin_addr.s_addr,
		 server->h_length);
	serv_addr.sin_port = htons(portno);
	if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
		error("ERROR connecting");
bInit=true;
}


void SendVizSocket(int x,int y)
{
	if(!bInit)
		return ;
	//sprintf(buffer,"send -1 RENDERER*TREE*$object*TRANSFORMATION*SCREEN_POSITION SET %d %d \r\n",x,y);
	sprintf(buffer,"send -1 MAIN_SCENE*TREE*$object*TRANSFORMATION*POSITION SET %d %d \r\n",x,y);

	n = write(sockfd,buffer,strlen(buffer)+1);
	if (n < 0)
		 error("ERROR writing to socket");
	bzero(buffer,256);
//	n = read(sockfd,buffer,255);
//	if (n < 0)
//		 error("ERROR reading from socket");
//	printf("%s\n",buffer);

}

int TestVizSocket(void)
{
	//InitVizSocket();

	    for(int x=960;x<1200;x++)
	    	for(int y=400;y<500;y++)
	    	{
	    		SendVizSocket(x,y);
	  //  printf("Please enter the message: ");

	    	}
	    close(sockfd);
	    return 0;

	return 0;
}

