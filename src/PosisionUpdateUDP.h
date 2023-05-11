#ifndef POSISIONUPDATE_UDP_H_
#define POSISIONUPDATE_UDP_H_

#include <opencv2/opencv.hpp>

void initPosUDPData();
void SendSocketResetFrame();
void SendSocketUpdatePos(int ID,unsigned char r,unsigned char g,unsigned char b,int x,int y);

#endif
