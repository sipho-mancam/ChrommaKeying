/*
 * InputLoopThrough.h
 *
 *  Created on: 06 Apr 2022
 *      Author: jurie
 */

#ifndef INPUTLOOPTHROUGH_H_
#define INPUTLOOPTHROUGH_H_
#include <mutex>
#include "pevents.h"
#include <list>
void EndLoop();
using namespace neosmart;

class FrameHandeler
{
public:
	FrameHandeler()
	{
		events=CreateEvent();
	}
	std::list<void*> imagelist;
	void AddFrame(void* ptr,long Size);
	void AddFrame(void* ptr);
	void *GetFrame(bool bPop);
	void  ClearAll(int iPopTo=0);
	unsigned int GetFrameCount();
	std::mutex mtxVideo;           // mutex for critical section
	neosmart_event_t events;

};

class VideoIn
{
	public:
	VideoIn();
	~VideoIn();
	void WaitForFrames(int iDelayFrames);
	FrameHandeler imagelistVideo;
	FrameHandeler imagelistFill;
	FrameHandeler imagelistKey;
	FrameHandeler ImagelistOutput;
	long m_iRGBSize ;
	long m_iRGBSizeOF ;
	long m_iABGRSize ;
	long m_iFrameSizeUnpacked ;
	long m_iWidth ;
	long m_iHeight ;
	long m_RowLength ;
	long m_sizeOfFrame ;
	bool m_bExitApp;
};



#endif /* INPUTLOOPTHROUGH_H_ */
