/*
 * ui.cpp
 *
 *  Created on: 02 Jun 2023
 *      Author: jurie
 */


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <interfaces.hpp>
#include <ui.hpp>
#include <exception>

void mouseCallback(int event, int x, int y, int flags, void* data)
{

	MouseData* md = (MouseData*)data;
	static int iRecsize = 20;
	static int rectLim = 20;
	Rect tt = getWindowImageRect(md->windowName);
	double x1 = x; //double(x)/(double)(tt.width)  * 1920.0;//window correction
	double y1 = y; //double(y)/(double)(tt.height) * 1080.0;//window correction

	switch (event)
	{
		case EVENT_MOUSEWHEEL ://!< positive and negative values mean forward and backward scrolling, respectively.
			if (flags > 0)
			{
				iRecsize += iRecsize>1?-1:0;
			}
			else
			{
				iRecsize += iRecsize<rectLim-1? 1 : -iRecsize+rectLim;
			}

			md->iXUpDynamic = x1 - iRecsize;
			md->iYUpDynamic = y1 - iRecsize;
			md->iXDownDynamic = x1 + iRecsize;
			md->iYDownDynamic = y1 + iRecsize;
			break;

		case EVENT_LBUTTONDOWN:
			md->iXDown = x1;
			md->iYDown = y1;
			md->iXUpDynamic = x1 - iRecsize;
			md->iYUpDynamic = y1 - iRecsize;
			md->iXDownDynamic = x1 + iRecsize;
			md->iYDownDynamic = y1 + iRecsize;
			md->bHandleLDown = true;
			break;

		case EVENT_RBUTTONDOWN:
			md->iXDown = x1;
			md->iYDown = y1;
			md->bHandleRDown = true;
			break;

		case EVENT_LBUTTONUP:
			md->bHandleLDown = false;
			md->iXUp = x1;
			md->iYUp = y1;
			break;

		case EVENT_RBUTTONUP:
			md->bHandleRDown = false;
			md->iXUp = x1;
			md->iYUp = y1;
			break;

		case EVENT_MOUSEMOVE:
			md->iXUpDynamic = x1 - iRecsize;
			md->iYUpDynamic = y1 - iRecsize;
			md->iXDownDynamic = x1 + iRecsize;
			md->iYDownDynamic = y1 + iRecsize;
			md->x=x;
			md->y=y;
			break;
	}
}

void updateTrackbar(int pos, void* settingsWindowOb)
{
	SettingsWindow* sw = (SettingsWindow*)settingsWindowOb;
	sw->update();
}

void WindowI::setMouseCB(void* md, void (*cb_)(int, int, int, int, void*))
{
	cv::setMouseCallback(this->windowName, cb_, md);
}

int WindowI::parseKey()
{
	if(!this->keyEnabled) return -1;
	switch(this->key)
	{
	case 'q':
		return WINDOW_EVENT_CAPTURE;
		break;
	case 'Q':
		return WINDOW_EVENT_CAPTURE;
		break;
	case 's':
		return WINDOW_EVENT_SAVE_IMAGE;
		break;
	default:
		return -1;
	}
}


void KeyingWindow::process()
{
	if(this->rgbData==nullptr)return;
	this->gMat.data = (uchar*)rgbData;
	gMat.download(this->previewMat);
	cv::resizeWindow(this->windowName, this->iWidth, this->iHeight);

//	if(!this->captureKey)return;
//	if(!this->mEnabled) return;
}

void KeyingWindow::show()
{
	this->process();
	if(this->rgbData==nullptr)return;
	cv::imshow(this->windowName, this->previewMat);
}

void KeyingWindow::update()
{
	int rectangleSize = abs(this->mouseData.iXDownDynamic - this->mouseData.x); // the width/2 and height/2 of the rectangle (square)
	cv::Mat prevClone = this->previewMat.clone();
	if(this->rgbData==nullptr)return;
	cv::rectangle(prevClone, Point(this->mouseData.x-rectangleSize, mouseData.y-rectangleSize),
					Point(mouseData.x+rectangleSize, mouseData.y+rectangleSize), Scalar(255, 255, 255), 1, 8, 0);
	cv::circle(prevClone, cv::Point(mouseData.x, mouseData.y), sqrt((pow(rectangleSize, 2)+pow(rectangleSize, 2)))+4, cv::Scalar(255,255,255),
			3, 8, 0);

	cv::Rect roi(this->mouseData.iXUpDynamic, this->mouseData.iYUpDynamic, rectangleSize*2, rectangleSize*2);

	if((0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= prevClone.cols &&
		0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= prevClone.rows))
	{
		Mat roiMat = prevClone(roi);
		Mat roiLargeMat;
		Size ssize = roiMat.size();
		if (!ssize.empty())
		{
			cv::resize(roiMat, roiLargeMat, Size((rectangleSize) * 25, (rectangleSize) * 25), 0, 0, INTER_NEAREST);

			roiLargeMat.copyTo(prevClone.rowRange(0, roiLargeMat.rows).colRange(0, roiLargeMat.cols));
		}
	}
	cv::imshow(this->windowName, prevClone);
	prevClone.release();
}



void SettingsWindow::init()
{
	cv::setTrackbarPos(this->trackbars[0], this->windowName, this->windowSettings.m_BlendPos);
	cv::setTrackbarPos(this->trackbars[1], this->windowName, 1);
	cv::setTrackbarPos(this->trackbars[2], this->windowName, this->windowSettings.m_iUV_Diam);
	cv::setTrackbarPos(this->trackbars[3], this->windowName, this->windowSettings.m_iOuter_Diam);
	cv::setTrackbarPos(this->trackbars[4], this->windowName, this->windowSettings.m_iLum_Diam);
	cv::setTrackbarPos(this->trackbars[5], this->windowName, this->windowSettings.m_iErase_Diam);
	cv::setTrackbarPos(this->trackbars[6], this->windowName, this->windowSettings.m_iErase_Lum_Diam);
	cv::setTrackbarPos(this->trackbars[7], this->windowName, this->windowSettings.m_iErode);
	cv::setTrackbarPos(this->trackbars[8], this->windowName, this->windowSettings.m_iDilate);
	cv::setTrackbarPos(this->trackbars[9], this->windowName, this->windowSettings.m_iLowerlimit);
	cv::setTrackbarPos(this->trackbars[10], this->windowName, this->windowSettings.m_iUpperlimit);
	cv::setTrackbarPos(this->trackbars[11], this->windowName, 100);
}



void WindowsContainer::addWindow(WindowI *w)
{
	try{
		this->windows[w->getHandle()] = w;
	}catch(std::exception& w){
		std::cerr<<"Error adding window to container"<<std::endl;
	}

}

int WindowsContainer::dispatchKey()
{
	this->pressedKey = waitKey(10);
	if(windows.empty())return -1;
	if(this->pressedKey == -1)return -1;
	if(this->pressedKey == WINDOW_EVENT_EXIT)
	{
		this->eventQueue.push(WINDOW_EVENT_EXIT);
		return WINDOW_EVENT_EXIT;
	}

	int event;

	for(auto& window: this->windows)
	{
		if(!window.second->isKeysEnabled())continue;
		window.second->setKey(pressedKey);
		event = window.second->parseKey();
		if(event != -1)
		{
			this->eventQueue.push(event);
		}
	}

	return this->pressedKey;
}

void WindowsContainer::removeWindow(std::string windowHandle)
{
	if(windows.empty())return;

	int k = windows.erase(windowHandle);
}

WindowI* WindowsContainer::getWindow(std::string windowHandle)
{
	if(windows.empty())
		return nullptr;
	try{
		return windows[windowHandle];
	}catch(std::exception& err){
		return nullptr;
	}
}

void WindowsContainer::dispatchEvent()
{
	if(!this->eventQueue.empty())
	{
		this->currentEvent = this->eventQueue.front();
		this->eventQueue.pop();
	}
	else
		this->currentEvent = -1;
}

int WindowsContainer::getEvent()
{
	return this->currentEvent;
}

void WindowsContainer::updateWindows()
{
	for(auto &window : this->windows)
	{
		window.second->update();
	}
}

WindowsContainer::~WindowsContainer()
{
	cv::destroyAllWindows();
}
