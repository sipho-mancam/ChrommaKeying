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
	static int iRecsize = 10;
	Rect tt = getWindowImageRect(md->windowName);
	double x1 = double(x)/(double)(tt.width)  * 1920.0;//window correction
	double y1 = double(y)/(double)(tt.height) * 1080.0;//window correction

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
				iRecsize += iRecsize<19? 1 : -iRecsize+20;
			}

			md->iXUpDynamic = x1 - iRecsize;
			md->iYUpDynamic = y1 - iRecsize + 4;
			md->iXDownDynamic = x1 + iRecsize;
			md->iYDownDynamic = y1 + iRecsize + 4;
			break;

		case EVENT_LBUTTONDOWN:
			md->iXDown = x1;
			md->iYDown = y1;
			md->iXUpDynamic = x1 - iRecsize;
			md->iYUpDynamic = y1 - iRecsize + 4;
			md->iXDownDynamic = x1 + iRecsize;
			md->iYDownDynamic = y1 + iRecsize + 4;
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
			md->iXUpDynamic = x1-iRecsize;
			md->iYUpDynamic = y1-iRecsize+4;
			md->iXDownDynamic = x1+ iRecsize;
			md->iYDownDynamic = y1+ iRecsize+4;
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


void KeyingWindow::process()
{
	if(this->rgbData==nullptr)return;
	this->gMat.data = (uchar*)rgbData;
	gMat.download(this->previewMat);

	if(!this->captureKey)return;
	if(!this->mEnabled) return;


	// Draw a rectangle around mouse
	rectangle(this->previewMat, Point(this->mouseData.iXUpDynamic, mouseData.iYUpDynamic),
				Point(mouseData.iXDownDynamic, mouseData.iYDownDynamic), Scalar(255, 255, 255), 1, 8, 0);

}

void KeyingWindow::show()
{
	this->process();
	if(this->rgbData==nullptr)return;

	cv::imshow(this->windowName, this->previewMat);
}

void KeyingWindow::update()
{
//	std::cout<<"I execute"<<std::endl;
	int rectangleSize = 25;
	cv::Mat prevClone = this->previewMat.clone();
	if(this->rgbData==nullptr)return;
	cv::rectangle(prevClone, Point(this->mouseData.x-rectangleSize, mouseData.y-rectangleSize),
					Point(mouseData.x+rectangleSize, mouseData.y+rectangleSize), Scalar(255, 255, 255), 1, 8, 0);
	cv::circle(prevClone, cv::Point(mouseData.x, mouseData.y), sqrt((pow(rectangleSize, 2)+pow(rectangleSize, 2))), cv::Scalar(255,255,255),
			3, 8, 0);

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
	this->pressedKey = waitKey(2);
	if(windows.empty())return -1;
	if(this->pressedKey == -1)return -1;

	for(auto& window: this->windows)
	{
		window.second->setKey(pressedKey);
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
