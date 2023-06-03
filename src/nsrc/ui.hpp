/*
 * ui.hpp
 *
 *  Created on: 02 Jun 2023
 *      Author: jurie
 */

#ifndef SRC_NSRC_UI_HPP_
#define SRC_NSRC_UI_HPP_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <YUVUChroma.cuh>
#include <unordered_map>
#include <vector>


void mouseCallback(int event, int x, int y, int flag, void* data);
void updateTrackbar(int, void* );

class WindowI
{
protected:
	std::string windowName;
	bool created;
	MouseData mouseData;
	WindowSettings windowSettings;
	int key;
	cv::Rect windowRect;
	bool mEnabled;
	bool captureKey;
public:
	WindowI(std::string windowHandle)
	{
		this->windowName = windowHandle;
		this->created = false;
		cv::namedWindow(this->windowName, WINDOW_NORMAL);
		this->created = true;
		key = -1;
		windowRect = cv::getWindowImageRect(this->windowName);
		mouseData.windowName = this->windowName;
		mEnabled = false;
		captureKey = false;
	}

	std::string getHandle(){return this->windowName;}
	void setMouseCB( void* md, void(*cb_)(int, int, int, int, void*));
	void enableMouse(void (*cb_)(int, int, int, int, void*)=mouseCallback)
	{
		this->mEnabled = true;
		this->setMouseCB((void*)&this->mouseData, cb_);
	}
	MouseData getMD(){return this->mouseData;}
	int getPressKey(){ return key;}
	void setKey(int k){
		key = k;
		if(this->key == 'q' || this->key == 'Q') this->captureKey = true;
	}
	bool isCaptured(){return this->captureKey;}
	void setKeyCB(void(*kCB)(int));
};

class SettingsWindow :public WindowI
{
private:
	std::vector<std::string> trackbars = {
											"Blending", "Delay", "Erode", "Dilate", "Outer Diam",
											"UV Diam", "Lum Depth", "E UV", "E Lum", "Key Bot",
											"Key Top", "NMS"
										};
	std::unordered_map<std::string, int> trackbarValues;
public:
	SettingsWindow(std::string windowHandle): WindowI(windowHandle)
	{
		cv::createTrackbar(this->trackbars[0], this->windowName, 0, 2000, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[1], this->windowName, 0, 30, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[2], this->windowName, 0, 20, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[3], this->windowName, 0, 20, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[4], this->windowName, 0, 200, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[5], this->windowName, 0, 50, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[6], this->windowName, 0, 50, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[7], this->windowName, 0, 50, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[8], this->windowName, 0, 50, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[9], this->windowName, 0, 300, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[10], this->windowName, 0, 300, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[11], this->windowName, 0, 100, updateTrackbar, (void*)this);

		for(std::string& elem : trackbars)
		{
			trackbarValues[elem] = 0;
		}
		this->init();
		this->update();
	}
	void init();
	void update()
	{
		for(std::string& trackbar: trackbars)
		{
			trackbarValues[trackbar] = cv::getTrackbarPos(trackbar, this->windowName);
		}
	}
	std::unordered_map<std::string, int> getTrackbarValues(){return this->trackbarValues;}
};


class WindowsContainer
{
private:
	std::unordered_map<std::string, WindowI*> windows;
	int pressedKey;
public:
	WindowsContainer()
	{
		this->pressedKey = 0;
	}
	void addWindow(WindowI *w);
	void removeWindow(std::string windowHandle);
	int dispatchKey();
	WindowI * getWindow(std::string windowHandle);
};




#endif /* SRC_NSRC_UI_HPP_ */
