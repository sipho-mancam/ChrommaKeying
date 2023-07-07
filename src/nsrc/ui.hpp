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
#include <queue>
#include "events.hpp"


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
	bool keyEnabled;
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
		keyEnabled = false;
	}

	std::string getHandle(){return this->windowName;}
	void setMouseCB( void* md, void(*cb_)(int, int, int, int, void*));
	void enableMouse(void (*cb_)(int, int, int, int, void*)=mouseCallback)
	{
		this->mEnabled = true;
		this->setMouseCB((void*)&this->mouseData, cb_);
	}
	void enableKeys(){this->keyEnabled = true;}
	bool isKeysEnabled(){return this->keyEnabled;}
	void disableKeys(){this->keyEnabled = false;}
	MouseData* getMD(){return &this->mouseData;}
	int getPressKey(){ return key;}
	void setKey(int k){key = k;}
	bool isCaptured(){return this->captureKey;}
	void captured(){this->captureKey = true;}
	void setKeyCB(void(*kCB)(int));
	virtual void update(){}
	virtual ~WindowI() = default;
	virtual int parseKey(); // this method will transform the received key to an event.
							// All windows will implement the same parser, but they can override for custom parser.
};

class KeyingWindow: public WindowI
{
private:
	uchar3* rgbData; // this data is in the GPU memory
	cv::Mat previewMat;
	cv::cuda::GpuMat gMat;
	bool startCapture;

	int iWidth, iHeight;
	void process(); // apply operations to the image buffer

public:
	KeyingWindow(std::string windowHandle, int iW, int iH): WindowI(windowHandle)
	{
		this->iHeight = iH;
		this->iWidth = iW;
		rgbData = nullptr;
		this->gMat.create(this->iHeight, this->iWidth, CV_8UC3);
		this->gMat.step = 5760;
		cv::setWindowProperty(this->windowName, WND_PROP_ASPECT_RATIO,WINDOW_FULLSCREEN);
		this->startCapture = false;
	}

	void enableCapture(){this->startCapture = true;}
	void disableCapture(){this->startCapture = false;}
	bool captureStatus(){return this->startCapture;}
	void loadImage(uchar3* d){ this->rgbData = d;}
	void show();
	void update() override;// update graphics on the screen

};

class SettingsWindow :public WindowI
{
private:
	std::vector<std::string> trackbars = {
											WINDOW_TRACKBAR_BLENDING,
											WINDOW_TRACKBAR_DELAY,
											WINDOW_TRACKBAR_ERODE,
											WINDOW_TRACKBAR_DILATE,
											WINDOW_TRACKBAR_OUTER_DIAM,
											WINDOW_TRACKBAR_UV_DIAM,
											WINDOW_TRACKBAR_LUM_DEPTH,
											WINDOW_TRACKBAR_UV,
											WINDOW_TRACKBAR_LUM,
											WINDOW_TRACKBAR_KEYTOP,
											WINDOW_TRACKBAR_KEYBOTTOM,
											WINDOW_TRACKBAR_BRIGHTNESS,
											WINDOW_TRACKBAR_SAT,
										};
	std::unordered_map<std::string, int> trackbarValues;
public:
	SettingsWindow(std::string windowHandle): WindowI(windowHandle)
	{
		cv::createTrackbar(this->trackbars[0], this->windowName, 0, 900, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[1], this->windowName, 0, 30, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[2], this->windowName, 0, 20, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[3], this->windowName, 0, 20, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[4], this->windowName, 0, 200, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[5], this->windowName, 0, 50, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[6], this->windowName, 0, 50, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[7], this->windowName, 0, 50, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[8], this->windowName, 0, 100, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[9], this->windowName, 0, 300, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[10], this->windowName, 0, 300, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[11], this->windowName, 0, 100, updateTrackbar, (void*)this);
		cv::createTrackbar(this->trackbars[12], this->windowName, 0, 100, updateTrackbar, (void*)this);

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
	std::queue<int> eventQueue;
	int currentEvent;

public:
	WindowsContainer()
	{
		this->pressedKey = 0;
		this->currentEvent = -1;
	}
	void addWindow(WindowI *w);
	void removeWindow(std::string windowHandle);
	int dispatchKey();
	void dispatchEvent();
	int getEvent();
	void updateWindows();
	WindowI * getWindow(std::string windowHandle);
	int getKey(){return this->pressedKey;}
	~WindowsContainer();
};




#endif /* SRC_NSRC_UI_HPP_ */
