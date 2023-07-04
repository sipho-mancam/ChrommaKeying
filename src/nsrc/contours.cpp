/*
 * contours.cpp
 *
 *  Created on: Jul 4, 2023
 *      Author: sipho-mancam
 */

#include <opencv2/cudaimgproc.hpp>
#include "contours.hpp"
#include <vector>
#include <iostream>


ContourDetector::ContourDetector()
{
	this->init();
}

void ContourDetector::setImage(uchar* mask)
{
	cv::cuda::GpuMat mat(cv::Size(1920, 1080), CV_8UC1, mask);
	mat.download(this->img_mat);


}

void ContourDetector::init()
{
//	this->detector =;
}

void ContourDetector::detect(bool& cd)
{
	if(!cd)return;

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(this->img_mat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);


	cv::Mat counterDraw = cv::Mat::zeros(this->img_mat.size(), CV_8UC3);
	for(int i=0; i<contours.size(); i++)
	cv::drawContours(counterDraw, contours, i, cv::Scalar(0, 240, 0));

	cv::imshow("Contours", counterDraw);

	cd = true;


}




