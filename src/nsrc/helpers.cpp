/*
 * helpers.cpp
 *
 *  Created on: 20 Jun 2023
 *      Author: jurie
 */


#include "helpers.cpp"
#include <iostream>
#include <opencv2/opencv.hpp>


void drawHistogram(std::vector<float> hist, cv::Mat & output)
{
	int width = 500, height = 500, lineHeight = height*0.8;
	float edgeOffset = 0.1;
	output.create(cv::Size(width, height), CV_8UC3);
	output = cv::Scalar(255,255,255);

	// y-axis
	cv::line(output, cv::Point(width*edgeOffset, height*edgeOffset), cv::Point(width*edgeOffset,height*edgeOffset-lineHeight ), cv::Scalar(0,0,0), 2);

	cv::imshow("histogram", output);
	cv::waitKey(0);

}
