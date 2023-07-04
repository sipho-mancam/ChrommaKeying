/*
 * color_balance.hpp
 *
 *  Created on: Jul 4, 2023
 *      Author: sipho-mancam
 */

#ifndef SRC_COLOR_BALANCE_HPP_
#define SRC_COLOR_BALANCE_HPP_

#include <opencv2/opencv.hpp>

class ColorBalancer
{
private:
	cv::Mat* img_mat;
	std::vector<cv::Mat> channels;

	int lastBValue;
	cv::Mat lookupTable;

public:
	ColorBalancer();
	void setImage(cv::Mat* img);
	cv::Mat* getImage();
	void Brightness(int value); // value between 0 and 100
	void saturation(int value);
	void hue(int value);
	void finish();
};




#endif /* SRC_COLOR_BALANCE_HPP_ */
