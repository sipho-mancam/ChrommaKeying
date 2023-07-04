/*
 * contours.hpp
 *
 *  Created on: Jul 4, 2023
 *      Author: sipho-mancam
 */

#ifndef INCLUDE_CONTOURS_HPP_
#define INCLUDE_CONTOURS_HPP_

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

class ContourDetector
{
private:
	cv::Mat img_mat;
	cv::Ptr<cv::cuda::CannyEdgeDetector> detector;

public:
	ContourDetector();
	void setImage(uchar* );
	void init();
	void detect(bool&);
};



#endif /* INCLUDE_CONTOURS_HPP_ */
