
#include "color_balance.hpp"
#include <opencv2/opencv.hpp>
#include <cassert>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/saturate.hpp>


ColorBalancer::ColorBalancer()
{
	this->img_mat = nullptr;
	this->lastBValue = 0;
	this->lookupTable.create(1, 256, CV_8U);
}


void ColorBalancer::setImage(cv::Mat * img)
{
	assert(img!=nullptr);
	this->img_mat = img;

	cv::cvtColor(*this->img_mat, *this->img_mat, cv::COLOR_RGB2HSV);

	cv::split(*(this->img_mat), this->channels);
}

cv::Mat* ColorBalancer::getImage(){return this->img_mat;}

void ColorBalancer::Brightness(int value)
{
	assert(this->img_mat!=nullptr);

	float gamma = ((100-value)*1.0)/100.0f;

	if(this->lastBValue != value)
	{
		uchar* p = lookupTable.ptr();

		for(int i=0; i<256; i++)
		{
			p[i] = cv::saturate_cast<uchar>(pow( i / 255.0, gamma)*255.0);
		}
	}

	cv::LUT(this->channels[2], lookupTable,this->channels[2]);



	this->lastBValue = value;

}


void ColorBalancer::saturation(int value)
{
	for(int i=0; i<this->channels[1].cols*this->channels[1].rows; i++)
	{
		this->channels[1].ptr()[i] = value;
	}
}

void ColorBalancer::hue(int value)
{
	for(int i=0; i<this->channels[0].cols*this->channels[0].rows; i++)
	{
		this->channels[1].ptr()[i] = value;
	}
}


void ColorBalancer::finish()
{
	cv::merge(this->channels, *(this->img_mat));

	cv::cvtColor(*(this->img_mat), *this->img_mat, cv::COLOR_HSV2RGB);
}

