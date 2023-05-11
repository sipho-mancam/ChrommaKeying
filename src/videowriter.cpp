#include <iostream>

#include "opencv2/opencv_modules.hpp"



#include <vector>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
const double FPS = 2.0;
//cv::Ptr<cv::cudacodec::VideoWriter> d_writer;
cv::VideoWriter writer;


int writeframe(cv::Mat frame)
{
	if (!writer.isOpened())
	{
		std::cout << "Frame Size : " << frame.cols << "x" << frame.rows << std::endl;
		std::cout << "Open CPU Writer" << std::endl;
		if (!writer.open("/home/jurie/Videos/output_cpu1.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), FPS, frame.size()))
			return -1;
	}

	  writer.write(frame);
//	std::cout << "Open CUDA Writer" << std::endl;
//		const cv::String outputFilename = "/home/jurie/Videos/output_gpu.avi";
//		d_writer = cv::cudacodec::createVideoWriter(outputFilename, d_frame.size(), FPS);
//		d_writer->write(d_frame);
//        return 0;
}


