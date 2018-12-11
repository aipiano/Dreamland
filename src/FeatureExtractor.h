#pragma once

#include <opencv2\features2d.hpp>
#include <vector>

class FeatureExtractor
{
public:
	virtual bool extract(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) = 0;
};
