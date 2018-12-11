#pragma once

#include <opencv2\features2d.hpp>
#include <vector>

class FeatureMatcher
{
public:
	virtual void train(int train_id) = 0;
	virtual void match(int query_id, std::vector<cv::DMatch>& matches) = 0;
};
