#pragma once

#include <vector>
#include <opencv2\features2d.hpp>

class MatchFrame
{
public:
	MatchFrame();
	MatchFrame(const MatchFrame& other);
	MatchFrame(MatchFrame&& other);
	MatchFrame(const std::vector<cv::DMatch>& matches, int train_id, int query_id /*, cv::Mat& mask*/);
	MatchFrame(std::vector<cv::DMatch>&& matches, int train_id, int query_id /*, cv::Mat& mask*/);
	MatchFrame& operator=(const MatchFrame& other);
	void operator=(MatchFrame&& other);

public:
	int train_id = -1;
	int query_id = -1;
	std::vector<cv::DMatch> matches;
	//cv::Mat mask;	//not used.
};
