#include "MatcherUtils.h"

void ratio_test(std::vector<std::vector<cv::DMatch>>& knn_matches, double ratio, std::vector<cv::DMatch>& out_matches)
{
	out_matches.clear();
	out_matches.reserve(knn_matches.size() * 0.5);
	for (int i = 0; i < knn_matches.size(); ++i)
	{
		//Ratio test
		if (knn_matches[i][0].distance > knn_matches[i][1].distance * ratio)
			continue;
		out_matches.push_back(knn_matches[i][0]);
	}
}

void cross_test(std::vector<cv::DMatch>& matches1, std::vector<cv::DMatch>& matches2, std::vector<cv::DMatch>& out_matches)
{
	out_matches.clear();
	out_matches.reserve(matches1.size() * 0.5);
	for (size_t i = 0; i < matches1.size(); ++i)
	{
		if (matches1[i].queryIdx == matches2[matches1[i].trainIdx].trainIdx)
			out_matches.push_back(matches1[i]);
	}
}
