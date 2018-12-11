#pragma once

#include <vector>
#include <opencv2/features2d.hpp>

void ratio_test(std::vector<std::vector<cv::DMatch>>& knn_matches, double ratio, std::vector<cv::DMatch>& out_matches);

void cross_test(std::vector<cv::DMatch>& matches1, std::vector<cv::DMatch>& matches2, std::vector<cv::DMatch>& out_matches);
