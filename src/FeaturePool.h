#pragma once

#include <vector>
#include <memory>
#include <opencv2\features2d.hpp>
//#include <Eigen\Eigen>

class FeaturePool
{
public:
	typedef std::vector<cv::KeyPoint> KeyPoints;
	typedef std::shared_ptr<const KeyPoints> KeyPointsPtr;
	typedef std::shared_ptr<const cv::Mat> DescriptorsPtr;
	virtual void push(const KeyPoints& keypoints, const cv::Mat& descriptors, int view_id) = 0;
	virtual void push(KeyPoints&& keypoints, cv::Mat&& descriptors, int view_id) = 0;
	// ����ָ����������ã�ʹ���Ժ����֧��IO cache
	virtual const KeyPointsPtr get_keypoints(int view_id) = 0;
	virtual const DescriptorsPtr get_descriptors(int view_id) = 0;	// Mat�Լ������ü���
	virtual size_t size() = 0;
	virtual void clear() = 0;
};
