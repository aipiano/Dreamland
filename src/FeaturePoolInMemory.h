#pragma once

#include <opencv2\features2d.hpp>
#include <vector>
#include <map>
#include <mutex>
#include "FeaturePool.h"

class FeaturePoolInMemory final : public FeaturePool
{
public:
	FeaturePoolInMemory() {};
	virtual void push(const KeyPoints& keypoints, const cv::Mat& descriptors, int view_id) override;
	virtual void push(KeyPoints&& keypoints, cv::Mat&& descriptors, int view_id) override;
	virtual const KeyPointsPtr get_keypoints(int view_id) override;
	virtual const DescriptorsPtr get_descriptors(int view_id) override;
	virtual size_t size() override;
	virtual void clear() override;

private:
	typedef std::map<int, KeyPointsPtr> KeyPointsMap;
	typedef std::map<int, DescriptorsPtr> DescriptorsMap;

	std::mutex m_mutex;
	KeyPointsMap m_keypoints_map;
	DescriptorsMap m_descriptors_map;

	//用于表示为找到对应项时返回的空值
	//const std::vector<cv::KeyPoint> m_keypoints_empty;
	//const cv::Mat m_descriptors_empty;
};