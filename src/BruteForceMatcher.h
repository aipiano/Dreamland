#pragma once

#include "FeatureMatcher.h"
#include "FeaturePool.h"
#include <memory>

class BruteForceMatcher final : public FeatureMatcher
{
public:
	BruteForceMatcher(std::shared_ptr<FeaturePool> features, cv::NormTypes norm_type = cv::NormTypes::NORM_L2, /*bool cross_check = true, */float ratio = 0.6f);
	// Í¨¹ý FeatureMatcher ¼Ì³Ð
	virtual void train(int train_id) override;
	virtual void match(int query_id, std::vector<cv::DMatch>& matches) override;

private:
	float m_ratio = 0.6f;
	std::vector<std::vector<cv::DMatch>> m_knn_matches;
	std::shared_ptr<FeaturePool> m_ptr_features;
	std::shared_ptr<cv::BFMatcher> m_ptr_matcher;

	cv::Mat m_trained_descriptors;
};
