#include "FlannMatcher.h"
#include "MatcherUtils.h"
using namespace cv;
using namespace std;

FlannMatcher::FlannMatcher(std::shared_ptr<FeaturePool> features, float distinct_ratio)
{
	m_ptr_features = features;
	m_ptr_matcher = make_shared<FlannBasedMatcher>();
	m_ratio = distinct_ratio;
}

void FlannMatcher::train(int train_id)
{
	auto descriptors_ptr = m_ptr_features->get_descriptors(train_id);
	m_ptr_matcher->clear();
	m_ptr_matcher->add(*descriptors_ptr);
	m_ptr_matcher->train();
}

void FlannMatcher::match(int query_id, std::vector<cv::DMatch>& matches)
{
	const auto& query_descriptors = *m_ptr_features->get_descriptors(query_id);

	m_knn_matches.clear();
	m_knn_matches.reserve(query_descriptors.rows);
	m_ptr_matcher->knnMatch(query_descriptors, m_knn_matches, 2);

	ratio_test(m_knn_matches, m_ratio, matches);
}
