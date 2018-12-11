#pragma once

//------------------
//-- Bibliography --
//------------------
//- [1] "Fast and Accurate Image Matching with Cascade Hashing for 3D Reconstruction"
//- Authors: Jian Cheng, Cong Leng, Jiaxiang Wu, Hainan Cui, Hanqing Lu.
//- Date: 2014.
//- Conference: CVPR.
//
//- 与原算法相比有些许改进
//- 只用一套hash函数计算特征的hash值（长hash），用于粗匹配的短hash通过对长hash的截取得到
//- 所以要求 num_bucket_groups * num_bits_per_bucket < 128
//- 这样不仅减少计算量，也简化了代码实现

#include "FeatureMatcher.h"
#include "FeatureMatcher.h"
#include "FeaturePool.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <random>
#include <array>
#include <boost/multiprecision/cpp_int.hpp>

class CasHashMatcher final : public FeatureMatcher
{
public:
	/* 
	num_bits_per_bucket < 32. 
	num_bucket_groups * num_bits_per_bucket < 128.
	num_bucket_groups should become larger when num_bits_per_bucket is large.
	*/
	CasHashMatcher(std::shared_ptr<FeaturePool> features, int num_bucket_groups = 6, int num_bits_per_bucket = 10, float distinct_ratio = 0.6f);

	// 通过 FeatureMatcher 继承
	virtual void train(int train_id) override;
	virtual void match(int query_id, std::vector<cv::DMatch>& matches) override;

private:
	typedef std::vector<int> Bucket;
	typedef std::vector<Bucket> BucketGroup;
	typedef std::vector<BucketGroup> BucketGroups;
	typedef std::vector<boost::multiprecision::int128_t> HashedDescriptors;
	typedef cv::Mat LSHProjector;	// each row is a hyperplane

	void hash_descriptors(const cv::Mat& descriptors, HashedDescriptors& hashed_descriptors, LSHProjector& lsh_projector);
	void build_buckets(HashedDescriptors& hashed_descriptors, BucketGroups& bucket_groups);

	void calc_mean_descriptor(const cv::Mat& descriptors, cv::Mat& mean_descriptor);

	inline boost::multiprecision::int128_t calc_hash_code(const cv::Mat& descriptor, const LSHProjector& lsh_projector);

private:
	std::shared_ptr<FeaturePool> m_ptr_features;
	std::shared_ptr<const cv::Mat> m_train_descriptors;
	cv::Mat m_mean_descriptor;
	int m_num_bucket_groups;
	int m_num_bits_per_bucket;
	int m_num_buckets_per_group;
	int m_bucket_id_mask;
	float m_ratio;

	HashedDescriptors m_hashed_descriptors;
	LSHProjector m_lsh_projector;
	BucketGroups m_bucket_groups;

	//std::mt19937 m_rng;
	//std::normal_distribution<double> m_normal_distribution;
};
