#include "CasHashMatcher.h"
#include "Utils.h"
#include <unordered_set>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace boost::multiprecision;

#define NUM_FINE_HASH_BITS 128

CasHashMatcher::CasHashMatcher(std::shared_ptr<FeaturePool> features, int num_bucket_groups, int num_bits_per_bucket, float distinct_ratio)
	: m_ptr_features(features), m_num_bucket_groups(num_bucket_groups), m_num_bits_per_bucket(num_bits_per_bucket), m_ratio(distinct_ratio)
{
	//std::random_device rd;
	//m_rng = std::mt19937(rd());
	assert(num_bits_per_bucket < 32);
	assert(num_bucket_groups * num_bits_per_bucket < NUM_FINE_HASH_BITS);

	m_num_buckets_per_group = 1 << num_bits_per_bucket;
	m_bucket_id_mask = 0xFFFFFFFF >> (32 - num_bits_per_bucket);

	omp_set_num_threads(omp_get_num_procs());
}

void CasHashMatcher::train(int train_id)
{
	m_train_descriptors = m_ptr_features->get_descriptors(train_id);
	
	calc_mean_descriptor(*m_train_descriptors, m_mean_descriptor);
	Mat unbiased_descriptors(m_train_descriptors->size(), m_train_descriptors->type());
	for (int i = 0; i < unbiased_descriptors.rows; ++i)
	{
		unbiased_descriptors.row(i) = m_train_descriptors->row(i) - m_mean_descriptor;
	}

	hash_descriptors(unbiased_descriptors, m_hashed_descriptors, m_lsh_projector);
	build_buckets(m_hashed_descriptors, m_bucket_groups);
}

void CasHashMatcher::match(int query_id, std::vector<cv::DMatch>& matches)
{
	const auto& query_descriptors = *m_ptr_features->get_descriptors(query_id);
	assert(m_train_descriptors != nullptr);
	matches.clear();

	unordered_set<int> candidate_indices;
	vector<Bucket> candidate_indices_by_hamming;
	Mat unbiased_descriptor;

	#pragma omp parallel for schedule(dynamic) private(candidate_indices, candidate_indices_by_hamming, unbiased_descriptor)
	for (int query_idx = 0; query_idx < query_descriptors.rows; ++query_idx)
	{
		candidate_indices.clear();
		auto& descriptor = query_descriptors.row(query_idx);
		unbiased_descriptor = descriptor - m_mean_descriptor;

		// get candidate ids from buckets
		int128_t query_hash = calc_hash_code(unbiased_descriptor, m_lsh_projector);
		int128_t hash_code = query_hash;	// make a copy
		for (int group_id = 0; group_id < m_num_bucket_groups; ++group_id)
		{
			// get bucket id by slicing the long hash code.
			unsigned int bucket_id = (unsigned int)(hash_code & m_bucket_id_mask);
			hash_code >>= m_num_bits_per_bucket;
			auto& bucket = m_bucket_groups[group_id][bucket_id];
			for (int desc_idx : bucket)
				candidate_indices.insert(desc_idx);
		}
		if (candidate_indices.size() < 10) continue;

		candidate_indices_by_hamming.resize(NUM_FINE_HASH_BITS + 1);
		for (auto& bucket : candidate_indices_by_hamming)
			bucket.clear();

		// sort candidate descriptors by hamming distance.
		// descriptors with the same hamming distance will be stored in the same bucket.
		for (auto train_idx : candidate_indices)
		{
			int hamming = count_bits(m_hashed_descriptors[train_idx] ^ query_hash);
			candidate_indices_by_hamming[hamming].push_back(train_idx);
		}

		// match top k(>= 10) candidates.
		int k = 0;
		int best_match_idx = -1;
		double min_distance = DBL_MAX;
		double sec_distance = DBL_MAX;	// nearest neighbor
		for (auto& indices : candidate_indices_by_hamming)
		{
			for (auto idx : indices)
			{
				double distance = cv::norm(m_train_descriptors->row(idx), descriptor, NORM_L2);
				if (distance < min_distance)
				{
					min_distance = distance;
					best_match_idx = idx;
				}
				else if (distance < sec_distance)
				{
					sec_distance = distance;
				}
			}
			k += indices.size();
			if (k >= 10) break;
		}

		if (k >= 10 && min_distance < m_ratio * sec_distance)	// ratio test
		{
			#pragma omp critical
			{
				matches.push_back(DMatch(query_idx, best_match_idx, min_distance));
			}
		}
	}
}

void CasHashMatcher::hash_descriptors(const cv::Mat & descriptors, HashedDescriptors & hashed_descriptors, LSHProjector & lsh_projector)
{
	lsh_projector.create(NUM_FINE_HASH_BITS, descriptors.cols, CV_32FC1);
	cv::randn(lsh_projector, 0, 1);	// set elements by normal distribution

	hashed_descriptors.clear();
	hashed_descriptors.resize(descriptors.rows);

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < descriptors.rows; ++i)
	{
		int128_t hash_code = calc_hash_code(descriptors.row(i), lsh_projector);
		hashed_descriptors[i] = hash_code;
	}
}

void CasHashMatcher::build_buckets(HashedDescriptors & hashed_descriptors, BucketGroups & bucket_groups)
{
	bucket_groups.clear();
	bucket_groups.resize(m_num_bucket_groups);
	for (auto& bucket_group : bucket_groups)
	{
		bucket_group.resize(m_num_buckets_per_group);
	}

	for (int i = 0; i < hashed_descriptors.size(); ++i)
	{
		auto hash_code = hashed_descriptors[i];
		for (int group_id = 0; group_id < m_num_bucket_groups; ++group_id)
		{
			// get bucket id by slicing the long hash code.
			unsigned int bucket_id = (unsigned int)(hash_code & m_bucket_id_mask);
			hash_code >>= m_num_bits_per_bucket;
			bucket_groups[group_id][bucket_id].push_back(i);
		}
	}
}

void CasHashMatcher::calc_mean_descriptor(const cv::Mat & descriptors, cv::Mat & mean_descriptor)
{
	mean_descriptor.create(1, descriptors.cols, descriptors.type());
	mean_descriptor.setTo(0);
	for (int i = 0; i < descriptors.rows; ++i)
	{
		mean_descriptor += descriptors.row(i);
	}
	mean_descriptor /= descriptors.rows;
}

inline int128_t CasHashMatcher::calc_hash_code(const cv::Mat & descriptor, const LSHProjector& lsh_projector)
{
	int128_t hash_code = 0;
	/*for (auto& hyperplane : lsh_projector)
	{
		hash <<= 1;
		hash |= (descriptor.dot(hyperplane) > 0.);
	}*/
	
	Mat projected_descriptor = lsh_projector * descriptor.t();
	float* ptr_proj_desc = projected_descriptor.ptr<float>();
	for (int i = 0; i < lsh_projector.rows; ++i)
	{
		hash_code <<= 1;
		hash_code |= (ptr_proj_desc[i] > 0.);
	}
	return hash_code;
}

