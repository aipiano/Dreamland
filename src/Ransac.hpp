#pragma once

#include "RansacKernel.hpp"
#include <memory>
#include <random>

template <class SampleType, class ModelType>
class Ransac
{
public:
	Ransac() {}
	Ransac(std::shared_ptr<RansacKernel<SampleType, ModelType>> kernel, int model_points, double threshold, double confidence = 0.99, int max_iters = 1000)
	{
		set_kernel(kernel, model_points, threshold, confidence, max_iters);
	}
	void set_kernel(std::shared_ptr<RansacKernel<SampleType, ModelType>> kernel, int model_points, double threshold, double confidence = 0.99, int max_iters = 1000)
	{
		m_kernel = kernel;
		m_model_points = model_points;
		m_threshold = threshold;
		m_confidence = confidence;
		m_max_iters = max_iters;
	}

	bool run(
		const std::vector<SampleType>& samples,
		ModelType& model,
		std::vector<unsigned char>& mask
	)
	{
		bool result = false;
		ModelType best_model;
		std::vector<ModelType> models;
		Eigen::VectorXd err;

		int iter, niters = std::max(m_max_iters, 1);
		int count = samples.size(), max_good_count = 0;
		std::random_device rd;
		std::mt19937 rng(rd());

		assert(m_kernel != nullptr);
		assert(m_confidence > 0 && m_confidence < 1);
		assert(m_model_points > 0);
		assert(count >= 0);

		std::vector<SampleType> sub_samples(m_model_points);

		if (count < m_model_points)
			return false;

		if (count == m_model_points)
		{
			models.clear();
			m_kernel->run_kernel(samples, models);
			if (models.size() == 0)
				return false;
			model = models[0];
			mask.resize(count);
			std::fill(mask.begin(), mask.end(), 1);
			return true;
		}

		std::vector<unsigned char> best_mask(count, 1);
		for (iter = 0; iter < niters; iter++)
		{
			int i, good_count;
			if (count > m_model_points)
			{
				bool found = get_subset(samples, sub_samples, rng);
				if (!found)
				{
					if (iter == 0)
						return false;
					break;
				}
			}

			models.clear();
			m_kernel->run_kernel(sub_samples, models);
			if (models.size() <= 0)
				continue;
			
			for (i = 0; i < models.size(); i++)
			{
				good_count = find_inliers(samples, models[i], err, mask, m_threshold);

				if (good_count > max(max_good_count, m_model_points - 1))
				{
					std::swap(mask, best_mask);
					best_model = models[i];
					max_good_count = good_count;
					niters = update_num_iters(m_confidence, (double)(count - good_count) / count, m_model_points, niters);
				}
			}
		}

		if (max_good_count > 0)
		{
			mask = std::move(best_mask);
			model = std::move(best_model);
			result = true;
		}

		return result;
	}

private:
	bool get_subset(
		const std::vector<SampleType>& samples,
		std::vector<SampleType>& sub_samples,
		std::mt19937& rng
	) const
	{
		std::vector<int> idx(m_model_points);
		int i = 0, j;
		int count = samples.size();
		assert(count >= m_model_points);

		std::uniform_int_distribution<int> uni(0, count - 1);	//uni的参数是闭区间

		sub_samples.resize(m_model_points);

		for (i = 0; i < m_model_points;)
		{
			int idx_i = 0;
			for (;;)
			{
				idx_i = idx[i] = uni(rng);
				for (j = 0; j < i; j++)	// 防止重复采样
					if (idx_i == idx[j])
						break;
				if (j == i)	// 若没有重复的，则跳出循环，添加到子集中
					break;
			}
			sub_samples[i] = samples[idx_i];
			++i;
		}

		return i == m_model_points;
	}

	int find_inliers(
		const std::vector<SampleType>& samples,
		const ModelType& model,
		Eigen::VectorXd& err,
		std::vector<unsigned char>& mask,
		double thresh
	) const
	{
		err.resize(samples.size());
		mask.resize(err.rows());
		m_kernel->compute_error(samples, model, err);

		int i, n = err.rows(), nz = 0;
		for (i = 0; i < n; i++)
		{
			int f = err[i] <= thresh;
			mask[i] = f;
			nz += f;
		}
		return nz;
	}

	int update_num_iters(double p, double ep, int modelPoints, int maxIters)
	{
		p = max(p, 0.);
		p = min(p, 1.);
		ep = max(ep, 0.);
		ep = min(ep, 1.);

		// avoid inf's & nan's
		double num = max(1. - p, DBL_MIN);
		double denom = 1. - std::pow(1. - ep, modelPoints);
		if (denom < DBL_MIN)
			return 0;

		num = std::log(num);
		denom = std::log(denom);

		return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : (int)round(num / denom);
	}

private:
	std::shared_ptr<RansacKernel<SampleType, ModelType>> m_kernel;
	int m_model_points = 0;
	double m_threshold = 0;
	double m_confidence = 0.99;
	int m_max_iters = 1000;
	//std::random_device m_rd;
};

