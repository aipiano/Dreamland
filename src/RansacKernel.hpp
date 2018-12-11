#pragma once

#include <vector>
#include <Eigen/Eigen>

template <class SampleType, class ModelType>
class RansacKernel
{
public:
	virtual void run_kernel(
		const std::vector<SampleType>& samples,
		std::vector<ModelType>& models
	) = 0;
	virtual void compute_error(
		const std::vector<SampleType>& samples,
		const ModelType& model,
		Eigen::VectorXd& errors
	) = 0;
};
