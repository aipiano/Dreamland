#pragma once

#include "SampleConsensusKernel.h"
#include <Eigen/Eigen>
#include <vector>
#include <memory>

class SampleConsensusEstimator
{
public:
	virtual bool run(
		const Eigen::Ref<const Eigen::MatrixXd>& m1,
		const Eigen::Ref<const Eigen::MatrixXd>& m2,
		Eigen::MatrixXd& model,
		std::vector<unsigned char>& mask
	) = 0;
	virtual void set_kernel(std::shared_ptr<SampleConsensusKernel> registrator)
	{
		m_kernel = registrator;
	}

protected:
	std::shared_ptr<SampleConsensusKernel> m_kernel;
};
