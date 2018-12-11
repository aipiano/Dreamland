#pragma once

#include <Eigen\Eigen>

class SampleConsensusKernel
{
public:
	// ����ģ�͵�����������Eigen������Ԫ��ÿ��ģ����models�а�x�����Ų�
	virtual int run_kernel(
		const Eigen::Ref<const Eigen::MatrixXd>& m1,
		const Eigen::Ref<const Eigen::MatrixXd>& m2,
		Eigen::MatrixXd& models
	) = 0;
	virtual void compute_error(
		const Eigen::Ref<const Eigen::MatrixXd>& m1,
		const Eigen::Ref<const Eigen::MatrixXd>& m2,
		const Eigen::Ref<const Eigen::MatrixXd>& model,
		Eigen::VectorXd& errors
	) = 0;
	virtual bool check_subset(
		const Eigen::Ref<const Eigen::MatrixXd>&,
		const Eigen::Ref<const Eigen::MatrixXd>&,
		int
	) {	return true; }
};
