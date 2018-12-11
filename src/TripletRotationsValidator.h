#pragma once

#include <Eigen\Eigen>

class TripletRotationsValidator
{
public:
	struct Options
	{
		double max_consensus_degree = 5.0; // max consensus angle in degree
	};

	TripletRotationsValidator(Options options);
	bool acceptable(
		const Eigen::Ref<const Eigen::Matrix3d>& Rij,
		const Eigen::Ref<const Eigen::Matrix3d>& Rik,
		const Eigen::Ref<const Eigen::Matrix3d>& Rjk
	);

private:
	Options m_options;
};
