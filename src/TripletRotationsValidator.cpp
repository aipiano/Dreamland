#include "TripletRotationsValidator.h"
#include "Utils.h"
using namespace Eigen;

TripletRotationsValidator::TripletRotationsValidator(Options options)
	: m_options(options)
{
	m_options.max_consensus_degree = deg2rad(m_options.max_consensus_degree);
}

bool TripletRotationsValidator::acceptable(const Eigen::Ref<const Eigen::Matrix3d>& Rij, const Eigen::Ref<const Eigen::Matrix3d>& Rik, const Eigen::Ref<const Eigen::Matrix3d>& Rjk)
{
	Matrix3d R_err = Rij * Rjk * Rik.transpose();
	Vector3d r_err;
	rotation_matrix_to_angle_axis(R_err, r_err);
	if (r_err.norm() > m_options.max_consensus_degree)
		return false;

	return true;
}
