#pragma once

#include "CameraExtrinsic.h"
#include "RelativeTransform.h"
#include "GlobalTransform.h"
#include "Camera.h"
#include "Tracks.h"
#include "Scene.h"
#include <opencv2\features2d.hpp>
#include <vector>
#include <Eigen\Eigen>
#include <boost/multiprecision/cpp_int.hpp>


const char *const g_bits_per_byte =
"\0\1\1\2\1\2\2\3\1\2\2\3\2\3\3\4"
"\1\2\2\3\2\3\3\4\2\3\3\4\3\4\4\5"
"\1\2\2\3\2\3\3\4\2\3\3\4\3\4\4\5"
"\2\3\3\4\3\4\4\5\3\4\4\5\4\5\5\6"
"\1\2\2\3\2\3\3\4\2\3\3\4\3\4\4\5"
"\2\3\3\4\3\4\4\5\3\4\4\5\4\5\5\6"
"\2\3\3\4\3\4\4\5\3\4\4\5\4\5\5\6"
"\3\4\4\5\4\5\5\6\4\5\5\6\5\6\6\7"
"\1\2\2\3\2\3\3\4\2\3\3\4\3\4\4\5"
"\2\3\3\4\3\4\4\5\3\4\4\5\4\5\5\6"
"\2\3\3\4\3\4\4\5\3\4\4\5\4\5\5\6"
"\3\4\4\5\4\5\5\6\4\5\5\6\5\6\6\7"
"\2\3\3\4\3\4\4\5\3\4\4\5\4\5\5\6"
"\3\4\4\5\4\5\5\6\4\5\5\6\5\6\6\7"
"\3\4\4\5\4\5\5\6\4\5\5\6\5\6\6\7"
"\4\5\5\6\5\6\6\7\5\6\6\7\6\7\7\x8";

int count_bits(boost::multiprecision::int128_t n);


//求src到dst的相对变换
CameraExtrinsic relative_transform_between(CameraExtrinsic & src, CameraExtrinsic & dst);

CameraExtrinsic reversed_transform(CameraExtrinsic & rt);

inline Eigen::Matrix3d cross_mat(const Eigen::Ref<const Eigen::Vector3d>& x);
inline cv::Matx33d cross_mat(cv::Matx31d& x);

void kron(
	const Eigen::Ref<const Eigen::MatrixXd>& a,
	const Eigen::Ref<const Eigen::MatrixXd>& b,
	Eigen::Ref<Eigen::MatrixXd> k
);

//使用DLT方法对一对点进行三角化，如果需要对track进行三角化，请使用Triangulator类
Eigen::Vector3d triangulate_DLT(
	const Eigen::Ref<const Eigen::Vector3d>& x1,
	const Eigen::Ref<const Eigen::Vector3d>& x2,
	const Eigen::Ref<const Eigen::Matrix3d>& R,
	const Eigen::Ref<const Eigen::Vector3d>& t
);

double calc_reproj_error(Scene& scene);

double rad2deg(double radius);
double deg2rad(double degree);

void rotation_matrix_to_angle_axis(const Eigen::Ref<const Eigen::Matrix3d>& matrix, Eigen::Ref<Eigen::Vector3d> angle_axis);
void angle_axis_to_rotation_matrix(const Eigen::Ref<const Eigen::Vector3d>& angle_axis, Eigen::Ref<Eigen::Matrix3d> matrix);
void angle_axis_to_quaternion(const Eigen::Ref<const Eigen::Vector3d>& angle_axis, Eigen::Quaterniond& q);
void quaternion_to_angle_axis(Eigen::Quaterniond& q, Eigen::Ref<Eigen::Vector3d> angle_axis);

Eigen::Vector3d extrinsic_to_position(CameraExtrinsic& extrinsic);

double angle_between(const Eigen::Ref<const Eigen::Vector3d>& v1, const Eigen::Ref<const Eigen::Vector3d>& v2);
double clamp(double v, double min, double max);