#include "Utils.h"
#include "CeresBA.h"
#include "Triangulator.h"
#include <ceres\rotation.h>

using namespace std;
using namespace Eigen;
using namespace cv;

int count_bits(boost::multiprecision::int128_t n)
{
	int count = 0;
	const unsigned char* pn = (const unsigned char*)&n;
	for (int i = 0; i < 16; ++i)
	{
		count += g_bits_per_byte[*pn];
		++pn;
	}
	return count;
}

CameraExtrinsic relative_transform_between(CameraExtrinsic & src, CameraExtrinsic & dst)
{
	Matrix3d R1, R2, R;
	angle_axis_to_rotation_matrix(src.topRows(3), R1);
	angle_axis_to_rotation_matrix(dst.topRows(3), R2);

	R1.transposeInPlace();
	R = R2 * R1;
	Vector3d rotated_global_src = R * src.bottomRows(3);

	CameraExtrinsic relative;
	rotation_matrix_to_angle_axis(R, relative);
	relative.bottomRows(3) = dst.bottomRows(3) - rotated_global_src;

	return relative;
}

CameraExtrinsic reversed_transform(CameraExtrinsic & rt)
{
	Matrix3d R;
	angle_axis_to_rotation_matrix(rt.topRows(3), R);
	R.transposeInPlace();

	CameraExtrinsic reversed_rt;
	rotation_matrix_to_angle_axis(R, reversed_rt.topRows(3));
	reversed_rt.bottomRows(3).noalias() = -R * rt.bottomRows(3);

	return reversed_rt;
}

inline Eigen::Matrix3d cross_mat(const Ref<const Eigen::Vector3d>& x)
{
	Matrix3d cross;
	cross <<
		0, -x[2], x[1],
		x[2], 0, -x[0],
		-x[1], x[0], 0;
	return std::move(cross);
}

inline cv::Matx33d cross_mat(cv::Matx31d & x)
{
	return cv::Matx33d(
		0, -x.val[2], x.val[1],
		x.val[2], 0, -x.val[0],
		-x.val[1], x.val[0], 0
	);
}

void kron(const Eigen::Ref<const Eigen::MatrixXd>& a, const Eigen::Ref<const Eigen::MatrixXd>& b, Eigen::Ref<Eigen::MatrixXd> k)
{
	const size_t a_rows = a.rows();
	const size_t a_cols = a.cols();
	const size_t b_rows = b.rows();
	const size_t b_cols = b.cols();

	size_t rows = a_rows * b_rows;
	size_t cols = a_cols * b_cols;

	k.resize(rows, cols);
	for (size_t r = 0; r < a_rows; ++r)
	{
		size_t row_start = r*b_rows;
		for (size_t c = 0; c < a_cols; ++c)
		{
			k.block(row_start, c*b_cols, b_rows, b_cols) = a(r, c)*b;
		}
	}
}

Eigen::Vector3d triangulate_DLT(
	const Ref<const Eigen::Vector3d>& x1, 
	const Ref<const Eigen::Vector3d>& x2, 
	const Ref<const Eigen::Matrix3d>& R, 
	const Ref<const Eigen::Vector3d>& t
)
{
	Matrix<double, 6, 4> A;
	Matrix<double, 3, 4> P1;

	P1.block(0, 0, 3, 3) = R;
	P1.col(3) = t;

	A.topRows(3).noalias() = cross_mat(x1) * Matrix<double, 3, 4>::Identity();;
	A.bottomRows(3).noalias() = cross_mat(x2) * P1;

	JacobiSVD<Matrix<double, 6, 4>> svd(A, ComputeFullV);
	Vector4d _X = svd.matrixV().rightCols(1);	// V的最后一列对应最小特征向量

	return _X.topRows(3) / _X[3];
}

double calc_reproj_error(Scene & scene)
{
	Matrix3d R;
	Matrix<double, 3, 4> P;
	Vector3d x;
	auto& views = scene.views;
	JacobiSVD<Matrix<double, -1, 4>> svd;

	double reproj_error = 0;
	size_t reproj_count = 0;
	for (auto& track : scene.tracks)
	{
		if (!track.is_inlier) continue;

		Vector3d X = track.point3d;
		auto& nodes = track.nodes;

		// compute reprojected error
		for (size_t n = 0; n < nodes.size(); ++n)
		{
			auto& view = views[scene.view_idx_by_view_id[nodes[n].view_id]];
			auto& camera = scene.view_camera_binder.get_camera(view.view_id);

			angle_axis_to_rotation_matrix(view.rt.topRows(3), R);
			Vector3d _X = R * X + view.rt.bottomRows(3);
			Vector2d reproj_x = camera->world_to_image(_X);
			reproj_error += (reproj_x - nodes[n].observation).norm();
		}
		reproj_count += nodes.size();
	}

	return reproj_error / reproj_count;
}

double rad2deg(double radius)
{
	return radius / CV_PI * 180.0;
}

double deg2rad(double degree)
{
	return degree / 180 * CV_PI;
}

void rotation_matrix_to_angle_axis(const Eigen::Ref<const Eigen::Matrix3d>& matrix, Eigen::Ref<Eigen::Vector3d> angle_axis)
{
	AngleAxisd a(matrix);
	angle_axis = a.axis() * a.angle();
}

void angle_axis_to_rotation_matrix(const Eigen::Ref<const Eigen::Vector3d>& angle_axis, Eigen::Ref<Eigen::Matrix3d> matrix)
{
	double angle = angle_axis.norm();
	Vector3d axis;
	if (abs(angle) < DBL_EPSILON)
		axis = Vector3d::UnitX();
	else
		axis = angle_axis / angle;

	AngleAxisd a(angle, axis);
	matrix = a.toRotationMatrix();
}

void angle_axis_to_quaternion(const Eigen::Ref<const Eigen::Vector3d>& angle_axis, Eigen::Quaterniond & q)
{
	double angle = angle_axis.norm();
	Vector3d axis;
	if (abs(angle) < DBL_EPSILON)
		axis = Vector3d::UnitX();
	else
		axis = angle_axis / angle;

	AngleAxisd a(angle, axis);
	q = Quaterniond(a);
}

void quaternion_to_angle_axis(Eigen::Quaterniond & q, Eigen::Ref<Eigen::Vector3d> angle_axis)
{
	AngleAxisd a(q);
	angle_axis = a.axis() * a.angle();
}

Eigen::Vector3d extrinsic_to_position(CameraExtrinsic & extrinsic)
{
	Quaterniond q;
	angle_axis_to_quaternion(extrinsic.topRows(3), q);
	return -(q.conjugate() * extrinsic.bottomRows(3));
}

double angle_between(const Eigen::Ref<const Eigen::Vector3d>& v1, const Eigen::Ref<const Eigen::Vector3d>& v2)
{
	return acos(clamp(v1.dot(v2) / (v1.norm() * v2.norm()), -1.0 + 1e-8, 1.0 - 1e-8));
}

double clamp(double v, double minv, double maxv)
{
	return min(maxv, max(v, minv));
}
