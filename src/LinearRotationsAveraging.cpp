#include "LinearRotationsAveraging.h"
#include "Utils.h"

using namespace std;
using namespace Eigen;
using namespace boost;

bool LinearRotationsAveraging::estimate(std::shared_ptr<FeaturePool> feature_pool, std::shared_ptr<MatchPool> match_pool, ViewCameraBinder & view_camera_binder, EpipolarGraph & epi_graph)
{
	size_t num_vertex = num_vertices(epi_graph);

	MatrixXd G = MatrixXd::Identity(3 * num_vertex, 3 * num_vertex);
	vector<double> d(num_vertex, 1.0);

	EpipolarGraph::edge_iterator e_begin, e_end;
	tie(e_begin, e_end) = edges(epi_graph);
	for (auto it_e = e_begin; it_e != e_end; ++it_e)
	{
		size_t i = it_e->m_source;
		size_t j = it_e->m_target;
		if (i > j)	//�� i > j �򽻻�
		{
			i = i^j;
			j = i^j;
			i = i^j;
		}

		RelativeTransform* transform = (RelativeTransform*)it_e->m_eproperty;
		double w = transform->weight;

		Matrix3d R;
		angle_axis_to_rotation_matrix(transform->rt.topRows(3), R);
		/*if (epi_graph[i].view_id == transform->src_id)
		{
			R.transposeInPlace();
		}*/
		// �ۼ�Ȩ��
		d[i] += w;
		d[j] += w;

		// G�ǶԳƾ���ֻ�����������ǲ���
		G.block(i * 3, j * 3, 3, 3).noalias() = w * R;
	}

	// ��ȡD�������
	MatrixXd D_inv = MatrixXd::Identity(3 * num_vertex, 3 * num_vertex);
	for (size_t i = 0; i < num_vertex; ++i)
	{
		D_inv.block(i * 3, i * 3, 3, 3) /= d[i];
	}

	// �� D^(-1) * G ����������������ֵ
	EigenSolver<MatrixXd> eigen_solver(D_inv * G.selfadjointView<Upper>(), ComputeEigenvectors);
	MatrixXd eigen_vectors = eigen_solver.eigenvectors().real();
	VectorXd eigen_values = eigen_solver.eigenvalues().real();

	// ������ֵ��������������
	vector<size_t> sort_idx(eigen_values.size());
	for (size_t i = 0; i < sort_idx.size(); ++i)
	{
		sort_idx[i] = i;
	}
	std::sort(sort_idx.begin(), sort_idx.end(), [&eigen_values](size_t a, size_t b) { return eigen_values[a] > eigen_values[b]; });

	// ȡ�������ֵ��Ӧ��������������
	Matrix<double, -1, 3> rotations(eigen_vectors.rows(), 3);
	rotations.col(0) = eigen_vectors.col(sort_idx[0]);
	rotations.col(1) = eigen_vectors.col(sort_idx[1]);
	rotations.col(2) = eigen_vectors.col(sort_idx[2]);

	// ��ȡ��һ���������ת�任����Ϊ��׼
	// Ѱ�ҵ�һ������Ⱥ���
	size_t r0;
	for (size_t i = 0; i < num_vertex; ++i)
	{
		r0 = i;
		if (epi_graph[i].is_inlier)
			break;
	}
	if (r0 == num_vertex) return false;

	JacobiSVD<Matrix<double, 3, 3>> svd(rotations.middleRows(r0*3, 3), ComputeFullU | ComputeFullV);
	Matrix3d base_R = svd.matrixU() * svd.matrixV().transpose();
	base_R.transposeInPlace();
	if (base_R.determinant() < 0)
		base_R *= -1;

	// ��ͼ�еĵ�һ������Ⱥ�����Ϊ��׼����������ϵ������Ӧ����ת�任Ϊ��λ��
	epi_graph[r0].rt.topRows(3) = Vector3d::Zero();
	Matrix3d global_R;
	for (size_t i = r0 + 1; i < num_vertex; ++i)
	{
		if (!epi_graph[i].is_inlier) continue;

		svd.compute(rotations.middleRows(i * 3, 3), ComputeFullU | ComputeFullV);
		global_R = svd.matrixU() * svd.matrixV().transpose() * base_R;

		//��֤��ת���������ʽ������һ
		if (global_R.determinant() < 0)
			global_R *= -1;

		rotation_matrix_to_angle_axis(global_R, epi_graph[i].rt.topRows(3));
	}

	return true;
}
