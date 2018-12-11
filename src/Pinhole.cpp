#include "Pinhole.hpp"

ceres::CostFunction * Pinhole::create_cost_function(Eigen::Vector2d& observation)
{
	return Pinhole::Cost::Create(observation);
}

Eigen::Vector2d Pinhole::image_to_world(Eigen::Vector2d & img_point)
{
	return Eigen::Vector2d();
}

Eigen::Vector2d Pinhole::world_to_image(Eigen::Vector2d & world_point)
{
	return Eigen::Vector2d();
}
