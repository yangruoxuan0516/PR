#ifndef NEAREST_NEIGHBOR_H
#define NEAREST_NEIGHBOR_H


#include <vector>
#include <Eigen/Core>

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> nearest_neighbor(const Eigen::MatrixXd& points);

#endif // NEAREST_NEIGHBOR_H