#ifndef SEPERATE_INTO_COMPONENTS_H
#define SEPERATE_INTO_COMPONENTS_H
#include <queue>
#include <vector>
#include <Eigen/Dense>

std::vector<std::vector<int>> seperate_into_components(const Eigen::MatrixXd& V, double radius);

#endif // SEPERATE_INTO_COMPONENTS_H