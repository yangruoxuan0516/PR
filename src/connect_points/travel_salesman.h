#ifndef TRAVEL_SALESMAN_H
#define TRAVEL_SALESMAN_H

#include <vector>
#include <Eigen/Core>

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> travel_salesman(const Eigen::MatrixXd& points);

#endif // TRAVEL_SALESMAN_H