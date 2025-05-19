#ifndef WEIGHTED_TREE_PARTITION_H
#define WEIGHTED_TREE_PARTITION_H

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <limits>

std::unordered_map<int, int> pointwise_partition_with_dijkstra(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E_mst,
    const std::unordered_map<int, int>& labeled_points
);


std::unordered_map<int, int> segment_based_partition_based_on_dijkstra(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E_mst,
    const std::unordered_map<int, int>& labeled_points,
    Eigen::MatrixXd* debug_colors = nullptr
);

#endif // WEIGHTED_TREE_PARTITION_H