#ifndef WEIGHTED_TREE_PARTITION_H
#define WEIGHTED_TREE_PARTITION_H

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <vector>
#include <Eigen/Dense>

struct QueueElement {
    double dist;
    int node;
    int class_id;

    bool operator>(const QueueElement& other) const {
        return dist > other.dist;
    }
};

std::unordered_map<int, int> weighted_tree_partition(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E_mst,
    const std::unordered_map<int, int>& labeled_points
);

#endif // WEIGHTED_TREE_PARTITION_H