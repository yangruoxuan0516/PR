#include "find_hierarchy_with_root.h"

int find_root(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E) {
    int root;

    // find all the ending points in the tree
    std::vector<int> end_points;
    std::vector<int> degree(V.rows(), 0);
    for (int i = 0; i < E.rows(); ++i) {
        degree[E(i, 0)]++;
        degree[E(i, 1)]++;
    }

    for (int i = 0; i < degree.size(); ++i) {
        if (degree[i] == 1) {
            end_points.push_back(i);
        }
    }

    // calculate the distance from each end point to its first split, by adding the distance to the root
    std::vector<double> distances(end_points.size(), 0);
    for (int i = 0; i < end_points.size(); ++i) {
        int idx = end_points[i];

        // create a list to store the visited points
        std::vector<bool> visited(V.rows(), false);
        visited[idx] = true;

        // find the edge that connects to this end point
        for (int j = 0; j < E.rows(); ++j) {
            if (E(j, 0) == idx || E(j, 1) == idx) {
                visited[E(j, 0)] = true;
                visited[E(j, 1)] = true;
                idx = E(j, 0) == idx ? E(j, 1) : E(j, 0);
                break;
            }
        }
        // add the distance to the root
        distances[i] = (V.row(end_points[i]) - V.row(idx)).norm();
        // find the next edges
        while (degree[idx] == 2) {
            for (int j = 0; j < E.rows(); ++j) {
                if ( (E(j, 0) == idx && !visited[E(j, 1)]) || (E(j, 1) == idx && !visited[E(j, 0)]) ) {
                    visited[E(j, 0)] = true;
                    visited[E(j, 1)] = true;
                    idx = E(j, 0) == idx ? E(j, 1) : E(j, 0);
                    break;
                }
            }
            distances[i] += (V.row(end_points[i]) - V.row(idx)).norm();
        }
    }

    // find the end point with the maximum distance
    double max_distance = 0;
    for (int i = 0; i < distances.size(); ++i) {
        if (distances[i] > max_distance) {
            max_distance = distances[i];
            root = end_points[i];
        }
    }

    return root;
}