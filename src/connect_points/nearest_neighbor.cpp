#include <iostream>
#include "connect_points/nearest_neighbor.h"


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> nearest_neighbor(const Eigen::MatrixXd& points) {
    
    int n = points.rows();

    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0));

    // Compute pairwise distances
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                dist[i][j] = (points.row(i) - points.row(j)).norm();
            }
        }
    }

    std::vector<Eigen::RowVector2i> edges;
    std::vector<bool> visited(n, false);
    visited[0] = true;
    for (int i = 0; i < n - 1; i++) {
        int u = -1, v = -1;
        double min_dist = std::numeric_limits<double>::infinity();
        for (int j = 0; j < n; j++) {
            if (visited[j]) {
                for (int k = 0; k < n; k++) {
                    if (!visited[k] && dist[j][k] < min_dist) {
                        min_dist = dist[j][k];
                        u = j;
                        v = k;
                    }
                }
            }
        }
        visited[v] = true;
        edges.push_back(Eigen::RowVector2i(u, v));
    }

    Eigen::MatrixXi E(edges.size(), 2);

    for (size_t i = 0; i < edges.size(); i++) {
        E.row(i) = edges[i];
    }

    return std::make_tuple(points,E);
}