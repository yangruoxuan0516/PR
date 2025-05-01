#ifndef DIJKSTRA_H
#define DIJKSTRA_H

#include <vector>
#include <queue>
#include <limits>
#include <iostream>
#include <Eigen/Dense>



std::vector<std::vector<std::pair<int, double>>> build_graph_from_edges(
    const Eigen::MatrixXd& V_dt,
    const Eigen::MatrixXi& E_dt
);

int find_closest_point(const Eigen::MatrixXd& V, const Eigen::RowVectorXd& query_point);


Eigen::MatrixXi dijkstra_edges(
    int n,
    const std::vector<std::vector<std::pair<int, double>>>& graph,
    int start,
    int end
);


#endif