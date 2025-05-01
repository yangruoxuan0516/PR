#ifndef MST_H
#define MST_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

// 最小生成树（Kruskal）：从 Delaunay 边集提取 MST
Eigen::MatrixXi extract_mst_from_delaunay(
    const Eigen::MatrixXd& V,            // 点集
    const Eigen::MatrixXi& E_delaunay    // Delaunay 边集（索引）
);

#endif // MST_H