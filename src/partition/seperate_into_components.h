#ifndef SEPERATE_INTO_COMPONENTS_H
#define SEPERATE_INTO_COMPONENTS_H
#include <queue>
#include <vector>
#include <Eigen/Dense>

#include "connect_points/delaunay.h"
#include "connect_points/mst.h"

#include "utils/color_utils.h" // 假设有一个颜色生成函数

struct ComponentGraph {
    Eigen::MatrixXd V_sub;             // 局部点坐标
    Eigen::VectorXi global_indices;    // 每个 V_sub 点对应 V 中的索引
    Eigen::MatrixXi E_mst;             // 局部边（索引以 V_sub 为基准）
    Eigen::RowVector3d color;          // 本 component 的颜色
    int component_id;          // 本 component 的 ID（可选）
    bool selected = true;
};

void mark_selected_components(
    std::vector<ComponentGraph>& component_graphs,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& C,
    const std::vector<Eigen::RowVector3d>& type_colors,
    const std::vector<int>& point_to_component_id);

std::vector<std::vector<int>> seperate_into_components(const Eigen::MatrixXd& V, double radius);

std::vector<ComponentGraph> get_component_graphs(const Eigen::MatrixXd& V, double radius);

void component_graph_vertices(
    const std::vector<ComponentGraph>& component_graphs,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXd& C_out,
    bool show_selected_components_only
);

void component_graph_edges(
    const std::vector<ComponentGraph>& component_graphs,
    Eigen::MatrixXd& P1_out,
    Eigen::MatrixXd& P2_out,
    Eigen::MatrixXd& C_out,
    bool show_selected_components_only
);

#endif // SEPERATE_INTO_COMPONENTS_H