#include <CGAL/Delaunay_triangulation_3.h>
#include <Eigen/Core>
#include <numeric>

#include "connect_points/delaunay.h"


void insert_points_into_delaunay(const Eigen::MatrixXd& V, Delaunay& dt)
{
    std::vector<Point> points;
    for (int i = 0; i < V.rows(); ++i)
    {
        points.emplace_back(V(i,0), V(i,1), V(i,2)); // x,y,z
    }
    dt.insert(points.begin(), points.end());
}

void extract_edges_from_delaunay(
    const Delaunay& dt,
    const Eigen::MatrixXd& V,
    Eigen::MatrixXi& E_dt
) {
    // 构建 Point → 索引 映射（V 中的每个点）
    std::map<Point, int> point_to_index;
    for (int i = 0; i < V.rows(); ++i) {
        point_to_index[Point(V(i, 0), V(i, 1), V(i, 2))] = i;
    }

    // 提取边
    std::vector<Eigen::Vector2i> edge_list;
    std::set<std::pair<int, int>> unique_edges;

    for (auto e = dt.finite_edges_begin(); e != dt.finite_edges_end(); ++e) {
        auto segment = dt.segment(*e);
        Point p1 = segment.point(0);
        Point p2 = segment.point(1);

        // 使用映射查找对应索引
        if (point_to_index.count(p1) && point_to_index.count(p2)) {
            int idx1 = point_to_index[p1];
            int idx2 = point_to_index[p2];
  
            // Undirected edge: use (min, max) to avoid AB/BA duplicates
            int a = std::min(idx1, idx2);
            int b = std::max(idx1, idx2);

            if (!unique_edges.count({a, b})) {
                unique_edges.insert({a, b});
                edge_list.emplace_back(a, b);
            }
        }
    }

    // 输出边集
    E_dt.resize(edge_list.size(), 2);
    for (int i = 0; i < edge_list.size(); ++i) {
        E_dt.row(i) = edge_list[i];
    }
}