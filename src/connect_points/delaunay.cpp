#include <CGAL/Delaunay_triangulation_3.h>
#include <Eigen/Core>
#include <numeric>

#include "delaunay.h"


void insert_points_into_delaunay(const Eigen::MatrixXd& V, Delaunay& dt)
{
    std::vector<Point> points;
    for (int i = 0; i < V.rows(); ++i)
    {
        points.emplace_back(V(i,0), V(i,1), V(i,2)); // x,y,z
    }
    dt.insert(points.begin(), points.end());
}

void extract_edges_from_delaunay(const Delaunay& dt, Eigen::MatrixXd& V_edges, Eigen::MatrixXi& E_edges)
{
    std::vector<Eigen::RowVector3d> points;
    std::vector<Eigen::Vector2i> edges;

    for(auto e = dt.finite_edges_begin(); e != dt.finite_edges_end(); ++e)
    {
        auto segment = dt.segment(*e);
        Point p1 = segment.point(0);
        Point p2 = segment.point(1);

        // 保存端点坐标
        points.push_back(Eigen::RowVector3d(p1.x(), p1.y(), p1.z()));
        points.push_back(Eigen::RowVector3d(p2.x(), p2.y(), p2.z()));

        int idx1 = points.size() - 2;
        int idx2 = points.size() - 1;

        // 保存边 (每条边用2个点）
        edges.push_back(Eigen::Vector2i(idx1, idx2));
    }

    // 转成Eigen矩阵
    V_edges.resize(points.size(), 3);
    for (int i = 0; i < points.size(); ++i)
    {
        V_edges.row(i) = points[i];
    }

    E_edges.resize(edges.size(), 2);
    for (int i = 0; i < edges.size(); ++i)
    {
        E_edges.row(i) = edges[i];
    }
}