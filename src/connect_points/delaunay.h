// delaunay.h
#ifndef DELAUNAY_UTILS_H
#define DELAUNAY_UTILS_H

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef K::Point_3 Point;

void insert_points_into_delaunay(const Eigen::MatrixXd& V, Delaunay& dt);
void extract_edges_from_delaunay(const Delaunay& dt, Eigen::MatrixXd& V_edges, Eigen::MatrixXi& E_edges);


#endif // DELAUNAY_H
