#pragma once
#include <nanoflann.hpp>
#include <Eigen/Core>
#include <vector>

using namespace nanoflann;

/// A wrapper around an Eigen::MatrixXd to make it accessible to nanoflann
struct PointCloudAdaptor {
    const Eigen::MatrixXd &pts;

    PointCloudAdaptor(const Eigen::MatrixXd &points) : pts(points) {}

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return pts.rows();
    }

    // Returns the dim'th component of the idx'th point in the class
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts(idx, dim);
    }

    // Optional bounding-box computation: return false to default to a standard bbox
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const {
        return false;
    }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
    PointCloudAdaptor, 3>; // 3D points

/// Finds the `k` nearest neighbors from `points` for each point in `query`
/// Outputs an NxK matrix of indices (rows = query points)
inline Eigen::MatrixXi knn_search_nanoflann(const Eigen::MatrixXd &points, const Eigen::MatrixXd &query, int k) {
    PointCloudAdaptor adaptor(points);
    KDTree index(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    Eigen::MatrixXi indices(query.rows(), k);
    std::vector<size_t> ret_index(k);
    std::vector<double> out_dist_sqr(k);

    for (int i = 0; i < query.rows(); ++i) {
        nanoflann::KNNResultSet<double> resultSet(k);
        resultSet.init(ret_index.data(), out_dist_sqr.data());
        // index.findNeighbors(resultSet, query.row(i).data(), nanoflann::SearchParams());
        index.findNeighbors(resultSet, query.row(i).data(), 10);  // 10 is the number of checks


        for (int j = 0; j < k; ++j) {
            indices(i, j) = static_cast<int>(ret_index[j]);
        }
    }

    return indices;
}
