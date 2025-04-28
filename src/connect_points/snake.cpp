#include <iostream>
#include "connect_points/snake.h"
#include "params.h"

static int current_iter = 0;
static int current_i = 1;
static Eigen::MatrixXd new_curve;
static Eigen::MatrixXd resampled_points_backup;
static Eigen::MatrixXd points_backup;
static igl::opengl::glfw::Viewer* viewer_backup = nullptr;
static GUIParams* params_backup = nullptr;
static bool snake_running = false;
static Eigen::RowVector3d color_backup;

// Compute cumulative arc length
Eigen::VectorXd compute_arc_length(const Eigen::MatrixXd& V) {
    Eigen::VectorXd L(V.rows());
    L(0) = 0.0;
    for (int i = 1; i < V.rows(); ++i)
        L(i) = L(i-1) + (V.row(i) - V.row(i-1)).norm();
    return L;
}

// Uniform resampling
Eigen::MatrixXd resample_polyline(const Eigen::MatrixXd& V, int num_samples) {
    Eigen::VectorXd arc_length = compute_arc_length(V);
    double total_length = arc_length(arc_length.size() - 1);
    Eigen::MatrixXd V_resampled(num_samples, 3);
    for (int i = 0, j = 0; i < num_samples; ++i) {
        double target = i * total_length / (num_samples - 1);
        while (j < arc_length.size()-2 && arc_length(j+1) < target) ++j;
        double t = (target - arc_length(j)) / (arc_length(j+1) - arc_length(j));
        V_resampled.row(i) = (1 - t) * V.row(j) + t * V.row(j+1);
    }
    return V_resampled;
}

// Compute snake energy
double compute_energy(GUIParams& params, const Eigen::MatrixXd& curve, const Eigen::MatrixXd& points, int i, const Eigen::MatrixXi& knn_indices) {
    double elastic_energy = 0.0, curvature_energy = 0.0, attraction_energy = 0.0;

    if (i > 0) elastic_energy += (curve.row(i) - curve.row(i - 1)).squaredNorm();
    if (i < curve.rows() - 1) elastic_energy += (curve.row(i) - curve.row(i + 1)).squaredNorm();

    if (i > 0 && i < curve.rows() - 1) {
        Eigen::Vector3d v1 = curve.row(i) - curve.row(i - 1);
        Eigen::Vector3d v2 = curve.row(i + 1) - curve.row(i);
        curvature_energy += (v1 - v2).squaredNorm();
    }

    double min_dist = std::numeric_limits<double>::max();
    for (int j = 0; j < knn_indices.cols(); ++j) {
        int idx = knn_indices(0, j);
        double dist = (curve.row(i) - points.row(idx)).squaredNorm();
        if (dist < min_dist) min_dist = dist;
    }
    attraction_energy += min_dist;

    return params.weight_elastic * elastic_energy +
           params.weight_curvature * curvature_energy +
           params.weight_attraction * attraction_energy;
}

// Step-by-step optimizer
void optimize_snake(GUIParams& params, Eigen::MatrixXd& resampled_points, const Eigen::MatrixXd& points, igl::opengl::glfw::Viewer& viewer, const Eigen::RowVector3d& color) {
    current_iter = 0;
    current_i = 1;
    new_curve = resample_polyline(resampled_points, params.snake_resample_num);
    resampled_points_backup = resampled_points;
    points_backup = points;
    viewer_backup = &viewer;
    params_backup = &params;
    color_backup = color;
    snake_running = true;
}

// Perform one optimization step per Enter key
void optimize_snake_step() {
    if (!snake_running) return;
    
    if (current_iter >= params_backup->snake_iteration_num) {
        snake_running = false;
        viewer_backup->data_list[1].clear();
        viewer_backup->data_list[1].dirty |= igl::opengl::MeshGL::DIRTY_ALL;
        snake_running = false;
        return;
    }

    if (current_i >= resampled_points_backup.rows() - 1) {
        resampled_points_backup = new_curve;
        new_curve = resample_polyline(resampled_points_backup, params_backup->snake_resample_num);
        current_i = 1;
        current_iter++;
        if (current_iter >= params_backup->snake_iteration_num) {
            return;
        }
    }

    int k = 10;
    Eigen::MatrixXi knn_indices = knn_search_nanoflann(points_backup, new_curve.row(current_i), k);

    viewer_backup->data_list[1].clear();
    viewer_backup->data_list[2].clear();
    viewer_backup->data_list[1].point_size = 11;

    for (int j = 0; j < knn_indices.cols(); ++j) {
        int idx = knn_indices(0, j);
        viewer_backup->data_list[1].add_points(points_backup.row(idx), Eigen::RowVector3d(1.0, 0.0, 0.0));
    }

    viewer_backup->data_list[2].add_edges(
        new_curve.topRows(new_curve.rows()-1),
        new_curve.bottomRows(new_curve.rows()-1),
        color_backup);

        viewer_backup->data_list[1].dirty |= igl::opengl::MeshGL::DIRTY_ALL;
        viewer_backup->data_list[2].dirty |= igl::opengl::MeshGL::DIRTY_ALL;

    // Gradient descent step
    Eigen::Vector3d gradient(0, 0, 0);
    for (int dim = 0; dim < 3; ++dim) {
        Eigen::MatrixXd test_curve = new_curve;
        test_curve(current_i, dim) += 0.01;
        double energy_plus = compute_energy(*params_backup, test_curve, points_backup, current_i, knn_indices);

        test_curve(current_i, dim) -= 0.02;
        double energy_minus = compute_energy(*params_backup, test_curve, points_backup, current_i, knn_indices);

        gradient(dim) = (energy_plus - energy_minus) / 0.02;
    }

    new_curve.row(current_i) -= params_backup->snake_step * gradient.transpose();

    current_i++;
}

// Snake interface
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> snake(GUIParams& params, const Eigen::MatrixXd& points, const Eigen::RowVectorXd start_point, const Eigen::RowVectorXd end_point, igl::opengl::glfw::Viewer& viewer, const Eigen::RowVector3d& color) {
    Eigen::MatrixXd initial_line(2, 3);
    initial_line.row(0) = start_point;
    initial_line.row(1) = end_point;

    Eigen::MatrixXd resampled_points = resample_polyline(initial_line, params.snake_resample_num);
    optimize_snake(params, resampled_points, points, viewer, color);

    Eigen::MatrixXi E(resampled_points.rows() - 1, 2);
    for (int i = 0; i < resampled_points.rows() - 1; i++) {
        E(i, 0) = i;
        E(i, 1) = i + 1;
    }

    return std::make_tuple(resampled_points, E);
}
