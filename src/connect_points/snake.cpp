#include <iostream>
#include "connect_points/snake.h"
#include "params.h"


// Computes the cumulative arc length at each point
Eigen::VectorXd compute_arc_length(const Eigen::MatrixXd &V) {
    Eigen::VectorXd L(V.rows());
    L(0) = 0.0;
    for (int i = 1; i < V.rows(); ++i)
        L(i) = L(i-1) + (V.row(i) - V.row(i-1)).norm();
    return L;
}

// Uniformly resample a polyline
Eigen::MatrixXd resample_polyline(const Eigen::MatrixXd &V, int num_samples) {
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



// Function to compute energy for snake optimization
double compute_energy(GUIParams& params, const Eigen::MatrixXd &curve, const Eigen::MatrixXd &points, int i, const Eigen::MatrixXi& knn_indices) {
    double energy = 0.0;
    double elastic_energy = 0.0;
    double curvature_energy = 0.0;
    double attraction_energy = 0.0;

    double weight_elastic = 1.0;
    double weight_curvature = 100.0;
    double weight_attraction = 1.0;
    
    // Elasticity: Prefer shorter edges
    if (i > 0) {
        elastic_energy += (curve.row(i) - curve.row(i - 1)).squaredNorm();
    }
    if (i < curve.rows() - 1) {
        elastic_energy += (curve.row(i) - curve.row(i + 1)).squaredNorm();
    }

    // print the elastic energy
    // std::cout << "Elastic energy: " << elastic_energy << std::endl;
    
    // Curvature: Minimize sharp bends
    if (i > 0 && i < curve.rows() - 1) {
        Eigen::Vector3d v1 = curve.row(i) - curve.row(i - 1);
        Eigen::Vector3d v2 = curve.row(i + 1) - curve.row(i);
        curvature_energy += (v1 - v2).squaredNorm();
    }

    // print the curvature energy
    // std::cout << "Curvature energy: " << curvature_energy << std::endl;
    
    // Attraction: Move towards closest vessel point
    double min_dist = std::numeric_limits<double>::max();

    for (int j = 0; j < knn_indices.cols(); ++j) {
        int idx = knn_indices(i, j);
        double dist = (curve.row(i) - points.row(idx)).squaredNorm();
        if (dist < min_dist) {
            min_dist = dist;
        }
    }
    attraction_energy += min_dist;

    // print the attraction energy
    // std::cout << "Attraction energy: " << attraction_energy << std::endl;

    energy = params.weight_elastic * elastic_energy + params.weight_curvature * curvature_energy + params.weight_attraction * attraction_energy;

    return energy;
}


// Function to optimize the snake curve
void optimize_snake(GUIParams& params, Eigen::MatrixXd &resampled_points, const Eigen::MatrixXd &points) {
    for (int iter = 0; iter < params.snake_iteration_num; ++iter) {

        Eigen::MatrixXd new_curve = resample_polyline(resampled_points, params.snake_resample_num);

        int k = 10;
        Eigen::MatrixXi knn_indices = knn_search_nanoflann(points, new_curve, k);

        for (int i = 1; i < resampled_points.rows() - 1; ++i) { // Exclude endpoints
            Eigen::Vector3d gradient(0, 0, 0);

            for (int dim = 0; dim < 3; ++dim) {
                Eigen::MatrixXd test_curve = new_curve;
                
                // Small perturbation
                test_curve(i, dim) += 0.01;

                // std::cout<<"in iteration "<<iter<<" and in point "<<i<<" and in dim "<<dim<<std::endl;
                double energy_plus = compute_energy(params, test_curve, points, i, knn_indices);
                
                test_curve(i, dim) -= 0.02;
                double energy_minus = compute_energy(params, test_curve, points, i, knn_indices);

                gradient(dim) = (energy_plus - energy_minus) / 0.02;
            }

            new_curve.row(i) -= params.snake_step * gradient.transpose(); // Gradient descent
        }
        resampled_points = new_curve;
    }
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> snake(GUIParams& params, const Eigen::MatrixXd& points, const Eigen::RowVectorXd start_point, const Eigen::RowVectorXd end_point) {

    // Resample points
    Eigen::MatrixXd initial_line = Eigen::MatrixXd::Zero(2, 3);
    initial_line.row(0) = start_point;
    initial_line.row(1) = end_point;

    Eigen::MatrixXd resampled_points = resample_polyline(initial_line, params.snake_resample_num);

    // Optimize snake
    optimize_snake(params, resampled_points, points);

    Eigen::MatrixXi E(resampled_points.rows() - 1, 2);
    for (int i = 0; i < resampled_points.rows() - 1; i++) {
        E(i, 0) = i;
        E(i, 1) = i + 1;
    }

    return std::make_tuple(resampled_points, E);
}