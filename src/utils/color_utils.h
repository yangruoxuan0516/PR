#ifndef COLOR_UTILS_H
#define COLOR_UTILS_H

#include <Eigen/Dense>
#include <cmath>


inline Eigen::RowVector3d hsv2rgb(double h, double s, double v) {
    double c = v * s;
    double x = c * (1 - std::abs(fmod(h / 60.0, 2) - 1));
    double m = v - c;
    double r, g, b;

    if (h < 60)       r = c, g = x, b = 0;
    else if (h < 120) r = x, g = c, b = 0;
    else if (h < 180) r = 0, g = c, b = x;
    else if (h < 240) r = 0, g = x, b = c;
    else if (h < 300) r = x, g = 0, b = c;
    else              r = c, g = 0, b = x;

    return Eigen::RowVector3d(r + m, g + m, b + m);
}


inline Eigen::RowVector3d generate_distinct_color(int index, int total = 5) {
    double h = fmod((index * 360.0 / total), 360.0);  // 均匀分布在色相环上
    double s = 0.8;  // 高饱和
    double v = 0.9;  // 高亮度
    return hsv2rgb(h, s, v);
}

#endif // COLOR_UTILS_H