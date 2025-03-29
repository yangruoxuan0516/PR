// params.h
#ifndef PARAMS_H
#define PARAMS_H

struct GUIParams {

    int snake_iteration_num = 100;
    float snake_step = 0.001f;
    int snake_resample_num = 50;
    float weight_elastic = 1.0f;
    float weight_curvature = 1.0f;
    float weight_attraction = 10.0f;

};

#endif // PARAMS_H
