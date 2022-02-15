#pragma once
#ifndef MAT_H
#define MAT_H
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <math.h>
using namespace std;
using namespace std::chrono;

class conv_param
{
public:
    size_t pad;
    size_t stride;
    size_t kernel_size;
    size_t in_channels;
    size_t out_channels;
    float *p_weight;
    float *p_bias;
    conv_param(size_t pad, size_t stride, size_t kernel_size, size_t in_channels, size_t out_channels, float *p_weight, float *p_bias)
    {
        this->pad = pad;
        this->stride = stride;
        this->kernel_size = kernel_size;
        this->in_channels = in_channels;
        this->out_channels = out_channels;
        this->p_weight = p_weight;
        this->p_bias = p_bias;
    }
};

class fc_param
{
public:
    size_t in_features;
    size_t out_features;
    float *p_weight;
    float *p_bias;
    fc_param(size_t in_features, size_t out_features, float *p_weight, float *p_bias)
    {
        this->in_features = in_features;
        this->out_features = out_features;
        this->p_weight = p_weight;
        this->p_bias = p_bias;
    }
};

class Mat_conv
{
public:
    size_t row;
    size_t coloumn;
    size_t channel;
    int *counter;
    float *Matrix;

public:
    Mat_conv();
    Mat_conv(int row, int coloumn, int channel);
    Mat_conv(const Mat_conv &Mat);
    ~Mat_conv();
    bool conv(const Mat_conv &mat1, const conv_param &mat2, Mat_conv &Mat);
    bool dot(const Mat_conv &mat1, const conv_param &mat2, Mat_conv &matMul, int i);
    bool gemm(const Mat_conv &mat1,const Mat_conv &gmat1, const conv_param &mat2, Mat_conv &matMul,Mat_conv &gmatMul);
    bool setZero(int i, int j, size_t in_size, float (&t)[9], const Mat_conv &mat1, int flag1, size_t flag2, size_t flag3, size_t Ksize1);
    bool maxPool(const Mat_conv &mat, const Mat_conv &maxPool,float *(&bias));
    bool ReLu(const Mat_conv &mat);
    bool setBias(const Mat_conv &mat, float *(&bias));
    bool fullConnect(float *(&result), const Mat_conv &mat, fc_param &fullC);
    bool softMax(float *(&result), int size);
    bool CNN(const Mat_conv &mat, conv_param *(conv_params), fc_param &fc_params);
};

#endif
