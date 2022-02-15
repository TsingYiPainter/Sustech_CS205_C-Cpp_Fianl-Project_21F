#include "Mat.hpp"
#include "stdlib.h"
#include <cblas.h>
#include <immintrin.h>
#include <chrono>
#include <cstring>
#include "sys/time.h"
#include "time.h"
#define coutError                              \
    cerr << "---------------------" << endl;   \
    cerr << "error! nullptr argument" << endl; \
    cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 4 << endl;

#define flagCheck                                                                                          \
    if (!flag)                                                                                             \
    {                                                                                                      \
        cerr << "---------------------" << endl;                                                           \
        cerr << "error! return value is false" << endl;                                                    \
        cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 1 << endl; \
        return false;                                                                                      \
    }

extern void sgemm_(char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);

Mat_conv::Mat_conv()
{
    this->row = 0;
    this->coloumn = 0;
    this->channel = 0;
    this->counter = new int[1]{1};
    this->Matrix = NULL;
}

Mat_conv::Mat_conv(int r, int c, int channel)
{
    if (r <= 0 || c <= 0 || channel <= 0)
    {
        cout << "---------------------" << endl;
        cerr << "error! constructor's argument are invalid" << endl;
        cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 4 << endl;
        cerr << "r,c,channel are"
             << "    " << r << " " << c << " " << channel << endl;
    }
    else
    {
        this->row = r;
        this->coloumn = c;
        this->channel = channel;
        this->counter = new int[1]{1};
        this->Matrix = new float[r * c * channel];
        for (int i = 0; i < r * c * channel; i++)
        {
            this->Matrix[i] = 0;
        }
    }
}

Mat_conv::Mat_conv(const Mat_conv &Mat)
{

    // cout<<"copy"<<endl;
    // cout<<"mat1 "<<hex<<(void*)(Mat.Matrix)<<endl;
    if (Mat.Matrix == NULL)
    {
        coutError;
    }
    else
    {
        this->row = Mat.row;
        this->coloumn = Mat.coloumn;
        this->channel = Mat.channel;
        this->counter = Mat.counter;
        (*this->counter)++;
        this->Matrix = Mat.Matrix;
    }
};

Mat_conv::~Mat_conv()
{
    if ((*this->counter) <= 1)
    { //若当前仅一个Mat类享有这份数据，则直接delete
        // cout << "class is release" << endl;
        delete this->counter;
        //cout<<"free------"<<hex<<(void*)this->Matrix<<endl;
        this->counter == NULL;
        if (this->Matrix != NULL)
        {
            delete[](this->Matrix);
            this->Matrix == NULL;
        }
    }
    else
    { //若当前有多个Mat类共享这份数据，则将共享的类的个数减一，表示其中一个类被删除了
        (*this->counter)--;
        //cout<<"minus------"<<hex<<(void*)this->Matrix<<endl;
    }
}

bool Mat_conv::conv(const Mat_conv &mat1, const conv_param &mat2, Mat_conv &matMul)
{
    if (mat1.Matrix == NULL || matMul.Matrix == NULL || mat2.p_weight == NULL || mat2.p_bias == NULL)
    {
        coutError;
        return false;
    }
    if (mat1.channel != mat2.in_channels || mat2.out_channels == 0)
    {
        cerr << "---------------------" << endl;
        cerr << "Mat(Tensor) channel is not equal to weight channel or out_channel=0" << endl;
        cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 11 << endl;
        cerr << "mat1.channel=" << mat1.channel << "  mat2.in_channel" << mat2.in_channels << "  mat2.out_channel=" << mat2.out_channels << endl;
        return false;
    }

    for (int i = 0; i < mat2.out_channels; i++)
    {
        bool flag = dot(mat1, mat2, matMul, i);
        flagCheck;
    }
    return true;
}

bool Mat_conv::dot(const Mat_conv &mat1, const conv_param &mat2, Mat_conv &matMul, int index_channel)
{
    if (mat1.Matrix == NULL || mat2.p_weight == NULL || mat2.p_bias == NULL || matMul.Matrix == NULL)
    {
        coutError;
        return false;
    }
    if (index_channel < 0)
    {
        cout << "---------------------" << endl;
        cerr << "error! index_channel<0" << endl;
        cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 4 << endl;
        return false;
    }
    size_t in_size = mat1.row;                                                       //输入矩阵的大小
    size_t in_channel = mat1.channel;                                                //输入矩阵的channel
    size_t out_size = (in_size - mat2.kernel_size + 2 * mat2.pad) / mat2.stride + 1; //输出矩阵的大小
    size_t out_channel = mat2.out_channels;                                          //输出矩阵的channel
    size_t size1 = mat1.row * mat1.coloumn;                                          //矩阵换channel时指针偏移量
    size_t size2 = mat2.kernel_size * mat2.kernel_size * in_channel;                 //权重矩阵换channel时指针偏移量
    size_t size3 = index_channel * out_size * out_size;
    // __m256 vector1;
    // __m256 vector2;
    // __m256 result;
    // size_t s;
    // if (mat2.pad)
    // {
    //     s = 0;
    // }
    // else
    // {
    //     s = 1;
    // }
    // for (int k = 0; k < in_channel; k++)
    // {
    //     vector2 = _mm256_loadu_ps(&mat2.p_weight[k * 9 + index_channel * size2]);
    //     int Mulcounter = 0;
    //     for (int i = s; i < in_size - s; i += mat2.stride)
    //     {
    //         int flag1 = (i - 1) * in_size; //cout<<"flag1="<<flag1<<endl;
    //         int flag2 = (i)*in_size;       //cout<<"flag2="<<flag2<<endl;
    //         int flag3 = (i + 1) * in_size; //cout<<"flag3="<<flag3<<endl;
    //         for (int j = s; j < in_size - s; j += mat2.stride)
    //         {

    //             float t[9]{0, 0, 0, 0, 0, 0, 0, 0, 0};
    //             if (mat2.pad && (i == 0 || i == in_size - 1 || j == 0 || j == in_size - 1))
    //             {
    //                 bool flag = setZero(i, j, in_size, t, mat1, flag1, flag2, flag3, k * size1);
    //                 flagCheck;
    //                 vector1 = _mm256_loadu_ps(&t[0]);
    //             }
    //             else
    //             {
    //                 int i1 = flag1 + j + k * size1;
    //                 int i2 = flag2 + j + k * size1;
    //                 int i3 = flag3 + j + k * size1;
    //                 vector1 = _mm256_set_ps(mat1.Matrix[i3], mat1.Matrix[i3 - 1], mat1.Matrix[i2 + 1],
    //                                         mat1.Matrix[i2], mat1.Matrix[i2 - 1], mat1.Matrix[i1 + 1], mat1.Matrix[i1], mat1.Matrix[i1 - 1]);
    //                 t[8] = mat1.Matrix[i3 + 1];
    //             }

    //             result = _mm256_mul_ps(vector1, vector2);
    //             float sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7] + t[8] * mat2.p_weight[k * 9 + index_channel * size2 + 8];
    //             matMul.Matrix[Mulcounter + size3] += sum;

    //             Mulcounter++;
    //         }
    //     }
    //}
    return true;
};

bool Mat_conv::gemm(const Mat_conv &mat1, const Mat_conv &gmat1, const conv_param &mat2, Mat_conv &matMul, Mat_conv &gmatMul)
{
    if (mat1.Matrix == NULL || mat2.p_weight == NULL || mat2.p_bias == NULL || matMul.Matrix == NULL)
    {
        coutError;
        return false;
    }
    size_t r = ((mat1.row - 3 + 2 * mat2.pad) / mat2.stride + 1);
    if (mat1.row != mat1.coloumn || r * r != gmat1.row || gmat1.coloumn != 3 * 3 * mat1.channel || gmatMul.row != r * r || gmatMul.coloumn != mat2.out_channels || matMul.row != r || matMul.coloumn != r || matMul.channel != mat2.out_channels)
    {
        cerr << "---------------------" << endl;
        cerr << "error! invalid argument list" << endl;
        cerr << "mat1 row col channel     " << mat1.row << " " << mat1.coloumn << " " << mat1.channel << endl;
        cerr << "gmat1 row col channel    " << gmat1.row << " " << gmat1.coloumn << " " << gmat1.channel << endl;
        cerr << "matMul row col channel   " << matMul.row << " " << matMul.coloumn << " " << matMul.channel << endl;
        cerr << "gmatMul row col channel  " << gmatMul.row << " " << gmatMul.coloumn << " " << gmatMul.channel << endl;
        cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 3 << endl;
        return false;
    }
    size_t in_size = mat1.row;                                                       //输入矩阵的大小
    size_t in_channel = mat1.channel;                                                //输入矩阵的channel
    size_t out_size = (in_size - mat2.kernel_size + 2 * mat2.pad) / mat2.stride + 1; //输出矩阵的大小
    size_t out_channel = mat2.out_channels;                                          //输出矩阵的channel
    size_t size1 = mat1.row * mat1.coloumn;                                          //矩阵换channel时指针偏移量
    size_t size2 = mat2.kernel_size * mat2.kernel_size * in_channel;                 //权重矩阵换channel时指针偏移量
    int mycounter = 0;
    size_t s;
    if (!mat2.pad)
    {
        s = 1;
    }
    else
    {
        s = 0;
    }

    for (int i = s; i < in_size - s; i += mat2.stride)
    {
        int flag1 = (i - 1) * in_size; //cout<<"flag1="<<flag1<<endl;
        int flag2 = (i)*in_size;       //cout<<"flag2="<<flag2<<endl;
        int flag3 = (i + 1) * in_size; //cout<<"flag3="<<flag3<<endl;
        for (int j = s; j < in_size - s; j += mat2.stride)
        {
            for (int k = 0; k < mat1.channel; k++)
            {
                float t[9]{0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (mat2.pad == 1 && (i == 0 || i == in_size - 1 || j == 0 || j == in_size - 1))
                {
                    bool flag = setZero(i, j, in_size, t, mat1, flag1, flag2, flag3, k * size1);
                    flagCheck;
                    memcpy(&gmat1.Matrix[mycounter], &t, sizeof(float) * 9);
                    mycounter += 9;
                }
                else
                {
                    int i1 = flag1 + j + k * size1;
                    int i2 = flag2 + j + k * size1;
                    int i3 = flag3 + j + k * size1;
                    memcpy(&gmat1.Matrix[mycounter], &mat1.Matrix[i1 - 1], sizeof(float) * 3);
                    mycounter += 3;
                    memcpy(&gmat1.Matrix[mycounter], &mat1.Matrix[i2 - 1], sizeof(float) * 3);
                    mycounter += 3;
                    memcpy(&gmat1.Matrix[mycounter], &mat1.Matrix[i3 - 1], sizeof(float) * 3);
                    mycounter += 3;
                }
            }
        }
    }
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, gmat1.row, out_channel, in_channel * 3 * 3, 1.0, gmat1.Matrix, gmat1.coloumn, mat2.p_weight, in_channel * 3 * 3, 1.0, gmatMul.Matrix, out_channel);

    float *B = new float[in_channel * 3 * 3 * out_channel];
    for (int j = 0; j < in_channel * 3 * 3; j++)
    {
        for (int i = 0; i < out_channel; i++)
        {
            B[j * out_channel + i] = mat2.p_weight[i * in_channel * 3 * 3 + j];
        }
    }
    __m256 vector1;
    __m256 vector2;
    __m256 result;
    size_t len1 = gmat1.coloumn;
    size_t len2 = out_channel;
    for (int i = 0; i < gmat1.row; i++)
    {
        size_t skip = i * len2;
        for (int k = 0; k < gmat1.coloumn; k++)
        {
           vector1 = _mm256_set1_ps(gmat1.Matrix[i * len1 + k]);
            if (out_channel >= 8)
            {
                for (int j = 0; j < out_channel; j += 8)
                {
                     vector2 = _mm256_loadu_ps(&B[k * len2 + j]);
                     result = _mm256_mul_ps(vector1, vector2);
                    gmatMul.Matrix[skip + j] += result[0];
                    gmatMul.Matrix[skip + j + 1] += result[1];
                    gmatMul.Matrix[skip + j + 2] += result[2];
                    gmatMul.Matrix[skip + j + 3] += result[3];
                    gmatMul.Matrix[skip + j + 4] += result[4];
                    gmatMul.Matrix[skip + j + 5] += result[5];
                    gmatMul.Matrix[skip + j + 6] += result[6];
                    gmatMul.Matrix[skip + j + 7] += result[7];
                }
            }
            for (int j = out_channel - out_channel % 8; j < out_channel; j++)
            {
                //对非八的整数倍规模的矩阵做兼容处理
                gmatMul.Matrix[i * len2 + j] += gmat1.Matrix[i * len1 + k] * B[k * len2 + j];
            }
            
            // for (int j = 0; j < out_channel; j++)
            // {
            //     gmatMul.Matrix[i * len2 + j] += gmat1.Matrix[i * len1 + k] * B[k * len2 + j];
            // }
        }
    }
     delete[] B;B=NULL;
     
    for (int j = 0; j < out_channel; j++)
    {
        for (int i = 0; i < out_size * out_size; i++)
        {
            matMul.Matrix[j * out_size * out_size + i] = gmatMul.Matrix[i * out_channel + j];
        }
    }
    
    return true;
};

bool Mat_conv::setZero(int i, int j, size_t in_size, float (&t)[9], const Mat_conv &mat1, int flag1, size_t flag2, size_t flag3, size_t Ksize1)
{
    if (i < 0 || j < 0)
    {
        cerr << "---------------------" << endl;
        cerr << "error! invalid i or j " << endl;
        cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 4 << endl;
        cerr << "i=" << i << "    j=" << j << endl;
        return false;
    }
    int i1 = flag1 + j + Ksize1;
    int i2 = flag2 + j + Ksize1;
    int i3 = flag3 + j + Ksize1;
    if (i == 0 && j == 0)
    {
        t[0] = 0;
        t[1] = 0;
        t[2] = 0;
        t[3] = 0;
        t[6] = 0;
        t[4] = mat1.Matrix[i2];
        t[5] = mat1.Matrix[i2 + 1];
        t[7] = mat1.Matrix[i3];
        t[8] = mat1.Matrix[i3 + 1];
    }
    else if (i == in_size - 1 && j == in_size - 1)
    {
        t[2] = 0;
        t[5] = 0;
        t[6] = 0;
        t[7] = 0;
        t[8] = 0;
        t[0] = mat1.Matrix[i1 - 1];
        t[1] = mat1.Matrix[i1];
        t[3] = mat1.Matrix[i2 - 1];
        t[4] = mat1.Matrix[i2];
    }
    else if (i == 0 && j == in_size - 1)
    {
        t[0] = 0;
        t[1] = 0;
        t[2] = 0;
        t[5] = 0;
        t[8] = 0;
        t[3] = mat1.Matrix[i2 - 1];
        t[4] = mat1.Matrix[i2];
        t[6] = mat1.Matrix[i3 - 1];
        t[7] = mat1.Matrix[i3];
    }
    else if (i == in_size - 1 && j == 0)
    {
        t[0] = 0;
        t[3] = 0;
        t[6] = 0;
        t[7] = 0;
        t[8] = 0;
        t[1] = mat1.Matrix[i1];
        t[2] = mat1.Matrix[i1 + 1];
        t[4] = mat1.Matrix[i2];
        t[5] = mat1.Matrix[i2 + 1];
    }
    else if (i == 0)
    {
        t[0] = 0;
        t[1] = 0;
        t[2] = 0;
        t[3] = mat1.Matrix[i2 - 1];
        t[4] = mat1.Matrix[i2];
        t[5] = mat1.Matrix[i2 + 1];
        t[6] = mat1.Matrix[i3 - 1];
        t[7] = mat1.Matrix[i3];
        t[8] = mat1.Matrix[i3 + 1];
    }
    else if (j == 0)
    {
        t[0] = 0;
        t[3] = 0;
        t[6] = 0;
        t[1] = mat1.Matrix[i1];
        t[2] = mat1.Matrix[i1 + 1];
        t[4] = mat1.Matrix[i2];
        t[5] = mat1.Matrix[i2 + 1];
        t[7] = mat1.Matrix[i3];
        t[8] = mat1.Matrix[i3 + 1];
    }
    else if (i == in_size - 1)
    {
        t[6] = 0;
        t[7] = 0;
        t[8] = 0;
        t[0] = mat1.Matrix[i1 - 1];
        t[1] = mat1.Matrix[i1];
        t[2] = mat1.Matrix[i1 + 1];
        t[3] = mat1.Matrix[i2 - 1];
        t[4] = mat1.Matrix[i2];
        t[5] = mat1.Matrix[i2 + 1];
    }
    else
    {
        t[2] = 0;
        t[5] = 0;
        t[8] = 0;
        t[0] = mat1.Matrix[i1 - 1];
        t[1] = mat1.Matrix[i1];
        t[3] = mat1.Matrix[i2 - 1];
        t[4] = mat1.Matrix[i2];
        t[6] = mat1.Matrix[i3 - 1];
        t[7] = mat1.Matrix[i3];
    }
    return true;
}

bool Mat_conv::setBias(const Mat_conv &mat, float *(&bias))
{
    if (mat.Matrix == NULL || bias == NULL)
    {
        coutError;
        return false;
    }
    for (int k = 0; k < mat.channel; k++)
    {
        int size = k * mat.coloumn * mat.row;
        for (int i = 0; i < mat.row; i++)
        {
            for (int j = 0; j < mat.coloumn; j++)
            {
                mat.Matrix[i * mat.coloumn + j + size] += bias[k];
            }
        }
    }
    return true;
}

bool Mat_conv::ReLu(const Mat_conv &mat)
{
    if (mat.Matrix == NULL)
    {
        coutError;
        return false;
    }
    for (int k = 0; k < mat.channel; k++)
    {
        int size = k * mat.coloumn * mat.row;
        for (int i = 0; i < mat.row; i++)
        {
            for (int j = 0; j < mat.coloumn; j++)
            {
                float f=mat.Matrix[i * mat.coloumn + j + size];
                mat.Matrix[i * mat.coloumn + j + size] = f>0?f:0;
            }
        }
    }
    return true;
}

bool Mat_conv::maxPool(const Mat_conv &mat, const Mat_conv &maxPool, float *(&bias))
{
    if (mat.Matrix == NULL || maxPool.Matrix == NULL)
    {
        coutError;
        return false;
    }
    if (mat.row / 2 != maxPool.row || mat.coloumn / 2 != maxPool.coloumn || mat.channel != maxPool.channel)
    {
        cerr << "---------------------" << endl;
        cerr << "error! invalid member of argument" << endl;
        cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 4 << endl;
        cerr << "mat.row  mat.col   mat.channel=" << mat.row << "  " << mat.coloumn << "  " << mat.channel << endl;
        cerr << "maxPool.row  maxPool.col   maxPool.channel=" << maxPool.row << "  " << maxPool.coloumn << "  " << maxPool.channel << endl;
        return false;
    }
    size_t row = mat.row;
    size_t col = mat.coloumn;
    size_t size = row * col;
    size_t channel = mat.channel;
    size_t mycounter = 0;
    float a, b, c, d;
    int i1, i2;

    for (int k = 0; k < channel; k++)
    {
        float adder = bias[k];
        for (int i = 0; i < row; i += 2)
        {
            for (int j = 0; j < col; j += 2)
            {
                i1 = i * row + j + k * size;
                i2 = (i + 1) * row + j + k * size;
                a = mat.Matrix[i1];
                b = mat.Matrix[i1 + 1];
                c = mat.Matrix[i2];
                d = mat.Matrix[i2 + 1];
                float f = max(max(a, b), max(c, d)) + adder;
                maxPool.Matrix[mycounter++] = f>0?f:0;
            }
        }
    }
    return true;
}

bool Mat_conv::fullConnect(float *(&result), const Mat_conv &mat, fc_param &fullC)
{
    if (result == NULL || mat.Matrix == NULL || fullC.p_weight == NULL || fullC.p_bias == NULL)
    {
        coutError;
        return false;
    }
    if (mat.row * mat.coloumn * mat.channel != fullC.in_features)
    {
        cerr << "---------------------" << endl;
        cerr << "error! invalid member of argument. " << endl;
        cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 4 << endl;
        cerr << "mat.row=" << mat.row << "   mat.col=" << mat.coloumn << "   mat.channel=" << mat.channel << "   in_feature=" << fullC.in_features << endl;
        return false;
    }
    size_t in_features = fullC.in_features;
    size_t out_features = fullC.out_features;
    for (int k = 0; k < out_features; k++)
    {
        int flag = in_features * k;
        for (int i = 0; i < in_features; i++)
        {
            result[k] += mat.Matrix[i] * fullC.p_weight[i + flag];
        }
        result[k] += fullC.p_bias[k];
    }
    return true;
};

bool Mat_conv::softMax(float *(&result), int size)
{
    if (result == NULL)
    {
        coutError;
        return false;
    }
    if (size <= 0)
    {
        cerr << "---------------------" << endl;
        cerr << "error! invalid size" << endl;
        cerr << "file = " << __FILE__ << "   fun =  " << __func__ << "  line =  " << __LINE__ - 4 << endl;
        cout << "size = " << size << endl;
        return false;
    }
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += exp(result[i]);
    }
    for (int i = 0; i < size; i++)
    {
        result[i] = exp(result[i]) / sum;
    }
    return true;
};

bool Mat_conv::CNN(const Mat_conv &mat1, conv_param *(conv_params), fc_param &fc_params)
{
    bool flag = true;
    // struct timeval start, finish;
    // double duration;
    // srand((unsigned)time(NULL));
    // gettimeofday(&start, NULL);

    struct timeval start1, finish1;
    double duration1;
    srand((unsigned)time(NULL));
    gettimeofday(&start1, NULL);
    Mat_conv matfir(64, 64, 16);
    Mat_conv gmat1(64 * 64, 3 * 3 * 3, 1);
    Mat_conv gmatfir(64 * 64, 16, 1);
    flag = matfir.gemm(mat1, gmat1, conv_params[0], matfir, gmatfir);
    // flag = matfir.conv(mat1, conv_params[0], matfir);
    flagCheck;
    Mat_conv mat2(32, 32, 16);
    flag = matfir.maxPool(matfir, mat2, conv_params[0].p_bias);
    flagCheck;
    gettimeofday(&finish1, NULL);
    duration1 = (double)((double)(finish1.tv_usec - start1.tv_usec))*0.001;
    cout << "First conv  " << duration1<< "ms" << endl;

    struct timeval start2, finish2;
    double duration2;
    srand((unsigned)time(NULL));
    gettimeofday(&start2, NULL);
    Mat_conv matsec(30, 30, 32);
    Mat_conv gmat2(30 * 30, 16 * 3 * 3, 1);
    Mat_conv gmatsec(30 * 30, 32, 1);
    flag = matsec.gemm(mat2, gmat2, conv_params[1], matsec, gmatsec);
    // flag = matsec.conv(mat2, conv_params[1], matsec);
    flagCheck;
    Mat_conv mat3(15, 15, 32);
    flag = matsec.maxPool(matsec, mat3, conv_params[1].p_bias);
    flagCheck;
    gettimeofday(&finish2, NULL);
    duration2 =(double) ( (double)(finish2.tv_usec - start2.tv_usec))*0.001;
    cout << "Second conv  " << duration2 << "ms" << endl;

    struct timeval start3, finish3;
    double duration3;
    srand((unsigned)time(NULL));
    gettimeofday(&start3, NULL);
    Mat_conv matthird(8, 8, 32);
    Mat_conv gmat3(8 * 8, 32 * 3 * 3, 1);
    Mat_conv gmatthird(8 * 8, 32, 1);
    flag = matthird.gemm(mat3, gmat3, conv_params[2], matthird, gmatthird);
    flagCheck;
    flag = matthird.setBias(matthird, conv_params[2].p_bias);
    flagCheck;
    flag = matthird.ReLu(matthird);
    flagCheck;
    gettimeofday(&finish3, NULL);
    duration3 = (double)((double)(finish3.tv_usec - start3.tv_usec)) *0.001;
    cout << "Third conv  " << duration3 << "ms" << endl;

    struct timeval start4, finish4;
    double duration4;
    srand((unsigned)time(NULL));
    gettimeofday(&start4, NULL);
    float *result = new float[2]{0, 0};
    flag = matthird.fullConnect(result, matthird, fc_params);
    flagCheck;
    flag = matthird.softMax(result, 2);
    flagCheck;
    gettimeofday(&finish4, NULL);
    duration4 = (double)((double)(finish4.tv_usec - start4.tv_usec))*0.001;
    cout << "Full connection  " << duration4 << "ms" << endl;

    // gettimeofday(&finish, NULL);
    // duration = (double)((double)(finish.tv_usec - start.tv_usec)) / 1000;
    cout << "Total Time " << duration1+duration2+duration3+duration4 << "ms" << endl;

    printf("bg score %.6f, ",result[0]);
    printf("face score %.6f\n",result[1]);
    delete[] result;
    return true;
}
