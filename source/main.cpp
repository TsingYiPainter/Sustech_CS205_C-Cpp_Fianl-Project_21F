#include <opencv2/opencv.hpp>
#include "Mat.hpp"
#include "weight.hpp"
int main()
{
    cv::Mat image = cv::imread("face.jpg");
    Mat_conv mat(128, 128, 3);
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            mat.Matrix[i * 128 + j] = (float)image.at<cv::Vec3b>(i, j)[0] / 255.0f;
            mat.Matrix[i * 128 + j + 128 * 128 * 1] = (float)image.at<cv::Vec3b>(i, j)[1] / 255.0f;
            mat.Matrix[i * 128 + j + 128 * 128 * 2] = (float)image.at<cv::Vec3b>(i, j)[2] / 255.0f;
        }
    }
    bool flag=mat.CNN(mat, conv_params, fc_params);
    if(!flag){
        cout<<"Convolution failed, please see error.txt"<<endl;
    }
    return 0;
}
