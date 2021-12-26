/*
@File          :utils.hpp
@Description:  :
@Date          :2021/12/25 09:23:48
@Author        :xieyin
@version       :1.0
*/
#pragma once
#include<string>
#include<cmath>
#include<memory>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

#include"mwindow.hpp"

extern shared_ptr<MWindow> myWin;

// generate randow color basd on randow number generator
Scalar randColor(RNG& rng);

// calculate given img's light pattert with large kernel's Blur operation
Mat calLigthPattern(Mat img);

// use 2 tpyes of light removal method, 0 diff, 1 div, defalut is 0
Mat removeLight(Mat img, Mat pattern, int methodLight=0);

// packed opencv lib connectedComponents function
Mat connectedComponents(Mat img_thr);

// packed opencv lib connectedComponentsWithStats function
Mat connectedComponentsWithStats(Mat img_thr);

// packed opencv lib findContours function
Mat findContours(Mat img_thr);

// helper function for trainAndTest, readFolderAndExtractFeatures
bool readFolderAndExtractFeatures(string filePath, int label, int numTest, 
    vector<float>& trainingData, vector<int>& trainResponses, vector<float>& testData, vector<int>& testResponses);

// helper function for trainAndTest, ploat data error
void plotData(Mat trainingDataMat, Mat trainResponsesMat, string mode="svm", float* error=NULL);

// define svm parameters
void defineSVM(Ptr<SVM>& svm);

// the train and test process for mechain learning
template<typename T>
void trainAndTest(string mode="svm");

// train svm model
void trainSVM();

// predict features extracted from imgOut, and put text in left top position
template<typename T>
void predict(vector<vector<float>> features, vector<int> posLeft, vector<int> posTop, string mode, Mat& imgOut);

// preprocess test image
Mat preProcess(Mat img);

// extract feature from preprocess image and get left top location
vector<vector<float>> extractFeatures(Mat img, vector<int>* posLeft=NULL, vector<int>* posTop=NULL);

#include"utils.inl"