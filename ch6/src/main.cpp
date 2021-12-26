/*
@File          :main.cpp
@Description:  :
@Date          :2021/12/25 09:23:30
@Author        :xieyin
@version       :1.0
*/

#include<iostream>
#include<string>
#include<sstream>
#include<memory>
using namespace std;

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

#include"mwindow.hpp"
#include"utils.hpp"

const char* keys = {
    "{help h usage ? | | Print this message}"
    "{@image | | Image for test}"
    "{@lightPat | | light pattern for test image}"
    "{@mode | svm | machine learning mode, default svm}"
};

shared_ptr<MWindow> myWin;

int main(int argc, const char** argv){
    // command line parser
    CommandLineParser parser(argc, argv, keys);
    if(parser.has("help")){
        parser.printMessage();
        return 0;
    }
    if(!parser.check()){
        parser.printErrors();
        return 0;
    }

    // define mywin
    myWin = make_shared<MWindow>("Main Window", 2, 2, 700, 1000, 1);
    
    // get test image path
    String imgFile = parser.get<String>(0); 
    Mat img = imread(imgFile, 0);
    if(img.data == NULL){
        cout << "can not read image file." << endl;
        return 0;
    }

    // get light pattern image
    String ligPatFile = parser.get<String>(1);
    Mat lightPat = imread(ligPatFile, 0);
    if(lightPat.data == NULL){
        cout << "can not read image file." << endl;
        return 0;
    }
    // mdeianblur light pattern
    medianBlur(lightPat, lightPat, 3);

    // copy img to imgOut
    Mat imgOut = img.clone();
    cvtColor(imgOut, imgOut, COLOR_GRAY2BGR);

    // preprocess image
    Mat pre = preProcess(img);

    // get feature and top left location from image
    vector<int> posLeft, posTop;
    vector<vector<float>> features = extractFeatures(pre, &posLeft, &posTop);

    // get mode selection
    string mode = parser.get<string>(2);
    // train and predict model
    if (mode == "svm"){
        trainSVM();
        // trainAndTest<SVM>(mode);
        predict<SVM>(features, posLeft, posTop, mode, imgOut);
    }
    else if (mode == "bayes"){
        trainAndTest<NormalBayesClassifier>(mode);
        predict<NormalBayesClassifier>(features, posLeft, posTop, mode, imgOut);
    }
    else if(mode == "boost"){
        trainAndTest<Boost>(mode);
        predict<Boost>(features, posLeft, posTop, mode, imgOut);
    }
    else{
        cout << "not support model";
        return 0;
    }

    myWin->addImage("binary Image", pre);
    myWin->addImage("result", imgOut);
    myWin->render();
    waitKey(0);
    return 0;
}