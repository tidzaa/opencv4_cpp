/*
@File          :utils.cpp
@Description:  :
@Date          :2021/12/25 09:23:38
@Author        :xieyin
@version       :1.0
*/
#include<string>
#include<cmath>
#include<memory>
#include<iostream>
#include<vector>
using namespace std;

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

#include"utils.hpp"
#include"mwindow.hpp"

Scalar randColor(RNG& rng){
    /*
    @description  : generate randow color
    @param  : 
        rng : random number generator object
    @Returns  : 
        Sacalar() : BGR scalar
    */
    auto iColor = (unsigned)rng;
    return Scalar(iColor&255, (iColor >> 8)&255, (iColor >> 16)&255);
}

Mat calLigthPattern(Mat img){
    /*
    @description  : get source image's light pattern 
    @param  : 
        img : source BGR image or Gray image
    @Returns  : 
        pattern : the light pattern
    */
    Mat pattern;
    blur(img, pattern, Size(img.cols / 3, img.cols / 3));
    return pattern;
}

Mat removeLight(Mat img, Mat pattern, int methodLight){
    /*
    @description  : remove light between img and pattern based on method light
    @param  : 
        img : source BGR/Gray image
        pattern : pattern BGR/Gray image
        methodLight : choise options: 0 difference, 1 div
    @Returns  : 
        aux : light removed BGR/Gray image
    */
    Mat aux;
    if(methodLight == 1){
        // div operation in float 32 format CV_32F
        Mat img32, pattern32;
        img.convertTo(img32, 5);
        pattern.convertTo(pattern32, 5);
        aux = 1.0 - (img32 / pattern32);
        // covert to CV_8U and clip
        aux.convertTo(aux, 0, 255);
    }
    else{
        // difference
        aux = pattern - img;
    }
    return aux;
}


Mat connectedComponents(Mat img_thr){
    /*
    @description  : opencv connnected components
    @param  : 
        img : threshold image
    @Returns  : 
        None
    */
   Mat labels;
   auto num_objs = connectedComponents(img_thr, labels);
   Mat res;
   if(num_objs < 2){
       cout << "no object is detected. " << endl;
       return res;
   }
   res = Mat::zeros(img_thr.rows, img_thr.cols, CV_8UC3);
   RNG rng(0xFFFFFFFF);
   for(auto i = 1; i < num_objs; i++){
       Mat mask = labels == i;
       res.setTo(randColor(rng), mask);
   }
   return res;
}

Mat connectedComponentsWithStats(Mat img_thr){
    /*
    @description  : connnected components with stats
    @param  : 
        img : threshold image
    @Returns  : 
        None
    */
    Mat labels, stats, centroids;
    auto num_objs = connectedComponentsWithStats(img_thr, labels, stats, centroids);
    Mat res;
    if(num_objs < 2){
       cout << "no object is detected. " << endl;
       return res;
   }
   res = Mat::zeros(img_thr.rows, img_thr.cols, CV_8UC3);
   RNG rng(0xFFFFFFFF);
   for(auto i = 1; i < num_objs; i++){
       Mat mask = labels == i;
       res.setTo(randColor(rng), mask);
       stringstream ss;
       ss << "area: " << stats.at<int>(i, CC_STAT_AREA);
       // add text info
       putText(res, ss.str(), centroids.at<Point2d>(i), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0));
   }
   return res;
}


Mat findContours(Mat img_thr){
    /*
    @description  : find contours and put text
    @param  : 
        img : threshold image
    @Returns  : 
        None
    */
    vector<vector<Point>> contours;
    findContours(img_thr, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat res;
    if(contours.size() == 0){
        cout << "no contours are found ." << endl;
        return res;
    }
    RNG rng(0xFFFFFFFF);
    res = Mat::zeros(img_thr.rows, img_thr.cols, CV_8UC3);
    // calculate moments
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}
	// calculate centroids
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}
    for(auto i = 0; i < contours.size(); i++){
        drawContours(res, contours, i, randColor(rng));
        putText(res, "*", Point(mc[i].x, mc[i].y), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 255), 1);
    }
    return res;
}

// helper function for readFolderAndExtractFeatures, preprocess image to binary image
Mat preProcess(Mat img){
    /*
    @description  : preprocess img to denoise and remove light
    @param  : 
        img : image to process
    @Returns  : 
    */
    if(img.channels() == 3){
        cvtColor(img, img, COLOR_BGR2GRAY);
    }
    Mat imgOut, imgNoise, imgLight;
    medianBlur(img, imgNoise, 3);
    imgNoise.copyTo(imgLight);
    // read lightPat
    Mat lightPat = imread("data/pattern.pgm", 0);
    imgLight = removeLight(imgNoise, lightPat);
    threshold(imgLight, imgOut, 30, 255, THRESH_BINARY);
    return imgOut;
}

// helper function for trainAndTest, readFolderAndExtractFeatures
bool readFolderAndExtractFeatures(string filePath, int label, int numTest, 
    vector<float> &trainingData, vector<int> &trainResponses, vector<float> &testData, vector<int> &testResponses){
    /*
    @description  : read file data and extract area and aspect features
    @param  : 
        filePath : image file path
        label : image lable to classify
        numTest : number for test
        trainingData : trainingData feature: area, aspect
        trainResponses : trainingData label
        testData : testData feature: area, aspect
        testResponses : testData label
    @Returns  : 
        (ref return) : trainingData, trainResponses, testData, testResponses
    */
    vector<String> files;
    // get folder
    glob(filePath, files, true); 
    Mat frame;
    int imgIdx = 0;
    if(files.size() == 0){
        return false;
    }
    for(int i = 0; i < files.size(); i++){
        frame = imread(files[i]);
        // preprocess image
        Mat pre = preProcess(frame);
        // get n features pair for each image
        vector<vector<float>> features = extractFeatures(pre);
        for(int i = 0; i < features.size(); i++){
            // first numTest for model test
            if(imgIdx >= numTest){
                trainingData.push_back(features[i][0]);
                trainingData.push_back(features[i][1]);
                trainResponses.push_back(label);
            }else{
                testData.push_back(features[i][0]);
                testData.push_back(features[i][1]);
                testResponses.push_back(label);
            }
        }
        imgIdx++;
    }
    return true;
}

// helper function for trainAndTest, ploat data error
void plotData(Mat trainingDataMat, Mat trainResponsesMat, string mode, float* error){
    /*
    @description  : ploat train data feature (x: area, y: aspect) distributiion
    @param  : 
        trainingDataMat : trainingDataMat shape [N/2, 2], N is trainData vector size
        trainResponsesMat : trainResponsesMat shape [N, 1], N is trainData label vector size
        error : total error rate to display
    @Returns  : 
        None
    */
    float areaMax, areaMin, asMax, asMin;
    areaMax = asMax = 0.0;
    areaMin = asMin = 99999999;
    for(int i = 0; i < trainingDataMat.rows; i++){
        float area = trainingDataMat.at<float>(i, 0);
        float aspect = trainingDataMat.at<float>(i, 1);
        // get min, max value
        if(area > areaMax){
            areaMax = area;
        }
        if(aspect > asMax){
            asMax = aspect;
        }
        if(areaMin > area){
            areaMin = area;
        }
        if(asMin > area){
            asMin = aspect;
        }
    }
    // create image to display
    Mat fig = Mat::zeros(512, 512, CV_8UC3);
    for(int i = 0; i < trainingDataMat.rows; i++){
        float area = trainingDataMat.at<float>(i, 0);
        float aspect = trainingDataMat.at<float>(i, 1);
        // min-max norm [0~1] * 420 pixel
        int x = (int)(420.0f*((area - areaMin) / (areaMax - areaMin)));
        int y = (int)(420.0f*((aspect - asMin) / (asMax - asMin)));
        int label = trainResponsesMat.at<int>(i);
        Scalar color;
        if(label == 0){
            color = Scalar(255, 0, 0);
        }else if(label == 1){
            color = Scalar(0, 255, 0);
        }else if(label == 2){
            color = Scalar(0, 0, 255);
        }
        // cicle locate with start at(80, 80) to overcome border
        circle(fig, Point(x+80, y+80), 3, color, -1, 8);
    }
    if(error != NULL){
        stringstream ss;
        ss << mode << " error: " << *error << " \%";
        putText(fig, ss.str(), Point(20, 512-40), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(200, 200, 200), 1, LINE_AA);
    }
    myWin->addImage("Fig", fig);
}

void defineSVM(Ptr<SVM>& svm){
    /*
    @description  : define svm parameters
    @param  : 
        svm : svm model
    @Returns  : 
        (ref return) : svm with parameters
    */
    svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setNu(0.05);
    svm->setKernel(SVM::CHI2);
    svm->setDegree(1.0);
    svm->setGamma(2.0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
}

void trainSVM(){
    /*
    @description  : train a svm model and test its error rate
    @param  : 
        mode : machine learning mode
    @Returns  : 
        None
    */
    vector<float> trainingData;
    vector<int> trainResponses;
    vector<float> testData;
    vector<int> testResponses;

    int numTest = 20;
    string nutPath = "data/nut";
    string ringPath = "data/ring";
    string screwPath = "data/screw";
    // read data path and extract feature
    readFolderAndExtractFeatures(nutPath, 0, numTest, trainingData, trainResponses, testData, testResponses);
    readFolderAndExtractFeatures(ringPath, 1, numTest, trainingData, trainResponses, testData, testResponses);
    readFolderAndExtractFeatures(screwPath, 2, numTest, trainingData, trainResponses, testData, testResponses);
    // cout << "Num of train samples: " << trainingData.size() << endl;
    // cout << "Num of test samples: " << testData.size() << endl;
    Mat trainingDataMat(trainingData.size() / 2, 2, CV_32FC1, &trainingData[0]);
    Mat trainResponsesMat(trainResponses.size(), 1, CV_32SC1, &trainResponses[0]);
    Mat testDataMat(testData.size() / 2, 2, CV_32FC1, &testData[0]);
    Mat testResponsesMat(testResponses.size(), 1, CV_32SC1, &testResponses[0]);
    // set row sample
    Ptr<TrainData> tData = TrainData::create(trainingDataMat, ROW_SAMPLE, trainResponsesMat);

    // select model
    Ptr<SVM> model = SVM::create();
    defineSVM(model);
    model->train(tData);
    model->save("model/svm.xml");

    if(testResponses.size() > 0){
        Mat testPredict;
        // predict
        model->predict(testDataMat, testPredict);
        testPredict.convertTo(testPredict, CV_32SC1);
        Mat errMat = testPredict != testResponsesMat;
        float error = 100.0f * countNonZero(errMat) / testResponses.size();
        cout << "svm" << " Error rate: " << error << "\%" << endl;
        plotData(trainingDataMat, trainResponsesMat, "svm", &error);
    }
    else{
        plotData(trainingDataMat, trainResponsesMat, "svm");
    }
}

vector<vector<float>> extractFeatures(Mat img, vector<int>* posLeft, vector<int>* posTop){
    /*
    @description  : extract image features and get left top loation
    @param  : 
        img : image to get feature
        postLeft : left top_left location
        postTop : top top_left location
    @Returns  : 
        features: extracted feature
    */
    vector<vector<float>> features;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat temp = img.clone();
    // find contours
    findContours(temp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if(contours.size() == 0){
        return features;
    }
    for(int i = 0; i < contours.size(); i++){
        Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
        // draw contours
        drawContours(mask, contours, i, Scalar(1), FILLED, LINE_8, hierarchy, 1);
        // get area value
        Scalar areaSum = sum(mask);
        float area = areaSum[0];
        if(area > 500){
            // calculate aspect for area is larger than 500
            RotatedRect r = minAreaRect(contours[i]);
            float w = r.size.width;
            float h = r.size.height;
            float aspect = w < h ? h / w : w / h;
            vector<float> row;
            // load calculated feature
            row.push_back(area);
            row.push_back(aspect);
            features.push_back(row);
            // load top_left location
            if(posLeft != NULL){
                posLeft->push_back((int)r.center.x);
            }
            if(posTop != NULL){
                posTop->push_back((int)r.center.y);
            }
            myWin->addImage("Extracted Feature", mask * 255);
            myWin->render();
            waitKey(10);
        }
    }
    return features;
}