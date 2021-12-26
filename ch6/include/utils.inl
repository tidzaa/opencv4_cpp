/*
@File          :utils.inl
@Description:  :
@Date          :2021/12/25 20:29:26
@Author        :xieyin
@version       :1.0
*/
template<typename T>
void trainAndTest(string mode){
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
    Ptr<T> model = T::create();
    model->train(tData);
    model->save("model/" + mode + ".xml");

    if(testResponses.size() > 0){
        Mat testPredict;
        // predict
        model->predict(testDataMat, testPredict);
        testPredict.convertTo(testPredict, CV_32SC1);
        Mat errMat = testPredict != testResponsesMat;
        float error = 100.0f * countNonZero(errMat) / testResponses.size();
        cout << mode << " Error rate: " << error << "\%" << endl;
        plotData(trainingDataMat, trainResponsesMat, mode, &error);
    }
    else{
        plotData(trainingDataMat, trainResponsesMat, mode);
    }
}

template<typename T>
void predict(vector<vector<float>> features, vector<int> posLeft, vector<int> posTop, string mode, Mat& imgOut){
    /*
    @description  : predict features extracted from imgOut, and put text in left top position
    @param  : 
        features : extracted feature from imgOut
        posLeft : left_top left location
        posTop : left_top top location
        mode : machine learning mode
        imgOut : the img with text output
    @Returns  : 
        (ref return) : imgOut
    */
    for(int i = 0; i < features.size(); i++){
        Mat predDataMat(1, 2, CV_32FC1, &features[i][0]);
        Ptr<T> model = Algorithm::load<T>("model/" + mode + ".xml");
        float result = model->predict(predDataMat);
        cout << result << endl;
        stringstream ss;
        Scalar color;
        if(result == 0){
            color = Scalar(255, 0, 0);
            ss << "NUT";
        }
        else if(result == 1){
            color = Scalar(0, 255, 0);
            ss << "RING";
        }
        else if(result == 2){
            color = Scalar(0, 255, 0);
            ss << "SCREW";
        }
        putText(imgOut, ss.str(), Point2d(posLeft[i], posTop[i]), FONT_HERSHEY_SIMPLEX, 0.4, color);
    }
}
