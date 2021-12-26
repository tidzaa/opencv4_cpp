/*
@File          :mwindow.hpp
@Description:  :
@Date          :2021/12/25 09:23:14
@Author        :xieyin
@version       :1.0
*/
#pragma once

#include<iostream>
#include<string>
#include<vector>
using namespace std;

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
using namespace cv;

class MWindow{
    public:
        // consturtor
        MWindow(string windowTitle, int rows, int cols, int height=700, int width=1200, int flags=WINDOW_AUTOSIZE);
        // add image into canvas
        int addImage(string title, Mat img, bool render = false);
        // remove image from canvas
        void removeImage(int pos);
        // adjust all image size in canvas
        void render();
    private:
        string mWindowTitle;
        int mRows;
        int mCols;
        Mat mCanvas;
        vector<string> mSubTitles;
        vector<Mat> mSubImages;     
};