/*
@File          :mwindow.cpp
@Description:  :
@Date          :2021/12/25 09:23:22
@Author        :xieyin
@version       :1.0
*/

#include<iostream>
#include<string>
#include<vector>
using namespace std;

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
using namespace cv;

#include"mwindow.hpp"

MWindow::MWindow(string windowTitle, int rows, int cols, int height, int width, int flags):mWindowTitle(windowTitle), mRows(rows), mCols(cols){
    /*
    @description  : MWindow constructor
    @param  : 
        windowTitle : whole window title
        rows : sub window rows
        cols : sub window cols
        flags : namedWindow flags (eg, WINDOW_AUTOSIZE)
    @Returns  : 
    */
   // create canvas
    namedWindow(mWindowTitle, flags);
    mCanvas = Mat(height, width, CV_8UC3);
    imshow(mWindowTitle, mCanvas);
}

int MWindow::addImage(string title, Mat img, bool render){
    /*
    @description  : add title and image into canvas
    @param  : 
        title : sub image title
        img : image to be added
        render : render(flag) whether need to adjust the image for canvas
    @Returns  : 
        index : sub image index in total mRows * mCols
    */
   int index=-1;
    for(int i=0; i<mSubTitles.size(); i++){
        string t=this->mSubTitles[i];
        if(t.compare(title)==0){
            index=i;
            break;
        }
    }
    if(index==-1){
        mSubTitles.push_back(title);
        mSubImages.push_back(img);
    }else{
        mSubImages[index]= img;
    }
    if(render){
        MWindow::render();
    }
    return mSubImages.size() - 1;
}


void MWindow::removeImage(int pos){
    /*
    @description  : remove image from canvas based on index
    @param  : 
        pos : sub image index in total mRows * mCols
    @Returns  : 
        None
    */
    mSubTitles.erase(mSubTitles.begin() + pos);
    mSubImages.erase(mSubImages.begin() + pos);
}

void MWindow::render(){
    /*
    @description  : fill title and image into canvas in suitable way
    @param  : 
        None
    @Returns  :
        None 
    */
    mCanvas.setTo(Scalar(20, 20, 20));
    // get sub canvas size
    int cellH = mCanvas.rows / mRows;
    int cellW = mCanvas.cols / mCols;
    // set total number of images to load
    int n = mSubImages.size();
    int numImgs = n > mRows * mCols ? mRows * mCols : n;
    for(int i = 0; i < numImgs; i++){
        // get title
        string title = mSubTitles[i];
        // get sub canvas top left location
        int cellX = (cellW) * ((i) % mCols);
        int cellY = (cellH) * floor( (i) / (float) mCols);
        Rect mask(cellX, cellY, cellW, cellH);
        // set subcanvas size
        rectangle(mCanvas, Rect(cellX, cellY, cellW, cellH), Scalar(200, 200, 200), 1);
        Mat cell(mCanvas, mask);
        Mat imgResz;
        // get cell aspect
        double cellAspect = (double) cellW / (double) cellH;
        // get image
        Mat img = mSubImages[i];
        // get image aspect
        double imgAspect = (double) img.cols / (double) img.cols;
        double wAspect = (double) cellW / (double) img.cols;
        double hAspect = (double) cellH / (double) img.rows;
        // get suitable aspect and resize image
        double aspect = cellAspect < imgAspect ? wAspect : hAspect;
        resize(img, imgResz, Size(0, 0), aspect, aspect);
        // if gray image, convert to BGR
        if(imgResz.channels() == 1){
            cvtColor(imgResz, imgResz, COLOR_GRAY2BGR);
        }

        Mat subCell(mCanvas, Rect(cellX, cellY, imgResz.cols, imgResz.rows));
        imgResz.copyTo(subCell);
        putText(cell, title, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0));
    }
    // show total canvas
    imshow(mWindowTitle, mCanvas);

}