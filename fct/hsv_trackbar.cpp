// hsv_trackbar.cpp : modify the parameter of HSV on  "Control"panel
//                    white: inRange black: out of range.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>

using namespace cv;

void hsv_trackbar(Mat src)
{
    int iLowH = 0;  //B:100~110 G:72~80   R:0~2 dark:100~138 overW:0(defect 132~155)
    int iHighH = 179;
    int iLowS = 0;  //B:245up   G:225up     R:250up dark:80~210
    int iHighS = 255;
    int iLowV = 0;  //B:210up   G:134~225   R:234up dark:75~125
    int iHighV = 255;   
    //lowWhite: 10~75 0~30 75~145
    Mat imgHSV;
    Mat img_thres;
    cvtColor(src, imgHSV, COLOR_BGR2HSV);        
    namedWindow("Control", 1);

    createTrackbar( "LowH", "Control", &iLowH, 179);
    createTrackbar( "HighH", "Control", &iHighH, 179);

    createTrackbar( "LowS", "Control", &iLowS, 255);
    createTrackbar( "HighS", "Control", &iHighS, 255);

    createTrackbar( "LowV", "Control", &iLowV, 255);
    createTrackbar( "HighV","Control", &iHighV, 255);
    while(true){
        int lowH = getTrackbarPos("LowH", "Control");
        int HighH = getTrackbarPos("HighH", "Control");
        int LowS = getTrackbarPos("LowS", "Control");
        int HighS = getTrackbarPos("HighS", "Control");
        int LowV = getTrackbarPos("LowV", "Control");
        int HighV = getTrackbarPos("HighV", "Control");
        inRange(imgHSV,Scalar(lowH, LowS, LowV), Scalar(HighH, HighS, HighV),img_thres);
        imshow("HSVimg_thres",img_thres);
        waitKey(30);
    }    
}

int main( int argc, char** argv )
{    
    Mat src = imread("green.png");    
    resize(src,src,Size(320,240));
    imshow("src",src);
    hsv_trackbar(src);
    return 0;
}

