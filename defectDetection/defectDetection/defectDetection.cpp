// defectDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2\opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;


/// Global Variables
//trackbar
const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;

// for statistics
float lowBnd[3];
float upBnd[3];

/// Matrices to store images
Mat img_sampleBlue;
Mat img_sampleGreen;
Mat img_sampleRed;
Mat dst;

/**
* @function on_trackbar
* @brief Callback for trackbar
*/
void on_trackbar( int, void* )
{
    alpha = (double) alpha_slider/alpha_slider_max ;
    beta = ( 1.0 - alpha );

    addWeighted( img_sampleBlue, alpha, img_sampleRed, beta, 0.0, dst);

    imshow( "Linear Blend", dst );
}


bool isBlue(Mat src);
bool isGreen(Mat src);
bool isRed(Mat src);
void showHist(Mat img);
void showDefect_statistic(Mat src, int N_onEdges);
void showDefect_sample(Mat src, int colorType);

void grabImgPer_nFrames(Mat img, int nFrames);
vector<Rect> getEdge(Mat image, int threshold);


Rect LCDrect(Mat src){
    int threshold = 10;
    Mat src_gray;
    cvtColor( src, src_gray, CV_RGB2GRAY );

    cv::threshold(src_gray, src_gray, threshold, 255, cv::THRESH_BINARY);
    /*imshow("src_gray", src_gray);
    cvMoveWindow("src_gray", 660, 250);
    */
    vector<vector<Point>> contours;//儲存邊界需要的點
    vector<Vec4i> hierarchy;//定義儲存層次

    //檢測所有輪廓並重構層次
    //CV_RETR_TREE：取得所有輪廓，以全階層的方式儲存
    //CV_RETR_EXTERNAL : 取出最外層的輪廓
    findContours(src_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    //畫出輪廓
    // drawContours(frame, contours, -1, Scalar(255, 0, 0));
    Rect rect;//儲存符合條件的外接矩形，由boundingRect函數返回
    int idx = 0;//index of contour iterate with hierarchy[idx][0]
    if (contours.size())//to avoid exception when there is no rect in vision
    {
        for (; idx >= 0; idx = hierarchy[idx][0])//找到面积最大的轮廓（hierarchy[idx][0]会指向下一个轮廓，若没有下一个轮廓则hierarchy[idx][0]为负数）
        {            
            if (fabs(contourArea(Mat(contours[idx]))) > 10000 
                && fabs(contourArea(Mat(contours[idx]))) > rect.area()){
                    cout << "==="<<endl;
                    cout << "boundingRect of contour (x,y) = "<< boundingRect(contours[idx]).x
                        <<" , "<< boundingRect(contours[idx]).y <<endl;
                    cout << "idx: "<< idx <<" area\t"<< fabs(contourArea(Mat(contours[idx]))) <<endl;
                    rect =boundingRect(contours[idx]);
            }
            cout << "============================="<<endl;
        }
        return rect;
    }
}
int main(int argc, _TCHAR* argv[])
{
    VideoCapture cap("../images/brg_noDefect_WIN_20160707.mp4");
    if(!cap.isOpened()){
        return -1;
    }
    //VideoCapture cap("../images/testWIN10.mp4");   
    //CvCapture *capture = cvCreateFileCapture("C:\\VIDEO0013.mp4");

    //test trackbar
    /// Read image ( same size, same type )    
    img_sampleBlue = imread("../images/sample_b.png");
    img_sampleGreen = imread("../images/sample_g.png");
    img_sampleRed = imread("../images/sample_r.png");

    //for test convenience
    //resize(img_sampleBlue, img_sampleBlue,Size(320,240));
    //resize(img_sampleRed, img_sampleRed,Size(320,240));

    if( !img_sampleBlue.data) { printf("Error loading img_sampleBlue \n"); return -1;}
    if( !img_sampleRed.data ) { printf("Error loading img_sampleRed  \n"); return -1;}

    /// Initialize values
    alpha_slider = 0;
    int iLowH = 0;  //B:100~110 G:72~80   R:0~2 dark:100~138 overW:0(defect 132~155)
    int iHighH = 179;
    int iLowS = 0;  //B:245up   G:225up     R:250up dark:80~210
    int iHighS = 255;
    int iLowV = 0;  //B:210up   G:134~225   R:234up dark:75~125
    int iHighV = 255;   
    //lowWhite: 10~75 0~30 75~145

    /// Create Windows
    namedWindow("Control", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Control", 10, 260);

    createTrackbar( "LowH", "Control", &iLowH, 179, on_trackbar );
    createTrackbar( "LowH", "Control", &iLowH, 179);
    createTrackbar( "HighH", "Control", &iHighH, 179);

    createTrackbar( "LowS", "Control", &iLowS, 255);
    createTrackbar( "HighS", "Control", &iHighS, 255);

    createTrackbar( "LowV", "Control", &iLowV, 255);
    createTrackbar( "HighV","Control", &iHighV, 255);
    //Show some stuff
    while(true){
        //test with trackbar
        //Mat src = imread("../images/save_63.png");
        //Mat src = imread("../images/test_white03.jpg");
        /// Show some stuff
        /// Create Trackbars
        char TrackbarName[50];
        sprintf( TrackbarName, "Alpha x %d", alpha_slider_max );

        createTrackbar( TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar );
        on_trackbar( alpha_slider, 0);
        
        Mat src;
        cap >> src;
        if(src.empty()){
            break;
        }
        //resize(src,src, Size(320,240));

        imshow("src",src);
        grabImgPer_nFrames(src, 5);   //grab img per 5 frames
       /* cvMoveWindow("src", 10, 10);
        showDefect_sample(src, 0);
        showDefect_sample(src, 1);
        showDefect_sample(src, 2);
        Rect LCD = LCDrect(src);*/
        //Mat src_roi= src(LCD);
        //Mat imgHSV;
        //cvtColor(src_roi, imgHSV, COLOR_BGR2HSV);
        //Mat img_thres;

        ////attention
        //showDefect_statistic(src_roi, 20);
        //inRange(imgHSV,Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV),img_thres);
        //showHist(imgHSV);

        //imshow("img_thres",img_thres);
        //// imshow("src",src);
        ///// Wait until user press esc key
        //if(waitKey(30)==27){
        //    break;
        //}

        ////
        ////Mat src = imread("../images/red.png");
        //if(isBlue(src_roi)){
        //    cout << "it's blue";
        //}
        //if(isGreen(src_roi)){
        //    cout << "it's green";
        //}
        //if(isRed(src_roi)){
        //    cout << "it's red";
        //}
        if(waitKey(30)==27){
            break;
        }
    }
    return 0;
}

#define cvQueryHistValue_1D(hist, idx0) ((float)cvGetReal1D((hist)->bins, (idx0)))//for opencv version 2.31
Mat sample_Mat(Mat src, int n){
    Mat new_Mat = Mat(n,n,src.type());
    for(int j = 0; j <new_Mat.rows;j++){
        for(int i = 0; i <new_Mat.cols; i++){
            int x = src.cols*(i+1)/(n+1);
            int y = src.rows*(j+1)/(n+1);
            new_Mat.at<Vec3b>(j,i).val[0] = src.at<Vec3b>(y,x).val[0];
            new_Mat.at<Vec3b>(j,i).val[1] = src.at<Vec3b>(y,x).val[1];
            new_Mat.at<Vec3b>(j,i).val[2] = src.at<Vec3b>(y,x).val[2];
        }   
    }
    return new_Mat;
}
float avg(Mat sample){
    float sum = 0.0;
    for(int j = 0; j <sample.rows;j++){
        for(int i = 0; i <sample.cols; i++){            
            /*cout <<(int)sample.at<Vec3b>(j,i).val[0] <<","<<(int)sample.at<Vec3b>(j,i).val[1] <<","<<
            (int)sample.at<Vec3b>(j,i).val[2] <<endl;*/
            sum+= sample.at<Vec3b>(j,i).val[0];
        }   
    }
    return sum/(sample.rows*sample.cols);
}
float variance(Mat sample, float avg){
    float square_sum = 0.0;
    // float squ_devi = 0.0;
    for(int j = 0; j <sample.rows;j++){
        for(int i = 0; i <sample.cols; i++){            
            /*cout <<(int)sample.at<Vec3b>(j,i).val[0] <<","<<(int)sample.at<Vec3b>(j,i).val[1] <<","<<
            (int)sample.at<Vec3b>(j,i).val[2] <<endl;*/
            square_sum+= (sample.at<Vec3b>(j,i).val[0])*sample.at<Vec3b>(j,i).val[0];
            // squ_devi += (sample.at<Vec3b>(j,i).val[0]-avg)*(sample.at<Vec3b>(j,i).val[0]-avg);
        }

    }
    int num_samples = sample.rows*sample.cols;
    float var;
    if(num_samples>1){
        var =(square_sum/num_samples-avg*avg)*num_samples/(num_samples-1);
    }else{
        var =0;
        cout <<"num_samples might have problem"<<endl;
    } 
    //float var2 = squ_devi/num_samples;
    //cout << "var "<<var << "var2" <<var2<<endl;
    return var;
}
void showHist(Mat img){

    //resize(src,src, Size(320,240));
    if(img.channels() == 3){
        //test hsv val
        /*int centerX = (img.cols+1)/2;
        int centerY = (img.rows+1)/2;

        float value_ch1 = img.at<Vec3b>(centerY,centerX).val[0];
        float value_ch2 = img.at<Vec3b>(centerY,centerX).val[1];
        float value_ch3 = img.at<Vec3b>(centerY,centerX).val[2];*/

        int N =20;//cols, rows = n
        Mat sample_NxN = sample_Mat(img,N);
        //show sample mat
        /*for(int j = 0; j <sample_NxN.rows;j++){
        for(int i = 0; i <sample_NxN.cols; i++){            
        cout <<(int)sample_NxN.at<Vec3b>(j,i).val[0] <<","<<(int)sample_NxN.at<Vec3b>(j,i).val[1] <<","<<
        (int)sample_NxN.at<Vec3b>(j,i).val[2] <<endl;
        }   
        }*/

        Mat mean, stdDev;
        //output: mean, stdDev
        meanStdDev(sample_NxN, mean, stdDev);

        /*cout <<stdDev.at<double>(2)<<endl;
        cout <<mean<<endl;
        cout <<stdDev<<endl;*/

        //Separate the image into 3 places(H.S.V)
        vector<Mat> pqr_planes;
        split(img, pqr_planes);

        //histSize = 256
        int histSize = 256;

        float range[] = {0,256};
        const float* histRange = {range};

        bool uniform =true; bool accumulate = false;
        Mat p_hist, q_hist, r_hist;

        calcHist(&pqr_planes[0], 1, 0, Mat(), p_hist, 1, &histSize, &histRange, uniform, accumulate);
        calcHist(&pqr_planes[1], 1, 0, Mat(), q_hist, 1, &histSize, &histRange, uniform, accumulate);
        calcHist(&pqr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

        //
        //histogram attributes: w=512, h=400
        int hist_w =512, hist_h = 400;
        int bin_w = cvRound( (double)hist_w/histSize);

        Mat histImg(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));

        //normalize the result to [0, histImg.rows]
        normalize(p_hist, p_hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
        normalize(q_hist, q_hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
        normalize(r_hist, r_hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

        //show peak value
        /*double max_value = 0;  
        minMaxLoc(p_hist, 0,&max_value,0,0);
        cout <<"=========="<<endl;
        cout <<"max_value: " << max_value;
        */
        float sumOfBin =0;
        /// Draw for each channel
        for( int i = 1; i < histSize; i++ )
        {
            //show bin value
            //float bin_val = p_hist.at<float>(i-1); //像素i的概率  
            //sumOfBin += bin_val;
            //cout << "i, bin_val\t"<<i <<" ,"<<bin_val<<endl;
            //cout << "sum= " <<sumOfBin <<endl;

            line( histImg, 
                Point( bin_w*(i-1), hist_h - cvRound(p_hist.at<float>(i-1))),
                Point( bin_w*(i), hist_h - cvRound(p_hist.at<float>(i))),
                Scalar( 255, 0, 0), 2, 8, 0 );
            line( histImg, 
                Point( bin_w*(i-1), hist_h - cvRound(q_hist.at<float>(i-1))),
                Point( bin_w*(i), hist_h - cvRound(q_hist.at<float>(i))),
                Scalar( 0, 255, 0), 2, 8, 0 );
            line( histImg, 
                Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1))),
                Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
                Scalar( 0, 0, 255), 2, 8, 0 );

        }
        //test add a bound: (avrg +-vari)   

        for(int i=0; i < 3; i++){
            lowBnd[i] = mean.at<double>(i)-stdDev.at<double>(i)*2;
            upBnd[i] = mean.at<double>(i)+stdDev.at<double>(i)*2;

            cout << "lowBnd["<<i<< "\t"<<lowBnd[i] <<", "<<upBnd[i] <<endl;
            line( histImg, 
                Point( lowBnd[i]*bin_w, 0),
                Point( lowBnd[i]*bin_w, hist_h),
                Scalar( 0, 255-70*i, 255-70*i), 2-i, 8, 0 );
            line( histImg, 
                Point( upBnd[i]*bin_w, 0),
                Point( upBnd[i]*bin_w, hist_h),
                Scalar( 0, 255-70*i, 255-70*i), 2-i, 8, 0 );   
        }        
        //last item which is skipped 
        cout << "i, bin_val\t"<<256 <<" ,"<<p_hist.at<float>(255)<<endl;
        imshow("histImg", histImg);
    }
}

//colorType: [012] =[BGR]
void showDefect_sample(Mat src, int colorType){
    if(colorType > 2 || colorType < 0){
        return;
    }
    int thresh = 15;
    Mat dst;
    //test enum
    //enum _color{Blue,Green,Red};

    switch (colorType)
    {
    case 0:
        absdiff(img_sampleBlue, src, dst);  
        break;
    case 1:
        absdiff(img_sampleGreen, src, dst);
        break;
    case 2:
        absdiff(img_sampleRed, src, dst);
        break;
    default:
        break;
    }
    cv::threshold(dst, dst, thresh, 255, cv::THRESH_BINARY);
    char dstname[80];
    sprintf(dstname, "diff_dst_%d",colorType);
    imshow(dstname,dst);
    /*cout <<(int)src.at<Vec3b>(160,120).val[0]<<",";
    cout <<(int)src.at<Vec3b>(160,120).val[1]<<",";
    cout <<(int)src.at<Vec3b>(160,120).val[2]<<endl;
    cout <<(int)img_sampleBlue.at<Vec3b>(160,120).val[0]<<",";
    cout <<(int)img_sampleBlue.at<Vec3b>(160,120).val[1]<<",";
    cout <<(int)img_sampleBlue.at<Vec3b>(160,120).val[2]<<endl;*/

    cout <<(int)dst.at<Vec3b>(160,120).val[0]<<",";
    cout <<(int)dst.at<Vec3b>(160,120).val[1]<<",";
    cout <<(int)dst.at<Vec3b>(160,120).val[2]<<endl;
}

void showDefect_statistic(Mat src, int N_onEdges){
    Mat hsvImg, defect;
    cvtColor(src, hsvImg,CV_BGR2HSV);
    src.copyTo(defect);
    //int N =20;//cols, rows = n
    Mat sample_NxN = sample_Mat(hsvImg,N_onEdges);

    //show sample_NxN
    cout<< sample_NxN <<endl;
    for(int j = 0; j <sample_NxN.rows;j++){
        for(int i = 0; i <sample_NxN.cols; i++){            
            cout <<(int)sample_NxN.at<Vec3b>(j,i).val[0] <<","<<(int)sample_NxN.at<Vec3b>(j,i).val[1] <<","<<
                (int)sample_NxN.at<Vec3b>(j,i).val[2] <<endl;
        }   
    }

    /*float avrg = avg(test);
    cout << "avg= " << avrg<<endl;
    float vari = variance(test, avrg);*/

    Mat mean,stdDev;
    meanStdDev(sample_NxN, mean, stdDev);

    //cout <<stdDev.at<double>(2)<<endl;
    cout <<mean<<endl;
    cout <<stdDev<<endl;
    for(int i=0; i < 3; i++){

        lowBnd[i] = mean.at<double>(i)-stdDev.at<double>(i)*2;
        upBnd[i] = mean.at<double>(i)+stdDev.at<double>(i)*2;

        cout << "lowBnd["<<i<< "\t"<<lowBnd[i] <<", "<<upBnd[i] <<endl;
    }
    for(int j = 0; j <hsvImg.rows;j++){
        for(int i = 0; i <hsvImg.cols; i++){
            //grab hsv data 
            int hue = hsvImg.at<Vec3b>(j,i).val[0];
            int saturation = hsvImg.at<Vec3b>(j,i).val[1];
            int value = hsvImg.at<Vec3b>(j,i).val[2];
            if( hue >= lowBnd[0] && hue <= upBnd[0] &&
                saturation >= lowBnd[1] && saturation <= upBnd[1] &&
                value >= lowBnd[2] && value <= upBnd[2]){
                    //set those are in standard deviaiton to 0, preserve those defect part
                    defect.at<Vec3b>(j,i).val[0]=0;
                    defect.at<Vec3b>(j,i).val[1]=0;
                    defect.at<Vec3b>(j,i).val[2]=0;
            }
            else{// since the black defect part will confused with those are set to 0 above.
                // defect part BGR are SET TO 255 as white points as shown. 
                defect.at<Vec3b>(j,i).val[0]=255;
                defect.at<Vec3b>(j,i).val[1]=255;
                defect.at<Vec3b>(j,i).val[2]=255;
                //below is subtitute
                /*else if(lowBnd[0]==upBnd[0]&&lowBnd[1]==upBnd[1]&&lowBnd[2]==upBnd[2]){
                src.at<Vec3b>(j,i).val[0]=0;
                src.at<Vec3b>(j,i).val[1]=0;
                src.at<Vec3b>(j,i).val[2]=0;
                }*/
            }
        }   
    }
    imshow("defect_src",defect);
}

void grabImgPer_nFrames(Mat img, int nFrames){
    if(nFrames < 1 || !img.data) return;
    static int n = 0;

    if(n%nFrames == 0){
        int index = n/nFrames;
        char filename[80];
        sprintf(filename, "../images/test/save_%d.png",index);
        imwrite(filename, img);
    }
    n++;

}

bool isBlue(Mat src){
    //Mat src = imread("../images/blue.png");
    //resize(src,src, Size(320,240));

    Mat imgHSV;
    Mat img_thres;
    cvtColor(src, imgHSV, COLOR_BGR2HSV);        

    inRange(imgHSV,Scalar(100, 100, 100), Scalar(124, 255, 255),img_thres);

    imshow("img_thres_blue",img_thres);
    cvMoveWindow("img_thres_blue", 330, 10);

    int numOfParticles = 0;

    //total pixels with value in aiming rect
    for(int j = 0; j < img_thres.rows; j++){
        for(int i = 0; i< img_thres.cols; i++){
            int a =img_thres.at<uchar>(j,i);
            if(a){
                numOfParticles++;     
                //cout << (int)img_thres.at<uchar>(j,i);
            }
        }
    }
    cout << "numOfParticles\t" << numOfParticles <<endl;
    float total_area = img_thres.rows*img_thres.cols;
    cout << "total area\t" << total_area <<endl;    

    //chech the blue point is enough
    float ratioInBlue = float(numOfParticles)/total_area;
    if(ratioInBlue < 0.93){
        cout << "ratioInBlue is " <<ratioInBlue<<", lower than threshold value 93%" <<endl;
        return false;
    }
    else{
        cout << "ratioInBlue is " <<ratioInBlue<<endl;
    }
    return true;
}
bool isGreen(Mat src){

   // resize(src,src, Size(320,240));

    Mat imgHSV;
    cvtColor(src, imgHSV, COLOR_BGR2HSV);
    Mat img_thres;
    inRange(imgHSV,Scalar(56, 100, 100), Scalar(64, 255, 255),img_thres);

    imshow("img_thres_green",img_thres);
    cvMoveWindow("img_thres_green", 650, 10);
    int numOfParticles = 0;

    //total pixels with value in aiming rect
    for(int j = 0; j < img_thres.rows; j++){
        for(int i = 0; i< img_thres.cols; i++){
            int a =img_thres.at<uchar>(j,i);
            if(a){
                numOfParticles++;     
                //cout << (int)img_thres.at<uchar>(j,i);
            }
        }
    }
    cout << "numOfParticles\t" << numOfParticles <<endl;
    float total_area = img_thres.rows*img_thres.cols;
    cout << "total area\t" << total_area <<endl;

    //chech the blue point is enough
    float ratioInGreen = float(numOfParticles)/total_area;
    if(ratioInGreen < 0.93){
        cout << "ratioInGreen is " <<ratioInGreen<<", lower than threshold value 93%" <<endl;
        return false;
    }
    else{
        cout << "ratioInGreen is " <<ratioInGreen<<endl;
    }
    return true;
}
bool isRed(Mat src){

    //resize(src,src, Size(320,240));

    Mat imgHSV;
    cvtColor(src, imgHSV, COLOR_BGR2HSV);
    Mat img_thres;
    inRange(imgHSV,Scalar(0, 100, 100), Scalar(5, 255, 255),img_thres);

    //imshow("src",src);
    imshow("img_thres_red",img_thres);
    cvMoveWindow("img_thres_red", 970, 10);
    int numOfParticles = 0;

    //total pixels with value in aiming rect
    for(int j = 0; j < img_thres.rows; j++){
        for(int i = 0; i< img_thres.cols; i++){
            int a =img_thres.at<uchar>(j,i);
            if(a){
                numOfParticles++;     
                //cout << (int)img_thres.at<uchar>(j,i);
            }
        }
    }
    cout << "numOfParticles\t" << numOfParticles <<endl;
    float total_area = img_thres.rows*img_thres.cols;
    cout << "total area\t" << total_area <<endl;

    //chech the red point is enough
    float ratioInRed = float(numOfParticles)/total_area;
    if(ratioInRed < 0.93){
        cout << "ratioInRed is " <<ratioInRed<<", lower than threshold value 93%" <<endl;
        return false;
    }
    else{
        cout << "ratioInRed is " <<ratioInRed<<endl;
    }
    return true;
}


//not used now
void onMouse( int event, int x, int y, int flags, void* )
{    
    //if( x < 0 || x >= mhi_img->width || y < 0 || y >= mhi_img->height )
    //    return;
    ////左鍵放開或是 非拖曳狀態
    ////if( event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON) ){
    ////    // prevPt = Point(-1,-1);        
    ////}
    //else if( event == CV_EVENT_LBUTTONDOWN ){

    //    
    //}

    //draging
    //else if( event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) )
    //{
    //    isPainted = true;
    //    Point pt(x, y);
    //    if(x < min_X) 
    //        min_X = x;
    //    if(x > max_X) 
    //        max_X = x;
    //    if(y < min_Y) min_Y = y;
    //    if(y > max_Y) max_Y = y;
    //    if( prevPt.x < 0 )
    //        prevPt = pt;
    //    line( markerMask, prevPt, pt, Scalar::all(255), 2, 8, 0 );
    //    line( img_wtrShed, prevPt, pt, Scalar::all(255), 5, 8, 0 );
    //    //width_error += (pt.x-prevPt.x);
    //    prevPt = pt;
    //    imshow("image", img_wtrShed);



    //}

}

vector<Rect> getEdge(Mat image, int threshold) {
    Mat src_gray;
    GaussianBlur( image, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT );
    // 如果是彩色圖就轉換成灰階圖
    if (src_gray.channels() == 3) {
        cvtColor( src_gray, src_gray, CV_RGB2GRAY );
    }
    else if (src_gray.channels() == 1) {
        src_gray.copyTo(src_gray);//??????
    }
    else {
        cout << "image error" << endl;
    }

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad;//combined abs value of x with that of y
    Mat bin_grad;

    //Scharr濾波器計算x軸導數
    Scharr(src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
    // Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs(grad_x, abs_grad_x);
    //Scharr濾波器計算y軸導數
    Scharr(src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
    // Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    imshow("bf thres_grad", grad);
    cvMoveWindow("bf thres_grad", 660, 10);

    cv::threshold(grad, bin_grad, threshold, 255, cv::THRESH_BINARY);
    imshow("bin_grad", bin_grad);
    cvMoveWindow("bin_grad", 660, 250);

    vector<vector<Point>> contours;//定義儲存邊界需要的點
    vector<Vec4i> hierarchy;//定義儲存層次的向量

    //檢測所有輪廓並重構層次
    //CV_RETR_TREE：取得所有輪廓，以全階層的方式儲存
    //CV_RETR_EXTERNAL : 取出最外層的輪廓
    findContours(bin_grad, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    //畫出輪廓
    // drawContours(frame, contours, -1, Scalar(255, 0, 0));
    vector<Rect> rects;//儲存符合條件的外接矩形，由boundingRect函數返回
    int idx = 0;//輪廓個數 由hierarchy[idx][0] 疊代至下個輪廓的index
    if (contours.size())//必须加上此判断，否则当视频中只有背景时会出错
    {
        for (; idx >= 0; idx = hierarchy[idx][0])//找到面积最大的轮廓（hierarchy[idx][0]会指向下一个轮廓，若没有下一个轮廓则hierarchy[idx][0]为负数）
        {
            drawContours(grad, contours, idx, Scalar(255, 0, 0),3);//画出该轮廓线;;仅仅为了测试
            imshow("drawContours_grad", grad);
            cvMoveWindow("drawContours_grad", 660, 490);
            //imshow("grad_drawContour",grad);
            if (fabs(contourArea(Mat(contours[idx]))) > 10)//remain those contour with area over 400
            {
                cout << "==="<<endl;
                cout << "boundingRect of contour (x,y) = "<< boundingRect(contours[idx]).x
                    <<" , "<< boundingRect(contours[idx]).y <<endl;
                cout << "idx: "<< idx <<" area\t"<< fabs(contourArea(Mat(contours[idx]))) <<endl;
                rects.push_back(boundingRect(contours[idx]));//push 符合條件的外接矩形
            }
        }
        cout << "============================="<<endl;
    }
    return rects;
}