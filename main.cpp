#include "staple_tracker.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <ctime>
#include <unistd.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char * argv[])
{
  int loops = 0;
  if(argc!=3){
    if(argc!=4){
      cout<<
  //      " Usage:  argv[1] camara index \n"
        " Usage:  argv[1] video file argv[2] start frame index argv[3] loops time\n"
        << endl;
      return 0;      
    } else {
      loops = atoi(argv[3]);
    }
  } else {
    loops = 1000000;
  }   

  std::string video = argv[1];
  VideoCapture cap(video);
  if (!cap.isOpened()) {
      cout<<"can not open the camera"<<endl;
      cin.get();
      exit(1);
  }
  if(!cap.set(CAP_PROP_POS_FRAMES,int(atoi(argv[2])))){
    cout<< "fail to set video start frame index!" << endl;
    exit(1);
  }

  Mat frame1,frame2;
  // get bounding box
  cap >> frame1;
  cap >> frame2;
  //resize(frame1,frame2,frame1.size()/2);
  //cv::namedWindow("tracker",WINDOW_NORMAL);
  //cv::resizeWindow("tracker",800,640);
  Rect_<float> roi= selectROI("tracker", frame1, true, false);
  //Rect_<float> roi= selectROI("tracker", frame2, true, false);
  //quit if ROI was not selected
  if(roi.width==0 || roi.height==0)
    return 0;

  if(roi.width > roi.height){
    roi.y -= (roi.width - roi.height)/2;
    roi.height = roi.width;
  } else {
    roi.x -= ( roi.height - roi.width )/2;
    roi.width = roi.height;
  }

  STAPLE_TRACKER staple;
  // initialize the tracker
  int64 t1 = cv::getTickCount();

  staple.tracker_staple_initialize(frame2, roi);
  staple.tracker_staple_train(frame2, true);  
  int64 t2 = cv::getTickCount();
  int64 tick_counter = t2 - t1;
  int frame_idx = 1;
  // do the tracking
  printf("Start the tracking process, press ESC to quit.\n");

  clock_t timer;
  double  fps;
  String text;
  char buffer [128];
  bool pause = false;

  for (;;) {

    int res = waitKey(1);
    //quit on ESC button
    if(res==27)break;    
    if(pause==true){
      if(res==32){
        pause = false;
      } else {
        continue;
      }
    } else {
      if(res==32){
        pause = true;
        continue;
      }
    }
    // get frame from the video
    cap >> frame1;
    cap >> frame2;
    // stop the program if no more images
    if(frame1.rows==0 || frame1.cols==0)
      break;
    if(frame_idx>=loops)
      break;
    //resize(frame1,frame2,frame1.size()/2);
    double frameNO = cap.get(CAP_PROP_POS_FRAMES);
    

    // update the tracking result
    t1 = cv::getTickCount();
    timer=clock();
    roi = staple.tracker_staple_update(frame2);
    staple.tracker_staple_train(frame2, false);
    t2 = cv::getTickCount();
    timer=clock()-timer;
    tick_counter += t2 - t1;
    fps=(double)CLOCKS_PER_SEC/(double)timer;
    frame_idx++;

    rectangle( frame2, roi, Scalar( 255, 0, 0 ), 2, 1 );

    sprintf (buffer, "speed: %.0f fps frame index:%d", fps,int(frameNO));
    text = buffer;
    putText(frame2, text, Point(20,20), FONT_HERSHEY_PLAIN, 1.25, Scalar(0,0,0),2);
    sprintf (buffer, "roi length: %d ", int(roi.width));
    text = buffer;
    putText(frame2, text, Point(20,35), FONT_HERSHEY_PLAIN, 1.25, Scalar(0,0,0),2);    
    // show image with the tracked object
    imshow("tracker",frame2);

  }

  cout << "Elapsed sec: " << static_cast<double>(tick_counter) / cv::getTickFrequency() << endl;
  cout << "FPS: " << ((double)(frame_idx)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;

}





