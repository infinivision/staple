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
  if(argc!=3){
    cout<< "Usage:  argv[1] img file argv[2] loops count"
        << endl;
    return 0;
  }  
  // Load the first frame.     
  Mat frame = imread(argv[1]);
  int loops = atoi(argv[2]);
  
  Rect_<float> roi;
  roi.x = 100;
  roi.y = 100;
  roi.width = 120;
  roi.height = 120;

  STAPLE_TRACKER staple;
  // initialize the tracker
  int64 t1 = cv::getTickCount();
  staple.tracker_staple_initialize(frame, roi);
  staple.tracker_staple_train(frame, true);  
  int64 t2 = cv::getTickCount();
  int64 tick_counter = t2 - t1;
  int frame_idx = 1;
  // do the tracking
  printf("Start the tracking process, press ESC to quit.\n");

  for (;;) {
    // update the tracking result
    t1 = cv::getTickCount();
    roi = staple.tracker_staple_update(frame);
    staple.tracker_staple_train(frame, false);
    t2 = cv::getTickCount();
    tick_counter += t2 - t1;
    frame_idx++;
    if(frame_idx>=loops)break;
  }

  cout << "Elapsed sec: " << static_cast<double>(tick_counter) / cv::getTickFrequency() << endl;
  cout << "FPS: " << ((double)(frame_idx)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;

}





