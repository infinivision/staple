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
  if(argc!=4){
    cout<< "Usage:  argv[1] video file"<< endl;
    cout<< "        argv[2] ground truth file"<< endl;
    cout<< "        argv[3] loops count"<< endl;
    return 0;
  }
  
  std::string video = argv[1];
  VideoCapture cap(video);
  if (!cap.isOpened()) {
      cout<<"can not open the camera"<<endl;
      cin.get();
      exit(1);
  }

  int index, x, y, h, w;
  std::ifstream gt;
  gt.open(argv[2]);
  if (!gt.is_open()) {
    cout << "ground truth file: "<<argv[2] << " , open failed" << endl;
    exit(1);
  }
  std::string line;
  getline(gt, line);
  std::stringstream ss;
  ss.str(line);
  ss >> index >> x >> y >> h >> w;

  if(!cap.set(CAP_PROP_POS_FRAMES,index)){
    cout<< "fail to set video start frame index!" << endl;
    exit(1);
  } 

  int loops = atoi(argv[3]);

  // Load the first frame.
  Mat frame ;
  cap >> frame;

  Rect_<float> roi;
  roi.x = x;
  roi.y = y;
  roi.height = h;
  roi.width = w;

  char * cfgPath = getenv("staple_cfg");
  if (cfgPath == nullptr) {
    cout << "env staple_cfg is null, read default cfg file staple.yaml" 
          << endl;
    cfgPath = (char *)"staple.yaml";
  }
  FileStorage fs(cfgPath, FileStorage::READ);
  staple_cfg cfg;
  cfg.read(fs.root());
  STAPLE_TRACKER staple(cfg);
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
    if(frame_idx>=loops)break;
    //cap >> frame;
    if(frame.rows==0 || frame.cols==0)break;
    // update the tracking result
    t1 = cv::getTickCount();
    roi = staple.tracker_staple_update(frame);
    staple.tracker_staple_train(frame, false);
    t2 = cv::getTickCount();
    tick_counter += t2 - t1;
    frame_idx++;

  }

  cout << "Elapsed sec: " << static_cast<double>(tick_counter) / cv::getTickFrequency() << endl;
  cout << "FPS: " << ((double)(frame_idx)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;

}





