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

  Mat frame;
  Mat show_frame;
  // get bounding box
  cap >> frame;
  //cv::namedWindow("tracker",WINDOW_NORMAL);
  //cv::resizeWindow("tracker",800,640);
  cv::resize(frame,show_frame,cv::Size(1024,768));
  Rect_<float> roi= selectROI("tracker", show_frame, true, false);

  roi.x = roi.x / 1024.0 * frame.cols;
  roi.y = roi.y / 768.0 * frame.rows;
  roi.width   = roi.width  / 1024.0 * frame.cols;
  roi.height  = roi.height / 768.0 * frame.rows;

  //quit if ROI was not selected
  if(roi.width==0 || roi.height==0)
    return 0;
  /*
  FileStorage fs("staple.yaml",FileStorage::WRITE);
  staple_cfg cfg;
  cfg.write(fs);
  fs.release();
  */
  char * cfgPath = getenv("staple_cfg");
  if (cfgPath == nullptr) {
    cout << "env staple_cfg is null, read default cfg file staple.yaml" 
          << endl;
    cfgPath = (char *)"staple.yaml";
  } 
  FileStorage fs(cfgPath, FileStorage::READ);
  staple_cfg cfg;
  cfg.read(fs.root());

  STAPLE_TRACKER::importWisdom();

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
    cap >> frame;
    // stop the program if no more images
    if(frame.rows==0 || frame.cols==0)
      break;
    if(frame_idx>=loops)
      break;
    double frameNO = cap.get(CAP_PROP_POS_FRAMES);
    

    // update the tracking result
    t1 = cv::getTickCount();
    timer=clock();
    roi = staple.tracker_staple_update(frame);
    staple.tracker_staple_train(frame, false);
    t2 = cv::getTickCount();
    timer=clock()-timer;
    tick_counter += t2 - t1;
    fps=(double)CLOCKS_PER_SEC/(double)timer;
    frame_idx++;

    rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );

    sprintf (buffer, "speed: %.0f fps frame index:%d, frame height[%d] width[%d]", fps,int(frameNO),frame.size().height,frame.size().width);
    text = buffer;
    putText(frame, text, Point(20,20), FONT_HERSHEY_PLAIN, 1.25, Scalar(0,0,0),2);
    sprintf (buffer, "roi length: %d , roi.x[%4d],roi.y[%4d]", int(roi.width),int(roi.x),int(roi.y));
    text = buffer;
    putText(frame, text, Point(20,35), FONT_HERSHEY_PLAIN, 1.25, Scalar(0,0,0),2);    
    // show image with the tracked object
    cv::resize(frame,show_frame,cv::Size(1024,768));
    imshow("tracker",show_frame);

  }

  cout << "Elapsed sec: " << static_cast<double>(tick_counter) / cv::getTickFrequency() << endl;
  cout << "FPS: " << ((double)(frame_idx)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;

}





