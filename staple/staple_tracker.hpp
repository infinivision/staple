#ifndef STAPLE_TRACKER_HPP
#define STAPLE_TRACKER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "fftw3.h"
#include <Eigen/Dense>
using namespace Eigen;
typedef Matrix<float,Dynamic,Dynamic,RowMajor> MatrixF;
using namespace std;

#define _PI 3.141592653589793
#define _2PI 6.283185307179586

struct staple_cfg
{
    bool grayscale_sequence = false;    // suppose that sequence is colour
    int hog_cell_size = 4;
    int fixed_area = 128*128;           // standard area to which we resize the target
    int n_bins = 2*2*2*2*2;             // number of bins for the color histograms (bg and fg models)
    double learning_rate_pwp = 0.04;    // bg and fg color models learning rate
    const char * feature_type = "fhog"; // "fhog", ""gray""
    double inner_padding = 0.2;         // defines inner area used to sample colors from the foreground
    double output_sigma_factor = 1/16.0; // standard deviation for the desired translation filter output
    double lambda = 1e-3;               // egularization weight
    double learning_rate_cf = 0.01;     // HOG model learning rate
    double merge_factor = 0.3;          // fixed interpolation factor - how to linearly combine the two responses
    const char * merge_method = "const_factor";
    bool den_per_channel = false;
    float bg_padding_rate = 0.5;

    // scale related
    bool scale_adaptation = true;
    int hog_scale_cell_size = 4;         // Default DSST=4
    double learning_rate_scale = 0.025;
    double scale_sigma_factor = 1/4.0;
    int num_scales = 25;
    double scale_model_factor = 1.0;
    double scale_step = 1.02;
    double scale_model_max_area = 16*16;

    // debugging stuff
    int visualization = 0;              // show output bbox on frame
    int visualization_dbg = 0;          // show also per-pixel scores, desired response and filter output

    cv::Point_<float> init_pos;
    cv::Size target_sz;
};

class STAPLE_TRACKER
{
public:
    STAPLE_TRACKER(){ cfg = default_parameters_staple(cfg); frameno = 0; };
    ~STAPLE_TRACKER(){}

    void mexResize(const cv::Mat &im, cv::Mat &output, cv::Size newsz, const char *method);
    void tracker_staple_train(const cv::Mat &im, bool first);
    void tracker_staple_initialize(const cv::Mat &im, cv::Rect_<float> region);
    cv::Rect tracker_staple_update(const cv::Mat &im);

protected:
    staple_cfg default_parameters_staple(staple_cfg cfg);
    void initializeAllAreas(const cv::Mat &im);

    void inline fftwInit(int row, int col, int cn);
    void inline fftTool(const vector<cv::Mat> & src, vector<cv::Mat> & dest);
    void inline fftTool(const cv::Mat & src, cv::Mat & dest);
    void inline ifftTool(const cv::Mat & src, cv::Mat & dest);
    void inline sumChannels(std::vector<cv::Mat> & src, cv::Mat & dest, bool real);
    void getSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output);
    void getSubwindowFloor(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output);
    void updateHistModel(bool new_model, cv::Mat &patch, double learning_rate_pwp=0.0);
    void CalculateHann(cv::Size sz, cv::Mat &output);
    void gaussianResponse(cv::Size rect_size, double sigma, cv::Mat &output);
    void getFeatureMap(cv::Mat &im_patch, const char *feature_type, cv::MatND &output);
    void cropFilterResponse(const cv::Mat &response_cf, cv::Size response_size, cv::Mat& output);
    void getColourMap(const cv::Mat &patch, cv::Mat& output);
    void getCenterLikelihood(const cv::Mat &object_likelihood, cv::Size m, cv::Mat& center_likelihood);
    void mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp, cv::Mat &response);
    void getScaleSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Mat &output);

private:
    staple_cfg cfg;

    cv::Point_<float> pos;
    cv::Size target_sz;

    cv::Size bg_area;
    cv::Size fg_area;
    double area_resize_factor;
    cv::Size cf_response_size;

    cv::Size norm_bg_area;
    cv::Size norm_target_sz;
    cv::Size norm_delta_area;
    cv::Size norm_pwp_search_area;

    // add for memory optimization
    cv::Mat im_patch_pwp;
    cv::Mat im_patch_bg;

    cv::Mat fg_mask_new;
    cv::Mat bg_mask_new;

    cv::Mat likelihood_map;
    cv::Mat response_pwp;
    cv::Mat response;

    cv::MatND xt;
    cv::Mat xt2;
    cv::Mat xtf2,     xtfr2;
    cv::Mat xtf2_old, xtfr2_old;
    cv::Mat xtfr_sum;
    cv::Mat xtf2_windows;
    // temp matrix for getFeatureMap
    cv::Mat new_im_patch;
    cv::Mat grayimg;
    cv::Mat grayimgf;

    // add for eigen fftw
    cv::Mat y_fftw,yf_fftw;
    vector<cv::Mat> xtv;
    vector<cv::Mat> xtfrv;
    vector<cv::Mat> xtfv;
    vector<cv::Mat> xtfv_windows;

    cv::Mat response_cff_fftw;
    cv::Mat response_cf_fftw;
    fftwf_plan fftPlan;
    fftwf_plan ifftPlan;

    cv::MatND bg_hist;
    cv::MatND fg_hist;

    cv::Mat hann_window;
    cv::Mat yf;

    std::vector<cv::Mat> hf_den;
    std::vector<cv::Mat> hf_num;

    cv::Rect rect_position;

    float scale_factor;
    cv::Mat scale_window;
    cv::Mat scale_factors;
    cv::Size scale_model_sz;
    float min_scale_factor;
    float max_scale_factor;
    cv::Size base_target_sz;

    cv::Mat ysf;
    cv::Mat sf_den;
    cv::Mat sf_num;

    int frameno;
};

#endif