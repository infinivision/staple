#include "fftw3.h"
#include <iostream>
#include <string>
#include <cassert>
using namespace std;
int main(int argc,char * argv[]) {
    if(argc < 2){
      cout << "input error!"
           << endl;
      exit(1);
    }
    int row; 
    int col;
    float * realInput;
    fftwf_complex * complexOutput;

    fftwf_complex * complexInput;
    float * realOutput;

    for(int i= 1;i<argc;i++){
      col = atoi(argv[i]);
      row = col;
      realInput   = (float *)  fftwf_malloc(sizeof (float) * row * col );
      assert(realInput!=nullptr);
      complexOutput = (fftwf_complex *) fftwf_malloc(sizeof (fftwf_complex) * row * (col/2+1));
      assert(complexOutput!=nullptr);
      fftwf_plan r2c = fftwf_plan_dft_r2c_2d(row, col, realInput, complexOutput, FFTW_PATIENT);

      if(r2c == nullptr ){
        cout << "fftwf create r2c plan failed!" << endl;
        cout << "plan row: " << row << " col: " << col << endl;
        exit(1);
      } else {
        cout << "fftwf success to create fft r2c!" << endl;
        cout << "plan row: " << row << " col: " << col << endl;
      }

      complexInput = (fftwf_complex *) fftwf_malloc(sizeof (fftwf_complex) * row * (col/2+1) );
      assert(complexInput!=nullptr);
      realOutput = (float *) fftwf_malloc(sizeof (float) * row * col);
      assert(realOutput!=nullptr);
      fftwf_plan c2r = fftwf_plan_dft_c2r_2d(row, col, complexInput, realOutput, FFTW_PATIENT);

      if(c2r == nullptr){
        cout << "fftwf create c2r failed!" << endl;
        cout << "plan row: " << row << " col: " << col << endl;
        exit(1);
      } else {
        cout << "fftwf success to create c2r!" << endl;
        cout << "plan row: " << row << " col: " << col << endl;
      }
    }
    
    string wisdomFile = "wisdom";
    if(1==fftwf_export_wisdom_to_filename(wisdomFile.c_str()))
      cout << "fftwf_export_wisdom_to_filename wisdom success" << endl;
    else
      cout << "fftwf_export_wisdom_to_filename wisdom fail" << endl;
  }
