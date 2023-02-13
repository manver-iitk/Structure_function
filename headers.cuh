#pragma once
#define USE_DEFINED_PARA

#if defined(_MSC_VER)
#define inline_qualifier __inline
#define _USE_MATH_DEFINES
#include <direct.h>
#else
#define inline_qualifier inline
#endif

// C++ Includes
#include <iostream>
#include <string>
#include <chrono>
#include <memory>
#include <math.h>
#include <cmath>
#include <vector>
#include <initializer_list>
#include <type_traits>
#include <algorithm>
#include <iomanip>
#include <Python.h>
#include <sys/stat.h>
#include <cstdlib>
#include <signal.h>

// CUDA Includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <nvtx3/nvToolsExtCuda.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda_profiler_api.h>

// IO Includes
#include "hdf5.h"

// ########### Type Definations ###########
typedef float T_f;
typedef double T_d;
// #########################################

// ################### Global variables #################
// #### Grid dimensions ####
extern int dimension;                  // CPU
extern __constant__ int dimension_gpu; // GPU

extern int Nx, Ny, Nz;                          // CPU
extern __constant__ int Nx_gpu, Ny_gpu, Nz_gpu; // GPU

template <typename T>
extern T Lx;
template <typename T>
extern __constant__ T Lx_gpu;

template <typename T>
extern T Ly;
template <typename T>
extern __constant__ T Ly_gpu;

template <typename T>
extern T Lz;
template <typename T>
extern __constant__ T Lz_gpu;

template <typename T>
extern T dx;
template <typename T>
extern __constant__ T dx_gpu;

template <typename T>
extern T dy;
template <typename T>
extern __constant__ T dy_gpu;

template <typename T>
extern T dz;
template <typename T>
extern __constant__ T dz_gpu;
// #########################

// ######## Precision ########
extern std::string precision; // CPU
// ###########################

// ######## Counters #########
extern int iteration_count; // Counting Total No of iterations. (CPU)
// ###########################

// ## Cuda Device ID #####
extern int Device_Id;
// #######################

// ####### Python Object Parameters ########
extern PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
// #########################################
// ######################################################

// ###### Data Variables ########
template <typename T>
extern T *Vx_cpu;

template <typename T>
extern T *Vy_cpu;

template <typename T>
extern T *Vz_cpu;

template <typename T>
extern T *Vx_gpu;

template <typename T>
extern T *Vy_gpu;

template <typename T>
extern T *Vz_gpu;

template <typename T>
extern T *S_upll_gpu;

template <typename T>
extern T *S_u_r_gpu;

template <typename T>
extern T *S_ux_gpu;

template <typename T>
extern T *S_uy_gpu;

template <typename T>
extern T *S_uz_gpu;
// ##############################

// ###################### Functions definations #################

// ######## Functions IO ############
template <typename T>
void initialize_io();

template <typename T>
void reader_io();

template <typename T>
void writer_io();
// ##################################

// ######## Global Functions ########
extern "C++" void set_precision();

template <typename T>
void init_global_arrays();
// ##################################

// ################# Main functions #############
template <typename T>
void initialize_memory();

template <typename T>
void copy_to_gpu();
// #############################################

// Printing parameters
template <typename T>
void print_parameters();

// Define the function to be called when ctrl-c (SIGINT) is sent to process
extern "C++" __forceinline__ void signal_callback_handler(int signum);