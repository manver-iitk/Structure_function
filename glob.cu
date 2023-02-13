#pragma once
#include "headers.cuh"
#include "err_check.cuh"

// #### Grid dimensions ####
int dimension;                  // CPU
__constant__ int dimension_gpu; // GPU

int Nx, Ny, Nz;                          // CPU
__constant__ int Nx_gpu, Ny_gpu, Nz_gpu; // GPU

template <typename T>
T Lx;
template <typename T>
__constant__ T Lx_gpu;

template <typename T>
T Ly;
template <typename T>
__constant__ T Ly_gpu;

template <typename T>
T Lz;
template <typename T>
__constant__ T Lz_gpu;

template <typename T>
T dx;
template <typename T>
__constant__ T dx_gpu;

template <typename T>
T dy;
template <typename T>
__constant__ T dy_gpu;

template <typename T>
T dz;
template <typename T>
__constant__ T dz_gpu;
// #########################

// ######## Precision ########
std::string precision; // CPU
// ###########################

// ######## Counters #########
int iteration_count; // Counting Total No of iterations. (CPU)
// ###########################

// ## Cuda Device ID #####
int Device_Id;
// #######################

// ####### Python Object Parameters ########
PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
// #########################################

extern "C++" void set_precision()
{
    // std::cout << "\n Setting the precision and device id";

    // Opening Python Script file
    pName = PyUnicode_FromString((char *)"para");
    python_err_check(pName, __LINE__, __FILE__);
    pModule = PyImport_Import(pName);
    python_err_check(pModule, __LINE__, __FILE__);

    // Opening The function named id_precision_kind in python script
    pFunc = PyObject_GetAttrString(pModule, (char *)"id_precision_kind");
    python_err_check(pFunc, __LINE__, __FILE__);
    pValue = PyObject_CallObject(pFunc, nullptr);
    python_err_check(pValue, __LINE__, __FILE__);

    // Setting of CUDA device
    Device_Id = PyLong_AsLong(PyList_GET_ITEM(pValue, 0));
    cudaSetDevice(Device_Id);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    // Setting precision
    precision = _PyUnicode_AsString(PyList_GET_ITEM(pValue, 1));
    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
}

template <typename T>
void init_global_arrays()
{
    // std::cout << "\n Setting the Dimension and sizes ";

    // Opening Function Grid_parameters
    pFunc = PyObject_GetAttrString(pModule, (char *)"grid_parameters");
    python_err_check(pFunc, __LINE__, __FILE__);
    pValue = PyObject_CallObject(pFunc, nullptr);
    python_err_check(pValue, __LINE__, __FILE__);

    // #### Setting Grid and Dimension of simulation ####
    dimension = PyLong_AsLong(PyList_GET_ITEM(pValue, 0));
    Nx = PyLong_AsLong(PyList_GET_ITEM(pValue, 1));
    Ny = PyLong_AsLong(PyList_GET_ITEM(pValue, 2));
    Nz = PyLong_AsLong(PyList_GET_ITEM(pValue, 3));
    Lx<T> = PyLong_AsLong(PyList_GET_ITEM(pValue, 4));
    Ly<T> = PyLong_AsLong(PyList_GET_ITEM(pValue, 5));
    Lz<T> = PyLong_AsLong(PyList_GET_ITEM(pValue, 6));

    dx<T> = Lx<T> / Nx;
    dy<T> = Ly<T> / Ny;
    dz<T> = Lz<T> / Nz;

    cudaMemcpyToSymbol(Nx_gpu, &Nx, sizeof(int));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(Ny_gpu, &Ny, sizeof(int));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(Nz_gpu, &Nz, sizeof(int));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(Lx_gpu<T>, &(Lx<T>), sizeof(T));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(Ly_gpu<T>, &(Ly<T>), sizeof(T));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(Lz_gpu<T>, &(Lz<T>), sizeof(T));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(dx_gpu<T>, &(dx<T>), sizeof(T));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(dy_gpu<T>, &(dy<T>), sizeof(T));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(dz_gpu<T>, &(dz<T>), sizeof(T));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpyToSymbol(dimension_gpu, &(dimension), sizeof(int));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);
    // ##################################################
}

// ############# Explicit instantiation ################
template void init_global_arrays<T_f>();
template void init_global_arrays<T_d>();

template <>
T_f Lx<T_f>;
template <>
T_d Lx<T_d>;
template <>
__constant__ T_f Lx_gpu<T_f>;
template <>
__constant__ T_d Lx_gpu<T_d>;

template <>
T_f Ly<T_f>;
template <>
T_d Ly<T_d>;
template <>
__constant__ T_f Ly_gpu<T_f>;
template <>
__constant__ T_d Ly_gpu<T_d>;

template <>
T_f Lz<T_f>;
template <>
T_d Lz<T_d>;
template <>
__constant__ T_f Lz_gpu<T_f>;
template <>
__constant__ T_d Lz_gpu<T_d>;

template <>
T_f dx<T_f>;
template <>
T_d dx<T_d>;
template <>
__constant__ T_f dx_gpu<T_f>;
template <>
__constant__ T_d dx_gpu<T_d>;

template <>
T_f dy<T_f>;
template <>
T_d dy<T_d>;
template <>
__constant__ T_f dy_gpu<T_f>;
template <>
__constant__ T_d dy_gpu<T_d>;

template <>
T_f dz<T_f>;
template <>
T_d dz<T_d>;
template <>
__constant__ T_f dz_gpu<T_f>;
template <>
__constant__ T_d dz_gpu<T_d>;
// #####################################################
