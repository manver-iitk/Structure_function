#pragma once
#include <cuda.h>
#include <iostream>
#include <hdf5.h>
#include <Python.h>

// Error checker FUNCTIONS

extern "C++" __forceinline__ void gpuerrcheck_cudaerror(cudaError_t err, int line, std::string file_name) // CUDA ERROR CHECKER
{
    if (err != 0)
    {
        std::cout << "\n cuda error  = " << cudaGetErrorString(err) << " , At line " << line << "\n In File " << file_name << " , aborting " << std::endl;
        exit(0);
    }
}

extern "C++" __forceinline__ void HDF5_err_check(herr_t status, int line, std::string file_name) // HDF5 Error checker
{
    if (status)
    {
        std::cout << "Error in HDF5 Functions at line no " << line << "\n In File " << file_name << " , aborting " << std::endl;
        exit(0);
    }
}

extern "C++" __forceinline__ void python_err_check(PyObject *data_pointer, int line, std::string file_name) // Python Error Checker
{
    if (data_pointer == NULL)
    {
        std::cout << "Error in Python Functions call at line no " << line << "\n In File " << file_name << " , aborting " << std::endl;
        exit(0);
    }
}

// Define the function to be called when ctrl-c (SIGINT) is sent to process
extern "C++" __forceinline__ void signal_callback_handler(int signum);