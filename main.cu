#include "headers.cuh"
#include "err_check.cuh"

// ###### Data Variables ########
template <typename T>
T *Vx_cpu;
template <typename T>
T *Vx_gpu;

template <typename T>
T *Vy_cpu;
template <typename T>
T *Vy_gpu;

template <typename T>
T *Vz_cpu;
template <typename T>
T *Vz_gpu;

template <typename T>
T *S_upll_gpu;

template <typename T>
T *S_u_r_gpu;

template <typename T>
T *S_ux_gpu;

template <typename T>
T *S_uy_gpu;

template <typename T>
T *S_uz_gpu;
// ##############################

__forceinline__ void signal_callback_handler(int signum)
{
    std::cout << "Caught signal " << signum << std::endl;
    // Terminate program
    exit(signum);
}

template <typename T>
void initialize_memory()
{
    // std::cout<<"\n Allocating the memory in CPU and GPU  ";

    Vx_cpu<T> = (T *)malloc(sizeof(T) * Nx * Ny * Nz);
    Vy_cpu<T> = (T *)malloc(sizeof(T) * Nx * Ny * Nz);
    Vz_cpu<T> = (T *)malloc(sizeof(T) * Nx * Ny * Nz);

    cudaMalloc(&(Vx_gpu<T>), sizeof(T) * Nx * Ny * Nz);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMalloc(&(Vy_gpu<T>), sizeof(T) * Nx * Ny * Nz);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMalloc(&(Vz_gpu<T>), sizeof(T) * Nx * Ny * Nz);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMalloc(&(S_upll_gpu<T>), sizeof(T) * Nx * Ny * Nz);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMalloc(&(S_u_r_gpu<T>), sizeof(T) * Nx * Ny * Nz);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMalloc(&(S_ux_gpu<T>), sizeof(T) * Nx * Ny * Nz);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMalloc(&(S_uy_gpu<T>), sizeof(T) * Nx * Ny * Nz);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMalloc(&(S_uz_gpu<T>), sizeof(T) * Nx * Ny * Nz);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);
}

template <typename T>
void copy_to_gpu()
{
    // std::cout<<"\n Copying the data To GPU \n";

    cudaMemcpy(Vx_gpu<T>, Vx_cpu<T>, sizeof(T) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpy(Vy_gpu<T>, Vy_cpu<T>, sizeof(T) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);

    cudaMemcpy(Vz_gpu<T>, Vz_cpu<T>, sizeof(T) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1, __FILE__);
}

int main()
{
    Py_Initialize();

    PyRun_SimpleString("import sys\n"
                       "import os\n"
                       "sys.path.append(os.getcwd())\n");

    // Setting the signal for interupt
    signal(SIGINT, signal_callback_handler);

    // Setting the Precision
    set_precision();

    if (!precision.compare("single"))
    {
        // std::cout << "\n This is Single precision ";
        init_global_arrays<T_f>();
        initialize_io<T_f>();
        initialize_memory<T_f>();
        reader_io<T_f>();
        copy_to_gpu<T_f>();

        print_parameters<T_f>();
    }

    else if (!precision.compare("double"))
    {
        // std::cout << "\n This is Double precision ";

        init_global_arrays<T_d>();
        initialize_io<T_d>();
        initialize_memory<T_d>();
        reader_io<T_d>();
        copy_to_gpu<T_d>();

        print_parameters<T_d>();
    }

    // Finalize the python
    Py_Finalize();

    return 0;
}

// Explicit instantiation
template <>
T_f *Vx_cpu<T_f>;
template <>
T_d *Vx_cpu<T_d>;

template <>
T_f *Vy_cpu<T_f>;
template <>
T_d *Vy_cpu<T_d>;

template <>
T_f *Vz_cpu<T_f>;
template <>
T_d *Vz_cpu<T_d>;

template <>
T_f *Vx_gpu<T_f>;
template <>
T_d *Vx_gpu<T_d>;

template <>
T_f *Vy_gpu<T_f>;
template <>
T_d *Vy_gpu<T_d>;

template <>
T_f *Vz_gpu<T_f>;
template <>
T_d *Vz_gpu<T_d>;

template <>
T_f *S_upll_gpu<T_f>;
template <>
T_d *S_upll_gpu<T_d>;

template <>
T_f *S_u_r_gpu<T_f>;
template <>
T_d *S_u_r_gpu<T_d>;

template <>
T_f *S_ux_gpu<T_f>;
template <>
T_d *S_ux_gpu<T_d>;

template <>
T_f *S_uy_gpu<T_f>;
template <>
T_d *S_uy_gpu<T_d>;

template <>
T_f *S_uz_gpu<T_f>;
template <>
T_d *S_uz_gpu<T_d>;

template void initialize_memory<T_f>();
template void initialize_memory<T_d>();
