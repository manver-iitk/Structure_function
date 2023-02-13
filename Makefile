NVCC= nvcc -std=c++17 -arch=sm_75
CXX= g++ -std=c++17
CC= gcc -std=c17
python= python3

target?=linux

NVCC_FLAGS= --device-c --disable-warnings
CC_FLAGS= 

#################################
ifeq ($(target),windows)

HDF5_HOME=D:/HDF_Group/HDF5/1.12.2
HDF5_INC= $(HDF5_HOME)/include/
HDF5_LIB= $(HDF5_HOME)/lib/

CUDA_HOME=D:/NVIDIA_GPU_Computing_Toolkit/CUDA/11.8
NV_TOOLS_EXT="C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/"
CUDA_INC=$(CUDA_HOME)/include/
CUDA_LIB=$(CUDA_HOME)/lib/x64/,$(NV_TOOLS_EXT)

PYTHON_INC=D:/Python310/include/
PYTHON_LIB=D:/Python310/libs/

CUDA_link= -lcuda
CUFFT_link= -lcufft
HDF5_LINK= -lhdf5
NCCL_link=
NVTX= -lnvToolsExt64_1
RDC= -rdc=true
Python = -lpython3

Oflag= -Xptxas -O3 -DWIN32 -DH5_BUILT_AS_DYNAMIC_LIB

else

HDF5_HOME=/usr
HDF5_INC= $(HDF5_HOME)/include/
HDF5_LIB= $(HDF5_HOME)/lib/

CUDA_HOME=/usr/local/cuda-11.8
CUDA_INC=$(CUDA_HOME)/include
CUDA_LIB=$(CUDA_HOME)/lib64

PYTHON_INC=/usr/include/python3.10/
PYTHON_LIB=/usr/lib/python3.10/config-3.10-x86_64-linux-gnu/

CUFFT_link= #-lcufft
CUDA_link= -lcuda
HDF5_LINK= -lhdf5
NVTX= -lnvToolsExt
RDC= -rdc=true
Python = -lpython3.10

Oflag= -Xptxas -O3
endif
#######################################


INC_ALL= $(PYTHON_INC),$(CUDA_INC),$(HDF5_INC)
LIB_ALL= $(CUDA_LIB),$(HDF5_LIB),$(PYTHON_LIB)

Link_all= $(CUFFT_link) $(HDF5_LINK) $(Python) $(CUDA_link)

str_function: compile_str_func clean

############ Compilling the Str_function ##########
compile_str_func: glob.o IO.o main.o
	$(NVCC) glob.o io.o main.o $(Oflag) -L $(LIB_ALL) $(Link_all) -o str_func

glob.o: glob.cu headers.cuh err_check.cuh
	$(NVCC) glob.cu -I $(INC_ALL) -L $(LIB_ALL) $(NVCC_FLAGS) $(Oflag) -c -o glob.o

IO.o: io.cu headers.cuh err_check.cuh
	$(NVCC) io.cu -I $(INC_ALL) -L $(LIB_ALL) $(NVCC_FLAGS) $(Oflag) -c -o io.o

main.o: main.cu headers.cuh err_check.cuh
	$(NVCC) main.cu -I $(INC_ALL) -L $(LIB_ALL) $(NVCC_FLAGS) $(Oflag) -c -o main.o
###################################################

######### Run str_function #########
run: str_func
	./str_func
####################################

clean: 
	rm -rf *.o
