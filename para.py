###### CUDA DEVICE ID #######
device_id = 0
#############################

######## Dimensions Parameters #######
Dimension = 3
Nx = 256
Ny = 256
Nz = 256
######################################

############ Length Parameters ########
Lx = 1
Ly = 1
Lz = 1
######################################

##### Precision of Simulation ######
precision = "double" ## "single" or "double"
####################################

################# I/O Parameters ########################
input_dir = "/mnt/e/Codes/Git_codes/input" ##### Path of INPUT FEILDS #########
output_dir = "/mnt/e/Codes/Git_codes/output" ##### Path of INPUT FEILDS #########
Name_of_input_file = "Tarang_256^3.h5" ################### Name of input feild data #########

# Name of datasets in input and output .h5 files
Ux = "Ux"
Uy = "Uy"
Uz = "Uz"
################################################




##################### Please dont change these functions #################
def id_precision_kind():
    return [device_id,precision]

def grid_parameters():
    return [Dimension,Nx,Ny,Nz,Lx,Ly,Lz]

def IO_parameters():
    return [input_dir,output_dir,Name_of_input_file]

def name_of_datasets():
    return [Ux,Uy,Uz]
##########################################################################