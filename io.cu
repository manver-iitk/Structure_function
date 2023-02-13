#include "headers.cuh"
#include "err_check.cuh"

// ########## File Storage PATHS ##############
std::string output_dir;
std::string input_dir;
// ############################################

// ############# HDF5 Handles ###############
herr_t status; // Teels the status of any hdf5 command
hid_t plist_id;
hid_t fileHandle;
hid_t dataSet;
hid_t mem_type;
hid_t sourceDSpace, targetDSpace;
// ##########################################

// ####### offset Dimensions ###########
hsize_t dimsf[3];  /* dataset dimensions */
hsize_t offset[3]; /* offset of hyperslab */
// #####################################

// ########## Strings names ##########
std::string file_name;
std::string data_name[3];
// ###################################

// ## String for copying Prameters file to output folder ##
std::string command_to_create_field_folder;
// ########################################################

template <typename T>
void initialize_io()
{
    std::cout << "\n Initializing the IO data ";

    // ################# Setting Input Parameters #####################
    pFunc = PyObject_GetAttrString(pModule, (char *)"IO_parameters");
    python_err_check(pFunc, __LINE__, __FILE__);
    pValue = PyObject_CallObject(pFunc, nullptr);
    python_err_check(pValue, __LINE__, __FILE__);

    // Path of Input Field
    input_dir = _PyUnicode_AsString(PyList_GET_ITEM(pValue, 0));

    // Setting Input File  Name
    std::string tmp_name = _PyUnicode_AsString(PyList_GET_ITEM(pValue, 2));
    file_name = input_dir + "/" + tmp_name;

    // Setting global write path for fields and datasets
    output_dir = _PyUnicode_AsString(PyList_GET_ITEM(pValue, 1));
    // #################################################################

    // ########## Setting Dataset names for Input and output ###########
    // Opening Function name_of_datasets
    pFunc = PyObject_GetAttrString(pModule, (char *)"name_of_datasets");
    python_err_check(pFunc, __LINE__, __FILE__);
    pValue = PyObject_CallObject(pFunc, nullptr);
    python_err_check(pValue, __LINE__, __FILE__);

    data_name[0] = _PyUnicode_AsString(PyList_GET_ITEM(pValue, 0));
    data_name[1] = _PyUnicode_AsString(PyList_GET_ITEM(pValue, 1));
    data_name[2] = _PyUnicode_AsString(PyList_GET_ITEM(pValue, 2));

    // #################################################################

    // ############### Setting Print parameters Flag ###################
    // Setting dimensions and offset temporary
    dimsf[0] = Nx;
    dimsf[1] = Ny;
    dimsf[2] = Nz;

    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;

    // Checking for existence of Fields folder and creating it if nessacry
    struct stat sb;
    if (!(stat(output_dir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)))
    {
#if defined(_MSC_VER)
        _mkdir(output_dir.c_str());
#else
        mkdir(output_dir.c_str(), 0777);
#endif
    }

    // Setting the memory type
    mem_type = H5T_NATIVE_FLOAT;
    if (std::is_same<T, T_d>::value)
    {
        mem_type = H5T_NATIVE_DOUBLE;
    }
}

template <typename T>
void reader_io()
{
    std::cout << "\n Reading the data from file " << file_name;

    // ####### Opening of file for reading the dataset #########
    fileHandle = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    // #########################################################

    // ################### Target space setting ##################
    // Setting of the target-space and source-space For Real datatype input
    dimsf[0] = Nx;
    dimsf[1] = Ny;
    dimsf[2] = Nz;

    targetDSpace = H5Screate_simple(dimension, dimsf, NULL);

    // Setting dimensions for target reading
    dimsf[0] = Nx;
    dimsf[1] = Ny;
    dimsf[2] = Nz;
    // SETTING THE OFFSETS OF DATA FIELDS
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;

    // Creating the hyperslab
    status = H5Sselect_hyperslab(targetDSpace, H5S_SELECT_SET, offset, NULL, dimsf, NULL);
    HDF5_err_check(status, __LINE__ - 1, __FILE__);
    // #########################################################

    // ################# Setting your source space #############
    dimsf[0] = Nx;
    dimsf[1] = Ny;
    dimsf[2] = Nz;
    sourceDSpace = H5Screate_simple(dimension, dimsf, NULL);

    // Modify the view of the *source* dataspace by using a hyperslab - *this view will be used to read from memory*
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;

    status = H5Sselect_hyperslab(sourceDSpace, H5S_SELECT_SET, offset, NULL, dimsf, NULL);
    HDF5_err_check(status, __LINE__ - 1, __FILE__);
    // ########################################################

    // ############ Reading The dataset #######################
    // Reading The Vx
    dataSet = H5Dopen2(fileHandle, data_name[0].c_str(), H5P_DEFAULT);
    status = H5Dread(dataSet, mem_type, sourceDSpace, targetDSpace, H5P_DEFAULT, Vx_cpu<T>);
    HDF5_err_check(status, __LINE__ - 1, __FILE__);

    // Reading The Vy
    dataSet = H5Dopen2(fileHandle, data_name[1].c_str(), H5P_DEFAULT);
    status = H5Dread(dataSet, mem_type, sourceDSpace, targetDSpace, H5P_DEFAULT, Vy_cpu<T>);
    HDF5_err_check(status, __LINE__ - 1, __FILE__);

    // Reading The Vz
    dataSet = H5Dopen2(fileHandle, data_name[2].c_str(), H5P_DEFAULT);
    status = H5Dread(dataSet, mem_type, sourceDSpace, targetDSpace, H5P_DEFAULT, Vz_cpu<T>);
    HDF5_err_check(status, __LINE__ - 1, __FILE__);
}

template <typename T>
void writer_io()
{
    std::cout << "\n Writting the data to file " << file_name;
}

template <typename T>
void print_parameters()
{
    std::cout << "\n\n\n  Printing the Parameters \n ";

    std::cout << "\n ################# Grid Parameters ##############";
    std::cout << "\n Precision = " << precision;
    std::cout << "\n Device_ID = " << Device_Id;
    std::cout << "\n Dimension = " << dimension;
    std::cout << "\n Nx = " << Nx;
    std::cout << "\n Ny = " << Ny;
    std::cout << "\n Nz = " << Nz;
    std::cout << "\n Lx = " << Lx<T>;
    std::cout << "\n Ly = " << Ly<T>;
    std::cout << "\n Lz = " << Lz<T>;
    std::cout << "\n dx = " << dx<T>;
    std::cout << "\n dy = " << dy<T>;
    std::cout << "\n dz = " << dz<T>;

    std::cout << "\n ################# IO Parameters ##############";
    std::cout << "\n Input directory = " << input_dir;
    std::cout << "\n Output directory = " << output_dir;
    std::cout << "\n Name of input file = " << file_name;
    std::cout << std::endl;
}

// ############### Explicit instantiation #####################
template void initialize_io<T_f>();
template void initialize_io<T_d>();

template void reader_io<T_f>();
template void reader_io<T_d>();

template void writer_io<T_f>();
template void writer_io<T_d>();

template void print_parameters<T_f>();
template void print_parameters<T_d>();
// ############################################################