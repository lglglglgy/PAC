//#include "arrayMD.h"
#include <complex>
#include <chrono>
#include <cmath>
#include <iostream>
using namespace std;
using namespace chrono;
using namespace std::chrono;

#define nstart 0
#define nend 3

using DataType = double;

#define ComplexType std::complex<DataType>

// ArrayMD definitions
// #define ARRAY3D Array3D<ComplexType>
// #define ARRAY2D Array2D<ComplexType>
// #define ARRAY1D Array1D<ComplexType>
// #define ARRAY1D_int Array1D<int>
// #define ARRAY1D_DataType Array1D<DataType>

// Function Definitions

// void
// noflagOCC_solver(size_t number_bands,
//                  size_t ngpown,
//                  size_t ncouls,
//                  ARRAY1D_int& inv_igp_index,
//                  ARRAY1D_int& indinv,
//                  ARRAY1D_DataType& wx_array,
//                  ARRAY2D& wtilde_array,
//                  ARRAY2D& aqsmtemp,
//                  ARRAY2D& aqsntemp,
//                  ARRAY2D& I_eps_array,
//                  ARRAY1D_DataType& vcoul,
//                  ARRAY1D& achtemp);
void
noflagOCC_solver(size_t number_bands,
                 size_t ngpown,
                 size_t ncouls,
                 int * inv_igp_index,
                 int * indinv,
                 DataType * wx_array,
                 ComplexType * wtilde_array,
                 ComplexType * aqsmtemp,
                 ComplexType * aqsntemp,
                 ComplexType * I_eps_array,
                 DataType * vcoul,
                 ComplexType * achtemp,
                 DataType * ach_re0,
                 DataType * ach_re1,
                 DataType * ach_re2,
                 DataType * ach_im0,
                 DataType * ach_im1,
                 DataType * ach_im2);

inline void ComplexType_print(ComplexType &src)
{
  printf("(%f,%f) \n",src.real(),src.imag());
}

inline ComplexType ComplexType_conj(ComplexType& src)
{
  return (std::conj(src));
}
