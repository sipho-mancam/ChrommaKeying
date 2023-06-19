#ifndef TRTX_CUDA_UTILS_H_
#define TRTX_CUDA_UTILS_H_

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__<<std::endl;\
            std::cerr<<cudaGetErrorString(error_code)<<std::endl;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

#endif  // TRTX_CUDA_UTILS_H_

