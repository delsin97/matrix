#ifndef __COMMON_CUDA_H__
#define __COMMON_CUDA_H__

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "../common.h"
#include "../utils.h"

#define OMEGA 32
#define THREAD_BUNCH 32
#define THREAD_GROUP 128

#define AUTO_TUNED_SIGMA -1

#endif