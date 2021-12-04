#ifndef __UTILS_CUDA_H__
#define __UTILS_CUDA_H__

#include "common_cuda.h"

struct lib_timer
{
    cudaEvent_t start_event, stop_event;

    void start()
    {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, 0);
        cudaDeviceSynchronize();
    }

    float stop()
    {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
        return elapsedTime;
    }
};

// exclusive scan
template<typename T>
__inline__ __device__ 
void scan_32(volatile T *s_scan, const int local_id)
{
    int ai, bi;
    const int baseai = 2 * local_id + 1;
    const int basebi = baseai + 1;
    T temp;

    if(local_id < 16) 
    {   
        ai = baseai - 1;
        bi = basebi - 1;
        s_scan[bi] += s_scan[ai];
    }
    if(local_id < 8)
    {
        ai = 2 * baseai - 1;
        bi = 2 * basebi - 1;
        s_scan[bi] += s_scan[ai];
    }
    if(local_id < 4)
    {
        ai = 4 * baseai - 1;
        bi = 4 * basebi - 1;
        s_scan[bi] += s_scan[ai];
    }
    if(local_id < 2)
    {
        ai = 8 * baseai - 1;
        bi = 8 * basebi - 1;
        s_scan[bi] += s_scan[ai];
    }
    if(local_id == 0)
    {
        s_scan[31] = s_scan[15];
        s_scan[15] = 0;
    }
    if(local_id < 2)
    {
        ai = 8 * baseai - 1;
        bi = 8 * basebi - 1;
        temp = s_scan[ai];
        s_scan[ai] = s_scan[bi];
        s_scan[bi] += temp; 
    }
    if(local_id < 4)
    {
        ai = 4 * baseai - 1;
        bi = 4 * basebi - 1;
        temp = s_scan[ai];
        s_scan[ai] = s_scan[bi];
        s_scan[bi] += temp;
    }
    if(local_id < 8)
    {
        ai = 2 * baseai - 1;
        bi = 2 * basebi - 1;
        temp = s_scan[ai];
        s_scan[ai] = s_scan[bi];
        s_scan[bi] += temp;
    }
    if(local_id < 16)
    {
        ai = baseai - 1;
        bi = basebi - 1;
        temp = s_scan[ai];
        s_scan[ai] = s_scan[bi];
        s_scan[bi] += temp;
    }
}

template<typename iT, typename uiT>
inline __device__ 
void prescan(uiT* d_odata, iT *d_idata, iT n)
{
	volatile __shared__ uiT temp[OMEGA];
	int tid = threadIdx.x;
	int offset = 1;
	//转移到共享内存
	temp[2 * tid] = d_idata[2 * tid];
	temp[2 * tid + 1] = d_idata[2 * tid + 1];
	//第一步先向上扫描
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (tid == 0)
		temp[n - 1] = 0;
	//向下扫描
	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			uiT t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	//写出到global内存
	d_odata[2 * tid] = temp[2 * tid];
	d_odata[2 * tid + 1] = temp[2 * tid + 1];
}

template <typename T>
__global__ void
Hillis_Steele_Scan_Kernel(T* arr, int space, int step, int steps)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	int y = threadIdx.y + blockDim.y * blockIdx.y;

	// 2D Kernel Launch parameters
	int tid = x + (y * gridDim.x * blockDim.x);

	// Kernel runs in the parallel
	// TID is the unique thread ID
	if (tid >= space)
		arr[tid] += arr[tid - space];
}

__global__ 
void warmup_kernel(int *d_scan)
{
    volatile __shared__ int s_scan[OMEGA];
    s_scan[threadIdx.x] = 1;
    scan_32<int>(s_scan, threadIdx.x);
    if(!blockIdx.x)
        d_scan[threadIdx.x] = s_scan[threadIdx.x];
}

template<typename iT>
__forceinline__ __device__
void fetch_x(cudaTextureObject_t d_x_tex, const iT i, float *x)
{
    *x = tex1Dfetch<float>(d_x_tex, i);
}

template<typename iT>
__forceinline__ __device__
void fetch_x(cudaTextureObject_t d_x_tex, const iT i, double *x)
{
    int2 x_int2 = tex1Dfetch<int2>(d_x_tex, i);
    *x = __hiloint2double(x_int2.y, x_int2.x);
}

#endif