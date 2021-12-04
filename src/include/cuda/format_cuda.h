#ifndef __FORMAT_CUDA_H__
#define __FORMAT_CUDA_H__

#include "common_cuda.h"
#include "utils_cuda.h"

int format_warmup()
{
    int *d_scan;
    checkCudaErrors(cudaMalloc((void **)&d_scan, OMEGA * sizeof(int)));

    int num_threads = OMEGA;
    int num_blocks = 4000;

    for(int i = 0; i < 50; i++)
        warmup_kernel<<<num_blocks, num_threads>>>(d_scan);

    checkCudaErrors(cudaFree(d_scan));
}

template<typename iT, typename uiT>
__global__
void reset_row(uiT *d_ecsr_row, int sigma, const iT d_m)
{
    iT tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= d_m)
        return;
    d_ecsr_row[tid] = (d_ecsr_row[tid] + sigma - 1) / sigma * sigma;
}

template<typename LIB_IT>
int reset_row_ptr(LIB_IT *d_ecsr_row, LIB_IT d_m, int sigma)// 重设每行的元素长度
{
    dim3 block(THREAD_GROUP);
    dim3 grid((d_m + block.x - 1) / block.x);
    reset_row<LIB_IT, LIB_IT><<<grid, block>>>(d_ecsr_row, sigma, d_m);
    cudaDeviceSynchronize();
    return SUCCESS;
}

template<typename iT>
__global__
void pre_scan(iT *d_ecsr_num, iT d_m, iT *d_ecsr_row)
{
    for(int i = 1; i < d_m; i++)
        d_ecsr_num[i] = d_ecsr_num[i - 1] + d_ecsr_row[i];
}

template<typename LIB_IT>
int get_num(LIB_IT *d_ecsr_num, LIB_IT d_m, LIB_IT *d_ecsr_row)
{
    // int steps = int(log2(float(d_m))); 
    // 	// 2D Kernel Launch Parameters
	// dim3 THREADS(1024, 1, 1);
	// dim3 BLOCKS;
	// if (d_m >= 65536)
	// 	BLOCKS = dim3(64, d_m / 65536, 1);
	// else if (d_m <= 1024)
	// 	BLOCKS = dim3(1, 1, 1);
	// else
	// 	BLOCKS = dim3(d_m / 1024, 1, 1);

	// int space = 1;

    // for(int step = 0; step < steps; step++)
    // {
    //     Hillis_Steele_Scan_Kernel<<<BLOCKS, THREADS>>>(d_ecsr_num, space, step, steps);
    //     space *= 2;
    // }

    // cudaDeviceSynchronize();

    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    pre_scan<LIB_IT><<<grid, block>>>(d_ecsr_num, d_m, d_ecsr_row);

    return SUCCESS;
}

template<typename iT, typename uiT>
__global__
void each_line(const iT *d_ecsr_num, uiT *d_ecsr_row_bit, int width, int size, iT m) // 获得每一行元素转化为的行数
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= m)
        return;
    if(idx == 0)
        d_ecsr_row_bit[idx] = true;
    else
    {
        iT index = d_ecsr_num[idx - 1] / width;
        // iT index = idx;
        d_ecsr_row_bit[index] = true;
    }
        // d_ecsr_row_bit[d_ecsr_num[idx - 1] / width] = true;
}

// 生成对应的二进制矩阵
template<typename LIB_IT, typename LIB_UIT>
int set_bit(const LIB_IT *d_ecsr_num, LIB_UIT *d_ecsr_row_bit, int width, int size, LIB_IT m)
{
    dim3 block(THREAD_BUNCH);
    dim3 grid((m + block.x - 1) / block.x);
    each_line<<<grid, block>>>(d_ecsr_num, d_ecsr_row_bit, width, size, m);
    return SUCCESS;
}

//
template<typename iT>
__global__ 
void set_col(const iT *d_csr_row_pointer, const iT *d_ecsr_num, const iT *d_csr_column_index, iT *d_ecsr_col, const iT d_m)
{
    iT tid = blockDim.x * blockIdx.x + threadIdx.x;
    iT prev_idx;
    if(tid >= d_m)
        return;
    if(tid == 0)
        prev_idx = 0;
    else
        prev_idx = d_ecsr_num[tid - 1];

    for(iT idx = d_csr_row_pointer[tid], i = 0; idx < d_csr_row_pointer[tid + 1]; idx++)
    {
        d_ecsr_col[prev_idx + i] = d_csr_column_index[idx];
        i++;
    }
}

template<typename iT, typename vT>
__global__ 
void set_val(const iT *d_csr_row_pointer, const iT *d_ecsr_num, const vT *d_csr_value, vT *d_ecsr_val, const iT d_m)
{
    iT tid = blockDim.x * blockIdx.x + threadIdx.x;
    iT prev_idx;
    if(tid >= d_m)
        return;
    if(tid == 0)
        prev_idx = 0;
    else
        prev_idx = d_ecsr_num[tid - 1];

    for(iT idx = d_csr_row_pointer[tid], i = 0; idx < d_csr_row_pointer[tid + 1]; idx++)
    {
        d_ecsr_val[prev_idx + i] = d_csr_value[idx];
        i++;
    }
}

template<typename LIB_IT, typename LIB_VT>
int set_value(const LIB_IT *d_csr_row_pointer, const LIB_IT *d_ecsr_num, const LIB_IT *d_csr_column_index, const LIB_VT *d_csr_value, LIB_IT *d_ecsr_col, LIB_VT *d_ecsr_val, const LIB_IT d_m)
{
    dim3 block(THREAD_BUNCH);
    dim3 grid((d_m + block.x - 1) / block.x);
    set_col<LIB_IT><<<grid, block>>>(d_csr_row_pointer, d_ecsr_num, d_csr_column_index, d_ecsr_col, d_m);
    set_val<LIB_IT, LIB_VT><<<grid, block>>>(d_csr_row_pointer, d_ecsr_num, d_csr_value, d_ecsr_val, d_m);
    return SUCCESS;
}



template<typename T>
__global__
void transpose_kernel(const T *i_data, T *o_data, const int m, const int n)
{
    __shared__ T block[THREAD_BUNCH][THREAD_BUNCH];

    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < m && j < n)
    {
        block[threadIdx.y][threadIdx.x] = i_data[i * n + j];
        __syncthreads();
        o_data[j * m + i] = block[threadIdx.y][threadIdx.x];
    }
}

// 转化为以列为主的排列
template<typename LIB_IT, typename LIB_VT> 
int generate_block(const LIB_IT *d_ecsr_col, const LIB_VT *d_ecsr_val, LIB_IT *d_ecsr_col_b, LIB_VT *d_ecsr_val_b, int height, int width)
{
    dim3 block_2d(THREAD_BUNCH, THREAD_BUNCH);
    dim3 grid_2d((width + block_2d.x -1) / block_2d.x, (height + block_2d.y - 1) / block_2d.y);
    transpose_kernel<LIB_IT><<<grid_2d, block_2d>>>(d_ecsr_col, d_ecsr_col_b, height, width);
    transpose_kernel<LIB_VT><<<grid_2d, block_2d>>>(d_ecsr_val, d_ecsr_val_b, height, width);  
    return SUCCESS;
}

#endif