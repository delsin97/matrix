#ifndef __SPMV_H__
#define __SPMV_H__

#include "common_cuda.h"
#include "utils_cuda.h"

template<typename iT, typename vT>
__inline__ __device__
vT candidate(const vT *d_value_partition,
             const vT *d_x,
             cudaTextureObject_t d_x_tex,
             const iT *d_column_index_partition,
             const iT candidate_index,
             const vT alpha)
{
    vT x = 0;
    x = __ldg(&d_x[d_column_index_partition[candidate_index]]);
    return d_value_partition[candidate_index] * x;
}

template<typename iT, typename uiT, typename vT, int c_sigma>
__inline__ __device__
void spmv_patition(const iT *d_column_index_partition, 
                    const vT *d_value_partition,
                    const vT *d_x,
                    cudaTextureObject_t d_x_tex,
                    const uiT *d_partiton_pointer,
                    vT *d_y)
{
    
}

// 每一个block处理 threadIdx.x 行元素，每行有c_sigma个元素
template<typename iT, typename uiT, typename vT> // sigma即位每行的元素, size总共的行数
__global__
void spmv_compute_kernel(const iT *d_ecsr_col_b,
                         const vT *d_ecsr_val_b,
                         const vT *d_x,
                         cudaTextureObject_t d_x_tex,
                         vT *d_y_temp, // size 为 总共的行数
                         const int c_sigma,
                         const int size
                        )
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    vT x = 0;

    int bound = size / c_sigma;
    
    if(idx < bound)
    {
        for(int i = 0; i < c_sigma; i++)
        {
            iT candidate_index = idx + i * bound;
            x = __ldg(&d_x[d_ecsr_col_b[candidate_index]]);
            // x = 1;
            // x = d_x[d_ecsr_col_b[candidate_index]];
            d_y_temp[idx] += (x * d_ecsr_val_b[candidate_index]);
        }
    }
}

template<typename iT, typename uiT, typename vT>
__global__
void spmv_sum(const iT *d_ecsr_num,
              const uiT *d_ecsr_row_bit,
              const vT *d_y_temp,
              const int width,
              const int d_m,
              vT *d_y)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int pre_idx, curr_idx;
    if(idx >= d_m)
        return;
    if(idx == 0)
        pre_idx = 0;
    else
        pre_idx = d_ecsr_num[idx - 1] / width;
    curr_idx = d_ecsr_num[idx] / width;
    for(int i = pre_idx; i < curr_idx; i++)
    {
        d_y[idx] += d_y_temp[i];
    }

}

template<typename LIB_IT, typename LIB_UIT, typename LIB_VT>
int eCSR_spmv(const LIB_IT *d_ecsr_col_b, 
              const LIB_VT *d_ecsr_val_b, 
              const LIB_VT *d_x,
              cudaTextureObject_t d_x_tex,
              LIB_VT *d_y_temp,
              const LIB_IT *d_ecsr_num,
              const LIB_UIT *d_ecsr_row_bit,
              int m,
              int width,
              int size,
              LIB_VT *d_y)
{
    // alpha = alpha;
    dim3 num_threads = THREAD_BUNCH;
    dim3 num_blocks = (size / width + num_threads.x - 1) / num_threads.x;
    spmv_compute_kernel<LIB_IT, LIB_UIT, LIB_VT><<<num_blocks, num_threads>>>(d_ecsr_col_b, d_ecsr_val_b, d_x, d_x_tex, d_y_temp, width, size);
    num_blocks = (m + num_threads.x - 1) / num_threads.x;
    spmv_sum<LIB_IT, LIB_UIT, LIB_VT><<<num_blocks, num_threads>>>(d_ecsr_num, d_ecsr_row_bit, d_y_temp, width, m, d_y);
    return SUCCESS;
}

#endif