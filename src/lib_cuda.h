#ifndef __LIB_CUDA_H__
#define __LIB_CUDA_H__

#include "include/common.h"
#include "include/utils.h"

#include "include/cuda/common_cuda.h"
#include "include/cuda/utils_cuda.h"
#include "include/cuda/format_cuda.h"
#include "include/cuda/spmv.h"

template <class LIB_IT, class LIB_UIT, class LIB_VT>
class libHandle
{
public:
    libHandle(LIB_IT m, LIB_IT n)   {_m = m;    _n = n;}
    int warmup();
    int inputCSR(LIB_IT nnz, LIB_IT *csr_row_pointer, LIB_IT *csr_column_idx, LIB_VT *csr_value, LIB_IT *csr_row_pointer_counter);
    int aseCSR();
    int setX(LIB_VT *x);
    int destory();
    int spmv(const LIB_VT alpha, LIB_VT *y);
    void setSigma(int sigma);

private:
    int computeSigma();
    int _format;
    LIB_IT _m;
    LIB_IT _n;
    LIB_IT _nnz;

    LIB_IT *_csr_row_pointer;
    LIB_IT *_csr_column_index;
    LIB_VT *_csr_value;
    LIB_IT *_csr_row;

    int _sigma;
    int _size;

    LIB_IT *_ecsr_row; // 每行的元素数目
    LIB_IT *_ecsr_num; // 通过此值寻找总数据量 inclusive scan ecsr_row
    LIB_IT *_ecsr_col;
    LIB_VT *_ecsr_val; // 以行排列
    LIB_IT *_ecsr_col_b;
    LIB_VT *_ecsr_val_b; // 以列排列
    LIB_IT *_ecsr_row_bit; // 二进制排列
    LIB_VT *_y_temp;

    LIB_IT _p; // the number of partitions // tile的数目

    LIB_VT *_x;
    cudaTextureObject_t _x_tex;
};

template <class LIB_IT, class LIB_UIT, class LIB_VT>
int libHandle<LIB_IT, LIB_UIT, LIB_VT>::warmup()
{
    format_warmup();
}

template <class LIB_IT, class LIB_UIT, class LIB_VT>
int libHandle<LIB_IT, LIB_UIT, LIB_VT>::inputCSR(LIB_IT nnz, LIB_IT *csr_row_pointer, LIB_IT *csr_column_idx, LIB_VT *csr_value, LIB_IT *csr_row)
{
    _format = FORMAT_CSR;

    _nnz = nnz;

    _csr_row_pointer = csr_row_pointer;
    _csr_column_index = csr_column_idx;
    _csr_value = csr_value;
    _csr_row = csr_row;

    return SUCCESS;
}

template <class LIB_IT, class LIB_UIT, class LIB_VT>
int libHandle<LIB_IT, LIB_UIT, LIB_VT>::aseCSR()
{
    int err = SUCCESS;

    if(_format == FORMAT_eCSR)
        return err;
    
    if(_format == FORMAT_CSR)
    {
        // compute sigma
        _sigma = computeSigma();
        cout << "omege = " << OMEGA << ", sigma = " << _sigma << endl;
        

        // 格式转换
        // 1.将counter按照sigma大小进行重新设置,重新设置csr_row的数量，获得e_csr_row
        checkCudaErrors(cudaMallocManaged((void **)&_ecsr_row, _m * sizeof(LIB_IT)));
        checkCudaErrors(cudaMemcpy(_ecsr_row, _csr_row, _m * sizeof(LIB_IT), cudaMemcpyHostToDevice));
        
        err = reset_row_ptr(_ecsr_row, _m, _sigma);
        // 2.获得总共的数据量 并获得对应的bit-flag
        checkCudaErrors(cudaMallocManaged((void **)&_ecsr_num, _m * sizeof(LIB_IT)));
        LIB_IT *_ecsr_num_temp = (LIB_IT *)malloc(sizeof(LIB_IT) * _m);
        checkCudaErrors(cudaMemcpy(_ecsr_num_temp, _ecsr_row, _m * sizeof(LIB_IT), cudaMemcpyDeviceToHost));
        // err = get_num(_ecsr_num, _m, _ecsr_row);
        // _ecsr_num[0] = _ecsr_row[0];
        // cout << _ecsr_row[9] << endl; // 用于判断col值是否正确
        
        for(int i = 1; i < _m; i++)
            _ecsr_num_temp[i] = _ecsr_num_temp[i - 1] + _ecsr_row[i];
        cout << _ecsr_num_temp[_m - 1] << " ";
        cout << endl;
        cout << endl;
        int size = _ecsr_num_temp[_m - 1];
        _size = size;
        checkCudaErrors(cudaMemcpy(_ecsr_num, _ecsr_num_temp, _m * sizeof(LIB_IT), cudaMemcpyHostToDevice));
        // calculate the number of partitions
        _p = ceil((double)size / (double)_sigma);
        cout << _p << endl;
        checkCudaErrors(cudaMalloc((void **)&_ecsr_row_bit, _p * sizeof(LIB_UIT)));
        checkCudaErrors(cudaMemset(_ecsr_row_bit, false, _p * sizeof(LIB_UIT)));
        err = set_bit(_ecsr_num, _ecsr_row_bit, _sigma, _p, _m);
        // LIB_IT *_ecsr_bit_temp = (LIB_IT *)malloc(sizeof(LIB_IT) * _p); // 验证结果
        // checkCudaErrors(cudaMemcpy(_ecsr_bit_temp, _ecsr_row_bit, _p * sizeof(LIB_UIT), cudaMemcpyDeviceToHost));
        // for(int i = 0; i < 100; i++)
        //     cout << _ecsr_bit_temp[i] << " ";
        // cout << endl;
        

        // 2.重新设置col_idx以及val_idx，获得e_col_idx和e_val_idx
        checkCudaErrors(cudaMalloc((void **)&_ecsr_col, size * sizeof(LIB_IT)));
        checkCudaErrors(cudaMemset(_ecsr_col, 0, size * sizeof(LIB_IT)));
        checkCudaErrors(cudaMalloc((void **)&_ecsr_val, size * sizeof(LIB_VT)));
        checkCudaErrors(cudaMemset(_ecsr_val, 0, size * sizeof(LIB_VT)));
        err = set_value(_csr_row_pointer, _ecsr_num, _csr_column_index, _csr_value, _ecsr_col, _ecsr_val, _m);
        // LIB_VT *ecsr_val_temp = (LIB_VT *)malloc(size * sizeof(LIB_VT));
        // checkCudaErrors(cudaMemcpy(ecsr_val_temp, _ecsr_val, size * sizeof(LIB_VT), cudaMemcpyDeviceToHost));
        // for(int i = 0; i < 100; i++) // 测试是否正确
        //     cout << ecsr_val_temp[i] << " ";
        // cout << endl;
        

        // 3.将col_idx和val_idx
        checkCudaErrors(cudaMalloc((void **)&_ecsr_col_b, size * sizeof(LIB_IT)));
        checkCudaErrors(cudaMemset(_ecsr_col_b, 0, size * sizeof(LIB_IT)));
        checkCudaErrors(cudaMalloc((void **)&_ecsr_val_b, size * sizeof(LIB_VT)));
        checkCudaErrors(cudaMemset(_ecsr_val_b, 0, size * sizeof(LIB_VT)));
        err = generate_block(_ecsr_col, _ecsr_val, _ecsr_col_b, _ecsr_val_b, size / _sigma, _sigma);

        // LIB_VT *ecsr_val_temp = (LIB_VT *)malloc(size * sizeof(LIB_VT));
        // LIB_VT *ecsr_val_b_temp = (LIB_VT *)malloc(size * sizeof(LIB_VT));
        // checkCudaErrors(cudaMemcpy(ecsr_val_temp, _ecsr_val, size * sizeof(LIB_VT), cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaMemcpy(ecsr_val_b_temp, _ecsr_val_b, size * sizeof(LIB_VT), cudaMemcpyDeviceToHost));
        // for(int i = 0; i < 100; i++) // 测试是否正确
        //     cout << ecsr_val_temp[i] << " ";
        // cout << endl;
        // for(int i = 0; i < 100; i++) // 测试是否正确
        //     cout << ecsr_val_b_temp[i] << " ";
        // cout << endl;

        _format = FORMAT_eCSR;

    }

    return err;
}

template <class LIB_IT, class LIB_UIT, class LIB_VT>
int libHandle<LIB_IT, LIB_UIT, LIB_VT>::setX(LIB_VT *x)
{
    int err = SUCCESS;
    _x = x;

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = _x;
    resDesc.res.linear.sizeInBytes = _n * sizeof(LIB_VT);
    if(sizeof(LIB_VT) == sizeof(float))
    {
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32;// bits per channel
    }
    else if(sizeof(LIB_VT) == sizeof(double))
    {
        resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32;
        resDesc.res.linear.desc.y = 32;
    }
    else
        return UNSUPPORTED_VALUE_TYPE;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0 , sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // create texture object
    _x_tex = 0;
    cudaCreateTextureObject(&_x_tex, &resDesc, &texDesc, NULL);
    //需要产生的纹理对象    资源描述符，用于获取纹理数据    纹理描述符，用来描述纹理参数
    return err;
}

template <class LIB_IT, class LIB_UIT, class LIB_VT>
int libHandle<LIB_IT, LIB_UIT, LIB_VT>::spmv(const LIB_VT alpha, LIB_VT *y)
{
    int err = SUCCESS;
    checkCudaErrors(cudaMalloc((void **)&_y_temp, (_size / _sigma) * sizeof(LIB_VT)));
    checkCudaErrors(cudaMemset(_y_temp, 0, (_size / _sigma) * sizeof(LIB_VT)));
    if(_format == FORMAT_CSR)
        return UNSUPPORTED_CSR_SPMV;
    if(_format == FORMAT_eCSR)
    {
        eCSR_spmv(_ecsr_col_b, 
              _ecsr_val_b, 
              _x,
              _x_tex,
              _y_temp,//
              _ecsr_num,
              _ecsr_row_bit,
              _m,
              _sigma,
              _size,
              y);
    }
    checkCudaErrors(cudaFree(_y_temp));
    return err;
}


template <class LIB_IT, class LIB_UIT, class LIB_VT>
int libHandle<LIB_IT, LIB_UIT, LIB_VT>::destory()
{
    cudaDestroyTextureObject(_x_tex);
    checkCudaErrors(cudaFree(_ecsr_row));
    checkCudaErrors(cudaFree(_ecsr_num));
    checkCudaErrors(cudaFree(_ecsr_col));
    checkCudaErrors(cudaFree(_ecsr_val));
    checkCudaErrors(cudaFree(_ecsr_col_b));
    checkCudaErrors(cudaFree(_ecsr_val_b));
    checkCudaErrors(cudaFree(_ecsr_row_bit));
    //---------------------------------
    // checkCudaErrors(cudaFree(_y_temp));
    _format = FORMAT_CSR;
    return SUCCESS;
}

template <class LIB_IT, class LIB_UIT, class LIB_VT>
void libHandle<LIB_IT, LIB_UIT, LIB_VT>::setSigma(int sigma)
{
    if(sigma == AUTO_TUNED_SIGMA)
    {
        int s = 32;
        int t = 256;
        int u = 128;

        int nnz_per_row = _nnz / _m;
        if(nnz_per_row <= s)
            _sigma = nnz_per_row;
        else if(nnz_per_row > s && nnz_per_row <= t)
            _sigma = s;
        else // nnz_per_row > t
            _sigma = u;
    }
    else
        _sigma = sigma;
}

template <class LIB_IT, class LIB_UIT, class LIB_VT>
int libHandle<LIB_IT, LIB_UIT, LIB_VT>::computeSigma()
{
    return _sigma;
}

#endif