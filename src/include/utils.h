#ifndef __UTILS_H__
#define __UTILS_H__

#include "common.h"

#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>
#include "cuda/common_cuda.h"

template<typename iT, typename vT>
double getB(const iT m, const iT nnz)// 存取操作 为什么要增加2*nnz
{
    return (double)((m + 1 + nnz) * sizeof(iT) + (2 * nnz + m) * sizeof(vT));
}

template<typename iT>
double getFLOP(const iT nnz)
{
    return (double(2 * nnz));
}

int getSigma(int nnz, int row, int sigma, int m)
{
    if(sigma == AUTO_TUNED_SIGMA)
    {
        int s = 32;
        int t = 256;
        int u = 128;

        int nnz_per_row = nnz / m;
        if(nnz_per_row <= s)
            sigma = nnz_per_row;
        else if(nnz_per_row > s && nnz_per_row <= t)
            sigma = s;
        else // nnz_per_row > t
            sigma = u;
    }
    return sigma;
}

#endif