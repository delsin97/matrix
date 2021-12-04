#include <iostream>
#include "mmio.h"

#include "lib_cuda.h"

using namespace std;

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef RUN_NUM
#define RUN_NUM 1000
#endif

int call_lib(int m, int n, int nnzA, int *csrRowPtrA, int *csrColIdxA,
    VALUE_TYPE *csrValA, int *csrRowPtrA_row, VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha)
{
    int err = 0;
    cudaError_t err_cuda = cudaSuccess;

    // set device
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    cout << "Device [" << device_id << "]" << deviceProp.name << ", " << "@ " << deviceProp.clockRate * 1e-3f << "MHz. " << endl;

    double gb = getB<int, VALUE_TYPE>(m, nnzA);
    double gflop = getFLOP<int>(nnzA);

    // Define pointers of matrix A, vector x and y
    int *d_csrRowPtrA;
    int *d_csrColIdxA;
    VALUE_TYPE *d_csrValA;
    int *d_csrRowPtrA_row; // 每行的元素个数
    VALUE_TYPE *d_x;
    VALUE_TYPE *d_y;

    // Matrix A
    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, (m + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIdxA, nnzA * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA, nnzA * sizeof(VALUE_TYPE)));
    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA_row, m * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIdxA, csrColIdxA, nnzA * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA, csrValA, nnzA * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA_row, csrRowPtrA_row, m * sizeof(int), cudaMemcpyHostToDevice));

    // Vector x
    checkCudaErrors(cudaMalloc((void **)&d_x, n * sizeof(VALUE_TYPE)));
    checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

    // Vector y
    checkCudaErrors(cudaMalloc((void **)&d_y, m * sizeof(VALUE_TYPE)));
    checkCudaErrors(cudaMemset(d_y, 0, m * sizeof(VALUE_TYPE)));

    libHandle<int, unsigned int, VALUE_TYPE> A(m, n);
    err = A.inputCSR(nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_csrRowPtrA_row);
    cout << "inputCSR err = " << err << endl;

    err = A.setX(d_x);
    cout << "setX err = " << err << endl;

    A.setSigma(AUTO_TUNED_SIGMA);

    // warmup device
    A.warmup();

    lib_timer as_timer;
    as_timer.start();

    err = A.aseCSR();

    cout << "CSR->eCSR time = " << as_timer.stop() << " ms. " << endl;

    err = A.spmv(alpha, d_y);

    checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    if(NUM_RUN)
    {
        for(int i = 0; i < 50; i++)
            err = A.spmv(alpha, d_y);
    }

    err_cuda = cudaDeviceSynchronize();

    lib_timer spmv_timer;
    spmv_timer.start();

    for(int i = 0; i < NUM_RUN; i++)
        err = A.spmv(alpha, d_y);

    err_cuda = cudaDeviceSynchronize();

    double spmv_time = spmv_timer.stop() / (double)NUM_RUN;

    // cout << "SPMV time = " << spmv_time / iter << " ms. " << endl;

    if (NUM_RUN)
        cout << "SpMV time = " << spmv_time
             << " ms. Bandwidth = " << gb/(1.0e+6 * spmv_time)
             << " GB/s. GFlops = " << gflop/(1.0e+6 * spmv_time)  << " GFlops." << endl;

    

    // for(int i = 0; i < 100; i++)
    //     cout << y[i] << " ";
    // cout << endl;

    cout << "hello" << endl;

    A.destory();

    // cout << "hello" << endl;

    checkCudaErrors(cudaFree(d_csrRowPtrA));
    checkCudaErrors(cudaFree(d_csrColIdxA));
    checkCudaErrors(cudaFree(d_csrValA));
    checkCudaErrors(cudaFree(d_csrRowPtrA_row));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    return err;
}

int main(int argc, char *argv[])
{
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    // report precision of floating-point
    cout << "----------------------------------" << endl;
    const char *precision;
    if(sizeof(VALUE_TYPE) == 4)
        precision = "32-bit Single Precision";
    else if(sizeof(VALUE_TYPE) == 8)
        precision = "64-bit Double Precision";
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }

    cout << "PRECISION = " << precision << endl;
    cout << "----------------------------------" << endl;

    int argi = 1;

    char *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    cout << "----------" << filename << "----------" << endl;

    //read matrix from .mtx file
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnzA_mtx_report;

    // int max_size;
    
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if((f = fopen(filename, "r")) == NULL)
        return -1;

    if(mm_read_banner(f, &matcode) != 0)
    {
        cout << "Could not process the matrix market banner" << endl;
        return -2;
    }

    if(mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode))
    {
        cout << "Sorry, this application does not support ";
        cout << "Matrix Market type: " << mm_typecode_to_str(matcode) << endl;
        return -3;
    }

    if(mm_is_pattern(matcode))
    {
        isPattern = 1;
        cout << "type = Pattern" << endl;
    }

    if(mm_is_real(matcode))
    {
        isReal = 1;
        cout << "type = Real" << endl;
    }
    if(mm_is_integer(matcode))
    {
        isInteger = 1;
        cout << "type = Integer" << endl;
    }

    // find out the size of the matrix
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);

    // set sigma
    // int sigma = getSigma(nnzA_mtx_report, m, AUTO_TUNED_SIGMA);

    // max_size = nnzA_mtx_report + m * (sigma - 1);

    if(ret_code != 0)
        return -4;

    if(mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric = 1;
        cout << "symmetric = true" << endl;
    }
    else
        cout << "symmetric = false" << endl;

    lib_timer con_timer;
    con_timer.start();

    int *csrRowPtrA_counter = (int *)malloc((m + 1) * sizeof(int));
    int *csrRowPtrA_row = (int *)malloc(m * sizeof(int));// 每一行的元素数目
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));
    memset(csrRowPtrA_row, 0, m * sizeof(int));

    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    VALUE_TYPE *csrValA_tmp = (VALUE_TYPE *)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));
    
    for(int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;

        if(isReal)
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if(isInteger)
        {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if(isPattern)
        {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // 1-based -> 0-based
        idxi--;
        idxj--;
        csrRowPtrA_counter[idxi]++;
        csrRowPtrA_row[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if(f != stdin)
        fclose(f);
    
    if(isSymmetric)// matrix_market对称的话仅存一半的元素
    {
        for(int i = 0; i < nnzA_mtx_report; i++)
            if(csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
                csrRowPtrA_row[csrColIdxA_tmp[i]]++;
            }
    }

    

    // csrRowPtrA_counter 调整为sigma的整数倍
    // for(int i = 0; i < m + 1; i++)
    // {
    //     csrRowPtrA_counter[i] = (csrRowPtrA_counter[i] + sigma - 1) / sigma * sigma;
    // }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;
    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for(int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)malloc((m + 1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));// 已经重新设置为0

    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrValA = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
    // csrColIdxA = (int *)malloc(max_size * sizeof(int));
    // memset(csrColIdxA, 0, max_size * sizeof(int));
    // csrValA = (VALUE_TYPE *)malloc(max_size * sizeof(VALUE_TYPE));
    // memset(csrValA, 0, max_size * sizeof(VALUE_TYPE));

    if(isSymmetric)
    {
        for(int i = 0; i < nnzA_mtx_report; i++)
        {
            if(csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for(int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    //free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);

    double con_time = con_timer.stop();
    cout << "CPU change time = " << con_time << " ms." << endl;

    srand(time(NULL));

    // // set csrValA to 1, easy for checking floating point results
    // for(int i = 0; i < nnzA; i++)
    // {
    //     csrValA[i] = rand() % 10;
    // }

    cout << "(" << m << ", " << n << " ) nnz = " << nnzA << endl;

    VALUE_TYPE *x = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
    for(int i = 0; i < n; i++)
        x[i] = rand() % 10;

    VALUE_TYPE *y = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));
    VALUE_TYPE *y_ref = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));

    double gb = getB<int, VALUE_TYPE>(m, nnzA);

    cout << "GB: " << gb << endl;

    double gflop = getFLOP<int>(nnzA);

    cout << "gflop: " << gflop << endl;


    VALUE_TYPE alpha = 1.0;

    // compute the result on CPU
    lib_timer ref_timer;
    ref_timer.start();

    int ref_iter = 10;
    for(int iter = 0; iter < ref_iter; iter++)
    {
        for(int i = 0; i < m; i++)
        {
            VALUE_TYPE sum = 0;
            for(int j = csrRowPtrA[i]; j < csrRowPtrA[i + 1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j] * alpha;
            y_ref[i] = sum;
        }
    }

    // for(int i = 0; i < 100; i++)
    //     cout << y_ref[i] << " ";
    // cout << endl;

    double ref_time = ref_timer.stop() / (double)ref_iter;

    cout << "CPU sequential time = " << ref_time << " ms. Bandwidth = "
        << gb / (1.0e+6 * ref_time) << " GB/s. GFlops = " << gflop / (1.0e+6 * ref_time)
        << " GFlops." << endl << endl;

    // launch compute
    //
    //
    call_lib(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, csrRowPtrA_counter, x, y, alpha);


    //compare reference and results
    int error_count = 0;
    for(int i = 0; i < m; i++)
        if(abs(y_ref[i] - y[i]) > 0.01 *abs(y_ref[i]))
        {
            error_count++;
        }

    if(error_count == 0)
        cout << "Check... PASS!" << endl;
    else
        cout << "Check... NO PASS! Error = " << error_count << " out of " << m << " entries" << endl;

    cout << "--------------------------------" << endl;

    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA_counter);
    free(x);
    free(y);
    free(y_ref);

    return 0;
}