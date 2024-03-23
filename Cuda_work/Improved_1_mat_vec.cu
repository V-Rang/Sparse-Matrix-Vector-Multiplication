/*
Improved matrix-vector multiplication by assigning all computations related to a row to a warp.
*/

#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<algorithm>
#include<numeric>
#include<fstream>

using namespace std;

void sorter_result(int size_x, int *x, int *y, double *z, int *sorted_x, int *sorted_y, double *sorted_z)
{
    int *idx = new int[size_x];
    std::iota(idx,idx+size_x,0);

    stable_sort(idx, idx+size_x, [&x](int i1, int i2){ return x[i1] < x[i2]; });

    for(int i=0;i<size_x;i++)
    {
        sorted_x[i] = x[idx[i]];
        sorted_y[i] = y[idx[i]];
        sorted_z[i] = z[idx[i]];

    }
    delete idx;
    idx = nullptr;
}

__global__ void check()
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello from thread %d\n",tid);
}


__global__ void calc(double *res, double *mat, double *vec, int *row_offset, int *col_inds, int no_rows)
{
	extern __shared__ double block_sum[];
	
	//each warp calculates sum corresponding to a given row
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
    int warpid = tid/32;
	
	//thread Ids within each warp
	int lane = tid & 31;
	int row = warpid;

	if(row < no_rows)
	{
		block_sum[threadIdx.x] = 0;
		for(int j=row_offset[row] + lane ; j < row_offset[row+1]; j += 32)
		{
			block_sum[threadIdx.x] += mat[j] * vec[col_inds[j]];
		}
		
		//substitute below code by parallel reduction using __shfl_down_sync() in each warp
		
		if(lane < 16) block_sum[threadIdx.x] += block_sum[threadIdx.x + 16];
        __syncwarp();
		if(lane < 8) block_sum[threadIdx.x] += block_sum[threadIdx.x + 8];
        __syncwarp();
        if(lane < 4) block_sum[threadIdx.x] += block_sum[threadIdx.x + 4];
		__syncwarp();
        if(lane < 2) block_sum[threadIdx.x] += block_sum[threadIdx.x + 2];
		__syncwarp();
        if(lane < 1) block_sum[threadIdx.x] += block_sum[threadIdx.x + 1];

		if(lane == 0)
		{
			// res[row] += block_sum[threadIdx.x]; //+= should not be needed acc to me, next line code should suffice.
			res[row] = block_sum[threadIdx.x];
		}
	}
}

__global__ void calc(double *res, double *mat, double *vec, int *row_offset, int *col_inds, int no_rows)
{
	extern __shared__ double block_sum[];
    int tid =  blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = tid/32;
    int row = warp_id;

    int lane = tid & 31;

    if(row < no_rows)
    {
        block_sum[threadIdx.x] = 0; 
        for(int j=row_offset[row]+lane; j< row_offset[row+1]; j += 32)
        {
            block_sum[threadIdx.x] += mat[j] * vec[col_inds[j]];
        }
        
        //reduce sum of each warp to its corresp row using __shfl_down_sync();
        for(int i=warpSize/2 ; i>0; i /= 2)
        {
            block_sum[threadIdx.x] += __shfl_down_sync(0xfffffff,block_sum[threadIdx.x],i);
        }

        if(lane == 0) res[row] = block_sum[threadIdx.x];

    }
	
}

int main(int argc, char **argv)
{
	int num_times_to_run = 1;
	if(argc > 1)
	{
		num_times_to_run = atoi(argv[1]);
	}

	int no_rows, no_cols, nnz;
	int *row_ind, *col_ind;
	double *values;

	int*sorted_rows, *sorted_cols,*row_offset;
	double *sorted_vals;
	double *x_glob;

	string filename = "C:\\Users\\venu1\\OneDrive\\Desktop\\Work\\Sparse_Matrix_Vector_Multiplication\\Matrix_Files\\1138_bus.mtx"; // any file in matrix market format, tested matrices taken from UF Sparse matrix collection.
	// string filename = "test_matrix.mtx";
	std::ifstream my_file;
	my_file.open(filename);
	string line;
	while(true)
	{
		if(my_file.eof()) break;
		getline(my_file,line);
		if(isdigit(line[0]) == 0) 
		{
			continue;
		}
		sscanf(line.c_str(),"%d %d %d",&no_rows,&no_cols,&nnz);
		break; 
	}
	
	row_ind = new int[nnz];
	col_ind = new int[nnz];
	values = new double[nnz];

	int index_keeper = 0;
	int row_ind_p1;
	int col_ind_p1;

	while(true)
	{
		if(my_file.eof()) break;
		getline(my_file,line);
		if(my_file.eof()) break; 
		sscanf(line.c_str(),"%d %d %lf",&row_ind_p1,&col_ind_p1,&values[index_keeper]);
		row_ind[index_keeper] = row_ind_p1-1;
		col_ind[index_keeper] = col_ind_p1-1;
		index_keeper += 1;
	}

	my_file.close();

	sorted_rows = new int[nnz];
	sorted_cols = new int[nnz];
	sorted_vals = new double[nnz];

	sorter_result(nnz,row_ind,col_ind,values,sorted_rows,sorted_cols,sorted_vals);

	delete[] row_ind,col_ind,values;
	row_ind = nullptr;
	col_ind = nullptr;
	values = nullptr;

	row_offset = new int[no_rows+1];
	int *counter = new int[no_rows];
	for(int i=0;i<no_rows;i++)
	{
		counter[i] = 0;
	}

	for(int i=0;i<nnz;i++)
	{
		counter[sorted_rows[i]] += 1;
	}

	row_offset[0] = 0;
	row_offset[no_rows] = nnz;

	for(int i = 1;i<=no_rows-1;i++)
	{
		row_offset[i] = row_offset[i-1] + counter[i-1];
	}
	
	delete[] counter;
	counter = nullptr;
	delete[] sorted_rows;
	sorted_rows = nullptr;
	
	x_glob = new double[no_cols];
	for(int i=0;i<no_cols;i++) x_glob[i] = i+1;

	
	double *d_result,*d_mat_vals,*d_vec_vals;
	int *d_row_offset,*d_col_inds;
	double *h_result = new double[no_rows];
	
	cudaEvent_t start,stop;
	float milliseconds;

	cudaMalloc(&d_result,no_rows*sizeof(double));
	cudaMalloc(&d_mat_vals,nnz*sizeof(double));
	cudaMalloc(&d_vec_vals,no_cols*sizeof(double));
	cudaMalloc(&d_row_offset,(no_rows+1)*sizeof(int));
	cudaMalloc(&d_col_inds,nnz*sizeof(int));

	double total_time = 0;

	int nt = 256;
	int nb = ceil(double(no_rows)/ceil((double)nt/32));

	for(int counter = 0; counter < num_times_to_run; counter ++)
	{

		cudaMemcpy(d_mat_vals,sorted_vals,nnz*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_vec_vals,x_glob,no_cols*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_row_offset,row_offset,(no_rows+1)*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_col_inds,sorted_cols,nnz*sizeof(int),cudaMemcpyHostToDevice);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		calc<<<nb,nt,nt*sizeof(double)>>>(d_result,d_mat_vals,d_vec_vals,d_row_offset,d_col_inds,no_rows);
    	cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		total_time += milliseconds;
		cudaMemcpy(h_result,d_result,no_rows*sizeof(double),cudaMemcpyDeviceToHost);
	}

    cudaDeviceSynchronize();
	printf("Average time = %lf microseconds\n",(total_time*1e3)/num_times_to_run);
	ofstream out_file_result;
	out_file_result.open("result.txt");
	for(int i=0;i<no_rows;i++)
	{
		out_file_result << h_result[i] << endl;
	}
	out_file_result.close();

	return 0;
}
