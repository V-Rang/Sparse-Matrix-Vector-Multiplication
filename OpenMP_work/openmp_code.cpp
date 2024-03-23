#include<iostream>
#include<omp.h>
#include<string.h>
#include<fstream>
#include<vector>
#include<numeric>
#include<algorithm>
#include<random>
#include<chrono>
#include<windows.h>
#include <cstdio>


#define tic chrono::high_resolution_clock::now()
#define toc chrono::high_resolution_clock::now()

#define milliseconds(x) std::chrono::duration_cast<std::chrono::milliseconds>(x)
#define microseconds(x) std::chrono::duration_cast<std::chrono::microseconds>(x)     

using namespace std;

struct indice_maker //structure such that arr[n] = n, without explicitly storing an array of size n. 
{
    int val;
    
    // typedef indice_maker self_type;

    inline indice_maker(int value): val(value) {};

    inline int operator[](int n)
    {
        return val+n;
    }

};

// struct example
// {
//     int val;
//     typedef example self_type;

//     inline example(int value): val(value) {};

//     inline int operator[](int n)
//     {
//         return val+n;
//     }
// };

struct Cputimer
{
    double start;
    double stop;

    void Start()
    {
        start = omp_get_wtime();
    }

    void Stop()
    {
        stop = omp_get_wtime();
    }

    float EllapsedMicros()
    {
        return (stop - start)*1e6;
    }
};

struct thread_coord
{
    int x,y; //each thread will have a x and y coordinate when it traverses the grid of merge arrays.
};

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


void thread_start_end_coordinates(int diagonal, int *a, indice_maker b, int num_rows, int nnz, thread_coord &thread)
{
    int x_min = max(diagonal-nnz,0);
    int x_max = min(diagonal,num_rows);

    while(x_min < x_max)
    {
        int x_pivot = (x_min + x_max) >> 1;
        if(a[x_pivot] <= b[diagonal-x_pivot-1])
        {
            x_min = x_pivot + 1;
        }
        else
        {
            x_max = x_pivot;
        }
    }
    thread.x = min(x_min,num_rows);
    thread.y = diagonal - x_min;
}

void merge_spmv(int num_threads, int *row_end_offsets, int *col_ind, double *values, double *x_vec, double *y_vec, int no_rows, int nnz)
{
    int rco[256];
    double vco[256];
    
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int tid=0;tid<num_threads;tid++)
    {
        indice_maker nnz_indices(0);

        int num_merge_items = no_rows + nnz;
        int items_per_thread = (num_merge_items + num_threads - 1)/num_threads;


        thread_coord tc;
        thread_coord tc_end;

        int start_diagonal = min(items_per_thread*tid, num_merge_items);
        int end_diagonal = min(start_diagonal + items_per_thread,num_merge_items);

      

        thread_start_end_coordinates(start_diagonal,row_end_offsets,nnz_indices,no_rows,nnz,tc);
        thread_start_end_coordinates(end_diagonal,row_end_offsets,nnz_indices,no_rows,nnz,tc_end);


        for(;tc.x<tc_end.x;++tc.x)
        {
            double rt = 0;
            for(;tc.y<row_end_offsets[tc.x];++tc.y)
            {
                rt += values[tc.y] * x_vec[col_ind[tc.y]];
            }
            y_vec[tc.x] = rt;
        }

        double rt = 0;

        for(;tc.y<tc_end.y;++tc.y)
        {
            rt += values[tc.y] * x_vec[col_ind[tc.y]];
        }

        rco[tid] = tc_end.x;
        vco[tid] = rt;
    }

    for(int tid = 0; tid < num_threads-1;++tid)
    {
        if(rco[tid] < no_rows)  
        {
            y_vec[rco[tid]] += vco[tid];
        }
    }

}


int main()
{
    int i,j,k;
    const double minval  = 0.0;
    const double maxval = 100.0; 
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);

    string filename = "tx2010.mtx"; // 1k X 1k

    int no_rows,no_cols,nnz;

    std::ifstream my_file;
    my_file.open(filename);
    string line;
   
    while(true)
    {
        if(my_file.eof()) break;
        getline(my_file,line);
        // count += 1;
        if(line[0] == '%') 
        {
            continue;
        }
        sscanf(line.c_str(),"%d %d %d",&no_rows,&no_cols,&nnz);
        break;
        
    }

    // cout << no_rows << " " << no_cols << " " << nnz << endl;
    // cout << count << endl;

    int *row_ind = new (std::nothrow) int[nnz];
    int *col_ind = new (std::nothrow) int[nnz];
    double *values = new (std::nothrow)  double[nnz];

    // if(!row_ind)
    // {
    //     printf("failed to allocate row_ind\n");
    //     exit(1);
    // }
    // else
    // {
    //     printf("succesfully allocated row_ind\n");
    // }


    // if(!col_ind)
    // {
    //     printf("failed to allocate col_ind\n");
    //     exit(1);
    // }
    // else
    // {
    //     printf("succesfully allocated col_ind\n");
    // }


    // if(!values)
    // {
    //     printf("failed to allocate values\n");
    //     exit(1);
    // }
    // else
    // {
    //     printf("succesfully allocated values\n");
    // }

    // std::ofstream out_file;
    // out_file.open("test_data.mtx");

    int index_keeper = 0;
    int row_ind_p1;
    int col_ind_p1;
    while(true)
    {
        if(my_file.eof()) break;
        getline(my_file,line);
        sscanf(line.c_str(),"%d %d %lf",&row_ind_p1,&col_ind_p1,&values[index_keeper]);
        row_ind[index_keeper] = row_ind_p1-1;
        col_ind[index_keeper] = col_ind_p1-1;
        // out_file << row_ind_p1-1 << " " << col_ind_p1-1 << " " << values[index_keeper] << endl; 
        index_keeper += 1;
    
    }
    my_file.close();

    int *sorted_rows = new int[nnz];
    int *sorted_cols = new int[nnz];
    double *sorted_vals = new double[nnz];

    sorter_result(nnz,row_ind,col_ind,values,sorted_rows,sorted_cols,sorted_vals);
   
    delete[] row_ind,col_ind,values;
    row_ind = nullptr;
    col_ind = nullptr;
    values = nullptr;

    int *counter = new (std::nothrow) int[no_rows];
  

    for(i=0;i<no_rows;i++)
    {
        counter[i] = 0;
    }

    for(i=0;i<nnz;i++)
    {
        counter[sorted_rows[i]] += 1;
    }
    
    // out_file.open("outdata.mtx");
    // for(int i=0;i<no_rows;i++)
    // {
    //     out_file << i << " " << counter[i] << endl;
    // }
    // out_file.close();

    int *ro = new (std::nothrow) int[no_rows+1];  
    // if(!ro)
    // {
    //     printf("failed to allocate row offset\n");
    //     exit(1);
    // }
    // else
    // {
    //     printf("succesfully allocated row offset\n");
    // } 

    ro[0] = 0;
    ro[no_rows] = nnz;

    for(i = 1;i<=no_rows-1;i++)
    {
        ro[i] = ro[i-1] + counter[i-1];
    } 

    delete[] counter;
    counter = nullptr;

    double *x_vec = new (std::nothrow) double[no_cols]; 
    // if(!x_vec)
    // {
    //     printf("failed to allocate x_vec\n");
    //     exit(1);
    // }
    // else
    // {
    //     printf("succesfully allocated x_vec\n");
    // } 

    for(i=0;i<no_cols;i++)
    {
        // x_vec[i] = dist(gen);
        x_vec[i] = i+1;
        // cout << w << endl;
    }

    double *y_vec = new (std::nothrow) double[no_rows];
    // if(!y_vec)
    // {
    //     printf("failed to allocate y_vec\n");
    //     exit(1);
    // }
    // else
    // {
    //     printf("succesfully allocated y_vec\n");
    // } 

    double *y_vec_par = new (std::nothrow) double[no_rows];
    // if(!y_vec_par)
    // {
    //     printf("failed to allocate y_vec_par\n");
    //     exit(1);
    // }
    // else
    // {
    //    printf("succesfully allocated y_vec_par\n");
    // } 

    Cputimer timer;

    timer.Start();
    for(int row=0;row<no_rows;row++)
    {
        y_vec[row] = 0.0;
        for(int nz=ro[row]; nz<ro[row+1];nz++)
        {
            y_vec[row] += sorted_vals[nz]*x_vec[sorted_cols[nz]];
        }
    }
    timer.Stop();
    cout << "Time taken by serial code: " << timer.EllapsedMicros() << " microseconds" << endl;

    int num_threads = 3;
    omp_set_num_threads(num_threads);
    int row,nz;
    
    timer.Start();
    #pragma omp parallel for private(row,nz) shared(y_vec_par,no_rows,ro,sorted_vals,x_vec,sorted_cols) default(none)
    for(row=0;row<no_rows;row++)
    {
        y_vec_par[row] = 0.0;
        for(nz=ro[row];nz<ro[row+1];nz++)
        {
            y_vec_par[row   ] += sorted_vals[nz]*x_vec[sorted_cols[nz]];
        }
    }
    timer.Stop();
    cout << "Time taken by parallel code: " << timer.EllapsedMicros() << " microseconds" << endl;
    
    

    double *y_vec_par_merge = new double[no_rows];
    
    timer.Start();
    merge_spmv(num_threads,ro+1,sorted_cols,sorted_vals,x_vec,y_vec_par_merge,no_rows,nnz);
    timer.Stop();
    cout << "Time taken by parallel merge code: " << timer.EllapsedMicros() << " microseconds" << endl;


   

    int n_diff = 0;
    vector<int>incorrect_index;
    for(int i=0;i<no_rows;i++)
    {
        if(abs(y_vec[i] - y_vec_par[i]) > 1e-4) 
        {
            n_diff += 1;
            incorrect_index.push_back(i);
        }

    }

    printf("No of values that are different for standard parallel case are %d\n",n_diff);
    printf("Values that are different =:\n");

    for(auto p: incorrect_index)
    {
        cout << p << " " << y_vec[p] << " " << y_vec_par[p] << endl; 
    }


    delete[] y_vec_par;
    y_vec_par = nullptr;

    int n_diff2 = 0;
    vector<int>incorrect_index2;
    for(int i=0;i<no_rows;i++)
    {
        if(abs(y_vec_par_merge[i] - y_vec[i]) > 1e-4) 
        {
            n_diff2 += 1;
            incorrect_index2.push_back(i);
        }

    }

    printf("No of values that are different for merge case are %d\n",n_diff2);
    printf("Values that are different =:\n");
    for(auto p: incorrect_index2)
    {
        cout << p << " " << y_vec[p] << " " << y_vec_par_merge[p] << endl; 
    }


    delete[] y_vec_par_merge,y_vec;
    y_vec_par_merge=nullptr;
    y_vec = nullptr;


    delete[] sorted_rows,sorted_cols,sorted_vals;x_vec,ro;
    sorted_rows = nullptr;
    sorted_cols = nullptr;
    sorted_vals = nullptr;
    x_vec = nullptr;
    ro = nullptr;


    // printf("Reached end\n");



    return 0;
}