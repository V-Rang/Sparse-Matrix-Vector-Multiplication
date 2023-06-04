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

int main()
{
    int i,j,k;
    const double minval  = 0.0;
    const double maxval = 100.0; 
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);

    string filename = "tx2010.mtx"; // 1k X 1k
    // int length_junk_text = 13;

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

    if(!row_ind)
    {
        printf("failed to allocate row_ind\n");
        exit(1);
    }
    else
    {
        printf("succesfully allocated row_ind\n");
    }


    if(!col_ind)
    {
        printf("failed to allocate col_ind\n");
        exit(1);
    }
    else
    {
        printf("succesfully allocated col_ind\n");
    }


    if(!values)
    {
        printf("failed to allocate values\n");
        exit(1);
    }
    else
    {
        printf("succesfully allocated values\n");
    }

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
    if(!ro)
    {
        printf("failed to allocate row offset\n");
        exit(1);
    }
    else
    {
        printf("succesfully allocated row offset\n");
    } 

    ro[0] = 0;
    ro[no_rows] = nnz;

    for(i = 1;i<=no_rows-1;i++)
    {
        ro[i] = ro[i-1] + counter[i-1];
    } 

    delete[] counter;
    counter = nullptr;

    double *x_vec = new (std::nothrow) double[no_cols]; 
    if(!x_vec)
    {
        printf("failed to allocate x_vec\n");
        exit(1);
    }
    else
    {
        printf("succesfully allocated x_vec\n");
    } 

    for(i=0;i<no_cols;i++)
    {
        x_vec[i] = dist(gen);
        // cout << w << endl;
    }

    double *y_vec = new (std::nothrow) double[no_rows];
    if(!y_vec)
    {
        printf("failed to allocate y_vec\n");
        exit(1);
    }
    else
    {
        printf("succesfully allocated y_vec\n");
    } 

    double *y_vec_par = new (std::nothrow) double[no_rows];
    if(!y_vec_par)
    {
        printf("failed to allocate y_vec_par\n");
        exit(1);
    }
    else
    {
       printf("succesfully allocated y_vec_par\n");
    } 

    Cputimer timer;


    // auto start = tic;
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
    // auto end = toc;
    // cout<<"Time taken by serial code: "<<std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()<< " microseconds."<<endl;
    cout << "Time taken by serial code: " << timer.EllapsedMicros() << " microseconds" << endl;

    omp_set_num_threads(5);
    int row,nz;
    
    // start = tic;
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
    // end = toc;
    // cout<<"Time taken by parallel code: "<<std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()<< " microseconds."<<endl;
    cout << "Time taken by parallel code: " << timer.EllapsedMicros() << " microseconds" << endl;
    delete[] sorted_rows,sorted_cols,sorted_vals;x_vec,ro;
    sorted_rows = nullptr;
    sorted_cols = nullptr;
    sorted_vals = nullptr;
    x_vec = nullptr;
    ro = nullptr;

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

    printf("No of values that are different are %d\n",n_diff);
    printf("Values that are different =:\n");

    for(auto p: incorrect_index)
    {
        cout << p << " " << y_vec[p] << " " << y_vec_par[p] << endl; 
    }


    delete[] y_vec,y_vec_par;
    y_vec = nullptr;
    y_vec_par = nullptr;

    // out_file.open("row_off.mtx");
    // for(int i=0;i<no_rows+1;i++)
    // {
    //     out_file << ro[i] << endl;
    // }
    // out_file.close();


    printf("Reached here 1\n");



    return 0;
}