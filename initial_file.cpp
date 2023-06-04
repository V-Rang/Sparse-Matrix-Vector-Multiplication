    /*
if declare using pointer int *x = new int[some value]. The more you have of such statements, code more likely to 
not proceed. 
*/

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


// template <typename T>
// vector<T> sort_indexes(const vector<T> &v) 
// {

//   vector<int> idx(v.size());
//   iota(idx.begin(), idx.end(), 0);
//   stable_sort(idx.begin(), idx.end(),[&v](int i1, int i2) {return v[i1] < v[i2];});
//   return idx;
// }


// void sorter_result(vector<int> &x, vector<int> &y, vector<double> &z, vector<int>&sorted_x, vector<int>&sorted_y, vector<double>&sorted_z )
// {
    
//     vector<int>idx(x.size());
//     std::iota(idx.begin(),idx.end(),0);

//     stable_sort(idx.begin(), idx.end(), [&x](int i1, int i2){ return x[i1] < x[i2]; });

//     for(auto p: idx)
//     {
//         sorted_x.push_back(x[p]);
//         sorted_y.push_back(y[p]);
//         sorted_z.push_back(z[p]);
//     }
// }

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
    // omp_set_num_threads(4);
    int i,j,k;
    const double minval  = 0.0;
    const double maxval = 100.0; 
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);


    // omp_set_num_threads(5);
    // #pragma omp parallel
    // {
    //     printf("Hello from thread %d\n",omp_get_thread_num());
    // }

    //vanilla attempt to read in a matrix market file and store in CSR format - values, col_inds and row_offsets.
    
    // string filename = "bips98_1450.mtx"; // 11k x 11k
    // int length_junk_text = 105;
    //******************************************//


    string filename = "1138_bus.mtx"; // 1k X 1k
    // int *length_junk_text = new int(13);
    int length_junk_text = 13;
    
    int no_row,no_cols,nnz;
    std::ifstream my_file;
    my_file.open(filename);
    string line;
    // string filename = "tx2010.mtx"; //900k X 900k
    // int *length_junk_text = new int(52);

    // if(!length_junk_text)
    // {
    //     printf("Failed to allocate memeory for length_junk_text\n");
    //     exit(1);
    // }

    // string filename = "mark3jac080.mtx"; //36k X 36k
    // int length_junk_text = 13;

    // string filename = "language.mtx"; //400k X 400k //has very small values 1e-37...
    // int length_junk_text = 13;
    // int *length_junk_text = new int(1); //standard for all cases.

    // string filename = "mi2010.mtx"; //330k X 330k
    // int *length_junk_text = new int(52);

    // string filename = "tx2010.mtx"; //900k X 900k
    // int *length_junk_text = new int(52);

    // std::ifstream my_file;
    // my_file.open(filename);

    // string contents;   
    // int i,j,k;
    // i = 0;


    // //first 13 lines not important
    // if(my_file.is_open())
    // {
    //     // while(my_file.good())
    //     while(i<*length_junk_text)
    //     {
    //         getline(my_file,contents);
    //         // cout << contents << endl;
    //         i += 1;   
    //     }
    // }
    // delete length_junk_text;
    // length_junk_text = nullptr;

    // //line 14 has the number of rows, number of columns and number of non-zero values in the sparse matrix
    
    // int no_rows=50;
    // int no_cols=50;
    // int nnz=100;

    // // int *no_rows = new int(1);
    // // int *no_cols = new int(1);
    // // int *nnz = new int(1);

    // int no_rows,no_cols,nnz;

    // getline(my_file,contents);

    // double vals[3]; //vals[0]->row index(int), vals[1]->col index(int), vals[2]->value(double)
    // i = 0;
    // j = 0;
    // k = 0;

    // while(i<contents.length())
    // {
    //     if(contents[i] == ' ')
    //     {
    //         vals[j] = stoi(contents.substr(k,i+1));
    //         j += 1;
    //         k = i;
    //     }
    //     i += 1;
    //     if(j == 2)
    //     {
    //         vals[j] = stoi(contents.substr(k+1));
    //         break;
    //     }
    // }

    // no_rows,no_cols,nnz = vals[0],vals[1],vals[2];

    // no_rows = vals[0];
    // no_cols = vals[1];
    // nnz = vals[2];

    // cout << no_rows << " " << no_cols << " " << nnz << endl;

    // int row_ind[nnz];
    // int col_ind[nnz];
    // double values[nnz];

    // cout << nnz << endl;
    // int index = 0;


    //reading rest of file to populate the row_ind | col_ind | values
    // vector<int>row_ind,col_ind;
    // vector<double>values;

    int *row_ind = new (std::nothrow) int[nnz];
    int *col_ind = new (std::nothrow) int[nnz];
    double *values = new (std::nothrow)  double[nnz];

    // if(!row_ind)
    // {
    //     printf("Failed to allocate memeory for row_ind\n");
    //     exit(1);
    // }

    // if(!col_ind)
    // {
    //     printf("Failed to allocate memeory for col_ind\n");
    //     exit(1);
    // }

    
    // if(!values)
    // {
    //     printf("Failed to allocate memeory for values\n");
    //     exit(1);
    // }


    // int ind_counter = 0;


    // while(true)
    // {

    //     // getline(my_file,contents);
    //     getline(my_file,contents);
    //     i = 0;
    //     j = 0;
    //     k = 0;
    //     while(i<contents.length())
    //     {
    //         if(contents[i] == ' ')
    //         {
    //             vals[j] = stoi(contents.substr(k,i));
    //             j += 1;
    //             k = i;
    //         }
    //         i += 1;
    //         if(j == 2)
    //         {
    //             vals[j] = stof(contents.substr(k+1));
    //             break;
    //         }
    //     }
    //     if(my_file.eof()) break;

    // //     // //three values in val
    //     // row_ind[index] = vals[0];
    //     // col_ind[index] = vals[1];
    //     // values[index] = vals[2];
    //     // cout << index << endl;
    //     *(row_ind + ind_counter) = vals[0] - 1;
    //     *(col_ind + ind_counter) = vals[1] - 1;
    //     *(values + ind_counter) = vals[2];

    //     ind_counter += 1;
        
    //     // row_ind.push_back(vals[0]-1); // subtract 1 as file values are indexed starting from 1
    //     // col_ind.push_back(vals[1]-1);
    //     // values.push_back(vals[2]);
        
    //     // index += 1;
    // //    cout << vals[0] << " " << vals[1] << " " << vals[2] << endl;

    // }

    int index_keeper = 0;
    while(true)
    {
        string line;
        getline(my_file,line);
        // test_function(line,row_ind,col_ind,values,index_keeper);

        i = 0; //iterator through the line
        k = 0; // iterator to keep track of position within line to read from to store values of row index, col index and values
        j = 0; // to keep track of how many of the total 3 values in each line have been read. 
        while(i<line.length())
        {
            if(line[i] == ' ')
            {
                if(j == 0)
                {
                    row_ind[index_keeper] = stoi(line.substr(k,i-k+1));
                }
                if(j == 1)
                {
                    col_ind[index_keeper] = stoi(line.substr(k,i-k+1));
                }
                k = i+1;
                j += 1;
                if( j == 2)
                {
                    values[index_keeper] = stof(line.substr(k));
                    break;
                }
            }

            i += 1;
        }
        if(my_file.eof()) break;
        // cout << row_ind[index_keeper] << " " << col_ind[index_keeper] << " " << values[index_keeper] << endl;
        // cout << counter << " " << line << endl;
        // counter += 1;
        // write row_ind, col_ind and values to a file and see if you got all of them. terminal may be incomplete.
        // outdata << row_ind[index_keeper] << " " << col_ind[index_keeper] << " " << values[index_keeper]<< endl;
        // outdata << line << endl;
        // Sleep(1000);
        // index_keeper += 1;
        // if(my_file.eof()) break;
    }
    // my_file.close();
    printf("Reached here 1\n");
    // outdata.close();
    // delete[] row_ind, col_ind, values;

    // cout << no_rows << " " << no_cols << " " << nnz << endl;

    // vector<int>sorted_rows;
    // vector<int>sorted_cols;
    // vector<double>sorted_vals;

    int *sorted_rows = new(std::nothrow) int[nnz];
    int *sorted_cols = new (std::nothrow) int[nnz];
    double *sorted_vals = new (std::nothrow) double[nnz];

    
    if(!sorted_rows)
    {
        printf("Failed to allocate memeory for sorted_rows\n");
        exit(1);
    }
    else
    {
        printf("Successfully allocated memory for sorted_rows\n");
    }
    
    if(!sorted_cols)
    {
        printf("Failed to allocate memeory for sorted_cols\n");
        exit(1);
    }
    else
    {
        printf("Successfully allocated memory for sorted_cols\n");
    
    }

    
    if(!sorted_vals)
    {
        printf("Failed to allocate memeory for sorted_vals\n");
        exit(1);
    }
    else
    {
        printf("Successfully allocated memory for sorted_vals\n");
    
    }

    // if(!sorted_rows || !sorted_cols || !sorted_vals)
    // {
    //     cout << "memory allocation failed" << endl;
    //     exit(1);
    // }

    // sorter_result(nnz,row_ind,col_ind,values,sorted_rows,sorted_cols,sorted_vals);
    printf("Reached here 2\n");
  
    // delete[] row_ind,col_ind,values;
    // row_ind = nullptr;
    // col_ind = nullptr;
    // values = nullptr;

//     printf("Before sorting\n");
//     for(int w = 0;w<5;w++)
//     {
//         cout << row_ind[w] << " " << col_ind[w] << " " << values[w] << endl;
//     }


//     printf("After sorting\n");
//    for(int w = 0;w<5;w++)
//     {
//         cout << sorted_rows[w] << " " << sorted_cols[w] << " " << sorted_vals[w] << endl;
//     }




    //create row offset vector using prefix scan - try to place it in function sorter results
    // int counter[*no_rows] = {0};



    // for(int i=0;i<*nnz;i++)
    // {
    //     counter[sorted_rows[i]] += 1;
    // }

    int *counter = new (std::nothrow) int[no_rows];
    // int counter[no_rows]; 
    if(!counter)
    {
        printf("Memory allocation failed for counter = %d\n",no_rows);
    }
    else
    {
         printf("Memory successfully allocated for counter = %d\n",no_rows);
    }


    // cout << *no_rows << " " << *no_cols << " " << *nnz << endl;
    // omp_set_num_threads(4); //if commented, code only reads till reach 6, else reads till end of program
    // #pragma omp parallel for private(i) shared(no_rows,counter) default(none)
    for(int i=0;i<no_rows;i++)
    {
        // *(counter + i) = 0;
        counter[i] = 0;
    }

    // #pragma omp parallel for private(i) shared(nnz,sorted_rows,counter) default(none)
    for(int i=0;i<nnz;i++)
    {
        counter[sorted_rows[i]] += 1;
    }
    printf("Reached here 3\n");
  
    // code works till here for 400k X 400k
    // cout << *no_rows + 1 << endl;
    // int ro[no_rows+1] = {0};
    
    // int ro[no_rows+1]; //code not run beyond this point
    // printf("Reached below ro\n");
   
    // int *counter = new int[no_rows];
    
    // try
    // {
    //     int *ro = new int[no_rows+1]; //nothing below this is output. Why? //if used, code doesn't compile beyond this point
    //     cout << "Memory for ro is allocated successfully" << endl;
    // }
    // catch ()

    int *ro = new (std::nothrow) int[no_rows+1];
    // int ro[no_rows+1];
    printf("Reached below ro\n"); 
    fflush(stdout);  
    
    if(!ro)
    {
        printf("Failed to allocate memeory for row offset\n");
        exit(1);
    }
    else
    {
        printf("Successfully allocated memory for row offset\n");
    }


    // int *ro = new int( (*no_rows) + 1);


    // cout << *no_rows << " " << *no_cols << " " << *nnz << endl; // code doesn't get to this point, no errors either
    // ro[0] = 0;
    // ro[no_rows] = nnz;

    // for(int i=1;i<=no_rows-1;i++)
    // {
    //     ro[i] = ro[i-1] + counter[i-1];
    // }
    printf("Reached here 4\n");  

    // // int *counter = new int[no_rows];

    // // delete[] counter; // if uncommented, code doesn't proceed beyond this point for 1138_bus.mtx (no error either) Why?
    // // counter = nullptr;
    // printf("I am here\n"); //is not printed if above line is uncommented

    // //Have CSR repr - row_offset, sorted_cols, sorted_vals

    // //vector for GEMV
    // // double x_vec[*no_cols];
    // // double *x_vec = new double[no_cols]; //if used, code doesn't compile beyond this point
    // double x_vec[no_cols];
    
    // // if(!x_vec)
    // // {
    // //     printf("Failed to allocate memeory for x_vec\n");
    // //     exit(1);
    // // }

    // for(int w=0;w<no_cols;w++)
    // {
    //     x_vec[w] = dist(gen);
    //     cout << w << endl;
    // }

    // printf("Reached here 5\n");
  

    // // double *y_vec = new double[no_rows]; //if used, code doesn't compile beyond this point
    // double y_vec[no_rows];
    // printf("Reached below y_vec\n");
    // if(!y_vec)
    // {
    //     printf("Failed to allocate memeory for y_vec\n");
    //     exit(1);
    // }
    // //seq Mv
    
    // auto start = tic;
    // for(int row=0;row<no_rows;row++)
    // {
    //     y_vec[row] = 0.0;
    //     for(int nz=ro[row]; nz<ro[row+1];nz++)
    //     {
    //         y_vec[row] += sorted_vals[nz]*x_vec[sorted_cols[nz]];
    //     }
    // }
    // auto end = toc;
    // cout<<"Time taken by serial code: "<<std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()<< " microseconds."<<endl;

    // printf("Reached here 6\n");
  
    // //parallel code
    // // double y_vec_par[*no_rows] = {0};
    // // double *y_vec_par = new double[no_rows]; //if used, code doesn't compile beyond this point
    // double y_vec_par[no_rows]; 

    // if(!y_vec_par)
    // {
    //     printf("Failed to allocate memeory for y_vec_par\n");
    //     exit(1);
    // }

    // // omp_set_num_threads(4);

    // int row;
    // int nz;
    // start = tic;
    // //par Mv
    // // #pragma omp parallel private(row,nz) shared(ro,no_rows,x_vec,sorted_vals,sorted_cols,y_vec_par) default(none) 
    // // {

    //     // #pragma omp for nowait schedule(static,4) 
    //     for(row=0;row<no_rows;row++)
    //     {
    //         y_vec_par[row] = 0.0;
    //         // #pragma omp for private(nz) reduction(y_vec)
    //         for(nz=ro[row]; nz<ro[row+1]; nz++)
    //         {
    //             y_vec_par[row] += sorted_vals[nz]*x_vec[sorted_cols[nz]];
    //         }
    //     }
    //     // #pragma omp barrier
    // // }
    // end = toc;

    // cout<<"Time taken by parallel code: "<<std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()<< " microseconds."<<endl;

    // printf("Reached here 7\n");
  
    // //to check if serial code is same as parallel code
    // int n_diff=  0;
    // for(int i=0;i<no_rows;i++)
    // {
    //     if(abs(y_vec[i] - y_vec_par[i]) > 1e-4) n_diff += 1;
    // }

    // printf("No of values that are different are %d\n",n_diff);

    // for(i=0;i<100;i++)
    // {
    //     cout << y_vec[i] << " " << y_vec_par[i] << endl;
    // }

    // delete no_rows,no_cols,nnz;
    // delete[] counter;
    
    // delete[] y_vec;
    // delete[] y_vec_par;
    // delete[] x_vec;
    // delete[] ro;
    delete[] sorted_rows;
    delete[] sorted_cols;
    delete[] sorted_vals; 
    // y_vec = nullptr;
    // y_vec_par = nullptr;
    // x_vec = nullptr;
    // ro = nullptr;
    // sorted_rows = nullptr;
    // sorted_cols = nullptr;
    // sorted_vals = nullptr;

    // delete[] counter;

    //seq mat-vec
    // // double a_mat[no_rows][no_cols] = {0.0};


    // double y[no_rows];

    // for(int row = 0; row < no_rows; row++)
    // {
    //     y[row] = 0.0;
    //     for(int nnz = row_offset[row]; nnz < row_offset[row+1]; nnz++)
    //     {
    //         y[row] += sorted_vals[nnz]*x_vec[sorted_cols[nnz]];
    //     }
    // }

    // for(int w = 0; w< no_rows; w++)
    // {
    //     cout << y[w] << endl;
    // }


    // for(int w= 0;w<no_rows;w++)
    // {
    //     cout << row_offset[w] << " ";
    // }
    // cout << endl;

//     {
//         cout << sorted_rows[w] << " " << sorted_cols[w] << " " << sorted_vals[w] << endl;
//     }


    //sorted_cols and sorted_vals - final

    // for(int index = 0; index< nnz;index ++)
    // {
    //     cout << row_ind[index] << " ";
    // }
    // cout << endl;


    // cout << row_ind[0] << endl;



    return 0;
}