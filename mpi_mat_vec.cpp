/*
Matrix-vector multiplication of a square matrix using MPI, p = total_nodes. (p-1) nodes are used to perform the multiplication.
Hence, n must be divisible by (p-1). 
Steps:
1. Node 0 reads in a file of format .mtx that is in COO format: row_ind | col_ind | value.
2. The .mtx files are taken from UF Sparse Matrix collection that are stored in ascending order by columns.
3. Node 0 first sorts the above values of row_ind, col_ind and values in ascending order by rows.
4. Node 0 initializes the global vector x_glob and distributes n/(p-1) length to each of the other (p-1) processes.
5. Node 0 then reads in block of n/(p-1) rows, p iterations needed to finish reading the entire matrix.
6. Each block is distributed to (p-1) processes, each process receives n/(p-1) columns.
7. For each iteration, each process does local mat-vec and stores result in y_loc.
8. y_loc of all processes is Reduced in the global array in node 0 - y_glob.
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
#include<mpi.h>

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

// sorter_result(nnz,row_ind,col_ind,values,sorted_rows,sorted_cols,sorted_vals);
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


/*function to create a derived data type - mpi_custom_block.
It allows distributing- both sending from proc 0 and receiving into the other procs
of n/p X n/p blocks of the n/p X n block of rows read in by proc 0.
This is done by reading n/p length elements from each of the n/p rows and sending it to the jth proc.(1 -> total_nodes -1).

Key inference: Since a_loc_glob is a pointer to a pointer, the first element of the (i+1)th row is not at the next memory address
to the the last element of the ith row. This can be seen in the incorrect working of the function mat_dist used below.

*/
void custom_glb_block(int n, int p, double **a_loc_glob, MPI_Datatype *mpi_custom_block)
{

    // j -> to send to jth proc

    int blocklengths[n/(p-1)];
    MPI_Aint displacements[n / (p - 1)];
    MPI_Datatype typelists[n / (p - 1)];

    for (int i = 0; i < n / (p - 1); i++)
    {
        blocklengths[i] = n/(p-1);
        typelists[i] = MPI_DOUBLE;
    }

    displacements[0] = 0;
    MPI_Aint address, start_address;

    MPI_Get_address(&(a_loc_glob[0][0]), &start_address);

    for (int i=1; i < n/(p-1); i++)
    {
        MPI_Get_address(&(a_loc_glob[i][0]), &address);
        displacements[i] = address - start_address;
    }

    MPI_Type_create_struct(n / (p - 1), blocklengths, displacements, typelists, mpi_custom_block);
    MPI_Type_commit(mpi_custom_block);
}

void custom_loc_block(int n, int p, double **a_loc, MPI_Datatype *mpi_custom_block)
{
    // j -> to send to jth proc

    int blocklengths[n / (p - 1)];
    MPI_Aint displacements[n / (p - 1)];
    MPI_Datatype typelists[n / (p - 1)];

    for (int i = 0; i < n / (p - 1); i++)
    {
        blocklengths[i] = n / (p - 1);
        typelists[i] = MPI_DOUBLE;
    }

    displacements[0] = 0;
    MPI_Aint address, start_address;

    MPI_Get_address(&(a_loc[0][0]), &start_address);

    for (int i=1; i < n/(p-1); i++)
    {
        MPI_Get_address(&(a_loc[i][0]), &address);
        displacements[i] = address - start_address;
    }

    MPI_Type_create_struct(n / (p - 1), blocklengths, displacements, typelists, mpi_custom_block);
    MPI_Type_commit(mpi_custom_block);
}

void x_dist(double *x_glob, double *&x_loc,int my_node, int n, int total_nodes,MPI_Status &status)
{
   //*VVI: need to have int *&x_loc and not int *x_loc. Else values in x_loc will not be reflected outside function when using x_loc = new int[n/(p-1)]
    int p = total_nodes;
    if(my_node == 0)
    {
        for(int j=1;j<p;j++)
        {
            int tag = stoi(std::to_string(j) + "0");
            MPI_Send(&x_glob[(j-1)*(n/(p-1))],n/(p-1),MPI_DOUBLE,j,tag,MPI_COMM_WORLD);
        }
    }
    else
    {
        x_loc = new double[n/(p-1)];
        int tag = stoi(std::to_string(my_node) + "0");
        MPI_Recv(x_loc,n/(p-1),MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);

    }
}

/*
This function is not used and is incorrect in its goal. See description of function custom_block given above.
*/
void mat_dist(float **a_loc_glob, float **&a_loc, int n, int my_node, int total_nodes, int number_iter, MPI_Status &status)
{

    int p = total_nodes;
    
    MPI_Datatype mpi_block_glob;
    MPI_Type_vector(n / (p - 1), n / (p - 1), n, MPI_FLOAT, &mpi_block_glob);
    MPI_Type_commit(&mpi_block_glob);

    MPI_Datatype mpi_block_loc;
    MPI_Type_vector(n / (p - 1), n / (p - 1), n / (p - 1), MPI_FLOAT, &mpi_block_loc);
    MPI_Type_commit(&mpi_block_loc);

    if (my_node == 0)
    {
        for (int j = 1; j < p; j++)
        {
            int tag = stoi(std::to_string(j) + "0" + std::to_string(number_iter));

            MPI_Send(&(a_loc_glob[0][(j - 1) * (n / (p - 1))]), 1, mpi_block_glob, j, tag, MPI_COMM_WORLD);
        }
    }
    else
    {
        a_loc = new float*[n / (p - 1)];
        for (int v = 0; v < n / (p - 1); v++)
        {
            a_loc[v] = new float[n / (p - 1)];
        }

        int tag = stoi(std::to_string(my_node) + "0" + std::to_string(number_iter));
        MPI_Recv(&(a_loc[0][0]), (n / (p - 1)) * (n / (p - 1)), mpi_block_loc, 0, tag, MPI_COMM_WORLD, &status);
        
    }
}

/*
function to perform mat-vec multiplication of a_loc and x_loc, result is stored in y_loc.
Notice usage of &  in int *&y_loc, this is done so results in y_loc initialzied inside the function as
new int[n/(p-1)] are reflected outside the function. 
*/
// mat_vec(a_loc,x_loc,y_loc,n,p,my_node,i); //mat-vec for every proc - a_loc X x_loc, result stored in y_loc; done for every block of n/p rows read in by proc 0, So y_loc: n/p X 1 vector. (i)

void mat_vec(double **a_loc, double *x_loc, double *&y_loc, int n, int total_nodes,int my_node, int number_iter)
{
    int p = total_nodes;
    y_loc = new double[n/(p-1)];
    
    // for(int i=0;i<n/(p-1);i++)
    // {
    //     y_loc[i] = 0;
    //     for(int j=0;j<n/(p-1);j++)
    //     {
    //         y_loc[i] += a_loc[i][j] * x_loc[j];
    //     }
    // }    

    //mat-vec speedup using OpenMP
    int i,j;
    // omp_set_num_threads(3);
    // #pragma omp parallel for private(i) shared(j,y_loc,n,p,a_loc,x_loc) default(none)
    for(i=0;i<n/(p-1);i++)
    {
        y_loc[i] = 0;
        for(j=0;j<n/(p-1);j++)
        {
            y_loc[i] += a_loc[i][j] * x_loc[j];
            // if(y_loc[i] < 1e-3) y_loc[i] = 0;
        }
    } 

}

int main(int argc, char **argv)
{
    // omp_set_num_threads(4);
    // enum omp_proc_bind_true = 1;
    // bool omp_proc_bind_true = 1;
    // omp_proc_bind_true;
    // OMP_PROC_BIND()=true;
    // cout << omp_get_proc_bind() << endl;

    int my_node,total_nodes; //my_node - proc ID, total_nodes - total number of processes

    MPI_Status status;
    // MPI_Request req;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);

    int n; //order of square matrix.
    int p = total_nodes;
    double *x_glob; //global x of length n read in by proc 0, distributed to all the procs in x_loc (length = n/(p-1)).
    double *x_loc;
    double *y_glob; //global y of length n of proc 0 to which all the procs return their local mat-vec result to.
    double *y_loc;
    int no_rows,no_cols,nnz;
    int *row_ind, *col_ind;
    double *values;

    int *sorted_rows, *sorted_cols;
    double *sorted_vals;
   
    //proc 0 reads in the number of rows of the square matrix and bcasts it to all the other procs.
//    std::ifstream my_file;
    if(my_node == 0)
    {
        // printf("Enter number of rows of square matrix\n");
        // fflush(stdout);
        // scanf("%d",&n);

        string filename = "mi2010.mtx"; // 14 X 14
        std::ifstream my_file;
        my_file.open(filename);
        string line;
    
        while(true)
        {
            getline(my_file,line);
            if(my_file.eof()) break;
            // count += 1;
            if(line[0] == '%') 
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
        double value_read;
    
        while(true)
        {
            getline(my_file,line);
            if(my_file.eof()) break;
            sscanf(line.c_str(),"%d %d %lf",&row_ind_p1,&col_ind_p1,&values[index_keeper]);
            row_ind[index_keeper] = row_ind_p1-1;
            col_ind[index_keeper] = col_ind_p1-1;
            // if(value_read < 1e-3) value_read = 0;
            // values[index_keeper] = value_read;
            // out_file << row_ind_p1-1 << " " << col_ind_p1-1 << " " << values[index_keeper] << endl; 
            index_keeper += 1;
        }

        my_file.close();
        sorted_rows = new int[nnz];
        sorted_cols = new int[nnz];
        sorted_vals = new double[nnz];

        sorter_result(nnz,row_ind,col_ind,values,sorted_rows,sorted_cols,sorted_vals);
    //     // ofstream out_test;
    //     // out_test.open("./results/test_values.txt");
    //     // for(int i=0;i<nnz;i++) out_test << sorted_rows[i] << " " << sorted_cols[i] << " " << sorted_vals[i] << endl;
    //     // out_test.close();

        delete[] row_ind,col_ind,values;
        row_ind = nullptr;
        col_ind = nullptr;
        values = nullptr;
    }

    MPI_Bcast(&no_rows,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&no_cols,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&nnz,1,MPI_INT,0,MPI_COMM_WORLD);

    // if(my_node == 1)
    // {
    //     cout << no_rows << " " << no_cols << " " << nnz << endl;
    // }

    n = no_rows;
    // double **a_loc_glob;
    // double **a_loc;

    if(my_node == 0)
    {
        x_glob = new double[no_rows];
        for(int i=0;i<no_rows;i++)
        {
            x_glob[i] = i+1;
        } 
        
        y_glob = new double[n];

        y_loc = new double[n/(p-1)];
        for(int v = 0;v<n/(p-1);v++)
        {
            y_loc[v] = 0;
        }

       
    }


    // if(my_node == 0)
    // {
    //     for(int i=0;i<n;i++) cout << x_glob[i] <<  " ";
    //     cout << endl;
    // }
    x_dist(x_glob,x_loc,my_node,no_rows,total_nodes,status);
    if(my_node == 0)
    {
        delete[] x_glob;
        x_glob = nullptr;
    }

    // if(my_node == 1)
    // {
    //     for(int i=0;i<n/(p-1);i++) cout << x_loc[i] << " ";
    //     cout << endl;
    // }

    MPI_Datatype mpi_glb_block;
    MPI_Datatype mpi_loc_block;
   
    // // // //     //For MPI_Reduce - we finally reduce results of y_loc of every iteration( i.e. every block of n/p rows) to y_glob of proc 0. MPI_Reduce requires y_loc of EVERY proc to be initialized. Hence 0s for proc 0.
    // // // //     // y_loc = new int[n/(p-1)];
    // // // //     // for(int v = 0;v<n/(p-1);v++)
    // // // //     // {
    // // // //     //     y_loc[v] = 0;
    // // // //     // }

    // // // //     // a_loc_glob = new int*[n/(p-1)];
    // // // //     // for(int v = 0;v<n/(p-1);v++)
    // // // //     // {
    // // // //     //     a_loc_glob[v] = new int[n];
    // // // //     // } 
        
    int i;
    int counter = 0;
    // ofstream out_file;

    // //   if(my_node == 1)
    // // {
    // //     ofstream out_file;
    // //     out_file.open("./results/test_values_node1.txt",std::ios_base::app);
    // // }

    // // if(my_node == 2)
    // // {
    // //     out_file.open("./results/test_values_node2.txt",std::ios_base::app);
    // // }
    // ofstream out_file_1;

    

    // ofstream out_file_1;
    // if(my_node == 1)
    // {
    //     out_file_1.open("./results/test_values_node1.txt");
    // }

    // ofstream out_file_2;
    // if(my_node == 0)
    // {
    //     out_file_2.open("./results/test_glb_values.txt");   
    // }

    // if(my_node == 0)
    // {
    //     for(int i=0;i<nnz;i++)
    //     {
    //         out_file_2 << sorted_rows[i] << " " << sorted_cols[i] << "  " << sorted_vals[i] << endl;
    //     }
    //     out_file_2 << "end of first set of outputs\n";
    // }


    for(i=0; i < p-1 ; i++)
    {
        if(my_node == 0)
        {
            double **a_loc_glob;
            a_loc_glob = new double*[n/(p-1)];
            for(int v = 0;v<n/(p-1);v++)
            {
                a_loc_glob[v] = new double[n];
            }

            // double **a_loc;
    // //         // cout << i << endl;
    // //         // y_loc = new int[n/(p-1)];
    // //         // for(int v = 0;v<n/(p-1);v++)
    // //         // {
    // //         //     y_loc[v] = 0;
    // //         // }
    // //         // if(my_node == 0)
    // //         // {
    // //         //     for(int w = 0;w<n/(p-1);w++) cout << y_loc[w] << " ";
    // //         //     cout << endl;
    // //         // }

    // // //         // int counter = 0;

            for(int v = 0;v<n/(p-1);v++)
            {
                for(int w = 0; w< n; w++)
                {
                    a_loc_glob[v][w]  = 0;
                }
            }

            while(sorted_rows[counter] < (i+1)*(n/(p-1)))
            {
                // if(sorted_vals[counter] < 1e-3) sorted_vals[counter] = 0;
                // int row_index = sorted_rows[counter] % (n/(p-1));
                // out_file_2 << sorted_rows[counter] % (n/(p-1)) << " " << sorted_cols[counter] << " " << sorted_vals[counter] << endl;
                a_loc_glob[sorted_rows[counter] % (n/(p-1))][sorted_cols[counter]] = sorted_vals[counter];
                counter += 1;
                if(counter == nnz) 
                {
                    break;
                }
            }
            // out_file_2 << "end of second set of outputs\n";
    // // //         // printf("For i = %d\n",i);
            // out_file_2 << "For i = " << i << endl;
            // for(int v=0;v<n/(p-1);v++)
            // {
            //     for(int w = 0; w<n;w++)
            //     {
            //         if(fabs(a_loc_glob[v][w]) < 1e-3) out_file_2 << 0 << " ";
            //         else out_file_2 << a_loc_glob[v][w] << " ";
            //     }
            //     out_file_2 << endl;
            // }

            for (int j = 1; j < p; j++) //sending blocks of n/(p-1) columns to every other proc.
            {
                int tag = stoi(std::to_string(j) + "0" + std::to_string(i));
                custom_glb_block(n,p,a_loc_glob,&mpi_glb_block); //custom datatype to send n/(p-1) elements of each of the n/(p-1) rows.
                MPI_Send(&(a_loc_glob[0][(j-1)*(n/(p-1))]),1,mpi_glb_block,j,tag,MPI_COMM_WORLD);
            }
            delete[] a_loc_glob;
            a_loc_glob = nullptr;
        }
        else
        {
            // if(my_node == 1) cout << i << endl;
            double **a_loc;
            a_loc = new double*[n /(p-1)];
            for(int v= 0;v<n/(p-1);v++)
            {
                a_loc[v] = new double[n/(p-1)];
            }
            
            int tag = stoi(std::to_string(my_node) + "0" + std::to_string(i));
            custom_loc_block(n,p,a_loc,&mpi_loc_block);
            MPI_Recv(&(a_loc[0][0]),1,mpi_loc_block,0,tag,MPI_COMM_WORLD,&status); //every proc has n/p X n/p block in a_loc;
    //         // MPI_Wait(&req,&status);
            
            // if(my_node == 1)
            // {
            //     out_file_1 << "for i = " << i << endl;
            //     for(int v=0;v<n/(p-1);v++)
            //     {
            //         for(int w = 0; w < n/(p-1); w++)
            //         {
            //             if(fabs(a_loc[v][w]) < 1e-3) a_loc[v][w] = 0;
            //             out_file_1 << a_loc[v][w] << " ";
            //         }
            //         out_file_1 << endl;
            //     }
            // }

    //         // printf("For i = %d\n",i);
    //         // for(int v = 0; v<n/(p-1); v++)
    //         // {
    //         //     for(int w = 0; w<n/(p-1);w++)
    //         //     {
    //         //         cout << a_loc[v][w] << " "; 
    //         //     }
    //         //     cout << endl;
    //         // }
    // //         // }
    //         // y_loc = new float[n/(p-1)];
            mat_vec(a_loc,x_loc,y_loc,n,p,my_node,i); //mat-vec for every proc - a_loc X x_loc, result stored in y_loc; done for every block of n/p rows read in by proc 0, So y_loc: n/p X 1 vector. (i)
            delete[] a_loc,x_loc;
            a_loc = nullptr;
            x_loc = nullptr;

    //         // int j;
    //         // y_loc = new float[n/(p-1)];
    //         // // // omp_set_num_threads(4);
    //         // // #pragma omp parallel for private(i) shared(j,y_loc,n,p,a_loc,x_loc) default(none)
    //         // for(i=0;i<n/(p-1);i++)
    //         // {
    //         //     y_loc[i] = 0;
    //         //     for(j=0;j<n/(p-1);j++)
    //         //     {
    //         //         y_loc[i] += a_loc[i][j] * x_loc[j];
    //         //     }
    //         // } 

    //         // out_file << "for i = " << i << endl;
    //     //     if(my_node == 1)
    //     //     {
    //     //         ofstream out_file;
    //     //         out_file.open("./results/test_values_node1.txt");
    //     //         // for(int w = 0;w<n/(p-1);w++) out_file << y_loc[w] << endl;
    //     //         for(int v = 0; v<n/(p-1);v++)
    //     //         {
    //     //             for(int w = 0; w < n/(p-1); w++)
    //     //             {
    //     //                 out_file << a_loc[v][w] << " ";
    //     //             }
    //     //             out_file << endl;
    //     //         }
    //     //         out_file.close();
    //     //     }
        }
        MPI_Reduce(&(y_loc[0]),&(y_glob[i*(n/(p-1))]),n/(p-1),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);// y_glob contains the reduced values of y_loc for every proc; done for every block of n/p rows read in by proc 0. (i)
        delete[] y_loc;
        y_loc = nullptr;

    }

    // // //     // MPI_Barrier(MPI_COMM_WORLD);
    // //     // MPI_Reduce(&(y_loc[0]),&(y_glob[i*(n/(p-1))]),n/(p-1),MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);// y_glob contains the reduced values of y_loc for every proc; done for every block of n/p rows read in by proc 0. (i)    
    // // // if(my_node == 0) //final answer of mat-vec of nXn square matrix with vector of length n.
    // // // {
    // // //     for(int i=0;i<n;i++)
    // // //     {
    // // //         cout << y_glob[i] << " ";
    // // //     }
    // // //     cout << endl;
    // // // }
    // // //compare y_glob from above MPI Code to the result of the openmp code from OpenMP code
    
    if(my_node == 0)
    {
        ofstream out_file;
        out_file.open("./results/mpi_result.txt");
        for(int i=0;i<n;i++)
        {
            out_file << y_glob[i] << endl;
        }
        out_file.close();
        delete[] y_glob;
        y_glob = nullptr;
    }

    // if(my_node == 0)
    // {
    //     for(int i=0;i<n;i++)
    //     {
    //         cout << y_glob[i] << endl;
    //     }
    // }

    MPI_Finalize();

}
