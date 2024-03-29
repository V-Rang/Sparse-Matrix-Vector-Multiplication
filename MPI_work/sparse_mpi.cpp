#include<iostream>
#include<omp.h>
#include<string.h>
#include<fstream>
#include<vector>
#include<numeric>
#include<algorithm>
#include<random>
#include<chrono>
// #include<windows.h>
#include <cstdio>
#include<mpi.h>
#include<ctype.h>
#include<stdlib.h>
#include<stdio.h>

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
/*
function serves the following purpose:
Determine number of "additional" vector indicies that each node needs. i.e.
For example in the following mat-vec:
[[1,0,1,0]
 [0,0,0,0]
 [0,0,3,3]
 [4,4,4,4]]

 and
 vector:
 [1 2 3 4]

 if we perform the above with 3 nodes, the vector indices are distributed as follows:
 Node 0: 0,1
 Node 1: 2,3
 Node 2: _

 the matrix values are distrbuted as follows:
 Node 0: 1,1
 Node 1: 3,3,4,4,4,4
 Node 2: _

 To perform its portion of the mat-vec, Node 0 needs the vector value at index 2 (to multiply with 1 in column 2).
 Similarly, Node 1 needs the vector value at index 0 and 1 (to multiply with 4 in column 0 and 4 in column 1). 
*/

int investigator(int *local_col_inds, int my_node, int total_nodes, int no_rows, int n_loc_vals, int *needed_vec_count, int *needed_vec_displ)
{
    
    int gen_size = ceil((double)no_rows/total_nodes);
    int num_add_vectors_needed = 0;
    int local_int_index = my_node * gen_size;
    int local_fin_index = (my_node + 1)*gen_size;
    int temp_count = 0;
    needed_vec_displ[0] = 0;

    for(int i=0;i<total_nodes;i++) needed_vec_count[i] = 0;

    for(int i=0;i<n_loc_vals;i++)
    {
        if( (local_col_inds[i] < local_int_index) || (local_col_inds[i] >= local_fin_index) )
        {
            int node_to_get_from = local_col_inds[i]/gen_size;
            needed_vec_count[node_to_get_from] += 1;
            num_add_vectors_needed += 1;
        }
    }
    
    for(int i=1;i<total_nodes;i++)
    {
        needed_vec_displ[i] = needed_vec_displ[i-1] + needed_vec_count[i-1];
    }

    return num_add_vectors_needed;
}

/*
The matrix market files are arranged in the format: row index | col index | value. These values are sorted in ascending order of column. However, to 
create the row offset array, we need the indices and the matrix values to be rearranged to be sorted in ascending order of row. Following function
receives three arrays - x, y and z and creates three arrays - sorted_x, sorted_y and sorted_z which are sorted in ascending order of values in column x.
*/
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


int main(int argc, char**argv)
{
    // int num_times_to_run = 1;
    // if(argc > 1)
    // {
    //     num_times_to_run = atoi(argv[1]);
    // }

    double timer_start, timer_end,iteration_time,total_time;
    int my_node, total_nodes;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    
    int no_rows,no_cols,nnz;
    int *sorted_rows,*sorted_cols;
    double *sorted_vals;
    int *row_offset;
    double *x_glob;

    if(my_node == 0)
    {
        int *row_ind, *col_ind;
        double *values;
       

        string filename = "./Matrix_Files/1138_bus.mtx"; // any file in matrix market format, tested matrices taken from UF Sparse matrix collection.
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

    // to be distributed: nrows, rowoffset_array, matrix_values_array, vector_array, col_ind_array.

    //constructing row offset array
    // if(my_node == 0)
    // {
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
        //constructing vector values
        x_glob = new double[no_cols];
        for(int i=0;i<no_cols;i++) x_glob[i] = i+1;

    }
   
    MPI_Bcast(&no_rows,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&no_cols,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&nnz,1,MPI_INT,0,MPI_COMM_WORLD);

    //declare variables globally needed here.
    int *num_vec_vals;
    int *vec_vals_displ;
    int *num_mat_col_values;
    int *mat_col_displ;
    double *glob_result;
    double *loc_result;
    double *x_loc;
    double *loc_mat_vals;
    int  *loc_col_inds,*loc_row_offset;
    int *needed_vec_count,*needed_vec_index,*needed_vec_displ,*to_send_vec_count,*to_send_vec_displ,*to_send_vals_index;
    int n_loc_vals;
    int total_vals_to_send;
    double  *send_vec_vals;

    int gen_size,loc_size,loc_size_last;

    if(my_node == 0)
    {    

        num_vec_vals = new int[total_nodes];
        vec_vals_displ = new int[total_nodes];
        
        num_mat_col_values = new int[total_nodes];
        mat_col_displ = new int[total_nodes];

        gen_size = ceil((double)no_rows/total_nodes);
        loc_size_last = no_rows - (total_nodes-1)*gen_size;
        
        loc_size = gen_size;
        if(my_node == total_nodes-1) loc_size = no_rows - (total_nodes-1)*loc_size; 

        if(loc_size_last < 0) 
        {
            printf("Wrong choice of input number of nodes\n");
            exit(0);
        }

        /*
        decide on how many vector values each node will have -> use to construct displacement array for each node -> scatterv values
        accordingly from node 0.
        */
        for(int i=0;i<total_nodes-1;i++) num_vec_vals[i] = loc_size;
        num_vec_vals[total_nodes-1]  = no_rows - (total_nodes-1)*loc_size;
        vec_vals_displ[0] = 0;
        for(int i=1;i<total_nodes;i++) vec_vals_displ[i] = vec_vals_displ[i-1] + num_vec_vals[i-1];

        /*
        decide on how many matrix values and corresponding column indices each node will have -> use to construct displacement array 
        for each node -> scatterv values accordingly from node 0.
        */
        num_mat_col_values[0] = row_offset[num_vec_vals[0] + vec_vals_displ[0]];
        for(int i=1; i<total_nodes; i++)
        {
            num_mat_col_values[i] = row_offset[num_vec_vals[i] + vec_vals_displ[i]] - row_offset[num_vec_vals[i-1] + vec_vals_displ[i-1]];

        }
        mat_col_displ[0] = 0;
        for(int i=1;i<total_nodes;i++) mat_col_displ[i] = mat_col_displ[i-1] + num_mat_col_values[i-1];

        // let every ndoe know how many values from - sorted_cols and sorted_vals will be sent to it.
        MPI_Scatter(num_mat_col_values,1,MPI_INT,&n_loc_vals,1,MPI_INT,0,MPI_COMM_WORLD);
        loc_col_inds = new int[n_loc_vals];
        loc_mat_vals = new double[n_loc_vals];
        needed_vec_index = new int[n_loc_vals];

        //send values from sorted_cols and sorted_vals to every node.
        MPI_Scatterv(sorted_cols,num_mat_col_values,mat_col_displ,MPI_INT,loc_col_inds,n_loc_vals,MPI_INT,0,MPI_COMM_WORLD);
        delete[] sorted_cols;
        sorted_cols = nullptr;

        MPI_Scatterv(sorted_vals,num_mat_col_values,mat_col_displ,MPI_DOUBLE,loc_mat_vals,n_loc_vals,MPI_DOUBLE,0,MPI_COMM_WORLD);
        delete[] sorted_vals;
        sorted_vals = nullptr;
    
        //creating row_offset_vector corresponding to each node.
        loc_row_offset = new int[loc_size + 1];
        MPI_Scatterv(row_offset,num_vec_vals,vec_vals_displ,MPI_INT,loc_row_offset,loc_size,MPI_INT,0,MPI_COMM_WORLD);
        delete[] row_offset;
        row_offset = nullptr;

        loc_row_offset[loc_size] = n_loc_vals;
        // ofstream out_file_timing;
        // out_file_timing.open("./mpi_timings.txt",std::ios_base::app);
        
        x_loc = new double[loc_size];
        MPI_Scatterv(x_glob,num_vec_vals,vec_vals_displ,MPI_DOUBLE,x_loc,loc_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
        delete[] x_glob;
        x_glob = nullptr;

        //each node has -> loc_no_rows = local_size, loc_row_offset, loc_col_ind, loc_mat_vals, loc_vec_vals
        /*
        Purpose of following function:
        For EACH node:
        1. based of off the column indices that each node has, we determine the indices of the vector values that would be sent to the node from other
        nodes.
        2. store the vector indices that are needed in the needed_vec_index array.
        3. store the count of the number of indices that are needed from each node in needed_vec_count.
        4. create displacement array needed_vec_displ from the above needed_vec_count array.
        */

        // ofstream out_file_timing;
        // out_file_timing.open("./results/mpi_timing_vals.txt",std::ios_base::app);
        // timer_start = MPI_Wtime();
        needed_vec_count = new int[total_nodes];
        needed_vec_displ = new int[total_nodes];

        int num_add_vectors_needed = investigator(loc_col_inds, my_node, total_nodes, no_rows, n_loc_vals, needed_vec_count, needed_vec_displ);
        /*
        loc_col_inds array needs to be modified to account for the shuffle in the x_loc that each node has. For example, if x_loc of node 1 is expected to be:
        [1,2,3,4] and it initially had [3,4], then received [1,2] from node 0, the x_loc now looks like [3,4,1,2]. So, the column indices array needs to be
        modified so it can multiply the correct values of the mat_val array with the x_loc values.
        Every value l   oc_col_inds[i] is the index of x_glob that is expected to be multiplied by the mat_vals array.
        To do the modification, we do the following steps:
        1. for every  loc_col_inds[i], if the value was already in the x_loc (i.e. not received from any other node), we reduce the value by (my_node * loc_size).
        This is because the values local to the node (original values in x_loc) are at the beginning of the x_loc array.
        2. For other values, we determine the node from which the value was received from, then accordingly incrementing the value of loc_col_inds[i] by adding 
        loc_size  (i.e, first move to end of original x_loc) + 
        vec_vals_displ[sending_node] (i.e. how many values have to be recived before receiving values from this node) +
        incremental counter (i.e. a temporary value that has to be incremented for every value that is received from this same sending node). 
        */

        int local_int_index = my_node * gen_size;
        int local_fin_index = (my_node + 1)*gen_size;
        int temp_num_vals_sent_by_node[total_nodes];
        for(int i=0;i<total_nodes;i++) temp_num_vals_sent_by_node[i] = 0;

        for(int i=0;i<n_loc_vals;i++)
        {    
            if( (loc_col_inds[i] < local_int_index ) || ( loc_col_inds[i] >= local_fin_index ) )
            {
                int sending_node = loc_col_inds[i]/gen_size;
                needed_vec_index[needed_vec_displ[sending_node] + temp_num_vals_sent_by_node[sending_node]] = loc_col_inds[i];
                loc_col_inds[i] = loc_size  + needed_vec_displ[sending_node] + temp_num_vals_sent_by_node[sending_node];
                temp_num_vals_sent_by_node[sending_node]++;
            }
            else
            {
                loc_col_inds[i] -= (my_node *gen_size);
            }
        }

        /*
        After determining number of indices that each node needs from every other node, we do a collective communication so that each node knows
        how many indices it needs to send to every other node. This information is stored in the to_send_vec_count array.
        */        
        to_send_vec_count = new int[total_nodes];
        MPI_Alltoall(needed_vec_count,1,MPI_INT,to_send_vec_count,1,MPI_INT,MPI_COMM_WORLD);
        
        /*
        create displacement array for the above to_send_vec_count array.
        */
        to_send_vec_displ = new int[total_nodes];
        to_send_vec_displ[0] = 0;
        for(int i=1;i<total_nodes;i++) to_send_vec_displ[i] = to_send_vec_displ[i-1] + to_send_vec_count[i-1];

        /*
        Calculate the total number of indices that each node needs to send. This will be used to create the array to_send_vals_index that will store all 
        the indices that all the nodes will send to every other node from the array: needed_vec_index.
        */
        total_vals_to_send = 0;
        for(int i=0;i<total_nodes;i++) total_vals_to_send += to_send_vec_count[i];

        to_send_vals_index = new int[total_vals_to_send];
        MPI_Alltoallv(needed_vec_index,needed_vec_count,needed_vec_displ,MPI_INT,to_send_vals_index,to_send_vec_count,to_send_vec_displ,MPI_INT,MPI_COMM_WORLD);
        
        delete[]needed_vec_index;
        needed_vec_index = nullptr;
        /*
        create an array send_vec_vals that contains the vector values in the indices received in the to_send_vec_index array. There will be discrepancy for the 
        nodes other than node 0. example: if node 1 has the values [x2 x3] in its x_loc and node 0 requests index 2 (i.e. x2) value, then node 1 has to send
        the value at its index 0.
        */

        send_vec_vals = new double[total_vals_to_send];
        for(int i=0;i<total_vals_to_send;i++) send_vec_vals[i] = x_loc[to_send_vals_index[i]];
        delete[] to_send_vals_index;
        to_send_vals_index = nullptr;

        /*
        create a receiver array for each node that will receive the values that are sent by the send_vec_vals array. The values will be received in the
        x_loc array which is modified in size using the num_add_vecs_needed values and the pointer moved to start receiving values from the correct
        index.
        */
        x_loc = (double*)realloc(x_loc,(loc_size +num_add_vectors_needed)*sizeof(double));
        double* recv_array_pointer = x_loc + loc_size;
    
        MPI_Alltoallv(send_vec_vals,to_send_vec_count,to_send_vec_displ,MPI_DOUBLE,recv_array_pointer,needed_vec_count,needed_vec_displ,MPI_DOUBLE,MPI_COMM_WORLD);   
        delete[] send_vec_vals,needed_vec_displ,needed_vec_count,to_send_vec_count,to_send_vec_displ;
        send_vec_vals = nullptr;
        needed_vec_displ = nullptr;
        needed_vec_count = nullptr;
        to_send_vec_count = nullptr;
        to_send_vec_displ = nullptr;

        // compute mat-vec product
        loc_result = new double[loc_size];
        for(int i=0;i<loc_size;i++)
        {
            loc_result[i] = 0;
            for(int j = loc_row_offset[i]; j <loc_row_offset[i+1]; j++)
            {
                loc_result[i] += loc_mat_vals[j] * x_loc[loc_col_inds[j]];
            }
        } 
        glob_result = new double[no_rows];

        MPI_Gatherv(loc_result,loc_size,MPI_DOUBLE,glob_result,num_vec_vals,vec_vals_displ,MPI_DOUBLE,0,MPI_COMM_WORLD);  
    
        delete[] x_loc,loc_result;
        x_loc = nullptr;
        loc_result = nullptr;
        delete[] loc_row_offset,loc_col_inds,loc_mat_vals;
        loc_row_offset = nullptr;
        loc_col_inds = nullptr;
        loc_mat_vals = nullptr;
    }
    else
    {
        gen_size = ceil((double)no_rows/total_nodes);
        loc_size = gen_size;
        if(my_node == total_nodes-1) loc_size = no_rows - (total_nodes-1)*gen_size; 

        MPI_Scatter(num_mat_col_values,1,MPI_INT,&n_loc_vals,1,MPI_INT,0,MPI_COMM_WORLD);

        loc_col_inds = new int[n_loc_vals];
        loc_mat_vals = new double[n_loc_vals];

        MPI_Scatterv(sorted_cols,num_mat_col_values,mat_col_displ,MPI_INT,loc_col_inds,n_loc_vals,MPI_INT,0,MPI_COMM_WORLD);        
        MPI_Scatterv(sorted_vals,num_mat_col_values,mat_col_displ,MPI_DOUBLE,loc_mat_vals,n_loc_vals,MPI_DOUBLE,0,MPI_COMM_WORLD);

        loc_row_offset = new int[loc_size + 1];
        MPI_Scatterv(row_offset,num_vec_vals,vec_vals_displ,MPI_INT,loc_row_offset,loc_size,MPI_INT,0,MPI_COMM_WORLD);
        
        int row_off_corrector = loc_row_offset[0];
        for(int i=0;i<loc_size;i++) loc_row_offset[i] -= row_off_corrector;
        loc_row_offset[loc_size] = n_loc_vals;

        x_loc = new double[loc_size];
        MPI_Scatterv(x_glob,num_vec_vals,vec_vals_displ,MPI_DOUBLE,x_loc,loc_size,MPI_DOUBLE,0,MPI_COMM_WORLD);

        needed_vec_index = new int[n_loc_vals];
        needed_vec_displ = new int[total_nodes];
        
        needed_vec_count = new int[total_nodes];
        int num_add_vectors_needed = investigator(loc_col_inds, my_node, total_nodes, no_rows, n_loc_vals, needed_vec_count, needed_vec_displ);
        
        int local_int_index = my_node * gen_size;
        int local_fin_index = (my_node + 1)*gen_size;
        int temp_num_vals_sent_by_node[total_nodes];
        for(int i=0;i<total_nodes;i++) temp_num_vals_sent_by_node[i] = 0;
    
        for(int i=0;i<n_loc_vals;i++)
        {    
            if( (loc_col_inds[i] < local_int_index ) || ( loc_col_inds[i] >= local_fin_index ) )
            {
                int sending_node = loc_col_inds[i]/gen_size;
                needed_vec_index[needed_vec_displ[sending_node] + temp_num_vals_sent_by_node[sending_node]] = loc_col_inds[i];
                loc_col_inds[i] = loc_size  + needed_vec_displ[sending_node] + temp_num_vals_sent_by_node[sending_node];
                temp_num_vals_sent_by_node[sending_node]++;
            }
            else
            {
                loc_col_inds[i] -= (my_node *gen_size);
            }
        }

        to_send_vec_count = new int[total_nodes];
        MPI_Alltoall(needed_vec_count,1,MPI_INT,to_send_vec_count,1,MPI_INT,MPI_COMM_WORLD);

        to_send_vec_displ = new int[total_nodes];        
        to_send_vec_displ[0] = 0;
        for(int i=1;i<total_nodes;i++) to_send_vec_displ[i] = to_send_vec_displ[i-1] + to_send_vec_count[i-1];

        total_vals_to_send = 0;
        for(int i=0;i<total_nodes;i++) total_vals_to_send += to_send_vec_count[i];

        to_send_vals_index = new int[total_vals_to_send];
        MPI_Alltoallv(needed_vec_index,needed_vec_count,needed_vec_displ,MPI_INT,to_send_vals_index,to_send_vec_count,to_send_vec_displ,MPI_INT,MPI_COMM_WORLD);
        
        delete[] needed_vec_index;
        needed_vec_index = nullptr;

        send_vec_vals = new double[total_vals_to_send];
        for(int i=0;i<total_vals_to_send;i++) send_vec_vals[i] = x_loc[to_send_vals_index[i] - my_node*gen_size];

        delete[] to_send_vals_index;
        to_send_vals_index = nullptr;

        x_loc = (double*)realloc(x_loc,(loc_size +num_add_vectors_needed)*sizeof(double));
        double* recv_array_pointer = x_loc + loc_size;
    
        MPI_Alltoallv(send_vec_vals,to_send_vec_count,to_send_vec_displ,MPI_DOUBLE,recv_array_pointer,needed_vec_count,needed_vec_displ,MPI_DOUBLE,MPI_COMM_WORLD);
        delete [] send_vec_vals,needed_vec_count,needed_vec_displ,to_send_vec_displ;
        send_vec_vals = nullptr;
        needed_vec_count = nullptr;
        needed_vec_displ = nullptr;
        to_send_vec_displ = nullptr;
    
        loc_result = new double[loc_size];
        for(int i=0;i<loc_size;i++)
        {
            loc_result[i] = 0;
            for(int j = loc_row_offset[i]; j <loc_row_offset[i+1]; j++)
            {
                loc_result[i] += loc_mat_vals[j] * x_loc[loc_col_inds[j]];
            }
        } 
        MPI_Gatherv(loc_result,loc_size,MPI_DOUBLE,glob_result,num_vec_vals,vec_vals_displ,MPI_DOUBLE,0,MPI_COMM_WORLD); 
        delete[] x_loc,loc_result;
        x_loc = nullptr;
        loc_result = nullptr;

        delete[] loc_row_offset,loc_col_inds,loc_mat_vals;
        loc_row_offset = nullptr;
        loc_col_inds = nullptr;
        loc_mat_vals = nullptr;
    }
   
    if(my_node == 0)
    {
        ofstream out_file;
        out_file.open("mpi_result_2.txt");
        for(int i=0;i<no_rows;i++)
        {
            out_file << glob_result[i] << endl;
        }
        out_file.close();
     
        delete[] glob_result;
        glob_result = nullptr;
    }   

    MPI_Finalize();

    return 0;
}
