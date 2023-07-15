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
#include<ctype.h>

using namespace std;

int investigator(int *local_col_inds, int my_node, int total_nodes, int no_rows, int n_loc_vals,int *needed_vec_index, int *needed_vec_count, int *needed_vec_displ)
{

    // int num_add_vectors_needed = investigator();
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

            //determine - 1. index needed, proc that has the index. store both in seperate arrays.
            needed_vec_index[temp_count] = local_col_inds[i];
            temp_count += 1;
            int node_to_get_from = local_col_inds[i]/gen_size;
            needed_vec_count[node_to_get_from] += 1;
            num_add_vectors_needed += 1;
            

            //label needed from where and incerment the function return value
        }
    }
    
    for(int i=1;i<total_nodes;i++)
    {
        needed_vec_displ[i] = needed_vec_displ[i-1] + needed_vec_count[i-1];
    }

    return num_add_vectors_needed;
}


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
    int my_node, total_nodes;
    MPI_Init(&argc, &argv);
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

    // int investigator(int *local_col_inds, int my_node, int total_nodes, int loc_size, int n_loc_vals,int *needed_vec_index, int *needed_vec_count, int *needed_vec_displ)
    int *needed_vec_index;
    int *needed_vec_count = new int[total_nodes];
    int *needed_vec_displ = new int[total_nodes];
    int num_add_vectors_needed;
    int *to_send_vec_count = new int[total_nodes];
    int *to_send_vec_displ = new int[total_nodes];
    int total_vals_to_send;
    int *to_send_vals_index;
    double *send_vec_vals;

    double *loc_result;
    double *glob_result; 


        
    if(my_node == 0)
    {
        string filename = "./Matrix_Files/1138_bus.mtx"; // any file in matrix market format, tested matrices taken from UF Sparse matrix collection.
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
    }

    // to be distributed: nrows, rowoffset_array, matrix_values_array, vector_array, col_ind_array.

    //constructing row offset array
    int *row_offset;

    if(my_node == 0)
    {
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
    }

    //constructing vector values
    if(my_node == 0)
    {
        x_glob = new double[no_cols];
        for(int i=0;i<no_cols;i++) x_glob[i] = i+1;
    }

    int *loc_row_offset;
    double *loc_mat_vals;
    double *loc_vec_vals;
    int *loc_col_inds;

    int gen_size,loc_size,loc_size_last;

    MPI_Bcast(&no_rows,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&no_cols,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&nnz,1,MPI_INT,0,MPI_COMM_WORLD);

    int n_loc_vals;
    int num_vec_vals[total_nodes];
    int vec_vals_displ[total_nodes];

    int num_mat_col_values[total_nodes];
    int mat_col_displ[total_nodes];

    if(my_node == 0)
    {
        gen_size = ceil((double)no_rows/total_nodes);
        loc_size = gen_size;
        if(my_node == total_nodes-1) loc_size = no_rows - (total_nodes-1)*loc_size; 

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
        MPI_Scatterv(sorted_vals,num_mat_col_values,mat_col_displ,MPI_DOUBLE,loc_mat_vals,n_loc_vals,MPI_DOUBLE,0,MPI_COMM_WORLD);

        //creating row_offset_vector corresponding to each node.
        loc_row_offset = new int[loc_size + 1];
        MPI_Scatterv(row_offset,num_vec_vals,vec_vals_displ,MPI_INT,loc_row_offset,loc_size,MPI_INT,0,MPI_COMM_WORLD);
        loc_row_offset[loc_size] = n_loc_vals;
       
        x_loc = new double[loc_size];
        MPI_Scatterv(x_glob,num_vec_vals,vec_vals_displ,MPI_DOUBLE,x_loc,loc_size,MPI_DOUBLE,0,MPI_COMM_WORLD);

        //each node has -> loc_no_rows = local_size, loc_row_offset, loc_col_ind, loc_mat_vals, loc_vec_vals
        needed_vec_index = new int[n_loc_vals];
        /*
        Purpose of following function:
        For EACH node:
        1. based of off the column indices that each node has, we determine the indices of the vector values that would be sent to the node from other
        nodes.
        2. store the vector indices that are needed in the needed_vec_index array.
        3. store the count of the number of indices that are needed from each node in needed_vec_count.
        4. create displacement array needed_vec_displ from the above needed_vec_count array.
        */
        num_add_vectors_needed = investigator(loc_col_inds, my_node, total_nodes, no_rows, n_loc_vals, needed_vec_index, needed_vec_count, needed_vec_displ);

        /*
        After determining number of indices that each node needs from every other node, we do a collective communication so that each node knows
        how many indices it needs to send to every other node. This information is stored in the to_send_vec_count array.
        */        
        MPI_Alltoall(needed_vec_count,1,MPI_INT,to_send_vec_count,1,MPI_INT,MPI_COMM_WORLD);
        
        /*
        create displacement array for the above to_send_vec_count array.
        */
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

        /*
        create an array send_vec_vals that contains the vector values in the indices received in the to_send_vec_index array. There will be discrepancy for the 
        nodes other than node 0. example: if node 1 has the values [x2 x3] in its x_loc and node 0 requests index 2 (i.e. x2) value, then node 1 has to send
        the value at its index 0.
        */
        send_vec_vals = new double[total_vals_to_send];
        for(int i=0;i<total_vals_to_send;i++) send_vec_vals[i] = x_loc[to_send_vals_index[i]];

        /*
        create a receiver array for each node that will receive the values that are sent by the send_vec_vals array. The values will be received in the
        x_loc array which is modified in size using the num_add_vecs_needed values and the pointer moved to start receiving values from the correct
        index.
        */
        x_loc = (double*)realloc(x_loc,(loc_size +num_add_vectors_needed)*sizeof(double));
        double* recv_array_pointer = x_loc + loc_size;
    
        MPI_Alltoallv(send_vec_vals,to_send_vec_count,to_send_vec_displ,MPI_DOUBLE,recv_array_pointer,needed_vec_count,needed_vec_displ,MPI_DOUBLE,MPI_COMM_WORLD);

        /*
        loc_col_inds array needs to be modified to account for the shuffle in the x_loc that each node has. For example, if x_loc of node 1 is expected to be:
        [1,2,3,4] and it initially had [3,4], then received [1,2] from node 0, the x_loc now looks like [3,4,1,2]. So, the column indices array needs to be
        modified so it can multiply the correct values of the mat_val array with the x_loc values.
        Every value loc_col_inds[i] is the index of x_glob that is expected to be multiplied by the mat_vals array.
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
                loc_col_inds[i] = loc_size  + needed_vec_displ[sending_node] + temp_num_vals_sent_by_node[sending_node];
                temp_num_vals_sent_by_node[sending_node]++;
            }
            else
            {
                loc_col_inds[i] -= (my_node *loc_size);
            }
        }

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
        num_add_vectors_needed = investigator(loc_col_inds, my_node, total_nodes, no_rows, n_loc_vals, needed_vec_index, needed_vec_count, needed_vec_displ);

        MPI_Alltoall(needed_vec_count,1,MPI_INT,to_send_vec_count,1,MPI_INT,MPI_COMM_WORLD);
                
        to_send_vec_displ[0] = 0;
        for(int i=1;i<total_nodes;i++) to_send_vec_displ[i] = to_send_vec_displ[i-1] + to_send_vec_count[i-1];

        total_vals_to_send = 0;
        for(int i=0;i<total_nodes;i++) total_vals_to_send += to_send_vec_count[i];

        to_send_vals_index = new int[total_vals_to_send];
        MPI_Alltoallv(needed_vec_index,needed_vec_count,needed_vec_displ,MPI_INT,to_send_vals_index,to_send_vec_count,to_send_vec_displ,MPI_INT,MPI_COMM_WORLD);
        
        send_vec_vals = new double[total_vals_to_send];
        for(int i=0;i<total_vals_to_send;i++) send_vec_vals[i] = x_loc[to_send_vals_index[i] - my_node*gen_size];

        x_loc = (double*)realloc(x_loc,(loc_size +num_add_vectors_needed)*sizeof(double));
        double* recv_array_pointer = x_loc + loc_size;
    
        MPI_Alltoallv(send_vec_vals,to_send_vec_count,to_send_vec_displ,MPI_DOUBLE,recv_array_pointer,needed_vec_count,needed_vec_displ,MPI_DOUBLE,MPI_COMM_WORLD);

        int local_int_index = my_node * gen_size;
        int local_fin_index = (my_node + 1)*gen_size;
        int temp_num_vals_sent_by_node[total_nodes];
        for(int i=0;i<total_nodes;i++) temp_num_vals_sent_by_node[i] = 0;

        for(int i=0;i<n_loc_vals;i++)
        {   
            if( (loc_col_inds[i] < local_int_index ) || ( loc_col_inds[i] >= local_fin_index ) )
            {
                int sending_node = loc_col_inds[i]/gen_size;
                loc_col_inds[i] = loc_size + needed_vec_displ[sending_node] + temp_num_vals_sent_by_node[sending_node];
                temp_num_vals_sent_by_node[sending_node]++;
            }
            else
            {
                loc_col_inds[i] -= (my_node *loc_size);
            }
        }

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

    }
    if(my_node == 0)
    {
        delete[] sorted_rows,sorted_cols,sorted_vals, needed_vec_count, needed_vec_displ, needed_vec_index, to_send_vec_count, to_send_vec_displ, to_send_vals_index;
        delete[] send_vec_vals, loc_result, row_offset, loc_row_offset, loc_mat_vals, loc_vec_vals, loc_col_inds;

        sorted_vals = nullptr;
        sorted_cols = nullptr;
        sorted_vals = nullptr;
        needed_vec_count = nullptr;
        needed_vec_displ = nullptr;
        needed_vec_index = nullptr;
        to_send_vec_count =  nullptr;
        to_send_vec_displ = nullptr;
        to_send_vals_index = nullptr;
        send_vec_vals = nullptr;
        loc_result = nullptr;
        row_offset = nullptr;
        loc_row_offset = nullptr;
        loc_mat_vals = nullptr;
        loc_vec_vals = nullptr;
        loc_col_inds = nullptr;

        // ofstream out_file;
        // out_file.open("./results/mpi_result.txt");
        for(int i=0;i<no_rows;i++)
        {
            cout << glob_result[i] << endl;
        }
        // out_file.close();
     
        delete[] glob_result;
        glob_result = nullptr;
    }   
    MPI_Finalize();

    return 0;
}