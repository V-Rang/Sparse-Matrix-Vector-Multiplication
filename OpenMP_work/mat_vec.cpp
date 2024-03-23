/*
Matrix-vector multiplication of a square matrix using MPI, p = total_nodes. (p-1) nodes are used to perform the multiplication.
Hence, n must be divisible by (p-1). 
Steps:
1. P0 reads in order of matrix(n). Then bcasts it all the other procs.
2. P0 reads in n elements of the vector x_glob. This is distributed as n/(p-1) elements to all the procs in their x_loc.
3. P0 reads in block of n/(p-1) rows of length n each - in a_loc_glob. This is done (p-1) times to read in entire matrix.
4. P0 distributes a_loc_glob as blocks of n/(p-1) columns to each of the other (p-1) procs in their a_loc.
5. Every proc performs it's local mat-vec: a_loc X x_loc and stores results in y_loc.
6. All the y_locs for a given a_loc_glob are reduced to the y_glob array of length n.
7. Steps 3 - 6 are repeated (p-1) for processing of n/(p-1) block of rows.
*/

#include<iostream>
#include<mpi.h>
#include<string.h>
#include<omp.h>

using namespace std;

/*function to create a derived data type - mpi_custom_block.
It allows distributing- both sending from proc 0 and receiving into the other procs
of n/p X n/p blocks of the n/p X n block of rows read in by proc 0.
This is done by reading n/p length elements from each of the n/p rows and sending it to the jth proc.(1 -> total_nodes -1).

Key inference: Since a_loc_glob is a pointer to a pointer, the first element of the (i+1)th row is not at the next memory address
to the the last element of the ith row. This can be seen in the incorrect working of the function mat_dist used below.

*/
void custom_block(int n, int p, int **a_loc_glob, MPI_Datatype *mpi_custom_block, int j)
{

    // j -> to send to jth proc

    int blocklengths[n / (p - 1)];
    MPI_Aint displacements[n / (p - 1)];
    MPI_Datatype typelists[n / (p - 1)];

    for (int i = 0; i < n / (p - 1); i++)
    {
        blocklengths[i] = n / (p - 1);
        typelists[i] = MPI_INT;
    }

    displacements[0] = 0;
    MPI_Aint address, start_address;

    MPI_Get_address(&(a_loc_glob[0][(j - 1) * (n / (p - 1))]), &start_address);

    for (int i=1; i < n/(p-1); i++)
    {
        MPI_Get_address(&(a_loc_glob[i][(j - 1) * (n / (p - 1))]), &address);
        displacements[i] = address - start_address;
    }

    MPI_Type_create_struct(n / (p - 1), blocklengths, displacements, typelists, mpi_custom_block);
    MPI_Type_commit(mpi_custom_block);
}

void x_dist(int *x_glob, int *&x_loc,int my_node, int n, int total_nodes,MPI_Status &status)
{
   //*VVI: need to have int *&x_loc and not int *x_loc. Else values in x_loc will not be reflected outside function when using x_loc = new int[n/(p-1)]
    int p = total_nodes;
    if(my_node == 0)
    {
        for(int j=1;j<p;j++)
        {
            int tag = stoi(std::to_string(j) + "0");
            MPI_Send(&x_glob[(j-1)*(n/(p-1))],n/(p-1),MPI_INT,j,tag,MPI_COMM_WORLD);
        }
    }
    else
    {
        x_loc = new int[n/(p-1)];
        int tag = stoi(std::to_string(my_node) + "0");
        MPI_Recv(x_loc,n/(p-1),MPI_INT,0,tag,MPI_COMM_WORLD,&status);

    }
}

/*
This function is not used and is incorrect in its goal. See description of function custom_block given above.
*/
void mat_dist(int **a_loc_glob, int **&a_loc, int n, int my_node, int total_nodes, int number_iter, MPI_Status &status)
{

    int p = total_nodes;
    
    MPI_Datatype mpi_block_glob;
    MPI_Type_vector(n / (p - 1), n / (p - 1), n, MPI_INT, &mpi_block_glob);
    MPI_Type_commit(&mpi_block_glob);

    MPI_Datatype mpi_block_loc;
    MPI_Type_vector(n / (p - 1), n / (p - 1), n / (p - 1), MPI_INT, &mpi_block_loc);
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
        a_loc = new int *[n / (p - 1)];
        for (int v = 0; v < n / (p - 1); v++)
        {
            a_loc[v] = new int[n / (p - 1)];
        }

        int tag = stoi(std::to_string(my_node) + "0" + std::to_string(number_iter));
        MPI_Recv(&(a_loc[0][0]), (n / (p - 1)) * (n / (p - 1)), MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        
    }
}

/*
function to perform mat-vec multiplication of a_loc and x_loc, result is stored in y_loc.
Notice usage of &  in int *&y_loc, this is done so results in y_loc initialzied inside the function as
new int[n/(p-1)] are reflected outside the function. 
*/
void mat_vec(int **a_loc, int *x_loc, int *&y_loc, int n, int total_nodes,int my_node, int number_iter)
{
    int p = total_nodes;
    y_loc = new int[n/(p-1)];
    
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
    #pragma omp parallel for private(i) shared(j,y_loc,n,p,a_loc,x_loc) default(none)
    for(i=0;i<n/(p-1);i++)
    {
        y_loc[i] = 0;
        for(j=0;j<n/(p-1);j++)
        {
            y_loc[i] += a_loc[i][j] * x_loc[j];
        }
    } 



}

int main(int argc, char **argv)
{
    int my_node,total_nodes; //my_node - proc ID, total_nodes - total number of processes
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    int n; //order of square matrix.
    int p = total_nodes;
    int *x_glob; //global x of length n read in by proc 0, distributed to all the procs in x_loc (length = n/(p-1)).
    int *x_loc;
    int *y_glob; //global y of length n of proc 0 to which all the procs return their local mat-vec result to.
    int *y_loc;

    //proc 0 reads in the number of rows of the square matrix and bcasts it to all the other procs.
    if(my_node == 0)
    {
        printf("Enter number of rows of square matrix\n");
        fflush(stdout);
        scanf("%d",&n);
    }

    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

    //read in vector and distribute it - needs to be done only once.
    if(my_node == 0)
    {
        y_glob = new int[n];
        x_glob = new int[n];
        printf("Enter %d values of the vector\n",n);
        fflush(stdout);
        for(int i=0;i<n;i++)
        {  
            scanf("%d",&x_glob[i]);
        }
        //call a function to distribute n/p values to each proc
    }

   
    x_dist(x_glob,x_loc,my_node,n,total_nodes,status); //function to distribute the x_glob read in by process 0 to all the procs

  

    // //read in matrix and distribute it 
    int **a_loc_glob; // reading in blocks of n/p rows at once, and distribute to other procs, have to do this (p-1) times to read in the entire matrix
    int **a_loc; // n/p X n/p block local to each proc
    
    MPI_Datatype mpi_block; // the derived datatype to distribute the n/p X n a_loc_glob for every iteration to the n/p X n/p a_loc of every proc.

    for(int i=0;i<p-1;i++)
    {
        if(my_node == 0) // read in values
        {
            //For MPI_Reduce - we finally reduce results of y_loc of every iteration( i.e. every block of n/p rows) to y_glob of proc 0. MPI_Reduce requires y_loc of EVERY proc to be initialized. Hence 0s for proc 0.
            y_loc = new int[n/(p-1)];
            for(int v = 0;v<n/(p-1);v++)
            {
                y_loc[v] = 0;
            }

            a_loc_glob = new int*[n/(p-1)];
            
            for(int v = 0;v<n/(p-1);v++)
            {
                a_loc_glob[v] = new int[n];
            } 

            for(int v =0;v<(n/(p-1));v++)
            {
                printf("Enter %d values for row %d\n",n,v+ i*(n/(p-1)));
                fflush(stdout); 
                for(int w=0;w<n;w++)
                {
                    scanf("%d",&a_loc_glob[v][w]);
                }
            }//a_loc_glob constructed - contains n/(p-1) X n elements of the n/(p-1) block of rows. 
            for (int j = 1; j < p; j++) //sending blocks of n/(p-1) columns to every other proc.
            {
                int tag = stoi(std::to_string(j) + "0");
                custom_block(n,p,a_loc_glob,&mpi_block,j); //custom datatype to send n/(p-1) elements of each of the n/(p-1) rows.
                MPI_Send(&(a_loc_glob[0][(j-1)*(n/(p-1))]),1,mpi_block,j,tag,MPI_COMM_WORLD);

            }
        }
        else
        {
            a_loc = new int*[n /(p-1)];
            for(int v= 0;v<n/(p-1);v++)
            {
                a_loc[v] = new int[n/(p-1)];
            }
            int tag = stoi(std::to_string(my_node) + "0");
            custom_block(n,p,a_loc,&mpi_block,my_node);
            MPI_Recv(&(a_loc[0][0]),1,mpi_block,0,tag,MPI_COMM_WORLD,&status); //every proc has n/p X n/p block in a_loc;

            mat_vec(a_loc,x_loc,y_loc,n,p,my_node,i); //mat-vec for every proc - a_loc X x_loc, result stored in y_loc; done for every block of n/p rows read in by proc 0, So y_loc: n/p X 1 vector. (i)
        }

        MPI_Reduce(&(y_loc[0]),&(y_glob[i*(n/(p-1))]),n/(p-1),MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);// y_glob contains the reduced values of y_loc for every proc; done for every block of n/p rows read in by proc 0. (i)
    }


    if(my_node == 0) //final answer of mat-vec of nXn square matrix with vector of length n.
    {
        for(int i=0;i<n;i++)
        {
            cout << y_glob[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize();

}