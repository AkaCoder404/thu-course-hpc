#include "bfs_common.h"
#include "graph.h"
#include <cstddef>
#include <cstring>
#include <omp.h>
#include <cstdio>
#include <chrono>
#include <mpi.h>

#define ROOT_NODE_ID          0
#define NOT_VISITED_MARKER    -1
#define BU_NOT_VISITED_MARKER 0
#define NUM_THREADS           28
#define THRESHOLD            100
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (B))

void vertex_set_clear_omp_mpi(vertex_set *list) {
   list->count = 0; 
} 

void vertex_set_init_omp_mpi(vertex_set *list, int count) {
  list->max_vertices = count;
  list->vertices = (int *)malloc(sizeof(int) *list->max_vertices);
  vertex_set_clear_omp_mpi(list);
}

void bfs_top_down_step_omp_mpi(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances) {
  int local_count; 
  #pragma omp parallel num_threads(NUM_THREADS) private(local_count)      
  {
    local_count = 0;
    int thread_id = omp_get_thread_num();
    int number_of_threads = omp_get_num_threads();
    int* local_frontier = (int*)malloc(sizeof(int) * (g->num_nodes/NUM_THREADS));
  
    for (int i= thread_id; i < frontier->count; i += number_of_threads) {                
      int node = frontier->vertices[i];

      int start_edge = g->outgoing_starts[node];
      int end_edge = (node == g->num_nodes-1) ? g->num_edges : g->outgoing_starts[node+1];

      // attempt to add all neighbors to the new frontier
      for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
        int outgoing = g->outgoing_edges[neighbor];

        if( __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)) {                    
          local_frontier[local_count] = outgoing;
          local_count++;
        }
      }
    }
    // ensure local frontier is to be synchronized with the global new_frontier
    #pragma omp critical                    
    {
        memcpy(new_frontier->vertices + new_frontier->count, local_frontier, local_count*sizeof(int));
        new_frontier->count += local_count;
    }
  }
  // for (int i = 0; i < frontier->count; i++) {
  //   // select node on frontier to explore
  //   int node = frontier->vertices[i];
  //   // first outgoing edge (in terms of CSR)
  //   int start_edge = g->outgoing_starts[node];
  //   // last outgoing edge (in terms of CSR)
  //   int end_edge = (node == g->num_nodes - 1) ? g->num_edges                    // reached last node
  //                                             : g->outgoing_starts[node + 1];   // stop until next node's outgoing_edge first outgoing edge

  //   // attempt to add all neighbors to the new frontier
  //   for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
  //     // nodes reaches by the outgoing edges
  //     int outgoing = g->outgoing_edges[neighbor];
  //     // update total distance to get to those nodes
  //     if (distances[outgoing] == NOT_VISITED_MARKER) {
  //       // distance is equal to distance to get to previous node + 1
  //       distances[outgoing] = distances[node] + 1;
  //       // populate the new_frontier with the nodes that are being visited
  //       int index = new_frontier->count++;
  //       new_frontier->vertices[index] = outgoing;
  //     }
  //   }
  // }
}

void bfs_bottom_up_step_opm_mpi(Graph g, vertex_set *frontier, vertex_set *thread_frontier, int *distances, int nthreads, int rank, int nodes_per_proc, int iteration) {

  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < nthreads; i++)
    thread_frontier[i].count = 0;

  // thread checks in-process nodes in parallel
  #pragma omp parallel for schedule(guided)
  for (int i = rank * nodes_per_proc; i < MIN((rank + 1) * nodes_per_proc, g->num_nodes - 1); i++) {
    if (distances[i] == NOT_VISITED_MARKER) { 
      int thread_num = omp_get_thread_num(); 
      int start_edge = g->incoming_starts[i];
      int end_edge = (i == g->num_nodes-1)  ? g->num_edges 
                                            : g->incoming_starts[i + 1];
      // int end_edge = g->incoming_starts[i + 1];
      // searching the frontier 
      for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
        // if the neighbor connected to this node is an expanded node of the previous layer, 
        // then this node node can be expanded at this time
        if (distances[g->incoming_edges[neighbor]] == iteration) {
          thread_frontier[thread_num].vertices[thread_frontier[thread_num].count++] = i;
          distances[i] = iteration + 1;
          break;
        }
      }
    }
  }

  frontier->count = 0;

  // combine the extended sequence of all threads in the process
  for (int i = 0; i < nthreads; i++)
    for (int j = 0; j < thread_frontier[i].count; j++)
      frontier->vertices[frontier->count++] = thread_frontier[i].vertices[j];
}

void bfs_omp_mpi(Graph graph, solution *sol) {
  /** Your code ... */

  // standard mpi syntax
  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // number of nodes per proc
  int nodes_per_proc = (graph->num_nodes + nprocs - 1) / nprocs; 

  // initialize frontier and new frontier
  vertex_set list1;
  vertex_set list2;
  vertex_set_init_omp_mpi(&list1, graph->num_nodes);
  vertex_set_init_omp_mpi(&list2, graph->num_nodes);
  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
  }

  // set up frontier with the root node, set up root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  // number of threads : 28
  int nthreads = 0;
  #pragma omp parallel
  {
    #pragma omp master
    nthreads = omp_get_num_threads();
  }

  int* dl = new int[nprocs]; 
  int* dl_acc = new int[nprocs]; 

  // threaded frontier space
  vertex_set *thread_frontier = new vertex_set[nthreads];
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < nthreads; i++) {
    vertex_set_init_omp_mpi(thread_frontier + i, nodes_per_proc);
  }
  
  int iteration = 0; 
  // top-down and up-down depending on frontier size
  while(frontier->count != 0) {
    while (frontier->count != 0) {
      vertex_set_clear_omp_mpi(new_frontier);
      bfs_top_down_step_omp_mpi(graph, frontier, new_frontier, sol->distances);

      iteration++; // necessary for bfs buttom up approach

      vertex_set *tmp = frontier;
      frontier = new_frontier;
      new_frontier = tmp;

      // determine whether to switch to bfs buttom up
      if (frontier->count != 0 && graph->num_nodes / new_frontier->count <= THRESHOLD) break;
    }

    while (frontier->count != 0) {
      bfs_bottom_up_step_opm_mpi(graph, new_frontier, thread_frontier, sol->distances, nthreads, rank, nodes_per_proc, iteration++);

      // sum of all processes and count the length of the current step expansion sequence
      MPI_Allreduce(&new_frontier->count, &frontier->count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      // current sequence to be expanded is more than the threshold, the distance will be updated and transmitted directly
      // and the sequence to be expanded is no longer transmitted
      if (frontier->count != 0 && graph->num_nodes / frontier->count <= THRESHOLD / 5) {
        MPI_Allgather(&sol->distances[rank * nodes_per_proc], nodes_per_proc, MPI_INT, sol->distances, nodes_per_proc, MPI_INT, MPI_COMM_WORLD); // 合并各进程更新的距离信息
      }
      else  {
        // according to number of nodes to be expanded, merge
        MPI_Allgather(&new_frontier->count, 1, MPI_INT, dl, 1, MPI_INT, MPI_COMM_WORLD);
        dl_acc[0] = 0;
        for (int i = 1; i < nprocs; i++) {
          dl_acc[i] = dl_acc[i - 1] + dl[i - 1];
        }

        // merge all process' frontier 
        MPI_Allgatherv(new_frontier->vertices, dl[rank], MPI_INT, frontier->vertices, dl, dl_acc, MPI_INT, MPI_COMM_WORLD); 
        if (rank > 0)
          for (int i = 0; i < dl_acc[rank]; i++) {
            sol->distances[frontier->vertices[i]] = iteration;
          }
        if (rank < nprocs - 1)
        for (int i = dl_acc[rank + 1]; i < frontier->count; i++) {
          sol->distances[frontier->vertices[i]] = iteration;
        }
        if (frontier->count != 0 && graph->num_nodes / frontier->count > THRESHOLD) break;
      }
    }

  }

  // (void)graph; // make compiler happy, delete it in your code!
  // (void)sol;   // make compiler happy, delete it in your code!
  
  delete[] thread_frontier;
}





