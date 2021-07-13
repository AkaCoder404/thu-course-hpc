#include "bfs_common.h"
#include "graph.h"
#include <cstddef>
#include <cstring>
#include <omp.h>
#include <cstdio>
#include <chrono>

#define ROOT_NODE_ID          0
#define NOT_VISITED_MARKER    -1
#define BU_NOT_VISITED_MARKER 0
#define NUM_THREADS           28

// copied from bfs_common.cpp
void vertex_set_clear_omp(vertex_set *list) {
  list->count = 0; 
} 

// copied from bfs_common.cpp
void vertex_set_init_omp(vertex_set *list, int count) {
  list->max_vertices = count;
  list->vertices = (int *)malloc(sizeof(int) *list->max_vertices);
  vertex_set_clear_omp(list);
}

// top down appraoch without parallelism (sequential)
void top_down_step_omp(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances) { 
  for (int i = 0; i < frontier->count; i++) {
    // select node on frontier to explore
    int node = frontier->vertices[i];
    // first outgoing edge (in terms of CSR)
    int start_edge = g->outgoing_starts[node];
    // last outgoing edge (in terms of CSR)
    int end_edge = (node == g->num_nodes - 1) ? g->num_edges                    // reached last node
                                              : g->outgoing_starts[node + 1];   // stop until next node's outgoing_edge first outgoing edge

    // attempt to add all neighbors to the new frontier
    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
      // nodes reaches by the outgoing edges
      int outgoing = g->outgoing_edges[neighbor];
      // update total distance to get to those nodes
      if (distances[outgoing] == NOT_VISITED_MARKER) {
        // distance is equal to distance to get to previous node + 1
        distances[outgoing] = distances[node] + 1;
        // populate the new_frontier with the nodes that are being visited
        int index = new_frontier->count++;
        new_frontier->vertices[index] = outgoing;
      }
    }
  }
}

// bottom up approach without parallelism (sequential)
void bottom_up_step_omp(graph* g, vertex_set* frontier, int* distances, int iteration) {
    
  for (int i = 0; i < g->num_nodes; i++) {
    if (frontier->vertices[i] == BU_NOT_VISITED_MARKER) {
      int start_edge = g->incoming_starts[i];
      int end_edge = (i == g->num_nodes - 1)  ? g->num_edges 
                                              : g->incoming_starts[i + 1];
      // iterate through each incoming edge                                      
      for(int neighbor = start_edge; neighbor < end_edge; neighbor++) {
        int incoming = g->incoming_edges[neighbor];
        if(frontier->vertices[incoming] == iteration) {
          distances[i] = distances[incoming] + 1;
          frontier->count++;
          frontier->vertices[i] = iteration + 1;
          break;
        }
      }
    }
  }
}

// top down appraoch with parallelism
void top_down_step_omp_parallel(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances) {
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
}

// bottom up approach with parallelism
void bottom_up_step_omp_parallel(Graph g, vertex_set *frontier, int* distances, int iteration) {
    int local_count = 0;
    // int padding[15];
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp for reduction(+:local_count)
        for (int i=0; i < g->num_nodes; i++) {                   
            if (frontier->vertices[i] == BU_NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes-1)? g->num_edges : g->incoming_starts[i + 1];
                for(int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int incoming = g->incoming_edges[neighbor];
                    // if(__sync_bool_compare_and_swap(&frontier->present[incoming], iteration, distances[node] + 1)) {
                    if(frontier->vertices[incoming] == iteration) {
                        distances[i] = distances[incoming] + 1;                        
                        local_count ++;
                        frontier->vertices[i] = iteration + 1;
                        break;
                    }
                }
            }
        }
        // #pragma omp atomic
        //     frontier->count += local_count;
    }    
    frontier->count = local_count;

}

// top down method
void bfs_omp_top_down(Graph graph, solution *sol) {

  vertex_set list1;
  vertex_set_init_omp(&list1, graph->num_nodes);    
  vertex_set list2;
  vertex_set_init_omp(&list2, graph->num_nodes);

  // list of nodes we know about, but have not visited
  // frontier contains the vertexes that have same distance from source
  vertex_set *frontier = &list1;
  // list of nodes that are neighboring the frontier
  // new frontier contains vertexes not explored yet but will be descovered
  vertex_set *new_frontier = &list2;

   // initialize all nodes as not visited
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
  }

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  // set the root distance with 0
  sol->distances[ROOT_NODE_ID] = 0;

  while (frontier->count != 0) {
    vertex_set_clear_omp(new_frontier);
    // top_down_step_omp(graph, frontier, new_frontier, sol->distances);
    top_down_step_omp_parallel(graph, frontier, new_frontier, sol->distances);

    // swap pointers
    vertex_set* tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}

// bottom up method 
void bfs_omp_bottom_up(Graph graph, solution *sol) {
  
  vertex_set list1;
  vertex_set_init_omp(&list1, graph->num_nodes);
  vertex_set* frontier = &list1;
  
  memset(frontier->vertices, 0, sizeof(int) * graph->num_nodes);

  frontier->vertices[frontier->count++] = 1; // ROOT_NODE_ID; // 1
  sol->distances[ROOT_NODE_ID] = 0;

  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = 0;

  int iteration = 1;
  while (frontier->count != 0) {
    frontier->count = 0;
    bottom_up_step_omp_parallel(graph, frontier, sol->distances, iteration);
    iteration++;
  }
}

// mixed method (incomplete)
// void bfs_omp_mixed(Graph graph, solution *sol) {

//   vertex_set list1;
//   vertex_set_init_omp(&list1, graph->num_nodes);
//   int iteration = 1;
//   vertex_set* frontier = &list1;    
//   memset(frontier->vertices, 0, sizeof(int) * graph->num_nodes);
//   frontier->vertices[frontier->count++] = 1;
//   sol->distances[ROOT_NODE_ID] = 0;

//   while (frontier->count != 0) {      
//   if(frontier->count >= 10000000) {
//         frontier->count = 0;
//         bottom_up_step_omp(graph, frontier, sol->distances, iteration);
//     }
//     else {
//         frontier->count = 0;
//         top_down_step_omp(graph, frontier, sol->distances, iteration);
//     }
//     iteration++;
//   }             
// }

void bfs_omp(Graph graph, solution *sol) {
  /** Your code ... */
  // int number_of_threads;
  // // default thread count
  // #pragma omp parallel
  // {
  //   #pragma omp master
  //   number_of_threads = omp_get_num_threads();
  // }
  // if (false) printf("thread_count = %d\n", number_of_threads);

  // printf("%d", NUM_THREADS);
  // method 1 BFS top down
  bfs_omp_top_down(graph, sol);

  // method 2 BFS bottom up
  // bfs_omp_bottom_up(graph, sol);

  // method 3 BFS mixed
  // bfs_omp_mixed(graph, sol);

  // (void)graph; // make compiler happy, delete it in your code!
  // (void)sol;   // make compiler happy, delete it in your code!

}

