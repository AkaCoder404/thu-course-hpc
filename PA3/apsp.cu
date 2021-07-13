// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include <stdio.h>
#include <ctime>
#include <chrono>
#define BLOCK_SIZE 32
#define MAX 10000
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace {   
    // part 1 independent 
    __global__ void _independent(int block, int n, int *graph) {
        __shared__ int cache_graph[BLOCK_SIZE][BLOCK_SIZE];

        int j = threadIdx.x;
        int i = threadIdx.y;

        int v1 = BLOCK_SIZE * block + i; 
        int v2 = BLOCK_SIZE * block + j;

        int indexIJ = v1 * n + v2; 
        if(v1 < n && v2 < n) cache_graph[i][j] = graph[indexIJ];
        else cache_graph[i][j] = MAX;

        int new_path;        
        // make sure that all values are loaded in block
        __syncthreads();
        
        
        for (int u = 0; u < BLOCK_SIZE; u++) {
            new_path = cache_graph[i][u] + cache_graph[u][j];
            
            __syncthreads(); // sync before calculating new value
            cache_graph[i][j] = MIN(cache_graph[i][j], new_path);
            __syncthreads(); // sync to make sure that all values are current 
        }
    
        if( v1 < n && v2 < n) graph[indexIJ] = cache_graph[i][j];
    }

    // part 2 singly dependent
    __global__ void _singly_dependent(int block, int n, int *graph) {
        if (blockIdx.x == block && blockIdx.y == block) return; 

        __shared__ int cache_graph_base[BLOCK_SIZE][BLOCK_SIZE];

        int j = threadIdx.x;
        int i = threadIdx.y;

        int v1 = BLOCK_SIZE * block + i;
        int v2 = BLOCK_SIZE * block + j; 
        
        int indexIJ = v1 * n + v2;
        
        if (v1 < n && v2 < n)  cache_graph_base[i][j] = graph[indexIJ];
        else cache_graph_base[i][j] = MAX;


        if (blockIdx.y == 0) v2 = BLOCK_SIZE *  blockIdx.x + j; // load i-aligned singly dependent blocks
        else v1 = BLOCK_SIZE * blockIdx.x + i; // load j-aligned singly dependent blocks

        __shared__ int cache_graph[BLOCK_SIZE][BLOCK_SIZE];
        
        // load current block for graph
        int current_path;

        indexIJ = v1 * n + v2;
        if(v1 < n && v2 < n) current_path = graph[indexIJ];
        else current_path = MAX;

        cache_graph[i][j] = current_path;

        // sync to make sure all values are saved in cache
        __syncthreads();

        int new_path;
        // compute i-aligned singly dependent blocks
        if (blockIdx.y == 0) {
            
            for(int u = 0; u < BLOCK_SIZE; u++) {
                new_path = cache_graph_base[i][u] + cache_graph[u][j];
                current_path = MIN(current_path, new_path);
                
                __syncthreads();    // sync to make sure all threads compare new value with old
                cache_graph[i][j] = current_path;    // update  with new values
                __syncthreads();    // sync local threads updating local cache
            }

        } 
        // compute j-aligned singly dependent blocks
        else {   
            for (int u = 0; u < BLOCK_SIZE; u++) {
                new_path = cache_graph[i][u] + cache_graph_base[u][j];
                current_path = MIN(current_path, new_path);
               
                __syncthreads();   
                cache_graph[i][j] = current_path;  
                __syncthreads(); 
            }
        }
        if (v1 < n && v2 < n) graph[indexIJ] = current_path;
    }   
    // part 3 doubly dependent
    __global__ void _doubly_dependent(int block, int n, int *graph) {
        if(blockIdx.x == block || blockIdx.y == block) return; // not in same row/col

        int j = threadIdx.x;
        int i = threadIdx.y;

        int v1 = blockDim.y * blockIdx.y + i;
        int v2 = blockDim.x * blockIdx.x + j;

        __shared__ int cache_graph_baseRow[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int cache_graph_baseCol[BLOCK_SIZE][BLOCK_SIZE];

        int v1Row = BLOCK_SIZE * block + i;
        int v2Col = BLOCK_SIZE * block + j;

        // load data for block
        int indexIJ;
        if (v1Row < n && v2 < n) {
            indexIJ = v1Row * n + v2;
            cache_graph_baseRow[i][j] = graph[indexIJ];
        }
        else cache_graph_baseRow[i][j] = MAX;


        if (v1 < n && v2Col < n) {
            indexIJ = v1 * n + v2Col;
            cache_graph_baseCol[i][j] = graph[indexIJ];
        }
        else cache_graph_baseCol[i][j] = MAX;

        // sync to make sure all values are loaded in virtual block
        __syncthreads();

        int current_path, new_path;

        // compute data for one block
        if (v1 < n && v2 < n) {
            indexIJ = v1 * n + v2;
            current_path = graph[indexIJ];   
                    
            for (int u = 0; u < BLOCK_SIZE; u++) {
                new_path = cache_graph_baseCol[i][u] + cache_graph_baseRow[u][j];
                current_path = MIN(current_path, new_path);
            }
            graph[indexIJ] = current_path;
        }
    }
}

// void cuda_host(int n, int *graph) {
//     int bytes = n * n * sizeof(int);
//     int *graph_host = (int*)malloc(bytes);
//     copy device graph to host 
//     cudaMemcpy(graph_host, graph, bytes, cudaMemcpyDeviceToHost);
//     // cpu test
//     auto beg = std::chrono::high_resolution_clock::now();
//     for(int k = 0; k < n; k++) 
//         for(int i = 0; i < n; i++) 
//             for(int j = 0; j < n; j++) 
//                 graph_host[i * n + j] = MIN(graph_host[i * n + j], graph_host[i * n + k] + graph_host[k * n + j]);
//     auto end = std::chrono::high_resolution_clock::now();
//     double t = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count() * 1000;
//     printf("floyd on cpu = %f ms", t); // ms
//     for (int i = 0; i < n; i++) printf("%d ", graph_host[i]);
// }

// all pairs shortest path
void apsp(int n, /*device*/ int *graph) {  
    // graph already on device -> no need to move onto device
    
    dim3 grid_part_one(1, 1, 1);                                                    // first kernel
    dim3 grid_part_two((n - 1) / BLOCK_SIZE + 1, 2, 1);                             // second kernel
    dim3 grid_part_three((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1);       // third kernel
    dim3 blk(BLOCK_SIZE, BLOCK_SIZE, 1);

    // optimized blocking method
    for (int block = 0; block < (n - 1) / BLOCK_SIZE + 1; block++ ) {
        _independent<<<grid_part_one, blk>>>(block, n, graph);
        _singly_dependent<<<grid_part_two, blk>>>(block, n, graph);
        _doubly_dependent<<<grid_part_three, blk>>>(block, n, graph);
    } 

    // free device memory?
}
