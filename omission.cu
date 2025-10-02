#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>

#include "continentalnessLib.cuh"
#include "crunchLib.cuh"

typedef struct
{
    int xPos;
    int zPos;
} Pos2d;

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//==================DEVICE_FUNCTIONS==================

// Consolidated biome sampling function to reduce code duplication
__device__ bool biomeSamplesGeneric(DoublePerlinNoise* dpn, int octaveMax,
    int width, int density, double threshold, int countThreshold,
    Pos2d inPos, Pos2d* outPos)
{
    const int radius = width >> 1;
    int count = 0;
    int64_t xSum = 0;
    int64_t zSum = 0;
    
    // Pre-compute loop bounds
    const int xStart = -radius;
    const int xEnd = radius - density;
    const int zStart = -radius;
    const int zEnd = radius - density;
    
    // Scan world chunk
    for(int x = xStart; x < xEnd; x += density)
    {
        for(int z = zStart; z < zEnd; z += density)
        {
            double sample = sampleDoublePerlin(dpn, octaveMax,
                (double)(inPos.xPos + x), (double)(inPos.zPos + z));
            
            if(sample < threshold)
            {
                xSum += (inPos.xPos + x);
                zSum += (inPos.zPos + z);
                count++;
            }
        }
    }
    
    // Early exit if threshold not met
    if(count < countThreshold)
        return false;
    
    // Compute averaged position
    outPos->xPos = (int)(xSum / count);
    outPos->zPos = (int)(zSum / count);
    return true;
}

__device__ void biomeSamples4(DoublePerlinNoise* dpn, uint64_t* seed, Pos2d inPos)
{
    Pos2d temp;
    if(biomeSamplesGeneric(dpn, 18, 32768, 364, -1.05, 530, inPos, &temp))
    {
        printf("===== %ld %d %d =====\n", *seed, temp.xPos, temp.zPos);
    }
}

__device__ void biomeSamples3(DoublePerlinNoise* dpn, uint64_t* seed, Pos2d inPos)
{
    Pos2d temp;
    if(biomeSamplesGeneric(dpn, 18, 32768, 2048, -1.05, 9, inPos, &temp))
    {
        printf("%ld %d %d\n", *seed, temp.xPos, temp.zPos);
        biomeSamples4(dpn, seed, temp);
    }
}

__device__ void biomeSamples2(DoublePerlinNoise* dpn, uint64_t* seed, Pos2d inPos)
{
    Pos2d temp;
    if(biomeSamplesGeneric(dpn, 18, 32768, 1424, -0.74, 38, inPos, &temp))
    {
        biomeSamples3(dpn, seed, temp);
    }
}

__device__ void biomeSamples1(DoublePerlinNoise* dpn, uint64_t* seed, Pos2d inPos)
{
    Pos2d temp;
    if(biomeSamplesGeneric(dpn, 2, 32768, 2978, -0.74, 9, inPos, &temp))
    {
        biomeSamples2(dpn, seed, temp);
    }
}

__device__ void biomeSamples0(DoublePerlinNoise* dpn, uint64_t* seed, Pos2d inPos)
{
    Pos2d temp;
    if(biomeSamplesGeneric(dpn, 2, 32768, 6553, -0.74, 1, inPos, &temp))
    {
        biomeSamples1(dpn, seed, temp);
    }
}

__device__ void omission1Triangle0b(DoublePerlinNoise* dpn, uint64_t* seed, Pos2d* checkPos)
{
    const int octaveMax = 2;
    const int mostMinimum = (-30000000 / (1<<19)) * (1<<19);
    const int shift = (1 << 19);
    const double threshold = -0.4;
    
    for(int x = (mostMinimum + checkPos->xPos); x < 30000000; x += shift)
    {
        for(int z = (mostMinimum + checkPos->zPos); z < 30000000; z += shift)
        {
            double sampleRight = sampleDoublePerlin(dpn, octaveMax,
                (double)(x + 4096), (double)(z));
            
            // Early exit optimization - check most likely condition first
            if(sampleRight < threshold)
            {
                Pos2d temp = {x, z};
                biomeSamples0(dpn, seed, temp);
                continue;
            }
            
            double sampleTopLeft = sampleDoublePerlin(dpn, octaveMax,
                (double)(x - 2048), (double)(z + 3574));
            
            if(sampleTopLeft < threshold)
            {
                Pos2d temp = {x, z};
                biomeSamples0(dpn, seed, temp);
                continue;
            }
            
            double sampleBottomLeft = sampleDoublePerlin(dpn, octaveMax,
                (double)(x - 2048), (double)(z - 3574));
            
            if(sampleBottomLeft < threshold)
            {
                Pos2d temp = {x, z};
                biomeSamples0(dpn, seed, temp);
            }
        }
    }
}

__device__ void omission0Tiling0a(DoublePerlinNoise* dpn, uint64_t* seed)
{
    const int octaveMax = 1;
    const int step = 32768;
    const int limit = (1 << 19);
    const double threshold = -0.275;
    
    for(int x = 0; x < limit; x += step) {
        for(int z = 0; z < limit; z += step) {
            double sample = sampleDoublePerlin(dpn, octaveMax, (double)x, (double)z);
            
            if(sample < threshold)
            {
                Pos2d temp = {x, z};
                omission1Triangle0b(dpn, seed, &temp);
            }
        }
    }
}

//==================MAIN_KERNEL==================

__global__ void processSeedsKernel(uint64_t startSeed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread gets its own noise structures
    DoublePerlinNoise dpn;
    PerlinNoise octaves[18];

    for(uint64_t i = idx; ; i += stride)
    {
        uint64_t seed = startSeed + i;
        
        // Quick crunchiness check - early rejection
        if(inefficientScore(seed, 1, 1) > 0.01) 
            continue;
        
        // Initialize noise structures
        init_climate_seed(&dpn, octaves, seed, 1, -1);
        
        // Stage 0a: Initial tiling
        omission0Tiling0a(&dpn, &seed);
    }
}

//==================HOST_FUNCTIONS==================

int main(int argc, char** argv)
{
    // Optimized launch configuration
    int blockSize = 256;
    int numBlocks;
    
    // Get device properties for optimal configuration
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // Calculate optimal number of blocks based on SM count
    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                                   processSeedsKernel, 0, 0));
    numBlocks = minGridSize;
    
    printf("Processing seeds with %d blocks of %d threads...\n", numBlocks, blockSize);
    printf("GPU: %s with %d SMs\n", prop.name, prop.multiProcessorCount);
    
    processSeedsKernel<<<numBlocks, blockSize>>>(0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return 0;
}