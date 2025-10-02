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

__device__ void biomeSamples4(DoublePerlinNoise* dpn, uint64_t* seed, int octaveMax,
    int width, int density, double threshold, int countThreshold,
    Pos2d inPos)
{
    int radius = width >> 1;
    
    //inPos
    int count = 0;
    int64_t xSum = 0;
    int64_t zSum = 0;
    
    //scan a chunk of the world and count stuff
    for(int x = -radius; x + density < radius; x += density)
    {
        for(int z = -radius; z + density < radius; z += density)
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
    
    //printf("%d\n", count);

    //if seed passes...
    if(count >= countThreshold)
    {
        //get averaged points
        Pos2d temp;
        temp.xPos = (int)(xSum / count);
        temp.zPos = (int)(zSum / count);

        printf("===== %ld %d %d =====\n", *seed, temp.xPos, temp.zPos);
    }
}

__device__ void biomeSamples3(DoublePerlinNoise* dpn, uint64_t* seed, int octaveMax,
    int width, int density, double threshold, int countThreshold,
    Pos2d inPos)
{
    int radius = width >> 1;
    
    //inPos
    int count = 0;
    int64_t xSum = 0;
    int64_t zSum = 0;
    
    //scan a chunk of the world and count stuff
    for(int x = -radius; x + density < radius; x += density)
    {
        for(int z = -radius; z + density < radius; z += density)
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
    
    //if seed passes...
    if(count >= countThreshold)
    {
        //get averaged points
        Pos2d temp;
        temp.xPos = (int)(xSum / count);
        temp.zPos = (int)(zSum / count);
        printf("%ld %d %d\n", *seed, temp.xPos, temp.zPos);
        biomeSamples4(dpn, seed, 18, 32768, 364, -1.05, 530, temp);
    }
}

__device__ void biomeSamples2(DoublePerlinNoise* dpn, uint64_t* seed, int octaveMax,
    int width, int density, double threshold, int countThreshold,
    Pos2d inPos)
{
    int radius = width >> 1;
    
    //inPos
    int count = 0;
    int64_t xSum = 0;
    int64_t zSum = 0;
    
    //scan a chunk of the world and count stuff
    for(int x = -radius; x + density < radius; x += density)
    {
        for(int z = -radius; z + density < radius; z += density)
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
    
    //if seed passes...
    if(count >= countThreshold)
    {
        //get averaged points
        Pos2d temp;
        temp.xPos = (int)(xSum / count);
        temp.zPos = (int)(zSum / count);
        biomeSamples3(dpn, seed, 18, 32768, 2048, -1.05, 9, temp);
    }
}

__device__ void biomeSamples1(DoublePerlinNoise* dpn, uint64_t* seed, int octaveMax,
    int width, int density, double threshold, int countThreshold,
    Pos2d inPos)
{
    int radius = width >> 1;
    
    //inPos
    int count = 0;
    int64_t xSum = 0;
    int64_t zSum = 0;
    
    //scan a chunk of the world and count stuff
    for(int x = -radius; x + density < radius; x += density)
    {
        for(int z = -radius; z + density < radius; z += density)
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
    
    //if seed passes...
    if(count >= countThreshold)
    {
        //get averaged points
        Pos2d temp;
        temp.xPos = (int)(xSum / count);
        temp.zPos = (int)(zSum / count);
        biomeSamples2(dpn, seed, 2, 32768, 1424, -0.74, 38, temp);
    }
}

__device__ void biomeSamples0(DoublePerlinNoise* dpn, uint64_t* seed, int octaveMax,
    int width, int density, double threshold, int countThreshold,
    Pos2d inPos)
{
    int radius = width >> 1;
    
    //inPos
    int count = 0;
    int64_t xSum = 0;
    int64_t zSum = 0;
    
    //scan a chunk of the world and count stuff
    for(int x = -radius; x + density < radius; x += density)
    {
        for(int z = -radius; z + density < radius; z += density)
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
    
    //if seed passes...
    if(count >= countThreshold)
    {
        //get averaged points
        Pos2d temp;
        temp.xPos = (int)(xSum / count);
        temp.zPos = (int)(zSum / count);
        biomeSamples1(dpn, seed, 2, 32768, 2978, -0.74, 9, temp);
    }
}

__device__ void omission1Triangle0b(DoublePerlinNoise* dpn, uint64_t* seed, Pos2d* checkPos)
{
    int octaveMax = 2;
    int mostMinimum = (-30000000 / (1<<19)) * (1<<19);
    
    for(int x = (mostMinimum + checkPos->xPos); x < 30000000; x += (1 << 19))
    {
        for(int z = (mostMinimum + checkPos->zPos); z < 30000000; z += (1 << 19))
        {
            double sampleRight = sampleDoublePerlin(dpn, octaveMax,
                (double)(x + 4096), (double)(z));
            double sampleTopLeft = sampleDoublePerlin(dpn, octaveMax,
                (double)(x - 2048), (double)(z + 3574));
            double sampleBottomLeft = sampleDoublePerlin(dpn, octaveMax,
                (double)(x - 2048), (double)(z - 3574));
            
            if(sampleRight < -0.4 || sampleTopLeft < -0.4 || sampleBottomLeft < -0.4)
            {
                //next check
                Pos2d temp;
                temp.xPos = x;
                temp.zPos = z;
                biomeSamples0(dpn, seed, 2, 32768, 6553, -0.74, 1, temp);
            }
        }
    }
}

__device__ void omission0Tiling0a(DoublePerlinNoise* dpn, uint64_t* seed)
{
    int octaveMax = 1;
    
    for(int x = 0; x < (1 << 19); x += 32768) {
        for(int z = 0; z < (1 << 19); z += 32768) {
            double sample = sampleDoublePerlin(dpn, octaveMax, (double)x, (double)z);
            
            if(sample < -0.275)
            {
                Pos2d temp;
                temp.xPos = x;
                temp.zPos = z;
                //call next thing
                omission1Triangle0b(dpn, seed, &temp);
            }
        }
    }
}

//==================MAIN_KERNEL==================

__global__ void processSeedsKernel(uint64_t startSeed, uint64_t numSeeds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    DoublePerlinNoise dpn;
    PerlinNoise octaves[18];

    for(uint64_t i = idx; i < numSeeds; i += stride)
    {
        uint64_t seed = startSeed + i;
        int large = 1;
        
        //quick crunchiness check
        if(inefficientScore(seed, large, 1) > 0.01) continue;
        
        // Initialize noise structures
        init_climate_seed(&dpn, octaves, seed, large, -1);
        
        // Stage 0a: Initial tiling
        omission0Tiling0a(&dpn, &seed);
    }
}

//==================HOST_FUNCTIONS==================

int main(int argc, char** argv)
{
    const uint64_t TOTAL_SEEDS = 1000000ULL;
    const int BATCH_SIZE = 100000;
    
    // Launch configuration
    int blockSize = 256;
    int numBlocks = 256; // Enough to keep GPU busy
    
    printf("Processing %lu seeds with %d blocks of %d threads...\n", 
        TOTAL_SEEDS, numBlocks, blockSize);
    
    // Process in batches
    for(uint64_t batch = 0; batch < TOTAL_SEEDS; batch += BATCH_SIZE)
    {
        uint64_t batchSize = (batch + BATCH_SIZE > TOTAL_SEEDS) ? 
            (TOTAL_SEEDS - batch) : BATCH_SIZE;
        

        fflush(stdout);
        
        processSeedsKernel<<<blockSize, numBlocks>>>(batch, batchSize);

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return 0;
}
