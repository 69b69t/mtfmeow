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

__device__ int omission1Triangle0b(DoublePerlinNoise* dpn, Pos2d* inBuffer, int inCount,
    Pos2d* outBuffer, int maxSize)
{
    int count = 0;
    int octaveMax = 2;
    int mostMinimum = (-30000000 / (1<<19)) * (1<<19);
    
    for(int i = 0; i < inCount && count < maxSize; i++)
    {
        for(int x = (mostMinimum + inBuffer[i].xPos); x < 30000000; x += (1 << 19))
        {
            for(int z = (mostMinimum + inBuffer[i].zPos); z < 30000000; z += (1 << 19))
            {
                double sampleRight = sampleDoublePerlin(dpn, octaveMax,
                    (double)(x + 4096), (double)(z));
                double sampleTopLeft = sampleDoublePerlin(dpn, octaveMax,
                    (double)(x - 2048), (double)(z + 3574));
                double sampleBottomLeft = sampleDoublePerlin(dpn, octaveMax,
                    (double)(x - 2048), (double)(z - 3574));
                
                if(sampleRight < -0.4 || sampleTopLeft < -0.4 || sampleBottomLeft < -0.4)
                {
                    if(count < maxSize) {
                        outBuffer[count].xPos = x;
                        outBuffer[count].zPos = z;
                        count++;
                    }
                }
            }
        }
    }
    
    return count;
}

__device__ int omission0Tiling0a(DoublePerlinNoise* dpn, Pos2d* buffer, int maxSize)
{
    int count = 0;
    int octaveMax = 1;
    
    for(int x = 0; x < (1 << 19); x += 32768) {
        for(int z = 0; z < (1 << 19); z += 32768) {
            double sample = sampleDoublePerlin(dpn, octaveMax, (double)x, (double)z);
            
            if(sample < -0.275 && count < maxSize)
            {
                buffer[count].xPos = x;
                buffer[count].zPos = z;
                count++;
            }
        }
    }
    
    return count;
}

__device__ int biomeSamples(DoublePerlinNoise* dpn, int octaveMax,
    int width, int density, double threshold, int countThreshold,
    Pos2d* inBuffer, int inCount, Pos2d* outBuffer, int maxSize)
{
    int outCount = 0;
    int radius = width >> 1;
    
    for(int i = 0; i < inCount && outCount < maxSize; i++)
    {
        int count = 0;
        int64_t xSum = 0;
        int64_t zSum = 0;
        
        for(int x = -radius; x + density < radius; x += density)
        {
            for(int z = -radius; z + density < radius; z += density)
            {
                double sample = sampleDoublePerlin(dpn, octaveMax,
                    (double)(inBuffer[i].xPos + x), (double)(inBuffer[i].zPos + z));
                
                if(sample < threshold)
                {
                    xSum += (inBuffer[i].xPos + x);
                    zSum += (inBuffer[i].zPos + z);
                    count++;
                }
            }
        }
        
        if(count >= countThreshold)
        {
            outBuffer[outCount].xPos = (int)(xSum / count);
            outBuffer[outCount].zPos = (int)(zSum / count);
            outCount++;
        }
    }
    
    return outCount;
}

//==================MAIN_KERNEL==================

__global__ void processSeedsKernel(uint64_t startSeed, uint64_t numSeeds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    //Local buffers for this thread (allocated on heap)
    Pos2d* buffer0a = (Pos2d*)malloc(256 * sizeof(Pos2d));
    Pos2d* buffer0b = (Pos2d*)malloc(500000 * sizeof(Pos2d));
    Pos2d* bufferSamples0b0 = (Pos2d*)malloc(100000 * sizeof(Pos2d));
    Pos2d* bufferSamples0b1 = (Pos2d*)malloc(2000 * sizeof(Pos2d));
    Pos2d* bufferSamples0b2 = (Pos2d*)malloc(2000 * sizeof(Pos2d));
    Pos2d* bufferSamplesFull0 = (Pos2d*)malloc(2000 * sizeof(Pos2d));
    Pos2d* bufferSamplesFull1 = (Pos2d*)malloc(2000 * sizeof(Pos2d));

    DoublePerlinNoise dpn;
    PerlinNoise octaves[18];

    for(uint64_t i = idx; i < numSeeds; i += stride)
    {
        uint64_t seed = startSeed + i;
        int large = 1;
        
        // Quick crunchiness check
        Xoroshiro octASeed, octBSeed;
        initOctaveSeeds(&octASeed, &octBSeed, seed, large);
        if(doubleMad(octASeed, octBSeed, large, 1) > 0.1) continue;
        
        // Initialize noise structures

        init_climate_seed(&dpn, octaves, seed, large, -1);
        
        // Stage 0a: Initial tiling
        int count0a = omission0Tiling0a(&dpn, buffer0a, 256);
        if(count0a == 0) continue;
        
        // Stage 0b: Triangle check
        int count0b = omission1Triangle0b(&dpn, buffer0a, count0a, buffer0b, 500000);
        if(count0b == 0) continue;
        
        // BiomeSamples stage 1
        int countS0 = biomeSamples(&dpn, 2, 32768, 6553, -0.74, 1,
            buffer0b, count0b, bufferSamples0b0, 100000);
        if(countS0 == 0) continue;
        
        // BiomeSamples stage 2
        int countS1 = biomeSamples(&dpn, 2, 32768, 2978, -0.74, 9,
            bufferSamples0b0, countS0, bufferSamples0b1, 2000);
        if(countS1 == 0) continue;
        
        // BiomeSamples stage 3
        int countS2 = biomeSamples(&dpn, 2, 32768, 1424, -0.74, 38,
            bufferSamples0b1, countS1, bufferSamples0b2, 2000);
        if(countS2 == 0) continue;
        
        // BiomeSamples stage 4 (full octaves)
        int countF0 = biomeSamples(&dpn, 18, 32768, 2048, -1.05, 9,
            bufferSamples0b2, countS2, bufferSamplesFull0, 2000);
        if(countF0 == 0) continue;
        
        // BiomeSamples stage 5 (final)
        int countF1 = biomeSamples(&dpn, 18, 32768, 364, -1.05, 530,
            bufferSamplesFull0, countF0, bufferSamplesFull1, 2000);
        if(countF1 == 0) continue;
        
        printf("%ld %d %d\n", seed, bufferSamplesFull1[0].xPos, bufferSamplesFull1[0].zPos);
    }
    
    //free(buffer0a);
    //free(buffer0b);
}

//==================HOST_FUNCTIONS==================

int main(int argc, char** argv)
{
    const uint64_t TOTAL_SEEDS = 1000000ULL;
    const int BATCH_SIZE = 100000;
    
    // Launch configuration
    int blockSize = 1;
    int numBlocks = 2; // Enough to keep GPU busy
    
    printf("Processing %lu seeds with %d blocks of %d threads...\n", 
        TOTAL_SEEDS, numBlocks, blockSize);
    
    // Process in batches
    for(uint64_t batch = 0; batch < TOTAL_SEEDS; batch += BATCH_SIZE)
    {
        uint64_t batchSize = (batch + BATCH_SIZE > TOTAL_SEEDS) ? 
            (TOTAL_SEEDS - batch) : BATCH_SIZE;
        
        printf("\rBatch %lu-%lu... ", batch, batch + batchSize);
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
