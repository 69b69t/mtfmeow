#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>

#include "continentalnessLib.h"
#include "crunchLib.h"

#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

//this file will have filters in it which correspond to COMISSION 5.3
//the filters will be "semi-recursive", which is not how i would wish to do it, but we
//dont exactly have vectors to use
//thats really my only gripe with c. apart from the stack smashing thing

typedef struct
{
    int xPos;
    int zPos;
} Pos2d;

//helper function. this returns a boolean.
//it takes a noisemap, max octave, center, and width
//it also takes density, threshold, and countThreshold.
int biomeSamples(DoublePerlinNoise* dpn, int octaveMax, //noise based things
    int width, int density, double threshold, int countThreshold, //sampling things
    Pos2d** inBuffer, //input things
    Pos2d** outBuffer) //output buffer
{
    int radius = width >> 1;
    int bufferLength = 0;
    Pos2d temp;
    for(int i = 0; i < arrlen(*inBuffer); i++)
    {
        //sampling is done at 1:density scale
        //count points under threshold
        int count = 0;

        //averaging stuff to put the center point into outBuffer
        int64_t xSum = 0;
        int64_t zSum = 0;
        for(int x = -radius; x+density < radius; x += density)
        {
            for(int z = -radius; z+density < radius; z += density)
            {
                //sample
                double sample = sampleDoublePerlin(dpn, octaveMax,
                    (double)((*inBuffer)[i].xPos + x), (double)((*inBuffer)[i].zPos + z));
                if(sample < threshold)
                {
                    //increase the count of samples at this position that passed the check
                    //then add the x and z components to a running sum for averaging
                    //printf("%d\n", (inBuffer[i].zPos + z));
                    xSum += ((*inBuffer)[i].xPos + x);
                    zSum += ((*inBuffer)[i].zPos + z);
                    count++;
                }
            }
        }

        //after the check do stuff with the outBuffer to store it
        if(count >= countThreshold)
        {
            
            temp.xPos = (int)(xSum / count);
            temp.zPos = (int)(zSum / count);
            //printf("%ld/%d = %d and %ld/%d = %d\n", xSum, count, temp.xPos, zSum, count, temp.zPos);
            //printf("average %d %d\n", positions[i].xPos, positions[i].zPos);

            arrpush(*outBuffer, temp);
        }
    }
    return bufferLength;
}

int omission1Triangle0b(DoublePerlinNoise* dpn, Pos2d** inBuffer, Pos2d** outBuffer)
{
    //triangle check
    int octaveMax = 2;

    //this should be the first number before -30mil
    //that is divisable by 2^19
    int mostMinimum = (-30000000 / (1<<19)) * (1<<19);
    int bufferLength = 0;
    Pos2d temp;

    //im sorry never-nesters
    for(int i = 0; i < arrlen(*inBuffer); i++)
    {
        //search a slightly bigger area than the actual minecraft world
        for(int x = (mostMinimum + (*inBuffer)[i].xPos); x < 30000000; x += (1 << 19))
        {
            for(int z = (mostMinimum + (*inBuffer)[i].zPos); z < 30000000; z += (1 << 19))
            {
                //sampling a triangle
                double sampleRight = sampleDoublePerlin(dpn, octaveMax,
                    (double)(x + 4096), (double)(z));
                double sampleTopLeft = sampleDoublePerlin(dpn, octaveMax,
                    (double)(x - 2048), (double)(z + 3574));
                double sampleBottomLeft = sampleDoublePerlin(dpn, octaveMax,
                    (double)(x - 2048), (double)(z - 3574));

                //if one of the above is below threshold, print it
                if(sampleRight < -0.4 ||
                    sampleTopLeft < -0.4 ||
                    sampleBottomLeft < -0.4)
                {
                    temp.xPos = x;
                    temp.zPos = z;
                    arrpush(*outBuffer, temp);
                }
            }
        }
    }
    //change to return something else
    return bufferLength;
}

int omission0Tiling0a(DoublePerlinNoise* dpn, Pos2d** outBuffer)
{
    Pos2d temp;
    int octaveMax = 1;
    //this will segfault if we ever have more than a million points
    //in that case just increase the limit
    int bufferLength = 0;

    for(int x = 0; x < (1 << 19); x += 32768) {
        for(int z = 0; z < (1 << 19); z += 32768) {
            double sample = sampleDoublePerlin(dpn, octaveMax, (double)x, (double)z);

            if(sample < -0.275)
            {
                temp.xPos = x;
                temp.zPos = z;
                arrpush(*outBuffer, temp);
            }
        }
    }

    return bufferLength;
}

int contiguousCheck(DoublePerlinNoise* dpn, Pos2d samplePos)
{
    //this will stay on CPU as it will be unruly to run on a GPU, and gets called quite rarely

    //temporary struct
    struct
    {
        Pos2d key; //position
        int value; //is this position a shroom?
    }* posHashMap = NULL;

    /*
        CONTIG CHECK ALGO
        test webhook... again!
    */
}

void* spawnThread(void* arg)
{
    //clean up the threading immensely.
    //call threads and actually communicate with main thread
    struct
    {
        int threadId;
        int threadCount;
    } *args = arg;

    //this "simply" checks a see to see if it has a big shroom
    int large = 1;
    DoublePerlinNoise dpn;
    PerlinNoise octaves[18]; //this is all the noisemaps.
    //it must be 18 long as at most we have 18 perlin noisemaps

    //buffers
    Pos2d *buffer0a = NULL;
    Pos2d *buffer0b = NULL;
    Pos2d *bufferSamples0b0 = NULL;
    Pos2d *bufferSamples0b1 = NULL;
    Pos2d *bufferSamples0b2 = NULL;
    Pos2d *bufferSamplesFull0 = NULL;
    Pos2d *bufferSamplesFull1 = NULL;

    for(uint64_t i = args->threadId; i < 1000000ULL; i += args->threadCount)
    {
        if(inefficientScore(i, large, 1) > 0.1) continue;
        //climate init
        init_climate_seed(&dpn, octaves, i, large, -1);

        arrsetlen(buffer0a, 0);
        arrsetlen(buffer0b, 0);
        arrsetlen(bufferSamples0b0, 0);
        arrsetlen(bufferSamples0b1, 0);
        arrsetlen(bufferSamples0b2, 0);
        arrsetlen(bufferSamplesFull0, 0);
        arrsetlen(bufferSamplesFull1, 0);

        //calculating
        //double refs are passed because the array might be reallocated.
        //the address of the buffers are constant. the value is not.
        //the place it points is a pointer to another array
        omission0Tiling0a(&dpn, &buffer0a);
        omission1Triangle0b(&dpn, &buffer0a, &buffer0b);
        biomeSamples(&dpn, 2, 32768, 6553, -0.74, 1, &buffer0b, &bufferSamples0b0);
        biomeSamples(&dpn, 2, 32768, 2978, -0.74, 9, &bufferSamples0b0, &bufferSamples0b1);
        biomeSamples(&dpn, 2, 32768, 1424, -0.74, 38, &bufferSamples0b1, &bufferSamples0b2);
        biomeSamples(&dpn, 18, 32768, 2048, -1.05, 9, &bufferSamples0b2, &bufferSamplesFull0);
        biomeSamples(&dpn, 18, 32768, 364, -1.05, 530, &bufferSamplesFull0, &bufferSamplesFull1);
        //printf("bufferSamples0b2 is %d long\n", countSamples0b2);

        if(arrlen(bufferSamplesFull1) > 0)
        {
            printf("%ld %d %d\n", i, bufferSamplesFull1[0].xPos, bufferSamplesFull1[0].zPos);
            fflush(stdout);
        }
    }

    return NULL;
}

int main(int argc, char** argv)
{
    const int NUM_THREADS = 24;
    pthread_t threads[NUM_THREADS];

    //create threadArgs
    struct
    {
        int threadId;
        int threadCount;
    } threadArgs[NUM_THREADS];

    //define threadArgs and create threads on the fly
    for(int i = 0; i < NUM_THREADS; i++)
    {
        threadArgs[i].threadId = i;
        threadArgs[i].threadCount = NUM_THREADS;

        pthread_create(&threads[i], NULL, spawnThread, &threadArgs[i]);
    }

    //wait for threads to finish processing
    for(int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
}