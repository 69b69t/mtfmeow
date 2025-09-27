#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "continentalnessLib.h"

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
    Pos2d* positions, int positionCount, //input things
    Pos2d* buffer) //output buffer
{
    int radius = width >> 1;
    int bufferLength = 0;
    Pos2d temp;
    for(int i = 0; i < positionCount; i++)
    {
        //sampling is done at 1:density scale
        //count points under threshold
        int count = 0;

        //averaging stuff to put the center point into buffer
        int xSum = 0;
        int zSum = 0;
        for(int x = -radius; x+density < radius; x += density)
        {
            for(int z = -radius; z+density < radius; z += density)
            {
                //sample
                double sample = sampleDoublePerlin(dpn, octaveMax,
                    (double)(positions[i].xPos + x), (double)(positions[i].zPos + z));
                if(sample < threshold)
                {
                    //increase the count of samples at this position that passed the check
                    //then add the x and z components to a running sum for averaging
                    xSum += positions[i].xPos;
                    zSum += positions[i].zPos;
                    count++;
                    
                }
            }
        }

        //after the check do stuff with the buffer to store it
        if(count >= countThreshold)
        {
            temp.xPos = xSum / count;
            temp.zPos = zSum / count;
            buffer[bufferLength] = temp;
            bufferLength++;
        }
    }
    return bufferLength;
}

int omission1Triangle0b(DoublePerlinNoise* dpn, Pos2d* positions,
    int positionCount, Pos2d* buffer)
{
    //triangle check
    int octaveMax = 2;

    //this should be the first number before -30mil
    //that is divisable by 2^19
    int mostMinimum = (-30000000 / (1<<19)) * (1<<19);
    int bufferLength = 0;
    Pos2d temp;

    //im sorry never-nesters
    for(int i = 0; i < positionCount; i++)
    {
        //search a slightly bigger area than the actual minecraft world
        for(int x = (mostMinimum + positions[i].xPos); x < 30000000; x += (1 << 19))
        {
            for(int z = (mostMinimum + positions[i].zPos); z < 30000000; z += (1 << 19))
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
                    buffer[bufferLength] = temp;
                    bufferLength++;
                }
            }
        }
    }
    //change to return something else
    return bufferLength;
}

int omission0Tiling0a(DoublePerlinNoise* dpn, Pos2d* buffer)
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
                buffer[bufferLength] = temp;
                bufferLength++;
            }
        }
    }

    return bufferLength;
}

int main(int argc, char** argv)
{
    //this "simply" checks a see to see if it has a big shroom
    int large = 1;
    DoublePerlinNoise dpn;
    PerlinNoise octaves[18]; //this is all the noisemaps.
    //it must be 18 long as at most we have 18 perlin noisemaps


    init_climate_seed(&dpn, octaves, 694201337, large, -1);

    //buffers
    Pos2d *buffer0a = (Pos2d*)malloc(1000000 * sizeof(Pos2d));
    Pos2d *buffer0b = (Pos2d*)malloc(1000000 * sizeof(Pos2d));
    Pos2d *bufferSamples0b0 = (Pos2d*)malloc(1000000 * sizeof(Pos2d));
    Pos2d *bufferSamples0b1 = (Pos2d*)malloc(1000000 * sizeof(Pos2d));
    Pos2d *bufferSamples0b2 = (Pos2d*)malloc(1000000 * sizeof(Pos2d));
    Pos2d *bufferSamplesFull0 = (Pos2d*)malloc(1000000 * sizeof(Pos2d));
    Pos2d *bufferSamplesFull1 = (Pos2d*)malloc(1000000 * sizeof(Pos2d));

    //calculating
    int count0a = omission0Tiling0a(&dpn, buffer0a);
    int count0b = omission1Triangle0b(&dpn, buffer0a, count0a, buffer0b);
    int countSamples0b0 = biomeSamples(&dpn, 2, 32768, 6553, -0.74, 1, buffer0b, count0b, bufferSamples0b0);
    int countSamples0b1 = biomeSamples(&dpn, 2, 32768, 2978, -0.74, 9, bufferSamples0b0, countSamples0b0, bufferSamples0b1);
    int countSamples0b2 = biomeSamples(&dpn, 2, 32768, 1424, -0.74, 38, bufferSamples0b1, countSamples0b1, bufferSamples0b2);
    int countSamplesFull0 = biomeSamples(&dpn, 18, 32768, 2048, -1.05, 9, bufferSamples0b2, countSamples0b2, bufferSamplesFull0);
    int countSamplesFull1 = biomeSamples(&dpn, 18, 32768, 364, -1.05, 530, bufferSamplesFull0, countSamplesFull0, bufferSamplesFull1);

    printf("(%d -> %d) -> (%d -> %d -> %d) -> (%d -> %d)\n", count0a, count0b, countSamples0b0,
        countSamples0b1, countSamples0b2, countSamplesFull0, countSamplesFull1);
    printf("%d %d\n", bufferSamples0b2[0].xPos, bufferSamples0b2[0].zPos);

    return 0;
}
