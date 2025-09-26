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
//it takes a noisemap, max octave, position, and width
//it also takes density, threshold, and countThreshold.
int biomeSamples(DoublePerlinNoise* dpn, int octaveMax, Pos2d position,
    int size, int density, double threshold, int countThreshold)
{
    //sampling is done at 1:density scale
    //count points under threshold
    int radius = size >> 1;
    int count = 0;
    for(int x = -radius; x < radius; x += density)
    {
        for(int z = -radius; z < radius; z += density)
        {
            double sample = sampleDoublePerlin(dpn, octaveMax,
                (double)(position.xPos + x), (double)(position.zPos + z));
            if(sample < threshold) count++;
        }
    }

    //if we have enough points to pass threshold, return 1
    if(count >= countThreshold) return 1;
    else return 0;
}

int omission0b(DoublePerlinNoise* dpn, Pos2d* positions,
    int positionCount, Pos2d* buffer)
{
    //triangle check
    int octaveMax = 2;

    //this should be the first number before -30mil
    //that is divisable by 2^19
    int period = 1<<19;
    int mostMinimum = (-30000000 / period) * period;

    int bufferLength = 0;
    Pos2d temp;

    //search a slightly bigger area than the actual minecraft world
    for(int x = (mostMinimum + position.xPos); x < 30000000; x += (1 << 19))
    {
        for(int z = (mostMinimum + position.zPos); z < 30000000; z += (1 << 19))
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
            }
        }
    }

    //change to return something else
    return 0;
}

int omissionTiling0a(DoublePerlinNoise* dpn, Pos2d* buffer)
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
    Pos2d *buffer0a = (Pos2d*)malloc(1000000 * sizeof(Pos2d));
    int count0b = omissionTiling0a(&dpn, buffer0a);


    printf("%d\n", count);

    return 0;
}
