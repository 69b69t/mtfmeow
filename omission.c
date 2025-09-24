#include <stdio.h>
#include <stdint.h>

#include "continentalnessLib.h"

//this file will have filters in it which correspond to COMISSION 5.3
//the filters will be "semi-recursive", which is not how i would wish to do it, but we
//dont exactly have vectors to use
//thats really my only gripe with c. apart from the stack smashing thing

int omission0b(uint64_t seed, DoublePerlinNoise* dpn, int xOffset, int zOffset)
{
    //this "simply" checks a see to see if it has a big shroom
    int octave_max = 2;
    PerlinNoise octaves[18]; //this is all the noisemaps.
    //it must be 18 long as at most we have 18 perlin noisemaps

    //this function checks over the whole world at points designated by omissionTiling0a
    
    //this should be the first number before -30mil that is divisable by 2^19
    int mostMinimum = (-30000000 >> 19) << 19;
    
    //search a slightly bigger area than the actual minecraft world
    for(int x = (mostMinimum + xOffset); x < 30000000; x += (1 << 19))
    {
        for(int z = (mostMinimum + zOffset); z < 30000000; z += (1 << 19))
        {
            //showme7 says to only sample the triange, on an or statement
            double sampleRight = sampleDoublePerlin(dpn, octave_max, (double)(x + 4096), (double)(z));
            double sampleTopLeft = sampleDoublePerlin(dpn, octave_max, (double)(x - 2048), (double)(z + 3574));
            double sampleBottomLeft = sampleDoublePerlin(dpn, octave_max, (double)(x - 2048), (double)(z - 3574));

            //if one of the above is below threshold, print it
            if(sampleRight < -0.7 && sampleTopLeft < -0.7 && sampleBottomLeft < -0.7)
                printf("%ld %d %d\n", seed, x, z);
        }
    }
}

int omissionTiling0a(uint64_t seed)
{
    //this "simply" checks a see to see if it has a big shroom
    int large = 1;
    int octave_max = 1;
    DoublePerlinNoise dpn;
    PerlinNoise octaves[18]; //this is all the noisemaps.
    //it must be 18 long as at most we have 18 perlin noisemaps

    init_climate_seed(&dpn, octaves, seed, large, -1);

    //check a single perlin tile. it would perfectly tile IF SHOWME
    //actually gave the green light on it...
    for(int x = 0; x < (1 << 19); x += 32768) {
        for(int z = 0; z < (1 << 19); z += 32768) {
            double sample = sampleDoublePerlin(&dpn, octave_max, (double)x, (double)z);

            if(sample < -0.275) omission0b(seed, &dpn, x, z);
        }
    }

    //automatically false for now
    return 0;
}

int main(int argc, char** argv)
{
    for(uint64_t i = 0; ; i++)
    {
        omissionTiling0a(i);
    }
    
    return 0;
}