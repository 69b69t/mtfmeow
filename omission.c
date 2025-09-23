#include <stdio.h>
#include <stdint.h>

#include "xoroLib.h"
#include "continentalnessLib.h"

//this file will have filters in it which correspond to COMISSION 5.3

int main(int argc, char** argv)
{
    uint64_t seed = 2551209;
    int large = 0;
    int octave_max = -1;

    DoublePerlinNoise dpn;
    PerlinNoise octaves[18]; //this is all the noisemaps.
    //it must be 18 long as at most we have 18 perlin noisemaps

    init_climate_seed(&dpn, octaves, seed, large, octave_max);
    printf("%f\n", sampleDoublePerlin(&dpn, 0, 0));
    printf("%f\n", sampleDoublePerlin(&dpn, 1024/4, 0));

    return 0;

    for(int x = 0; x < (1 << 19); x += 25000) {
        for(int z = 0; z < (1 << 19); z += 25000) {
            double test = sampleDoublePerlin(&dpn, (double)x/4, (double)z/4);
            if(test < -0.3) printf("%d -- %d %d\n", (int)(10000*test), x, z);
        }
    }


    return 0;
}