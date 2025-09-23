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

    Xoroshiro pxr;
    xSetSeed(&pxr, seed);
    uint64_t xlo = xNextLong(&pxr);
    uint64_t xhi = xNextLong(&pxr);

    DoublePerlinNoise dpn;
    PerlinNoise octaves[18]; //this is all the noisemaps.
    //it must be 18 long as at most we have 18 perlin noisemaps


    init_climate_seed(&dpn, octaves, xlo, xhi, large, octave_max);

    
    double checksum = 0;
    for(int x = 0; x < 1024; x++) {
        for(int z = 0; z < 1024; z++) {
            checksum += sampleDoublePerlin(&dpn, (double)x, (double)z);
        }
    }
    printf("checksum: %lf (should be -180780.088673)\n", checksum);


    return 0;
}