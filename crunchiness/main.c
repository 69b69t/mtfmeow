#include <stdio.h>
#include <stdint.h>

#include "crunchLib.h"

int main()
{
    Xoroshiro octASeed, octBSeed;
    int large = 0;
    double mad;
    initOctaveSeeds(&octASeed, &octBSeed, -3192667955364718523ULL, large);
    printf("%f\n", doubleMad(octASeed, octBSeed, large, 2));

    return 0;

    for(uint64_t i = 0; i < 1000000000ULL; i++)
    {
        initOctaveSeeds(&octASeed, &octBSeed, i, large);
        mad = doubleMad(octASeed, octBSeed, large, 1);
        if (mad < 0.007) printf("%ld\n", i);
    }
    return 0;
}