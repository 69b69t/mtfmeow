#include <stdio.h>
#include <stdint.h>

#include "crunchLib.h"

int main() {
    uint64_t seed = 2551209;

    //large check
    int large = 0;

    Xoroshiro octASeed, octBSeed;
    initOctaveSeeds(&octASeed, &octBSeed, seed, large);

    //make a memory copy, this is necessary
    double crunch0a = calculateCrunchiness(octASeed, large, 0);
    double crunch0b = calculateCrunchiness(octBSeed, large, 0);

    printf("0a:%f 0b:%f", crunch0a, crunch0b);
    return 0;
}