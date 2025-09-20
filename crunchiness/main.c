#include <stdio.h>
#include <stdint.h>

#include "crunchLib.h"

int main() {
    //uint64_t seed = 2551209;

    //large check
    int large = 0;

    Xoroshiro octASeed, octBSeed;
    
    for(uint64_t i = 0; i < 1000000000ULL; i++)
    {
        initOctaveSeeds(&octASeed, &octBSeed, i, large);

        //double doubleMad(Xoroshiro octaveSeedA, Xoroshiro octaveSeedB, int large) {
        double mad = doubleMad(octASeed, octBSeed, large);
        if (mad < 0.003) printf("%ld\n", i);
    }
    return 0;
}