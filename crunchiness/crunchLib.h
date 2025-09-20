#ifndef CRUNCHLIB_H
#define CRUNCHLIB_H

typedef struct
{
    uint64_t lo, hi;
} Xoroshiro;

double calculateCrunchiness(Xoroshiro state, int large, int octaveNumber);

void initOctaveSeeds(Xoroshiro *octASeed, Xoroshiro *octBSeed, uint64_t seed, int large);

double doubleMad(Xoroshiro octaveSeedA, Xoroshiro octaveSeedB, int large);

#endif