#ifndef CRUNCHLIB_H
#define CRUNCHLIB_H

#include "xoroLib.h"

double calculateCrunchiness(Xoroshiro state, int large, int octaveNumber);

void initOctaveSeeds(Xoroshiro *octASeed, Xoroshiro *octBSeed, uint64_t seed, int large);

double doubleMad(Xoroshiro octaveSeedA, Xoroshiro octaveSeedB, int large, int octaveMax);

uint64_t getNextValid(uint64_t startSeed, double threshold, int large, int maxOctave);

double inefficientScore(uint64_t seed, int large, int octaveMax);

#endif