#ifndef CONTLIB_H
#define CONTLIB_H

#include "xoroLib.h"

//typedefs
//a single raw noisemap
typedef struct
{
    uint8_t lookupHash[256+1]; //shuffled bytes
    uint8_t yInt; //y offset floor
    double xOffset, yOffset, zOffset; //offsets of the current noisemap. x y and z
    double amplitude; //amplitude of the noisemap
    double lacunarity; //frequency of the noisemap
    double yOffsetFract; //y offset fractional
    double yFractSmoothstep; //precomputed noisestep of something. maybe y offset?
} PerlinNoise;

//a bunch of stacked noisemaps
typedef struct
{
    int octcnt; //number of octaves
    PerlinNoise *octaves;
} OctaveNoise;

//two octave noisemaps that get stacked
typedef struct
{
    double amplitude;
    OctaveNoise octA;
    OctaveNoise octB;
} DoublePerlinNoise;

//init
void xPerlinInit(PerlinNoise *noise, Xoroshiro *xr);

int xOctaveInit(OctaveNoise *noise, Xoroshiro *xr, PerlinNoise *octaves, int minimumOctave, int nmax);

int xDoublePerlinInit(DoublePerlinNoise *noise, Xoroshiro *xr, PerlinNoise *octaves, int minimumOctave, int nmax);

void init_climate_seed(DoublePerlinNoise *dpn, PerlinNoise *octaves, uint64_t xlo, uint64_t xhi, int large, int nmax);
//end init

//sampling
double samplePerlin(const PerlinNoise *noise, double x, double z);

double sampleOctave(const OctaveNoise *noise, double x, double z);

double sampleDoublePerlin(const DoublePerlinNoise *noise, double x, double z);
//end sampling

#endif