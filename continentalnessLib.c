#include <stdint.h>
#include <math.h>

#include "continentalnessLib.h"

//copies the cubiomes generation of the continentalness.
//for mushroom biome searching

static inline double lerp(double part, double from, double to)
{
    return from + part * (to - from);
}

static inline double indexedLerp(uint8_t idx, double a, double b, double c)
{
    switch (idx & 0xf)
    {
        case 0:  return  a + b;
        case 1:  return -a + b;
        case 2:  return  a - b;
        case 3:  return -a - b;
        case 4:  return  a + c;
        case 5:  return -a + c;
        case 6:  return  a - c;
        case 7:  return -a - c;
        case 8:  return  b + c;
        case 9:  return -b + c;
        case 10: return  b - c;
        case 11: return -b - c;
        case 12: return  a + b;
        case 13: return -b + c;
        case 14: return -a + b;
        case 15: return -b - c;
    }
    return 0;
}

//===================START_INIT=================

//generate a randomly shuffled vector, and x y and z offsets
void xPerlinInit(PerlinNoise *noise, Xoroshiro *xr)
{
    int i = 0;
    noise->xOffset = xNextDouble(xr) * 256.0;

    //the y offset is a randomly generated number between 0 and 256
    //this controls the height of the noisemap, and also makes it 3d
    noise->yOffset = xNextDouble(xr) * 256.0;
    noise->zOffset = xNextDouble(xr) * 256.0;

    noise->amplitude = 1.0;
    noise->lacunarity = 1.0;

    //get the pointer to noise values so we can easily work with them
    uint8_t *lookupHash = noise->lookupHash;

    //init 256 identity permutation
    for (i = 0; i < 256; i++)
    {
        lookupHash[i] = i;
    }

    //shuffle

    for (i = 0; i < 256; i++)
    {
        int j = xNextInt(xr, 256 - i) + i;
        uint8_t n = lookupHash[i];
        lookupHash[i] = lookupHash[j];
        lookupHash[j] = n;
    }

    //protect against overflow, if we happen to be indexing with something
    //bigger than a u8
    lookupHash[256] = lookupHash[0];

    //precompute integer and fractional part of the random y noise thing
    double yInt = floor(noise->yOffset);
    double yOffsetFract = noise->yOffset - yInt;

    //save values
    noise->yInt = (int) yInt;
    noise->yOffsetFract = yOffsetFract;

    //smooth(er)step between 0 and 1.
    noise->yFractSmoothstep = yOffsetFract*yOffsetFract*yOffsetFract * (yOffsetFract * (yOffsetFract*6.0-15.0) + 10.0);
    //noise->yFractSmoothstep = yOffsetFract;
}

int xOctaveInit(OctaveNoise *noise, Xoroshiro *xr, PerlinNoise *octaves, int minimumOctave, int nmax)
{
    //this function initializes stacked octaves

    //bunch of constants that shall be xor'd with xrng state
    static const uint64_t md5_octave_n[][2] = {
        {0xb198de63a8012672, 0x7b84cad43ef7b5a8}, // md5 "octave_-12"
        {0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8}, // md5 "octave_-11"
        {0x36d326eed40efeb2, 0x5be9ce18223c636a}, // md5 "octave_-10"
        {0x082fe255f8be6631, 0x4e96119e22dedc81}, // md5 "octave_-9"
        {0x0ef68ec68504005e, 0x48b6bf93a2789640}, // md5 "octave_-8"
        {0xf11268128982754f, 0x257a1d670430b0aa}, // md5 "octave_-7"
        {0xe51c98ce7d1de664, 0x5f9478a733040c45}, // md5 "octave_-6"
        {0x6d7b49e7e429850a, 0x2e3063c622a24777}, // md5 "octave_-5"
        {0xbd90d5377ba1b762, 0xc07317d419a7548d}, // md5 "octave_-4"
        {0x53d39c6752dac858, 0xbcd1c5a80ab65b3e}, // md5 "octave_-3"
        {0xb4a24d7a84e7677b, 0x023ff9668e89b5c4}, // md5 "octave_-2"
        {0xdffa22b534c5f608, 0xb9b67517d3665ca9}, // md5 "octave_-1"
        {0xd50708086cef4d7c, 0x6e1651ecc7f43309}, // md5 "octave_0"
    };

    //scalings
    static const double lacuna_ini[] = { // -minimumOctave = 3..12
        1, .5, .25, 1./8, 1./16, 1./32, 1./64, 1./128, 1./256, 1./512, 1./1024,
        1./2048, 1./4096,
    };

    double lacuna = lacuna_ini[-minimumOctave];
    double persist = 256./511;

    uint64_t xlo = xNextLong(xr);
    uint64_t xhi = xNextLong(xr);
    int i;

    static const double amplitudes[] = {1, 1, 2, 2, 2, 1, 1, 1, 1};

    for (i = 0; i < 9 && i != nmax; i++)
    {
        Xoroshiro pxr;
        pxr.lo = xlo ^ md5_octave_n[12 + minimumOctave + i][0];
        pxr.hi = xhi ^ md5_octave_n[12 + minimumOctave + i][1];
        xPerlinInit(&octaves[i], &pxr);
        octaves[i].amplitude = amplitudes[i] * persist;
        octaves[i].lacunarity = lacuna;

        lacuna *= 2.0;
        persist *= 0.5;
    }

    noise->octaves = octaves;
    noise->octcnt = i;
    return i;
}

int xDoublePerlinInit(DoublePerlinNoise *noise, Xoroshiro *xr,
        PerlinNoise *octaves, int minimumOctave, int nmax)
{
    int n = 0, na = -1, nb = -1, len = 9;

    //this splits the octaves so na + nb = nmax BUT na gets more if its odd.
    //this is so we evenly spread the octaves
    if (nmax > 0)
    {
        na = (nmax + 1) >> 1;
        nb = nmax - na;
    }

    //init the first octave noise in double perlin noise. na is octave count
    //append octaves initalized to octaves
    n += xOctaveInit(&noise->octA, xr, octaves, minimumOctave, na);

    //second octave noise
    n += xOctaveInit(&noise->octB, xr, octaves+n, minimumOctave, nb);


    static const double amp_ini[] = { // (5 ./ 3) * len / (len + 1), len = 2..9
        0, 5./6, 10./9, 15./12, 20./15, 25./18, 30./21, 35./24, 40./27, 45./30,
    };
    noise->amplitude = amp_ini[len];
    return n;
}

void init_climate_seed(
    DoublePerlinNoise *dpn, PerlinNoise *octaves,
    uint64_t seed, int large, int nmax
    )
{
    Xoroshiro pxr;
    xSetSeed(&pxr, seed);
    uint64_t xlo = xNextLong(&pxr);
    uint64_t xhi = xNextLong(&pxr);


    // md5 "minecraft:continentalness" or "minecraft:continentalness_large"
    pxr.lo = xlo ^ (large ? 0x9a3f51a113fce8dc : 0x83886c9d0ae3a662);
    pxr.hi = xhi ^ (large ? 0xee2dbd157e5dcdad : 0xafa638a61b42e8ad);

    xDoublePerlinInit(dpn, &pxr, octaves, large ? -11 : -9, nmax);
}

//===========================END_INIT=====================

//===========================SAMPLING==============================

//sample single perlin noisemap. this takes a noisemap and x z coordinates
//within the range 0.0 .. 256.0 on the y level 0
double samplePerlin(const PerlinNoise *noise, double x, double z)
{
    uint8_t xInt, yInt, zInt;
    double t1, yFractSmoothstep, t3;
    double yOffsetFract = 0.0;

    yOffsetFract = noise->yOffsetFract;
    yInt = noise->yInt;
    yFractSmoothstep = noise->yFractSmoothstep;

    //shift the noisemap by xoffset and zoffset
    x += noise->xOffset;
    z += noise->zOffset;

    //get the integer cell we're in
    double xIntTemp = floor(x);
    double zIntTemp = floor(z);

    //restrict x and z to be only the fractional part. 0.0 - 1.0
    x -= xIntTemp;
    z -= zIntTemp;
    xInt = (int) xIntTemp;
    zInt = (int) zIntTemp;

    //smoothstep function. potentially will remove
    t1 = x*x*x * (x * (x*6.0-15.0) + 10.0);
    t3 = z*z*z * (z * (z*6.0-15.0) + 10.0);

    const uint8_t *lookupHash = noise->lookupHash;

    //get psuedorandom values from a lookup table, to "randomize"
    //the 8 corners of the unit cube we're in

    //it looks like this because its optimized
    uint8_t a1 = lookupHash[xInt]   + yInt;
    uint8_t b1 = lookupHash[xInt+1] + yInt;

    uint8_t a2 = lookupHash[a1]   + zInt;
    uint8_t b2 = lookupHash[b1]   + zInt;
    uint8_t a3 = lookupHash[a1+1] + zInt;
    uint8_t b3 = lookupHash[b1+1] + zInt;

    //computes the gradients, and calculate the dot products across all 8 vectors
    double l1 = indexedLerp(lookupHash[a2],   x,   yOffsetFract,   z);
    double l2 = indexedLerp(lookupHash[b2],   x-1, yOffsetFract,   z);
    double l3 = indexedLerp(lookupHash[a3],   x,   yOffsetFract-1, z);
    double l4 = indexedLerp(lookupHash[b3],   x-1, yOffsetFract-1, z);
    double l5 = indexedLerp(lookupHash[a2+1], x,   yOffsetFract,   z-1);
    double l6 = indexedLerp(lookupHash[b2+1], x-1, yOffsetFract,   z-1);
    double l7 = indexedLerp(lookupHash[a3+1], x,   yOffsetFract-1, z-1);
    double l8 = indexedLerp(lookupHash[b3+1], x-1, yOffsetFract-1, z-1);

    //linearlly interpolate down to 2d(on the x axis)
    l1 = lerp(t1, l1, l2);
    l3 = lerp(t1, l3, l4);
    l5 = lerp(t1, l5, l6);
    l7 = lerp(t1, l7, l8);

    //linearlly interpolate down to 1d(on the y axis)
    l1 = lerp(yFractSmoothstep, l1, l3);
    l5 = lerp(yFractSmoothstep, l5, l7);

    //linearlly interpolate down to 0d(on the z axis)
    return lerp(t3, l1, l5);
}

//sample stacked perlin noisemaps(one OctaveNoise)
double sampleOctave(const OctaveNoise *noise, int octaveMax, double x, double z)
{
    double v = 0;
    int i;
    for (i = 0; i < noise->octcnt && i < octaveMax; i++)
    {
        PerlinNoise *p = noise->octaves + i;
        double lf = p->lacunarity;
        double ax = x * lf;
        double az = z * lf;
        double pv = samplePerlin(p, ax, az);
        v += p->amplitude * pv;
    }
    return v;
}

//sample together two stacked octave noisemaps to get the final continentalness
//noisemap at a layer
double sampleDoublePerlin(const DoublePerlinNoise *noise,
        int octaveMax, double x, double z)
{
    //biomes are actually sampled at 1:4. look here if scales are being weird
    x /= 4;
    z /= 4;
    double v = 0;

    //DRY this DRY that whatever
    int na = (octaveMax + 1) >> 1;
    int nb = octaveMax - na;

    const double f = 337.0 / 331.0;
    v += sampleOctave(&noise->octA, na, x, z);
    v += sampleOctave(&noise->octB, nb, x*f, z*f);

    return v * noise->amplitude;
}

//END_SAMPLING