#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lodepng.h"

//copies the cubiomes generation of the continentalness.
// for mushroom biome searching

#define STRUCT(S) typedef struct S S; struct S

enum
{
    NP_TEMPERATURE      = 0,
    NP_HUMIDITY         = 1,
    NP_CONTINENTALNESS  = 2,
    NP_EROSION          = 3,
    NP_SHIFT            = 4, 
    NP_WEIRDNESS        = 5,
    NP_MAX
};

//a single raw noisemap
STRUCT(PerlinNoise)
{
    uint8_t lookupHash[256+1]; //shuffled bytes
    uint8_t yInt; //y offset floor
    double xOffset, yOffset, zOffset; //offsets of the current noisemap. x y and z
    double amplitude; //amplitude of the noisemap
    double lacunarity; //frequency of the noisemap
    double yOffsetFract; //y offset fractional
    double yFractSmoothstep; //precomputed noisestep of something. maybe y offset?
};

//a bunch of stacked noisemaps
STRUCT(OctaveNoise)
{
    int octcnt; //number of octaves
    PerlinNoise *octaves;
};

//two octave noisemaps that get stacked
STRUCT(DoublePerlinNoise)
{
    double amplitude;
    OctaveNoise octA;
    OctaveNoise octB;
};

//==================RNG================

STRUCT(Xoroshiro)
{
    uint64_t lo, hi;
};

uint64_t rotl64(uint64_t x, uint8_t b)
{
    return (x << b) | (x >> (64-b));
}

static inline void xSetSeed(Xoroshiro *xr, uint64_t value)
{
    const uint64_t XL = 0x9e3779b97f4a7c15ULL;
    const uint64_t XH = 0x6a09e667f3bcc909ULL;
    const uint64_t A = 0xbf58476d1ce4e5b9ULL;
    const uint64_t B = 0x94d049bb133111ebULL;
    uint64_t l = value ^ XH;
    uint64_t h = l + XL;
    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;
    l = l ^ (l >> 31);
    h = h ^ (h >> 31);
    xr->lo = l;
    xr->hi = h;
}

static inline uint64_t xNextLong(Xoroshiro *xr)
{
    uint64_t l = xr->lo;
    uint64_t h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

static inline int xNextInt(Xoroshiro *xr, uint32_t n)
{
    uint64_t r = (xNextLong(xr) & 0xFFFFFFFF) * n;
    if ((uint32_t)r < n)
    {
        while ((uint32_t)r < (~n + 1) % n)
        {
            r = (xNextLong(xr) & 0xFFFFFFFF) * n;
        }
    }
    return r >> 32;
}

static inline double xNextDouble(Xoroshiro *xr)
{
    return (xNextLong(xr) >> (64-53)) * 1.1102230246251565E-16;
}

//=================END_RNG==================

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

    //debug
    // noise->xOffset = 0.0;
    //noise->yOffset = 0.0;
    // noise->zOffset = 0.0;

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

    for(int i = 0; i < 256; i++) {
        printf("%d ", lookupHash[i] & 0xf);
    }
    printf("\n");

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

        //this counts the last 9 octaves
        //printf("before hash: %.16lx %.16lx\n", xhi, xlo);
        pxr.lo = xlo ^ md5_octave_n[12 + minimumOctave + i][0];
        pxr.hi = xhi ^ md5_octave_n[12 + minimumOctave + i][1];
        xPerlinInit(&octaves[i], &pxr);
        octaves[i].amplitude = amplitudes[i] * persist;
        octaves[i].lacunarity = lacuna;

        lacuna *= 2.0;
        persist *= 0.5;
    }

    //printf("after xOctaveInit: %.16lx %.16lx\n", xhi, xlo);

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
    n += xOctaveInit(&noise->octA, xr, octaves, minimumOctave, na);

    //second octave noise
    n += xOctaveInit(&noise->octB, xr, octaves+n, minimumOctave, nb);


    static const double amp_ini[] = { // (5 ./ 3) * len / (len + 1), len = 2..9
        0, 5./6, 10./9, 15./12, 20./15, 25./18, 30./21, 35./24, 40./27, 45./30,
    };
    noise->amplitude = amp_ini[len];
    return n;
}

static void init_climate_seed(
    DoublePerlinNoise *dpn, PerlinNoise *oct,
    uint64_t xlo, uint64_t xhi, int large, int nmax
    )
{
    Xoroshiro pxr;

    // md5 "minecraft:continentalness" or "minecraft:continentalness_large"
    pxr.lo = xlo ^ (large ? 0x9a3f51a113fce8dc : 0x83886c9d0ae3a662);
    pxr.hi = xhi ^ (large ? 0xee2dbd157e5dcdad : 0xafa638a61b42e8ad);

    //printf("in initClimateSeed: %.16lx %.16lx\n", pxr.hi, pxr.lo);
    xDoublePerlinInit(dpn, &pxr, oct, large ? -11 : -9, nmax);
}

//===========================END_INIT=====================

//===========================SAMPLING==============================

//sample single perlin noisemap. this is the important bit
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

    //cast the integer values to 
    xInt = (int) xIntTemp;
    zInt = (int) zIntTemp;

    //smoothstep function. 
    t1 = x*x*x * (x * (x*6.0-15.0) + 10.0);
    t3 = z*z*z * (z * (z*6.0-15.0) + 10.0);
    //t1 = x;
    //t3 = z;

    const uint8_t *lookupHash = noise->lookupHash;

    //get psuedorandom values from a lookup table. this uses yInt to...?
    //seems like it would entirely modify the noise past integer boundaries?
    //upon further inspection, it seems to randomize gradient placement
    uint8_t a1 = lookupHash[xInt]   + yInt;
    uint8_t b1 = lookupHash[xInt+1] + yInt;

    //printf("a1:%.2x b1:%.2x yInt:%d\n", a1, b1, yInt);

    //gets more hashes, this time for the 
    uint8_t a2 = lookupHash[a1]   + zInt;
    uint8_t b2 = lookupHash[b1]   + zInt;
    uint8_t a3 = lookupHash[a1+1] + zInt;
    uint8_t b3 = lookupHash[b1+1] + zInt;
    //printf("a2:%.2x b2:%.2x a3:%.2x b3:%.2x \n\n", a2, b2, a3, b3);

    //computes the gradients 
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

//sample stacked perlin noisemaps
double sampleOctave(const OctaveNoise *noise, double x, double z)
{
    double v = 0;
    int i;
    for (i = 0; i < noise->octcnt; i++)
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

//sample two perlin noisemaps and add them together with v
double sampleDoublePerlin(const DoublePerlinNoise *noise,
        double x, double z)
{
    double v = 0;

    v += sampleOctave(&noise->octA, x, z);
    v += sampleOctave(&noise->octB, x*1.0181268882175227, z*1.0181268882175227);

    return v * noise->amplitude;
}

//END_SAMPLING

//helper function. delete if not using debug
void encodeOneStep(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
  /*Encode the image*/
  unsigned error = lodepng_encode_file(filename, image, width, height, LCT_GREY, 8);

  /*if there's an error, display it*/
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

void genImage(uint64_t frame) {
    //test code to generate a default noisemap

    double y = frame / 1024.0;
    char* fileName;
    asprintf(&fileName, "images/%.4ld.png", frame);

    Xoroshiro xrng;
    xSetSeed(&xrng, 1);

    PerlinNoise perlinTest;
    xPerlinInit(&perlinTest, &xrng);

    //manually modify noise height
    perlinTest.yInt = (int) floor(y);
    perlinTest.yOffsetFract = y - floor(y);

    //smooth(er)step between 0 and 1.
    perlinTest.yFractSmoothstep = perlinTest.yOffsetFract*perlinTest.yOffsetFract*perlinTest.yOffsetFract * (perlinTest.yOffsetFract * (perlinTest.yOffsetFract*6.0-15.0) + 10.0);

    unsigned char* buffer = (unsigned char*)malloc(1024 * 1024 * sizeof(unsigned char));
    double pixel;
    for(int x = 0; x < 1024; x++) {
        for(int z = 0; z < 1024; z++) {
            pixel = samplePerlin(&perlinTest, (double)x/16, (double)z/16);
            pixel += 1;
            pixel /= 2;
            buffer[x*1024 + z] = (unsigned char)(256 * pixel) & 0xff;
        }
    }
    encodeOneStep(fileName, buffer, 1024, 1024);
}

int main(int argc, char** argv)
{
    uint64_t seed = 5605115020223874862;
    int large = 0;
    int octave_max = 1;

    Xoroshiro pxr;
    xSetSeed(&pxr, seed);
    uint64_t xlo = xNextLong(&pxr);
    uint64_t xhi = xNextLong(&pxr);

    DoublePerlinNoise dpn;
    PerlinNoise octaves[2*23]; //this is all the noisemaps


    init_climate_seed(&dpn, octaves, xlo, xhi, large, octave_max);

    
    double checksum = 0;
    for(int x = 0; x < 1024; x++) {
        for(int z = 0; z < 1024; z++) {
            checksum += sampleDoublePerlin(&dpn, (double)x, (double)z);
        }
    }
    //printf("checksum: %lf (should be -180780.088673)\n", checksum);


    return 0;
}