#include <stdint.h>
#include <math.h>

#include "crunchLib.h"

inline uint64_t rotl64(uint64_t x, uint8_t b)
{
    return (x << b) | (x >> (64-b));
}

inline void xSetSeed(Xoroshiro *xr, uint64_t value)
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

inline uint64_t xNextLong(Xoroshiro *xr)
{
    uint64_t l = xr->lo;
    uint64_t h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

inline double xNextDouble(Xoroshiro *xr)
{
    return (xNextLong(xr) >> (64-53)) * 1.1102230246251565E-16;
}

inline double calculateCrunchiness(Xoroshiro state, int large, int octave)
{
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

    int minimumOctave = large ? -11 : -9;
    state.lo = state.lo ^ md5_octave_n[12 + minimumOctave + octave][0];
    state.hi = state.hi ^ md5_octave_n[12 + minimumOctave + octave][1];

    //burn a state. slightly faster than xNextDouble
    xNextLong(&state);

    double yOffset = xNextDouble(&state) * 256.0;
    return yOffset - floor(yOffset);
}

inline void initOctaveSeeds(Xoroshiro *octASeed, Xoroshiro *octBSeed, uint64_t seed, int large)
{
    //this gets the seeds for octaveA and octaveB
    Xoroshiro xr;
    xSetSeed(&xr, seed);

    //shuffle to get a new state
    uint64_t xlo = xNextLong(&xr);
    uint64_t xhi = xNextLong(&xr);

    //xor with different constants for different maps
    // md5 "minecraft:continentalness" or "minecraft:continentalness_large"
    xr.lo = xlo ^ (large ? 0x9a3f51a113fce8dc : 0x83886c9d0ae3a662);
    xr.hi = xhi ^ (large ? 0xee2dbd157e5dcdad : 0xafa638a61b42e8ad);

    //gets new states for both octave a and b
    octASeed->lo = xNextLong(&xr);
    octASeed->hi = xNextLong(&xr);
    octBSeed->lo = xNextLong(&xr);
    octBSeed->hi = xNextLong(&xr);
}

//calculate MAD for octaves up to n
inline double doubleMad(Xoroshiro octaveSeedA, Xoroshiro octaveSeedB, int large) {
    //double crunchiness = calculateCrunchiness(octaveSeed, large, octaveMax);

    double sum = 0.0;
    sum += 1 * fabs(calculateCrunchiness(octaveSeedA, large, 0) - 0.5);
    sum += 1 * fabs(calculateCrunchiness(octaveSeedB, large, 0) - 0.5);
    sum += 0.5 * fabs(calculateCrunchiness(octaveSeedA, large, 1) - 0.5);
    sum += 0.5 * fabs(calculateCrunchiness(octaveSeedB, large, 1) - 0.5);

    return sum / 3.0;
}

