#ifndef XOROLIB_H
#define XOROLIB_H

//move to seperate library
typedef struct
{
    uint64_t lo, hi;
} Xoroshiro;

void xSetSeed(Xoroshiro *xr, uint64_t value);

uint64_t xNextLong(Xoroshiro *xr);

int xNextInt(Xoroshiro *xr, uint32_t n);

double xNextDouble(Xoroshiro *xr);

#endif