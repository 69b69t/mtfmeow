#include <stdio.h>
#include <stdint.h>

#include "crunchLib.h"

int main()
{
    uint64_t test = getNextValid(0, 0.000001, 0, 1);
    printf("%ld with score %f\n", test, inefficientScore(test, 0, 1));
    return 0;
}