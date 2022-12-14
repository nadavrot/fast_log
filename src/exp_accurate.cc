#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <math.h>
#include <numbers>
#include <string>
#include <vector>

#include "exp_table.h"
#include "util.h"

/// @return True if \p x is a NAN.
bool is_nan(float x) {
    unsigned xb = bit_cast<unsigned, float>(x);
    xb >>= 23;
    return (xb & 0xff) == 0xff;
}

// Approximate the function \p exp in the range -0.004, 0.004.
// Q = fpminimax(exp(x), 5, [|D...|], [-0.0039, 0.0039])
double approximate_exp_pol_around_zero(float x) {
    return 1 +
           x * (1 + x * (0.49999999999985944576508245518198236823081970214844 +
                         x * (0.166666666666697105281258473041816614568233489990234 +
                              x * (4.1666696240209417922972789938285131938755512237549e-2 +
                                   x * 8.3333337622652735310335714302709675393998622894287e-3))));
}

float __attribute__((noinline)) my_exp(float x) {
    if (x >= 710) {
        return bit_cast<float, unsigned>(0x7f800000); // Inf
    } else if (x <= -710) {
        return 0;
    } else if (is_nan(x)) {
        return x;
    }

    // Split X into 3 numbers such that: x = I1 + (I2 << 8) + xt;
    int Int1 = int(x);
    x = x - Int1;
    int Int2 = int(x * 256);
    x = x - (float(Int2) / 256);

    return approximate_exp_pol_around_zero(x) * EXP_TABLE[Int1 + 710] * EXP_TABLE_r256[Int2 + 256];
}

// Wrap the standard exp(double) and use it as the ground truth.
float accurate_exp(float x) { return exp((double)x); }

// Wrap the standard exp(double) and use it as the ground truth.
float libc_exp(float x) { return expf(x); }

int main(int argc, char **argv) { print_ulp_deltas(my_exp, accurate_exp); }
