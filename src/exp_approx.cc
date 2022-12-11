#include <cassert>
#include <cmath>
#include <cstring>
#include <numbers>
#include <vector>

#include "exp_table.h"
#include "util.h"

double __attribute__((noinline)) nop(double x) { return x + 1; }

double __attribute__((noinline)) fast_exp(double x) {
    double integer = trunc(x);
    // X is now the fractional part of the number.
    x = x - integer;

    // Use a 4-part polynomial to approximate exp(x);
    double c[] = { 0.28033708, 0.425302, 1.01273643, 1.00020947 };

    // Use Horner's method to evaluate the polynomial.
    double val = c[3] + x * (c[2] + x * (c[1] + x * (c[0])));
    return val * EXP_TABLE[(unsigned)integer + 710];
}

int main(int argc, char **argv) {
    std::vector<double> iv = generate_test_vector(-10., 10., 10000);
    bench("nop", nop, iv);
    bench("trunc", trunc, iv);
    bench("fast_exp", fast_exp, iv);
    bench("libm_exp", exp, iv);
    return 0;
}
