#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <math.h>
#include <numbers>
#include <string>
#include <vector>

#include "util.h"

double __attribute__((noinline)) nop(double x) { return 0.00001; }

/// @returns the exponent and a normalized mantissa with the relationship:
/// [a * 2^b] = x
std::pair<double, int> my_frexp(double x) {
    uint64_t bits = bit_cast<uint64_t, double>(x);
    if (bits == 0) {
        return { 0., 0 };
    }
    // See:
    // https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats

    // Extract the 52-bit mantissa field.
    uint64_t mantissa = bits & 0xFFFFFFFFFFFFF;
    bits >>= 52;

    // Extract the 11-bit exponent field, and add the bias.
    int exponent = int(bits & 0x7ff) - 1023;
    bits >>= 11;

    // Extract the sign bit.
    uint64_t sign = bits;
    bits >>= 1;

    // Construct the normalized double;
    uint64_t res = sign;
    res <<= 11;
    res |= 1023 - 1;
    res <<= 52;
    res |= mantissa;

    double frac = bit_cast<double, uint64_t>(res);
    return { frac, exponent + 1 };
}

double __attribute__((noinline)) fastlog2(double x) {

    /// Extract the fraction, and the power-of-two exponent.

    auto a = my_frexp(x);
    x = a.first;
    int pow2 = a.second;

    // Use a 4-part polynom to approximate log2(x);
    double c[] = { 1.33755322, -4.42852392, 6.30371424, -3.21430967 };
    double log2 = 0.6931471805599453;

    // Use Horner's method to evaluate the polynomial.
    double val = c[3] + x * (c[2] + x * (c[1] + x * (c[0])));

    // Compute log2(x), and convert the result to base-e.
    return log2 * (pow2 + val);
}

// Find the max error.
void validate_error(const std::vector<double> &iv, double max_range = 20.0,
                    int iterations = 10000) {
    double max_error = 0;
    double error_val = 0;
    unsigned validated = 0;
    // Validate a sequence of numbers.
    for (int i = 1; i < iterations; i++) {
        validated++;
        double val = ((max_range * i) / iterations);
        double err = std::abs(log(val) - fastlog2(val));
        if (err > max_error) {
            error_val = val;
            max_error = err;
        }
    }

    // Validate the pre-computed random numbers.
    for (auto elem : iv) {
        validated++;
        double err = std::abs(log(elem) - fastlog2(elem));
        if (err > max_error) {
            error_val = elem;
            max_error = err;
        }
    }

    std::cout << "Tested " << validated << " values [0.." << max_range << "]\n";
    std::cout << "Max error " << max_error << " at " << error_val << "\n";
    std::cout << "# " << log(error_val) << " vs " << fastlog2(error_val) << "\n";
}

// Check if the function is monolithic.
void validate_monotonic(double max_range = 20.0, int iterations = 10000) {
    double prev = fastlog2(0);
    unsigned non_monotonic = 0;
    for (int i = 1; i < iterations; i++) {
        double val = ((max_range * i) / iterations);
        val = fastlog2(val);
        if (prev > val) {
            non_monotonic += 1;
        }
        prev = val;
    }

    std::cout << "Tested " << iterations << " values [0.." << max_range << "]\n";
    std::cout << "Found " << non_monotonic << " non-monotinic values\n";
}

void check() {
    auto a = my_frexp(4.5);
    auto b = my_frexp(3.2);
    auto c = my_frexp(-10);
    auto d = my_frexp(65536);
    assert(a.first == 0.5625 && a.second == 3);
    assert(b.first == 0.8 && b.second == 2);
    assert(c.first == -0.625 && c.second == 4);
    assert(d.first == 0.5 && d.second == 17);
}

int main(int argc, char **argv) {
    check();
    std::vector<double> iv = generate_test_vector(0.5, 10., 10000);
    validate_error(iv);
    validate_monotonic();

    bench("fast_log", fastlog2, iv);
    bench("libm_log", log, iv);
    bench("nop     ", nop, iv);
    return 0;
}
