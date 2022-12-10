#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

template <class To, class From> To bit_cast(const From &src) noexcept {
    static_assert(sizeof(To) == sizeof(From), "Size mismatch");
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

// Return the bitwise distance between the two doubles.
// Notice that a change in sign will return a high ULP difference,
// which is desirable.
template <class UnsignedTy, class FloatTy> UnsignedTy ulp_difference(FloatTy n1, FloatTy n2) {
    UnsignedTy b1 = bit_cast<UnsignedTy, FloatTy>(n1);
    UnsignedTy b2 = bit_cast<UnsignedTy, FloatTy>(n2);
    if (b1 == b2 || (std::isnan(n1) && std::isnan(n2))) {
        return 0;
    }

    // Return the delta between the two numbers in bits.
    return (b1 > b2) ? (b1 - b2) : (b2 - b1);
}

// Compare two functions and count the number of values with different ULPs.
// See https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
void print_ulp_deltas(float (*handle1)(float), float (*handle2)(float)) {
    // Count the numbers of the first 31 ULP deltas.
    uint64_t ULP_delta[32] = { 0 };

    // For each value in the 32bit range.
    for (uint64_t i = 0; i < 1L << 32; i++) {
        float val = bit_cast<float, unsigned>((unsigned)i);
        float r1 = handle1(val);
        float r2 = handle2(val);

        // Record the ULP delta.
        unsigned ud = ulp_difference<unsigned, float>(r1, r2);
        ud = std::min<unsigned>(31, ud);
        ULP_delta[ud]++;
    }

    // Print the ULP distribution:
    printf("ULP distribution:\n");
    for (int i = 0; i < 32; i++) {
        double percent = 100 * double(ULP_delta[i]) / double(1LL << 32);
        if (i < 31) {
            printf("%02d) %02.3f%% - %08lu\n", i, percent, ULP_delta[i]);
        } else {
            printf("Other: %02.3f%% - %08lu\n", percent, ULP_delta[i]);
        }
    }
}

// Prints a lookup table for [0x3fxx0000], that computes f(x)=log(1/x).
void print_log_table_for_3f_values() {
    // Count the numbers of the first 31 ULP deltas.
    unsigned table[256] = { 0 };

    for (unsigned i = 0; i < 256; i++) {
        unsigned valb = (0x3f << 24) | (i << 16);
        float val = bit_cast<float, unsigned>(valb);
        val = log(1 / val);
        table[i] = bit_cast<unsigned, float>(val);
    }

    // Print the ULP distribution:
    printf("unsigned masked_log_table[256] = {");
    for (int i = 0; i < 256; i++) {
        if (i % 8 == 0) {
            printf("\n\t");
        }
        printf("0x%x, ", table[i]);
    }
    printf("};\n");
}

// Prints a lookup table for [0x3fxx0000], that computes f(x)=1/x.
void print_recp_table_for_3f_values() {
    // Count the numbers of the first 31 ULP deltas.
    unsigned table[256] = { 0 };

    for (unsigned i = 0; i < 256; i++) {
        unsigned valb = (0x3f << 24) | (i << 16);
        float val = bit_cast<float, unsigned>(valb);
        val = 1 / val;
        table[i] = bit_cast<unsigned, float>(val);
    }

    // Print the ULP distribution:
    printf("unsigned masked_log_recp_table[256] = {");
    for (int i = 0; i < 256; i++) {
        if (i % 8 == 0) {
            printf("\n\t");
        }
        printf("0x%x, ", table[i]);
    }
    printf("};\n");
}

/// @return \p count random uniform numbers in the range \p start to \p end.
std::vector<double> generate_test_vector(double start, double end, unsigned count) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(start, end);
    std::vector<double> res;
    for (unsigned i = 0; i < count; i++) {
        res.push_back(dist(mt));
    }
    return res;
}

/// @brief Benchmark a program with the name \p name, and function pointer
/// \p handle. Run \p iterations iterations on inputs from the test vector
/// \p iv. Prints the result to stdout.
void bench(const std::string &name, double (*handle)(double), const std::vector<double> &iv,
           int iterations = 10000) {
    auto t1 = high_resolution_clock::now();

    double sum = 0;
    for (int iter = 0; iter < iterations; iter++) {
        for (auto elem : iv) {
            sum += handle(elem);
        }
    }

    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "name = " << name << ", ";
    std::cout << "sum = " << sum << ", ";
    std::cout << "time = " << ms_int.count() << "ms\n";
}
