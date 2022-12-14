#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define PRINT_DOUBLE(name, x)                                                                      \
    {                                                                                              \
        uint64_t ux = bit_cast<uint64_t, double>(x);                                               \
        printf("%s: (%.9f 0x%lx)\n", #name, x, ux);                                                \
    }
#define PRINT_FLOAT(name, x)                                                                       \
    {                                                                                              \
        uint32_t ux = bit_cast<uint32_t, float>(x);                                                \
        printf("%s: (%.9f 0x%lx)\n", #name, x, ux);                                                \
    }

#define PRINT_INT(name, x)                                                                         \
    { printf("%s: (0x%lx %ld)\n", #name, x, x); }

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

/// A generic histogram class.
template <unsigned NumBins> struct Histogram {
    uint64_t payload_[NumBins];
    Histogram() {
        for (unsigned i = 0; i < NumBins; i++) {
            payload_[i] = 0;
        }
    }
    // Add the counts
    void join(const Histogram &other) {
        for (unsigned i = 0; i < NumBins; i++) {
            payload_[i] += other.payload_[i];
        }
    }
    void add(unsigned idx, uint64_t val = 1) {
        idx = std::min<unsigned>(NumBins - 1, idx);
        payload_[idx] += val;
    }

    void dump(const char *message) {
        printf("%s", message);
        for (unsigned i = 0; i < NumBins; i++) {
            double percent = 100 * double(payload_[i]) / double(1LL << 32);
            if (i < (NumBins - 1)) {
                printf("%02d) %02.3f%% - %08lu\n", i, percent, payload_[i]);
            } else {
                printf("Other: %02.3f%% - %08lu\n", percent, payload_[i]);
            }
        }
    };
};

/// A helper class that performs multi-threaded computation of ULP differences
/// between two implementations.
template <class FloatTy = float, class UnsignedTy = unsigned, unsigned NumThreads = 8,
          unsigned NumBins = 32>
class Verifier {
    std::thread threads_[NumThreads];
    Histogram<NumBins> hist_[NumThreads];

  public:
    void print_ulp_deltas(FloatTy (*handle1)(FloatTy), FloatTy (*handle2)(FloatTy)) {
        auto scan = [&handle1, &handle2](uint64_t start, uint64_t end, Histogram<NumBins> &hist) {
            // For each value in the 32bit range.
            for (uint64_t i = start; i < end; i++) {
                FloatTy val = bit_cast<FloatTy, UnsignedTy>((unsigned)i);
                FloatTy r1 = handle1(val);
                FloatTy r2 = handle2(val);
                // Record the ULP delta.
                unsigned ud = ulp_difference<UnsignedTy, FloatTy>(r1, r2);
                hist.add(ud);
            }
        };

        uint64_t chunk_size = (1L << 32) / NumThreads;
        for (unsigned i = 0; i < NumThreads; i++) {
            uint64_t start = i * chunk_size;
            uint64_t end = (i + 1) * chunk_size;
            threads_[i] = std::thread(scan, start, end, std::ref(hist_[i]));
        }
        for (unsigned i = 0; i < NumThreads; i++) {
            threads_[i].join();
        }
        // Merge the histograms after the workers finished.
        for (unsigned i = 1; i < NumThreads; i++) {
            hist_[0].join(hist_[i]);
        }
        // Report the histogram.
        hist_[0].dump("\nULP delta:\n");
    }
};

// Compare two functions and count the number of values with different ULPs.
// See https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
void print_ulp_deltas(float (*handle1)(float), float (*handle2)(float)) {
    Verifier<float, unsigned, 64, 16> verifier;
    verifier.print_ulp_deltas(handle1, handle2);
}

// Prints a lookup table for [0x3fxx0000], that computes f(x)=log(1/x).
void print_log_recp_table_for_3f_values() {
    uint64_t table[256] = { 0 };

    for (unsigned i = 0; i < 256; i++) {
        unsigned valb = (0x3f << 24) | (i << 16);
        float val = bit_cast<float, unsigned>(valb);
        double val2 = log(1. / (double)val);
        table[i] = bit_cast<uint64_t, double>(val2);
    }

    // Print the ULP distribution:
    printf("uint64_t masked_log_recp_table[256] = {");
    for (int i = 0; i < 256; i++) {
        if (i % 8 == 0) {
            printf("\n\t");
        }
        printf("0x%lx, ", table[i]);
    }
    printf("};\n");
}

// Prints a lookup table for [0x3fxx0000], that computes f(x)=1/x.
void print_recp_table_for_3f_values() {
    uint64_t table[256] = { 0 };

    for (unsigned i = 0; i < 256; i++) {
        unsigned valb = (0x3f << 24) | (i << 16);
        float val = bit_cast<float, unsigned>(valb);
        double val2 = (1. / (double)val);
        table[i] = bit_cast<uint64_t, double>(val2);
    }

    // Print the ULP distribution:
    printf("uint64_t masked_recp_table[256] = {");
    for (int i = 0; i < 256; i++) {
        if (i % 8 == 0) {
            printf("\n\t");
        }
        printf("0x%lx, ", table[i]);
    }
    printf("};\n");
}

/// @return \p count random uniform numbers in the range \p start to \p end.
template <class FloatTy>
std::vector<FloatTy> generate_test_vector(FloatTy start, FloatTy end, unsigned count) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<FloatTy> dist(start, end);
    std::vector<FloatTy> res;
    for (unsigned i = 0; i < count; i++) {
        res.push_back(dist(mt));
    }
    return res;
}

/// @brief Benchmark a program with the name \p name, and function pointer
/// \p handle. Run \p iterations iterations on inputs from the test vector
/// \p iv. Prints the result to stdout.
template <class FloatTy>
void bench(const std::string &name, FloatTy (*handle)(FloatTy), const std::vector<FloatTy> &iv,
           int iterations = 10000) {
    auto t1 = high_resolution_clock::now();

    FloatTy sum = 0;
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
