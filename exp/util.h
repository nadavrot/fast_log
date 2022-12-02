#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

/// @return \p count random uniform numbers in the range \p start to \p end.
std::vector<double> generate_test_vector(double start, double end,
                                         unsigned count) {
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
void bench(const std::string &name, double (*handle)(double),
           const std::vector<double> &iv, int iterations = 10000) {
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
