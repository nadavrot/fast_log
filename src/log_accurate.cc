#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <math.h>
#include <numbers>
#include <string>
#include <vector>

#include "util.h"

/// @returns the exponent and a normalized mantissa with the relationship:
/// [m * 2^E] = x, where m is in [1..2].
/// This is similar to frexp(), except that the range for m is
/// in [1..2] and not [0.5 ..1].
std::pair<float, int> reduce_fp32(float x) {
    uint32_t bits = bit_cast<uint32_t, float>(x);
    if (bits == 0) {
        return { 0., 0 };
    }
    // See:
    // https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats

    // Extract the 23-bit mantissa field.
    uint64_t mantissa = bits & 0x7FFFFF;
    bits >>= 23;

    // Extract the 8-bit exponent field, and add the bias.
    int exponent = int(bits & 0xff) - 127;
    bits >>= 8;

    // Extract the sign bit.
    uint64_t sign = bits;
    bits >>= 1;

    //  Construct the normalized double;
    uint64_t res = sign;
    res <<= 11;
    res |= 127;
    res <<= 23;
    res |= mantissa;

    float frac = bit_cast<float, uint32_t>(res);
    return { frac, exponent };
}

// Compute the reciprocal of \p y in the range [sqrt(2)/2 .. sqrt(2)].
float recip_of_masked(float x) {
    unsigned masked_recp_table[256] = {
        0x40000000, 0x3ffe03f8, 0x3ffc0fc1, 0x3ffa232d, 0x3ff83e10, 0x3ff6603e, 0x3ff4898d,
        0x3ff2b9d6, 0x3ff0f0f1, 0x3fef2eb7, 0x3fed7304, 0x3febbdb3, 0x3fea0ea1, 0x3fe865ac,
        0x3fe6c2b4, 0x3fe52598, 0x3fe38e39, 0x3fe1fc78, 0x3fe07038, 0x3fdee95c, 0x3fdd67c9,
        0x3fdbeb62, 0x3fda740e, 0x3fd901b2, 0x3fd79436, 0x3fd62b81, 0x3fd4c77b, 0x3fd3680d,
        0x3fd20d21, 0x3fd0b6a0, 0x3fcf6475, 0x3fce168a, 0x3fcccccd, 0x3fcb8728, 0x3fca4588,
        0x3fc907da, 0x3fc7ce0c, 0x3fc6980c, 0x3fc565c8, 0x3fc43730, 0x3fc30c31, 0x3fc1e4bc,
        0x3fc0c0c1, 0x3fbfa030, 0x3fbe82fa, 0x3fbd6910, 0x3fbc5264, 0x3fbb3ee7, 0x3fba2e8c,
        0x3fb92144, 0x3fb81703, 0x3fb70fbb, 0x3fb60b61, 0x3fb509e7, 0x3fb40b41, 0x3fb30f63,
        0x3fb21643, 0x3fb11fd4, 0x3fb02c0b, 0x3faf3ade, 0x3fae4c41, 0x3fad602b, 0x3fac7692,
        0x3fab8f6a, 0x3faaaaab, 0x3fa9c84a, 0x3fa8e83f, 0x3fa80a81, 0x3fa72f05, 0x3fa655c4,
        0x3fa57eb5, 0x3fa4a9cf, 0x3fa3d70a, 0x3fa3065e, 0x3fa237c3, 0x3fa16b31, 0x3fa0a0a1,
        0x3f9fd80a, 0x3f9f1166, 0x3f9e4cad, 0x3f9d89d9, 0x3f9cc8e1, 0x3f9c09c1, 0x3f9b4c70,
        0x3f9a90e8, 0x3f99d723, 0x3f991f1a, 0x3f9868c8, 0x3f97b426, 0x3f97012e, 0x3f964fda,
        0x3f95a025, 0x3f94f209, 0x3f944581, 0x3f939a86, 0x3f92f114, 0x3f924925, 0x3f91a2b4,
        0x3f90fdbc, 0x3f905a38, 0x3f8fb824, 0x3f8f177a, 0x3f8e7835, 0x3f8dda52, 0x3f8d3dcb,
        0x3f8ca29c, 0x3f8c08c1, 0x3f8b7034, 0x3f8ad8f3, 0x3f8a42f8, 0x3f89ae41, 0x3f891ac7,
        0x3f888889, 0x3f87f781, 0x3f8767ab, 0x3f86d905, 0x3f864b8a, 0x3f85bf37, 0x3f853408,
        0x3f84a9fa, 0x3f842108, 0x3f839930, 0x3f83126f, 0x3f828cc0, 0x3f820821, 0x3f81848e,
        0x3f810204, 0x3f808081, 0x3f800000, 0x3f7e03f8, 0x3f7c0fc1, 0x3f7a232d, 0x3f783e10,
        0x3f76603e, 0x3f74898d, 0x3f72b9d6, 0x3f70f0f1, 0x3f6f2eb7, 0x3f6d7304, 0x3f6bbdb3,
        0x3f6a0ea1, 0x3f6865ac, 0x3f66c2b4, 0x3f652598, 0x3f638e39, 0x3f61fc78, 0x3f607038,
        0x3f5ee95c, 0x3f5d67c9, 0x3f5beb62, 0x3f5a740e, 0x3f5901b2, 0x3f579436, 0x3f562b81,
        0x3f54c77b, 0x3f53680d, 0x3f520d21, 0x3f50b6a0, 0x3f4f6475, 0x3f4e168a, 0x3f4ccccd,
        0x3f4b8728, 0x3f4a4588, 0x3f4907da, 0x3f47ce0c, 0x3f46980c, 0x3f4565c8, 0x3f443730,
        0x3f430c31, 0x3f41e4bc, 0x3f40c0c1, 0x3f3fa030, 0x3f3e82fa, 0x3f3d6910, 0x3f3c5264,
        0x3f3b3ee7, 0x3f3a2e8c, 0x3f392144, 0x3f381703, 0x3f370fbb, 0x3f360b61, 0x3f3509e7,
        0x3f340b41, 0x3f330f63, 0x3f321643, 0x3f311fd4, 0x3f302c0b, 0x3f2f3ade, 0x3f2e4c41,
        0x3f2d602b, 0x3f2c7692, 0x3f2b8f6a, 0x3f2aaaab, 0x3f29c84a, 0x3f28e83f, 0x3f280a81,
        0x3f272f05, 0x3f2655c4, 0x3f257eb5, 0x3f24a9cf, 0x3f23d70a, 0x3f23065e, 0x3f2237c3,
        0x3f216b31, 0x3f20a0a1, 0x3f1fd80a, 0x3f1f1166, 0x3f1e4cad, 0x3f1d89d9, 0x3f1cc8e1,
        0x3f1c09c1, 0x3f1b4c70, 0x3f1a90e8, 0x3f19d723, 0x3f191f1a, 0x3f1868c8, 0x3f17b426,
        0x3f17012e, 0x3f164fda, 0x3f15a025, 0x3f14f209, 0x3f144581, 0x3f139a86, 0x3f12f114,
        0x3f124925, 0x3f11a2b4, 0x3f10fdbc, 0x3f105a38, 0x3f0fb824, 0x3f0f177a, 0x3f0e7835,
        0x3f0dda52, 0x3f0d3dcb, 0x3f0ca29c, 0x3f0c08c1, 0x3f0b7034, 0x3f0ad8f3, 0x3f0a42f8,
        0x3f09ae41, 0x3f091ac7, 0x3f088889, 0x3f07f781, 0x3f0767ab, 0x3f06d905, 0x3f064b8a,
        0x3f05bf37, 0x3f053408, 0x3f04a9fa, 0x3f042108, 0x3f039930, 0x3f03126f, 0x3f028cc0,
        0x3f020821, 0x3f01848e, 0x3f010204, 0x3f008081,
    };
    unsigned xb = bit_cast<unsigned, float>(x);
    unsigned bval = masked_recp_table[(xb >> 16) & 0xff];
    return bit_cast<float, unsigned>(bval);
}

// Compute the reciprocal log of \p x in the range [sqrt(2)/2 .. sqrt(2)].
float log_of_masked(float x) {
    unsigned masked_log_recp_table[256] = {
        0x3f317218, 0x3f2f7415, 0x3f2d7a03, 0x3f2b83d1, 0x3f299172, 0x3f27a2d5, 0x3f25b7eb,
        0x3f23d0a9, 0x3f21ed00, 0x3f200ce1, 0x3f1e3041, 0x3f1c5711, 0x3f1a8146, 0x3f18aed2,
        0x3f16dfaa, 0x3f1513c3, 0x3f134b11, 0x3f118587, 0x3f0fc31b, 0x3f0e03c2, 0x3f0c4772,
        0x3f0a8e20, 0x3f08d7c2, 0x3f07244c, 0x3f0573b7, 0x3f03c5f8, 0x3f021b06, 0x3f0072d7,
        0x3efd9ac6, 0x3efa5540, 0x3ef7150c, 0x3ef3da15, 0x3ef0a451, 0x3eed73ab, 0x3eea4812,
        0x3ee72178, 0x3ee3ffcd, 0x3ee0e302, 0x3eddcb08, 0x3edab7d1, 0x3ed7a94b, 0x3ed49f6a,
        0x3ed19a21, 0x3ece9960, 0x3ecb9d1a, 0x3ec8a542, 0x3ec5b1cd, 0x3ec2c2ab, 0x3ebfd7d3,
        0x3ebcf133, 0x3eba0ec4, 0x3eb73076, 0x3eb45643, 0x3eb1801a, 0x3eaeadf0, 0x3eabdfba,
        0x3ea91571, 0x3ea64f06, 0x3ea38c6e, 0x3ea0cda2, 0x3e9e1293, 0x3e9b5b3b, 0x3e98a790,
        0x3e95f784, 0x3e934b12, 0x3e90a22b, 0x3e8dfcca, 0x3e8b5ae7, 0x3e88bc73, 0x3e86216b,
        0x3e8389c3, 0x3e80f572, 0x3e7cc8e2, 0x3e77ad6e, 0x3e729877, 0x3e6d89ec, 0x3e6881c2,
        0x3e637fde, 0x3e5e843a, 0x3e598ec2, 0x3e549f6c, 0x3e4fb61e, 0x3e4ad2d9, 0x3e45f582,
        0x3e411e0c, 0x3e3c4c6d, 0x3e378092, 0x3e32ba76, 0x3e2dfa04, 0x3e293f2f, 0x3e2489e9,
        0x3e1fda2a, 0x3e1b2fe3, 0x3e168b0b, 0x3e11eb8b, 0x3e0d515f, 0x3e08bc77, 0x3e042cc7,
        0x3dff4489, 0x3df639c7, 0x3ded393c, 0x3de442c2, 0x3ddb563e, 0x3dd273b2, 0x3dc99af2,
        0x3dc0cbf1, 0x3db8069f, 0x3daf4ace, 0x3da6988b, 0x3d9defa7, 0x3d95502c, 0x3d8cb9db,
        0x3d842ccd, 0x3d77519c, 0x3d665b93, 0x3d55778e, 0x3d44a542, 0x3d33e49c, 0x3d23356d,
        0x3d1297a0, 0x3d020ae4, 0x3ce31e83, 0x3cc24943, 0x3ca19559, 0x3c8102d2, 0x3c412270,
        0x3c0080a8, 0x3b8040aa, 0x0,        0xbbff015b, 0xbc7e0545, 0xbcbdc8d6, 0xbcfc14c8,
        0xbd1cf437, 0xbd3ba2ce, 0xbd5a16f0, 0xbd785185, 0xbd8b29b9, 0xbd9a0eba, 0xbda8d837,
        0xbdb78694, 0xbdc61a33, 0xbdd4936c, 0xbde2f2a6, 0xbdf1383a, 0xbdff648a, 0xbe06bbf4,
        0xbe0db958, 0xbe14aa96, 0xbe1b8fe1, 0xbe22695a, 0xbe29372f, 0xbe2ff983, 0xbe36b07e,
        0xbe3d5c48, 0xbe43fd04, 0xbe4a92d4, 0xbe511de0, 0xbe579e49, 0xbe5e1436, 0xbe647fbd,
        0xbe6ae10a, 0xbe71383b, 0xbe778570, 0xbe7dc8c6, 0xbe82012e, 0xbe851928, 0xbe882c5f,
        0xbe8b3ae5, 0xbe8e44c6, 0xbe914a0f, 0xbe944ad0, 0xbe974716, 0xbe9a3eee, 0xbe9d3263,
        0xbea02185, 0xbea30c5d, 0xbea5f2fd, 0xbea8d56c, 0xbeabb3ba, 0xbeae8ded, 0xbeb16416,
        0xbeb43640, 0xbeb70476, 0xbeb9cebf, 0xbebc952a, 0xbebf57c2, 0xbec2168e, 0xbec4d19d,
        0xbec788f5, 0xbeca3c9f, 0xbeccecac, 0xbecf991e, 0xbed24205, 0xbed4e766, 0xbed78949,
        0xbeda27bd, 0xbedcc2c5, 0xbedf5a6d, 0xbee1eebe, 0xbee47fbf, 0xbee70d79, 0xbee997f4,
        0xbeec1f3a, 0xbeeea34f, 0xbef12441, 0xbef3a213, 0xbef61ccf, 0xbef8947a, 0xbefb0921,
        0xbefd7ac3, 0xbeffe96f, 0xbf012a95, 0xbf025efd, 0xbf0391f3, 0xbf04c37b, 0xbf05f397,
        0xbf07224c, 0xbf084f9e, 0xbf097b8d, 0xbf0aa61f, 0xbf0bcf55, 0xbf0cf735, 0xbf0e1dc0,
        0xbf0f42fa, 0xbf1066e6, 0xbf118987, 0xbf12aadf, 0xbf13caf0, 0xbf14e9c0, 0xbf160750,
        0xbf1723a2, 0xbf183eba, 0xbf19589a, 0xbf1a7144, 0xbf1b88be, 0xbf1c9f07, 0xbf1db423,
        0xbf1ec812, 0xbf1fdadd, 0xbf20ec7e, 0xbf21fcfe, 0xbf230c5f, 0xbf241a9f, 0xbf2527c4,
        0xbf2633ce, 0xbf273ec1, 0xbf28489e, 0xbf29516a, 0xbf2a5924, 0xbf2b5fce, 0xbf2c656d,
        0xbf2d6a01, 0xbf2e6d8e, 0xbf2f7015, 0xbf307197,
    };
    unsigned xb = bit_cast<unsigned, float>(x);
    unsigned bval = masked_log_recp_table[(xb >> 16) & 0xff];
    return bit_cast<float, unsigned>(bval);
}

/// Evaluate a polynomial that approximates log(x+1) in the range [0-0.01].
double approximate_log_pol_1_to_1001(double x) {
    return -8.0159120687014415322143784594351322561698644901218e-21 +
           x * (0.99999999999996291855097751977154985070228576660156 +
                x * (-0.49999999982958548416789312796026933938264846801758 +
                     x * (0.33333320027787793904394675337243825197219848632812 +
                          x * (-0.249963278288175078101218673509720247238874435424805 +
                               x * 0.195840954071922368484592880122363567352294921875))));
}

/// @return True if \p x is a NAN.
bool is_nan(float x) {
    unsigned xb = bit_cast<unsigned, float>(x);
    xb >>= 23;
    return (xb & 0xff) == 0xff;
}

// Handbook of Floating-Point Arithmetic -- Jean-Michel Muller
// Chapter 11. Evaluating Floating-Point Elementary Functions (pg. 387)
float __attribute__((noinline)) my_log(float x) {
    // Handle the special values:
    if (x == 0) {
        return bit_cast<float, unsigned>(0xff800000); // -Inf
    } else if (is_nan(x)) {
        return x;
    } else if (x < 0) {
        return bit_cast<float, unsigned>(0xffc00000); // -Nan.
    }

    /// Extract the fraction, and the power-of-two exponent, such that:
    // (2^E) * m = x;
    auto a = reduce_fp32(x);
    float m = a.first;
    int E = a.second;

    // Reduce the range of m to [sqrt(2)/2 -- sqrt(2)]
    if (m > 1.4142136) {
        E = E + 1;
        m = m / 2;
    }

    float y = m;
    // Compute the reciprocal of y using a lookup table.
    float ri = recip_of_masked(y);
    float z = y * ri - 1;
    double log2 = 0.6931471805599453;

    // We use double here because float is not accurate enough for the final
    // reduction. We are missing just a few bits.

    // Compute log(1/ri) using a lookup table.
    double ln_ri = log_of_masked(y);
    // Approximate log(1+z) using a polynomial:
    double ln_1z = approximate_log_pol_1_to_1001(z);

    // Perform the final reduction.
    return E * log2 + ln_1z - ln_ri;
}

// Wrap the standard log(double) and use it as the ground truth.
float accurate_log(float x) { return log(x); }

int main(int argc, char **argv) {
    // print_log_table_for_3f_values();
    // print_recp_table_for_3f_values();
    print_ulp_deltas(my_log, accurate_log);
}
