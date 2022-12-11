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
    int exponent = int(bits & 0xff);
    int normalized_exponent = exponent - 127;
    bits >>= 8;

    // Handle denormals.
    if (exponent == 0) {
        // Scale the number to a manageable scale.
        auto r = reduce_fp32(x * 0x1p32);
        r.second -= 32;
        return r;
    }

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
    return { frac, normalized_exponent };
}

// Compute the reciprocal of \p y in the range [sqrt(2)/2 .. sqrt(2)].
double recip_of_masked(float x) {
    uint64_t masked_recp_table[256] = {
        0x4000000000000000, 0x3fffc07f01fc07f0, 0x3fff81f81f81f820, 0x3fff44659e4a4271,
        0x3fff07c1f07c1f08, 0x3ffecc07b301ecc0, 0x3ffe9131abf0b767, 0x3ffe573ac901e574,
        0x3ffe1e1e1e1e1e1e, 0x3ffde5d6e3f8868a, 0x3ffdae6076b981db, 0x3ffd77b654b82c34,
        0x3ffd41d41d41d41d, 0x3ffd0cb58f6ec074, 0x3ffcd85689039b0b, 0x3ffca4b3055ee191,
        0x3ffc71c71c71c71c, 0x3ffc3f8f01c3f8f0, 0x3ffc0e070381c0e0, 0x3ffbdd2b899406f7,
        0x3ffbacf914c1bad0, 0x3ffb7d6c3dda338b, 0x3ffb4e81b4e81b4f, 0x3ffb2036406c80d9,
        0x3ffaf286bca1af28, 0x3ffac5701ac5701b, 0x3ffa98ef606a63be, 0x3ffa6d01a6d01a6d,
        0x3ffa41a41a41a41a, 0x3ffa16d3f97a4b02, 0x3ff9ec8e951033d9, 0x3ff9c2d14ee4a102,
        0x3ff999999999999a, 0x3ff970e4f80cb872, 0x3ff948b0fcd6e9e0, 0x3ff920fb49d0e229,
        0x3ff8f9c18f9c18fa, 0x3ff8d3018d3018d3, 0x3ff8acb90f6bf3aa, 0x3ff886e5f0abb04a,
        0x3ff8618618618618, 0x3ff83c977ab2bedd, 0x3ff8181818181818, 0x3ff7f405fd017f40,
        0x3ff7d05f417d05f4, 0x3ff7ad2208e0ecc3, 0x3ff78a4c8178a4c8, 0x3ff767dce434a9b1,
        0x3ff745d1745d1746, 0x3ff724287f46debc, 0x3ff702e05c0b8170, 0x3ff6e1f76b4337c7,
        0x3ff6c16c16c16c17, 0x3ff6a13cd1537290, 0x3ff6816816816817, 0x3ff661ec6a5122f9,
        0x3ff642c8590b2164, 0x3ff623fa77016240, 0x3ff6058160581606, 0x3ff5e75bb8d015e7,
        0x3ff5c9882b931057, 0x3ff5ac056b015ac0, 0x3ff58ed2308158ed, 0x3ff571ed3c506b3a,
        0x3ff5555555555555, 0x3ff5390948f40feb, 0x3ff51d07eae2f815, 0x3ff5015015015015,
        0x3ff4e5e0a72f0539, 0x3ff4cab88725af6e, 0x3ff4afd6a052bf5b, 0x3ff49539e3b2d067,
        0x3ff47ae147ae147b, 0x3ff460cbc7f5cf9a, 0x3ff446f86562d9fb, 0x3ff42d6625d51f87,
        0x3ff4141414141414, 0x3ff3fb013fb013fb, 0x3ff3e22cbce4a902, 0x3ff3c995a47babe7,
        0x3ff3b13b13b13b14, 0x3ff3991c2c187f63, 0x3ff3813813813814, 0x3ff3698df3de0748,
        0x3ff3521cfb2b78c1, 0x3ff33ae45b57bcb2, 0x3ff323e34a2b10bf, 0x3ff30d190130d190,
        0x3ff2f684bda12f68, 0x3ff2e025c04b8097, 0x3ff2c9fb4d812ca0, 0x3ff2b404ad012b40,
        0x3ff29e4129e4129e, 0x3ff288b01288b013, 0x3ff27350b8812735, 0x3ff25e22708092f1,
        0x3ff2492492492492, 0x3ff23456789abcdf, 0x3ff21fb78121fb78, 0x3ff20b470c67c0d9,
        0x3ff1f7047dc11f70, 0x3ff1e2ef3b3fb874, 0x3ff1cf06ada2811d, 0x3ff1bb4a4046ed29,
        0x3ff1a7b9611a7b96, 0x3ff19453808ca29c, 0x3ff1811811811812, 0x3ff16e0689427379,
        0x3ff15b1e5f75270d, 0x3ff1485f0e0acd3b, 0x3ff135c81135c811, 0x3ff12358e75d3033,
        0x3ff1111111111111, 0x3ff0fef010fef011, 0x3ff0ecf56be69c90, 0x3ff0db20a88f4696,
        0x3ff0c9714fbcda3b, 0x3ff0b7e6ec259dc8, 0x3ff0a6810a6810a7, 0x3ff0953f39010954,
        0x3ff0842108421084, 0x3ff073260a47f7c6, 0x3ff0624dd2f1a9fc, 0x3ff05197f7d73404,
        0x3ff0410410410410, 0x3ff03091b51f5e1a, 0x3ff0204081020408, 0x3ff0101010101010,
        0x3ff0000000000000, 0x3fefc07f01fc07f0, 0x3fef81f81f81f820, 0x3fef44659e4a4271,
        0x3fef07c1f07c1f08, 0x3feecc07b301ecc0, 0x3fee9131abf0b767, 0x3fee573ac901e574,
        0x3fee1e1e1e1e1e1e, 0x3fede5d6e3f8868a, 0x3fedae6076b981db, 0x3fed77b654b82c34,
        0x3fed41d41d41d41d, 0x3fed0cb58f6ec074, 0x3fecd85689039b0b, 0x3feca4b3055ee191,
        0x3fec71c71c71c71c, 0x3fec3f8f01c3f8f0, 0x3fec0e070381c0e0, 0x3febdd2b899406f7,
        0x3febacf914c1bad0, 0x3feb7d6c3dda338b, 0x3feb4e81b4e81b4f, 0x3feb2036406c80d9,
        0x3feaf286bca1af28, 0x3feac5701ac5701b, 0x3fea98ef606a63be, 0x3fea6d01a6d01a6d,
        0x3fea41a41a41a41a, 0x3fea16d3f97a4b02, 0x3fe9ec8e951033d9, 0x3fe9c2d14ee4a102,
        0x3fe999999999999a, 0x3fe970e4f80cb872, 0x3fe948b0fcd6e9e0, 0x3fe920fb49d0e229,
        0x3fe8f9c18f9c18fa, 0x3fe8d3018d3018d3, 0x3fe8acb90f6bf3aa, 0x3fe886e5f0abb04a,
        0x3fe8618618618618, 0x3fe83c977ab2bedd, 0x3fe8181818181818, 0x3fe7f405fd017f40,
        0x3fe7d05f417d05f4, 0x3fe7ad2208e0ecc3, 0x3fe78a4c8178a4c8, 0x3fe767dce434a9b1,
        0x3fe745d1745d1746, 0x3fe724287f46debc, 0x3fe702e05c0b8170, 0x3fe6e1f76b4337c7,
        0x3fe6c16c16c16c17, 0x3fe6a13cd1537290, 0x3fe6816816816817, 0x3fe661ec6a5122f9,
        0x3fe642c8590b2164, 0x3fe623fa77016240, 0x3fe6058160581606, 0x3fe5e75bb8d015e7,
        0x3fe5c9882b931057, 0x3fe5ac056b015ac0, 0x3fe58ed2308158ed, 0x3fe571ed3c506b3a,
        0x3fe5555555555555, 0x3fe5390948f40feb, 0x3fe51d07eae2f815, 0x3fe5015015015015,
        0x3fe4e5e0a72f0539, 0x3fe4cab88725af6e, 0x3fe4afd6a052bf5b, 0x3fe49539e3b2d067,
        0x3fe47ae147ae147b, 0x3fe460cbc7f5cf9a, 0x3fe446f86562d9fb, 0x3fe42d6625d51f87,
        0x3fe4141414141414, 0x3fe3fb013fb013fb, 0x3fe3e22cbce4a902, 0x3fe3c995a47babe7,
        0x3fe3b13b13b13b14, 0x3fe3991c2c187f63, 0x3fe3813813813814, 0x3fe3698df3de0748,
        0x3fe3521cfb2b78c1, 0x3fe33ae45b57bcb2, 0x3fe323e34a2b10bf, 0x3fe30d190130d190,
        0x3fe2f684bda12f68, 0x3fe2e025c04b8097, 0x3fe2c9fb4d812ca0, 0x3fe2b404ad012b40,
        0x3fe29e4129e4129e, 0x3fe288b01288b013, 0x3fe27350b8812735, 0x3fe25e22708092f1,
        0x3fe2492492492492, 0x3fe23456789abcdf, 0x3fe21fb78121fb78, 0x3fe20b470c67c0d9,
        0x3fe1f7047dc11f70, 0x3fe1e2ef3b3fb874, 0x3fe1cf06ada2811d, 0x3fe1bb4a4046ed29,
        0x3fe1a7b9611a7b96, 0x3fe19453808ca29c, 0x3fe1811811811812, 0x3fe16e0689427379,
        0x3fe15b1e5f75270d, 0x3fe1485f0e0acd3b, 0x3fe135c81135c811, 0x3fe12358e75d3033,
        0x3fe1111111111111, 0x3fe0fef010fef011, 0x3fe0ecf56be69c90, 0x3fe0db20a88f4696,
        0x3fe0c9714fbcda3b, 0x3fe0b7e6ec259dc8, 0x3fe0a6810a6810a7, 0x3fe0953f39010954,
        0x3fe0842108421084, 0x3fe073260a47f7c6, 0x3fe0624dd2f1a9fc, 0x3fe05197f7d73404,
        0x3fe0410410410410, 0x3fe03091b51f5e1a, 0x3fe0204081020408, 0x3fe0101010101010,
    };

    unsigned xb = bit_cast<unsigned, float>(x);
    uint64_t bval = masked_recp_table[(xb >> 16) & 0xff];
    return bit_cast<double, uint64_t>(bval);
}

// Compute the reciprocal log of \p x in the range [sqrt(2)/2 .. sqrt(2)].
double log_recp_of_masked(float x) {
    uint64_t masked_log_recp_table[256] = {
        0x3fe62e42fefa39ef, 0x3fe5ee82aa241920, 0x3fe5af405c3649e0,
        0x3fe5707a26bb8c66, 0x3fe5322e26867857, 0x3fe4f45a835a4e19,
        0x3fe4b6fd6f970c1f, 0x3fe47a1527e8a2d4, 0x3fe43d9ff2f923c5,
        0x3fe4019c2125ca93, 0x3fe3c6080c36bfb5, 0x3fe38ae2171976e8,
        0x3fe35028ad9d8c85, 0x3fe315da4434068b, 0x3fe2dbf557b0df43,
        0x3fe2a2786d0ec107, 0x3fe269621134db92, 0x3fe230b0d8bebc98,
        0x3fe1f8635fc61658, 0x3fe1c07849ae6007, 0x3fe188ee40f23ca7,
        0x3fe151c3f6f29612, 0x3fe11af823c75aa8, 0x3fe0e4898611cce1,
        0x3fe0ae76e2d054fa, 0x3fe078bf0533c568, 0x3fe04360be7603ae,
        0x3fe00e5ae5b207ab, 0x3fdfb358af7a4884, 0x3fdf4aa7ee03192e,
        0x3fdee2a156b413e5, 0x3fde7b42c3ddad74, 0x3fde148a1a2726cf,
        0x3fddae75484c9615, 0x3fdd490246defa6a, 0x3fdce42f18064744,
        0x3fdc7ff9c74554ca, 0x3fdc1c60693fa39e, 0x3fdbb9611b80e2fc,
        0x3fdb56fa0446290a, 0x3fdaf5295248cdcf, 0x3fda93ed3c8ad9e3,
        0x3fda33440224fa79, 0x3fd9d32bea15ed3a, 0x3fd973a3431356ae,
        0x3fd914a8635bf689, 0x3fd8b639a88b2df4, 0x3fd85855776dcbfb,
        0x3fd7fafa3bd8151c, 0x3fd79e26687cfb3e, 0x3fd741d876c67bb1,
        0x3fd6e60ee6af1973, 0x3fd68ac83e9c6a15, 0x3fd630030b3aac48,
        0x3fd5d5bddf595f31, 0x3fd57bf753c8d1fb, 0x3fd522ae0738a3d7,
        0x3fd4c9e09e172c3d, 0x3fd4718dc271c41c, 0x3fd419b423d5e8c6,
        0x3fd3c25277333183, 0x3fd36b6776be1116, 0x3fd314f1e1d35ce3,
        0x3fd2bef07cdc9355, 0x3fd269621134db91, 0x3fd214456d0eb8d5,
        0x3fd1bf99635a6b95, 0x3fd16b5ccbacfb73, 0x3fd1178e8227e47a,
        0x3fd0c42d676162e2, 0x3fd07138604d5864, 0x3fd01eae5626c691,
        0x3fcf991c6cb3b37a, 0x3fcef5ade4dcffe5, 0x3fce530effe71013,
        0x3fcdb13db0d48941, 0x3fcd1037f2655e7b, 0x3fcc6ffbc6f00f71,
        0x3fcbd087383bd8aa, 0x3fcb31d8575bce3b, 0x3fca93ed3c8ad9e5,
        0x3fc9f6c407089663, 0x3fc95a5adcf70182, 0x3fc8beafeb38fe8f,
        0x3fc823c16551a3c0, 0x3fc7898d85444c74, 0x3fc6f0128b756ab9,
        0x3fc6574ebe8c1339, 0x3fc5bf406b543db0, 0x3fc527e5e4a1b58d,
        0x3fc4913d8333b563, 0x3fc3fb45a59928ca, 0x3fc365fcb0159014,
        0x3fc2d1610c86813d, 0x3fc23d712a49c201, 0x3fc1aa2b7e23f729,
        0x3fc1178e8227e47a, 0x3fc08598b59e3a07, 0x3fbfe89139dbd565,
        0x3fbec739830a1126, 0x3fbda7276384469e, 0x3fbc885801bc4b20,
        0x3fbb6ac88dad5b1d, 0x3fba4e7640b1bc38, 0x3fb9335e5d594988,
        0x3fb8197e2f40e3f0, 0x3fb700d30aeac0e8, 0x3fb5e95a4d9791cd,
        0x3fb4d3115d207eac, 0x3fb3bdf5a7d1ee5e, 0x3fb2aa04a44717a1,
        0x3fb1973bd1465561, 0x3fb08598b59e3a06, 0x3faeea31c006b87c,
        0x3faccb73cdddb2d0, 0x3faaaef2d0fb1108, 0x3fa894aa149fb34b,
        0x3fa67c94f2d4bb65, 0x3fa466aed42de3f9, 0x3fa252f32f8d1840,
        0x3fa0415d89e74440, 0x3f9c63d2ec14aad7, 0x3f98492528c8cac5,
        0x3f9432a925980cbc, 0x3f90205658935837, 0x3f882448a388a283,
        0x3f8010157588de69, 0x3f70080559588b25, 0x0,
        0xbf7fe02a6b106799, 0xbf8fc0a8b0fc03c4, 0xbf97b91b07d5b126,
        0xbf9f829b0e7832f8, 0xbfa39e87b9febd68, 0xbfa77458f632dcff,
        0xbfab42dd711971b9, 0xbfaf0a30c01162a8, 0xbfb16536eea37ae3,
        0xbfb341d7961bd1d0, 0xbfb51b073f06183c, 0xbfb6f0d28ae56b4e,
        0xbfb8c345d6319b23, 0xbfba926d3a4ad562, 0xbfbc5e548f5bc743,
        0xbfbe27076e2af2ea, 0xbfbfec9131dbeabc, 0xbfc0d77e7cd08e5b,
        0xbfc1b72ad52f67a2, 0xbfc29552f81ff521, 0xbfc371fc201e8f75,
        0xbfc44d2b6ccb7d1c, 0xbfc526e5e3a1b438, 0xbfc5ff3070a793d6,
        0xbfc6d60fe719d21b, 0xbfc7ab890210d907, 0xbfc87fa06520c911,
        0xbfc9525a9cf456b6, 0xbfca23bc1fe2b561, 0xbfcaf3c94e80bff3,
        0xbfcbc286742d8cd4, 0xbfcc8ff7c79a9a20, 0xbfcd5c216b4fbb94,
        0xbfce27076e2af2e8, 0xbfcef0adcbdc5935, 0xbfcfb9186d5e3e29,
        0xbfd0402594b4d041, 0xbfd0a324e27390e2, 0xbfd1058bf9ae4ad4,
        0xbfd1675cababa60f, 0xbfd1c898c16999fb, 0xbfd22941fbcf7966,
        0xbfd2895a13de86a4, 0xbfd2e8e2bae11d31, 0xbfd347dd9a987d56,
        0xbfd3a64c556945ea, 0xbfd404308686a7e4, 0xbfd4618bc21c5ec2,
        0xbfd4be5f957778a1, 0xbfd51aad872df82e, 0xbfd5767717455a6c,
        0xbfd5d1bdbf5809ca, 0xbfd62c82f2b9c796, 0xbfd686c81e9b14ad,
        0xbfd6e08eaa2ba1e4, 0xbfd739d7f6bbd007, 0xbfd792a55fdd47a1,
        0xbfd7eaf83b82afc2, 0xbfd842d1da1e8b18, 0xbfd89a3386c1425b,
        0xbfd8f11e873662c8, 0xbfd947941c2116fb, 0xbfd99d958117e08a,
        0xbfd9f323ecbf984d, 0xbfda484090e5bb09, 0xbfda9cec9a9a084a,
        0xbfdaf1293247786b, 0xbfdb44f77bcc8f64, 0xbfdb9858969310fd,
        0xbfdbeb4d9da71b7a, 0xbfdc3dd7a7cdad4d, 0xbfdc8ff7c79a9a21,
        0xbfdce1af0b85f3ec, 0xbfdd32fe7e00ebd5, 0xbfdd83e7258a2f3e,
        0xbfddd46a04c1c4a1, 0xbfde24881a7c6c26, 0xbfde744261d68789,
        0xbfdec399d2468cc1, 0xbfdf128f5faf06ec, 0xbfdf6123fa7028ad,
        0xbfdfaf588f78f31d, 0xbfdffd2e0857f497, 0xbfe02552a5a5d0ff,
        0xbfe04bdf9da926d2, 0xbfe0723e5c1cdf41, 0xbfe0986f4f573521,
        0xbfe0be72e4252a83, 0xbfe0e44985d1cc8c, 0xbfe109f39e2d4c96,
        0xbfe12f719593efbd, 0xbfe154c3d2f4d5ea, 0xbfe179eabbd899a0,
        0xbfe19ee6b467c96f, 0xbfe1c3b81f713c25, 0xbfe1e85f5e7040d1,
        0xbfe20cdcd192ab6e, 0xbfe23130d7bebf43, 0xbfe2555bce98f7ca,
        0xbfe2795e1289b11b, 0xbfe29d37fec2b08b, 0xbfe2c0e9ed448e8c,
        0xbfe2e47436e40268, 0xbfe307d7334f10be, 0xbfe32b1339121d71,
        0xbfe34e289d9ce1d2, 0xbfe37117b54747b6, 0xbfe393e0d3562a1a,
        0xbfe3b68449fffc23, 0xbfe3d9026a7156fb, 0xbfe3fb5b84d16f43,
        0xbfe41d8fe84672af, 0xbfe43f9fe2f9ce67, 0xbfe4618bc21c5ec2,
        0xbfe48353d1ea88df, 0xbfe4a4f85db03ebb, 0xbfe4c679afccee39,
        0xbfe4e7d811b75bb0, 0xbfe50913cc01686b, 0xbfe52a2d265bc5ab,
        0xbfe54b2467999498, 0xbfe56bf9d5b3f399, 0xbfe58cadb5cd7989,
        0xbfe5ad404c359f2d, 0xbfe5cdb1dc6c1765, 0xbfe5ee02a9241676,
        0xbfe60e32f44788d9,
    };

    unsigned xb = bit_cast<unsigned, float>(x);
    uint64_t bval = masked_log_recp_table[(xb >> 16) & 0xff];
    return bit_cast<double, uint64_t>(bval);
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
    } else if (x < 0) {
        return bit_cast<float, unsigned>(0xffc00000); // -Nan.
    } else if (is_nan(x)) {
        return x;
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

    // Compute the reciprocal of m using a lookup table.
    double ri = recip_of_masked(m);
    double z = m * ri - 1;
    double log2 = bit_cast<double, uint64_t>(0x3fe62e42fefa39ef);

    // We use double here because float is not accurate enough for the final
    // reduction. We are missing just a few bits.

    // Compute log(1/ri) using a lookup table.
    double ln_ri = log_recp_of_masked(m);
    // Approximate log(1+z) using a polynomial:
    double ln_1z = approximate_log_pol_1_to_1001(z);

    // Perform the final reduction.
    return (E * log2 + ln_1z) - ln_ri;
}

// Wrap the standard log(double) and use it as the ground truth.
float accurate_log(float x) { return log((double)x); }

// Wrap the standard log(double) and use it as the ground truth.
float libc_log(float x) { return logf(x); }

int main(int argc, char **argv) {
    //print_recp_table_for_3f_values();
    //print_log_recp_table_for_3f_values();
    print_ulp_deltas(my_log, accurate_log);
}
