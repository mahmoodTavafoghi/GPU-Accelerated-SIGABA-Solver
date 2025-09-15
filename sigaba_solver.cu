// ============================================================================
// SIGABA CUDA Solver 
// Author: Mahmood Tavafoghi
//
//  pipeline :
//  - Phase 1 (DP + Backtrack, steps 0..20):
//      Build dynamic-programming (DP) structures for the initial window (step 0 to step 20),
//      perform backtracking to extract paths that match the CRIB and  according to filters discover CTLs for 
//      every survived paths.
//  - Phase 2 (Device extension, extend to 65 steps):
//      Extend Phase-1 candidate paths on the device from 20 up to 65 steps.
//      At each extension step, apply filtering based on the CTL values discovered in Phase-1.
//     
//  - Phase 3 (Final verification):
//      Collect unique triples (CPH, CPHP, CTL) and run a final verification kernel that simulates
//      the full cipher across the remaining index space, emitting any full solutions found.
//
//    for compile :
//    nvcc -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp --extended-lambda -O3 -std=c++17 sigaba_solver_pr_03.cu -o sigaba_solver

//  for run:
//  ./sigaba_solver 0 38 916
//  the answer is in partition 38 ,and cph index = 916
 
//  or run this for search all cph in partition:
//  ./sigaba_solver 0 38 -1


#include <malloc.h>
#include <iostream>
#include <string>
#include <unordered_set>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cstdint>
#include <cassert>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <map>
#include <functional>
#include <parallel/algorithm>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <iomanip>




#define MAX_CRIB_LEN 65 // absolute maximum we support in this build
#define CRIB_LEN_P1 20  // phase-1 length

#define CHUNK_P1 8000000
#define MAX_ANSWERS_phaze1 20'000'000
#define MAX_ANSWERS_phaze2 2'000'000
#define MAX_SURVIVORS 20'000'000

#define CPH_LEN 10
#define CPH_P_LEN 5
#define TOTAL_CPH 1920
#define TOTAL_CTL 1920
#define CTL_LEN 10
#define ALL_INDEX_NUM 113400

// Phase 1 grid sizes 
#define NUM_POS 11881376      // 26^5
#define GRID_X (26 * 26 * 26) // 17,576
#define BLOCK_X 26
#define BLOCK_Y 26
#define DP_SIZE (CRIB_LEN_P1 * NUM_POS)
#define d_ag_masks_SLOTS CRIB_LEN_P1


#define CUDA_CHECK(expr)                                                                          \
    do                                                                                            \
    {                                                                                             \
        cudaError_t e = (expr);                                                                   \
        if (e != cudaSuccess)                                                                     \
        {                                                                                         \
            fprintf(stderr, "%s:%d: CUDA error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)


// Device Constants and Functions

// GPU-side compact entry
struct GpuEntry
{
    // Always allocate for MAX_CRIB_LEN.
    static constexpr int BITSEQ_BITS = (MAX_CRIB_LEN - 1) * 5;
    static constexpr int BITSEQ_WORDS = (BITSEQ_BITS + 31) / 32;

    // Packed bit sequence across the whole run (we only fill used prefix):
    uint32_t bit_seq[BITSEQ_WORDS]; // (MAX_CRIB_LEN-1) * 5 packed as bits

    char cph[CPH_LEN + 1];
    char cphP[CPH_P_LEN + 1];

    int ctl_id; // which CTL matched this entry (-1 before filtering)
};

// Comparator for Thrust sort: (cph, then cphP)
struct GpuEntryLess
{
    __host__ __device__ bool operator()(GpuEntry const &a, GpuEntry const &b) const
    {
        // Compare cph
        for (int i = 0; i < CPH_LEN; ++i)
        {
            if (a.cph[i] < b.cph[i])
                return true;
            if (a.cph[i] > b.cph[i])
                return false;
        }
        // Compare cphP
        for (int i = 0; i < CPH_P_LEN; ++i)
        {
            if (a.cphP[i] < b.cphP[i])
                return true;
            if (a.cphP[i] > b.cphP[i])
                return false;
        }
        return false;
    }
};

// for comma seprator in results 
struct comma_numpunct : std::numpunct<char>
{
protected:
    char do_thousands_sep() const override { return ','; }    // separator
    std::string do_grouping() const override { return "\3"; } // group by 3
};


__device__ double LASRY_LOWER;
__device__ double LASRY_UPPER;
__device__ char d_idx_idxp[ALL_INDEX_NUM * 10];
__device__ uint8_t *g_index_map;


static const char d_ROTOR_WIRINGS[10][27] = {
    "RZLVXBWPMEFCSUQJDOGYTHNAIK",
    "IWOTAKRFHJLDQGBCXUPZSEYMVN",
    "QAZURDIYJLVOTSGBFNWCEKPXMH",
    "EYVUQBWGAMPSTXZJLFKRDICONH",
    "XVKMZFTLBIUWECORHPQAGYSDJN",
    "ZIWUORVLTKNYJGMEFDBHCASXQP",
    "IYLGOXEWVURZCTQJSNHFPMKBDA",
    "YRTDWFCNPUOHJGIEZQVBSAXKML",
    "LCYXJHUFWMKTVDBGAQZOPSIENR",
    "WQBNXOLIZEJDSAVFYRUHGTMPCK"};

__constant__ char d_ROTOR_WIRINGS_P2[10][27] = {
    "RZLVXBWPMEFCSUQJDOGYTHNAIK",
    "IWOTAKRFHJLDQGBCXUPZSEYMVN",
    "QAZURDIYJLVOTSGBFNWCEKPXMH",
    "EYVUQBWGAMPSTXZJLFKRDICONH",
    "XVKMZFTLBIUWECORHPQAGYSDJN",
    "ZIWUORVLTKNYJGMEFDBHCASXQP",
    "IYLGOXEWVURZCTQJSNHFPMKBDA",
    "YRTDWFCNPUOHJGIEZQVBSAXKML",
    "LCYXJHUFWMKTVDBGAQZOPSIENR",
    "WQBNXOLIZEJDSAVFYRUHGTMPCK"};

static const int HOST_INDEX_WIRINGS[5][10] = {
    {1, 3, 9, 6, 8, 5, 0, 2, 4, 7},
    {8, 3, 2, 4, 6, 9, 1, 0, 7, 5},
    {5, 9, 7, 4, 6, 1, 3, 0, 2, 8},
    {6, 1, 7, 5, 0, 3, 9, 4, 2, 8},
    {5, 4, 2, 9, 8, 6, 3, 0, 7, 1}};

__constant__ int INDEX_OUT[10] = {1, 5, 5, 4, 4, 3, 3, 2, 2, 1};
__constant__ int INDEX_IN[26] = {9, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8};

__constant__ int d_ROTOR_WIRINGS_NORMAL[10][26];
__constant__ int d_ROTOR_WIRINGS_REVERSE[10][26];
__constant__ bool d_LEGAL[30 * 5] = {
    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0,
    0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
    0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0,
    1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
    1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};

// helpers functions=========================================================

// Read a single bit from packed bit_seq
__device__ inline int get_packed_bit(const uint32_t *words, int bit_idx)
{
    int w = bit_idx >> 5;
    int o = bit_idx & 31;
    return (words[w] >> o) & 1u;
}

// Write (set/clear) a single bit in packed bit_seq
__device__ inline void set_packed_bit(uint32_t *words, int bit_idx, int value)
{
    int w = bit_idx >> 5;
    int o = bit_idx & 31;
    uint32_t mask = (1u << o);
    if (value)
        words[w] |= mask;
    else
        words[w] &= ~mask;
}

// Convert a LEGAL 5-bit move pattern (bools) to a 5-bit integer [b0..b4] -> bits 4..0
// (same encoding used in process_packed_bit_sequences_kernel)
__device__ inline int legal5_to_u5(int move_idx)
{
    int base = move_idx * 5;
    int b0 = d_LEGAL[base + 0] ? 1 : 0;
    int b1 = d_LEGAL[base + 1] ? 1 : 0;
    int b2 = d_LEGAL[base + 2] ? 1 : 0;
    int b3 = d_LEGAL[base + 3] ? 1 : 0;
    int b4 = d_LEGAL[base + 4] ? 1 : 0;
    // pack b0..b4 to a 5-bit integer in the same order as process_packed_bit_sequences_kernel
    int val = 0;
    val = (val << 1) | b0;
    val = (val << 1) | b1;
    val = (val << 1) | b2;
    val = (val << 1) | b3;
    val = (val << 1) | b4;
    return val; // [0..31]
}

// Count ones in a LEGAL move (how many rotors advance)
__device__ inline int move_ones(int move_idx)
{
    int base = move_idx * 5;
    int ones = 0;
#pragma unroll
    for (int r = 0; r < 5; ++r)
        if (d_LEGAL[base + r])
            ++ones;
    return ones;
}


//compute Lasry_score , did not used in this approach but using as constructor for testing and etc...
// lasry_score was checked in backtrack kernel phaze 1 at the end , for now I comment it out.
__device__ inline float compute_lasry_score(const int *move_freq, int num_moves)
{
    // Accumulate rotor‐counts
    if (num_moves == 0)
        return 0.0f;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;

    for (int m = 0; m < 30; ++m)
    {

        // ) Build rotor counts if freq > 0
        int f = move_freq[m];
        if (f == 0)
            continue;

        int base = m * 5;
        if (d_LEGAL[base + 0])
            c0 += f;
        if (d_LEGAL[base + 1])
            c1 += f;
        if (d_LEGAL[base + 2])
            c2 += f;
        if (d_LEGAL[base + 3])
            c3 += f;
        if (d_LEGAL[base + 4])
            c4 += f;
    }

    // 3) Compute LASRY score = sum_r cr * ln(cr)
    float denom = float(num_moves) - 1;
    float score = 0.0;
    if (c0 > 0)
    {
        float cr = float(c0) / denom;
        score += cr * log(cr);
    }
    if (c1 > 0)
    {
        float cr = float(c1) / denom;
        score += cr * log(cr);
    }
    if (c2 > 0)
    {
        float cr = float(c2) / denom;
        score += cr * log(cr);
    }
    if (c3 > 0)
    {
        float cr = float(c3) / denom;
        score += cr * log(cr);
    }
    if (c4 > 0)
    {
        float cr = float(c4) / denom;
        score += cr * log(cr);
    }

    return score;
}

// for rotor data structure and rotor move ments and encryption , I used 2 method and functions, the second one is faster , but 
// i keep both of them , not big deal but we can make them in one form.the first version was used just in phaze 1 .

__device__ inline void advance_rotor_dev_p1(bool reversed, int &pos)
{
    pos = reversed ? (pos + 1) % 26 : (pos + 25) % 26;
}

__device__ inline int rotor_left_to_right_dev_p1(const int *wiring0, const int *wiring1, bool reversed, int pos, int v)
{
    if (!reversed)
        return (wiring0[(v + pos) % 26] - pos + 26) % 26;
    else
        return (pos - wiring1[(pos - v + 26) % 26] + 26) % 26;
}

__device__ inline void int_to_pos(int pos, int p[5])
{
    for (int r = 0; r < 5; ++r)
    {
        p[r] = pos % 26;
        pos /= 26;
    }
}

__device__ inline int pos_to_int(const int p[5])
{
    int pos = 0, base = 1;
    for (int r = 0; r < 5; ++r)
    {
        pos += p[r] * base;
        base *= 26;
    }
    return pos;
}

__device__ inline int encrypt_dev_p1(const int *forward, const int *inverse, const bool *reversed, const int p[5], int v)
{
    for (int r = 0; r < 5; ++r)
    {
        v = rotor_left_to_right_dev_p1(forward + r * 26, inverse + r * 26, reversed[r], p[r], v);
    }
    return v;
}

__device__ inline int rotor_l2r_dev(int wi, bool reversed, int pos, int v)
{
    if (!reversed)
    {
        const int *wiring = d_ROTOR_WIRINGS_NORMAL[wi];
        int idx = (v + pos) % 26;
        return (wiring[idx] - pos + 26) % 26;
    }
    else
    {
        const int *wiring = d_ROTOR_WIRINGS_REVERSE[wi];
        int idx = (pos - v + 26) % 26;
        return (pos - wiring[idx] + 26) % 26;
    }
}

__device__ inline int rotor_r2l_dev(int wi, bool reversed, int pos, int v)
{
    if (!reversed)
    {
        const int *wiring = d_ROTOR_WIRINGS_REVERSE[wi];
        int idx = (v + pos) % 26;
        return (wiring[idx] - pos + 26) % 26;
    }
    else
    {
        const int *wiring = d_ROTOR_WIRINGS_NORMAL[wi];
        int idx = (pos - v + 26) % 26;
        return (pos - wiring[idx] + 26) % 26;
    }
}

// generate all ctls -- in military key sheets they just used 2 reverse rotors , so the cases reduced to 1920
// it took 5 digits that represent the partition for example input ="24579"
std::vector<std::string> generate_all_ctl(const std::string &input)
{
    std::vector<std::string> ctls;
    ctls.reserve(TOTAL_CTL);
    std::unordered_set<char> used(input.begin(), input.end());
    std::vector<int> avail;
    avail.reserve(5);
    for (int d = 0; d < 10; ++d)
        if (!used.count(char('0' + d)))
            avail.push_back(d);

    int w[5];
    for (int a = 0; a < 5; ++a)
    {
        w[0] = avail[a];
        for (int b = 0; b < 5; ++b)
            if (b != a)
            {
                w[1] = avail[b];
                for (int c = 0; c < 5; ++c)
                    if (c != a && c != b)
                    {
                        w[2] = avail[c];
                        for (int d = 0; d < 5; ++d)
                            if (d != a && d != b && d != c)
                            {
                                w[3] = avail[d];
                                for (int e = 0; e < 5; ++e)
                                    if (e != a && e != b && c != e && d != e)
                                    {
                                        w[4] = avail[e];
                                        for (int mask = 0; mask < 32; ++mask)
                                        {
                                            if (__builtin_popcount(mask) > 2)  // here we filter more than 2 reversed rotors
                                                continue;
                                            std::string s;
                                            s.reserve(10);
                                            for (int i = 0; i < 5; ++i)
                                            {
                                                s.push_back(char('0' + w[i]));
                                                s.push_back((mask & (1 << i)) ? 'R' : 'N');
                                            }
                                            ctls.push_back(std::move(s));
                                        }
                                    }
                            }
                    }
            }
    }
    return ctls;
}

std::vector<std::string> generate_all_cph(const std::string &input)
{
    std::vector<std::string> cphs;
    cphs.reserve(TOTAL_CPH);
    std::unordered_set<char> used(input.begin(), input.end());
    std::vector<int> avail;
    avail.reserve(5);
    for (int d = 0; d < 10; ++d)
        if (used.count(char('0' + d)))
            avail.push_back(d);

    int w[5];
    for (int a = 0; a < 5; ++a)
    {
        w[0] = avail[a];
        for (int b = 0; b < 5; ++b)
            if (b != a)
            {
                w[1] = avail[b];
                for (int c = 0; c < 5; ++c)
                    if (c != a && c != b)
                    {
                        w[2] = avail[c];
                        for (int d = 0; d < 5; ++d)
                            if (d != a && d != b && d != c)
                            {
                                w[3] = avail[d];
                                for (int e = 0; e < 5; ++e)
                                    if (e != a && e != b && c != e && d != e)
                                    {
                                        w[4] = avail[e];
                                        for (int mask = 0; mask < 32; ++mask)
                                        {
                                            if (__builtin_popcount(mask) > 2)
                                                continue;
                                            std::string s;
                                            s.reserve(10);
                                            for (int i = 0; i < 5; ++i)
                                            {
                                                s.push_back(char('0' + w[i]));
                                                s.push_back((mask & (1 << i)) ? 'R' : 'N');
                                            }
                                            cphs.push_back(std::move(s));
                                        }
                                    }
                            }
                    }
            }
    }
    return cphs;
}

// Mark group starts (1 if i==0 or cphP[i] != cphP[i-1], else 0)
__global__ void mark_group_starts(const GpuEntry *arr, int n, int *flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (i == 0)
    {
        flags[0] = 1;
        return;
    }
    bool diff = false;

    for (int k = 0; k < CPH_P_LEN; ++k)
    {
        if (arr[i].cphP[k] != arr[i - 1].cphP[k])
        {
            diff = true;
            break;
        }
    }
    flags[i] = diff ? 1 : 0;
}


//  dp_fill_kernel now takes end_len (crib length for this pass) =====
//  Builds the dynamic programming table for the initial window used by backtracking.

__global__ void dp_fill_kernel(
    const char *d_cph, const char *d_crib, const char *d_cipher,
    const int *d_forward, const int *d_inverse, const bool *d_reversed, const int *d_delta,
    uint32_t *d_dp, int step, int end_len, int base_step)
{
    int x = blockIdx.x, y = threadIdx.x, z = threadIdx.y;
    int p0 = x % 26, p1 = (x / 26) % 26, p2 = (x / (26 * 26)) % 26, p3 = y, p4 = z;
    int pos = p0 + p1 * 26 + p2 * 26 * 26 + p3 * 26 * 26 * 26 + p4 * 26 * 26 * 26 * 26;
    if (pos >= NUM_POS)
        return;

    int p[5] = {p0, p1, p2, p3, p4};

    // Preserve your phase-1 initial constraints only when step==0 for that pass.
    if (step == 0)
    {
        if (p[0] == 25 || p[1] == 25 || p[2] == 25 || p[3] == 25 || p[4] == 25)  // Because in military formats theyy did not used "Z" 
            return;
        if ((p[0] == p[1] && p[1] == p[2]) || (p[1] == p[2] && p[2] == p[3]) || (p[2] == p[3] && p[3] == p[4]))  //  we did not expect key like "DFBBB"
            return;

            // this way they filled with zero and during BK we would not check them.
    }

    // Map step index to envelope used during encryption check (unchanged math).

    for (int r = 0; r < 5; ++r)
    {
        int adjust = (end_len - 1 - step) * d_delta[r];
        int pos_r = (p[r] + adjust) % 26;
        if (pos_r < 0)
            pos_r += 26;
    }

    int v = d_crib[step] - 'A';
    v = encrypt_dev_p1(d_forward, d_inverse, d_reversed, p, v);
    if (v != (d_cipher[step] - 'A'))
        return;

    if (step == end_len - 1)
    {
        d_dp[(step - base_step) * NUM_POS + pos] = 1;
    }
    else
    {
        uint32_t bitmask = 0;
        for (int m = 0; m < 30; ++m)
        {
            int next_p[5];
            for (int r = 0; r < 5; ++r)
            {
                next_p[r] = d_LEGAL[m * 5 + r] ? (p[r] + d_delta[r] + 26) % 26 : p[r];
            }
            int next_pos = pos_to_int(next_p);
            if (d_dp[(step + 1 - base_step) * NUM_POS + next_pos] != 0)
                bitmask |= (1 << m);
        }
        if (bitmask != 0)
            d_dp[(step - base_step) * NUM_POS + pos] = bitmask;
    }
}

//  dp_backtrack_kernel now takes end_len and packs only used prefix =====
//  Performs backtracking through the DP table to produce candidate move sequences

__global__ void dp_backtrack_kernel(
    const char *__restrict__ d_cph,
    const char *__restrict__ d_crib,
    const char *__restrict__ d_cipher,
    const int *__restrict__ d_forward,
    const int *__restrict__ d_inverse,
    const bool *__restrict__ d_reversed,
    const int *__restrict__ d_delta,
    uint32_t *__restrict__ d_dp,
    GpuEntry *__restrict__ d_out,
    int *__restrict__ d_answer_count,
    int end_len) // crib length for this run (Phase 1 uses 20)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_POS)
        return;

    int dp_idx0 = 0 * NUM_POS + tid;
    if (dp_idx0 < 0 || dp_idx0 >= DP_SIZE)
        return;

    uint32_t startMask = d_dp[0 * NUM_POS + tid];
    if (!startMask)
        return;

    const int steps = end_len - 1; // transitions
    int move_freq[30] = {0};
    int move_stack[MAX_CRIB_LEN]; // enough space
    int move_count = 0;

    struct Frame
    {
        int step, pos;
        uint32_t mask;
    };
    Frame stack[MAX_CRIB_LEN + 2];
    int depth = 0;
    stack[depth++] = {0, tid, startMask};

    while (depth > 0)
    {
        Frame &top = stack[depth - 1];
        int step = top.step;
        int pos = top.pos;
        uint32_t bm = top.mask;

        if (step == end_len - 1)
        {
            float lasry_score = compute_lasry_score(move_freq, steps);
            if (lasry_score >= LASRY_LOWER)
            

            {
                int slot = atomicAdd(d_answer_count, 1);
                if (slot < MAX_ANSWERS_phaze1)
                {
                    GpuEntry &e = d_out[slot];

                    for (int w = 0; w < GpuEntry::BITSEQ_WORDS; ++w)
                        e.bit_seq[w] = 0u;

                    // pack only first `steps`*5 bits
                    for (int i = 0; i < steps; ++i)
                    {
                        int m = move_stack[i];
                        int base = m * 5;
                        bool b0 = d_LEGAL[base + 0];
                        bool b1 = d_LEGAL[base + 1];
                        bool b2 = d_LEGAL[base + 2];
                        bool b3 = d_LEGAL[base + 3];
                        bool b4 = d_LEGAL[base + 4];

                        int bit_idx = i * 5;
                        int w0 = (bit_idx + 0) >> 5;
                        int o0 = (bit_idx + 0) & 31;
                        if (b0)
                            e.bit_seq[w0] |= (1u << o0);
                        int w1 = (bit_idx + 1) >> 5;
                        int o1 = (bit_idx + 1) & 31;
                        if (b1)
                            e.bit_seq[w1] |= (1u << o1);
                        int w2 = (bit_idx + 2) >> 5;
                        int o2 = (bit_idx + 2) & 31;
                        if (b2)
                            e.bit_seq[w2] |= (1u << o2);
                        int w3 = (bit_idx + 3) >> 5;
                        int o3 = (bit_idx + 3) & 31;
                        if (b3)
                            e.bit_seq[w3] |= (1u << o3);
                        int w4 = (bit_idx + 4) >> 5;
                        int o4 = (bit_idx + 4) & 31;
                        if (b4)
                            e.bit_seq[w4] |= (1u << o4);
                    }

                    for (int i = 0; i < CPH_LEN; ++i)
                        e.cph[i] = d_cph[i];

                    // cphP from initial position (stack[0].pos)
                    int tmp = stack[0].pos;
                    e.cphP[0] = 'A' + (tmp % 26);
                    tmp /= 26;
                    e.cphP[1] = 'A' + (tmp % 26);
                    tmp /= 26;
                    e.cphP[2] = 'A' + (tmp % 26);
                    tmp /= 26;
                    e.cphP[3] = 'A' + (tmp % 26);
                    tmp /= 26;
                    e.cphP[4] = 'A' + (tmp % 26);

                    e.ctl_id = -1;
                }
            }
            // pop
            depth--;
            if (move_count > 0)
            {
                int last = move_stack[--move_count];
                move_freq[last]--;
            }
            continue;
        }

        if (bm == 0)
        {
            depth--;
            if (move_count > 0)
            {
                int last = move_stack[--move_count];
                move_freq[last]--;
            }
            continue;
        }

        int m = __ffs(bm) - 1;
        top.mask = bm & (bm - 1);

        // decode pos -> p0..p4
        int tmp = pos;
        int p0 = tmp % 26;
        tmp /= 26;
        int p1 = tmp % 26;
        tmp /= 26;
        int p2 = tmp % 26;
        tmp /= 26;
        int p3 = tmp % 26;
        tmp /= 26;
        int p4 = tmp % 26;

        int np0 = d_LEGAL[m * 5 + 0] ? (p0 + d_delta[0] + 26) % 26 : p0;
        int np1 = d_LEGAL[m * 5 + 1] ? (p1 + d_delta[1] + 26) % 26 : p1;
        int np2 = d_LEGAL[m * 5 + 2] ? (p2 + d_delta[2] + 26) % 26 : p2;
        int np3 = d_LEGAL[m * 5 + 3] ? (p3 + d_delta[3] + 26) % 26 : p3;
        int np4 = d_LEGAL[m * 5 + 4] ? (p4 + d_delta[4] + 26) % 26 : p4;

        int next_pos = np0 + np1 * 26 + np2 * 26 * 26 + np3 * 26 * 26 * 26 + np4 * 26 * 26 * 26 * 26;
        int dp_idx = (step + 1) * NUM_POS + next_pos;
        uint32_t childMask = d_dp[dp_idx];
        if (!childMask)
            continue;

        int tf = move_freq[m] + 1;
        move_stack[move_count++] = m;
        move_freq[m] = tf;

        stack[depth++] = {step + 1, next_pos, childMask};
    }
}

// Phase-2 backtrack: extend P1 survivors to 65, joining at step 19 (after 19 transitions)
__global__ void dp_backtrack_kernel_p2(
    // fixed wiring/crib/cipher context for this CPH
    const char *__restrict__ d_cph,
    const char *__restrict__ d_crib,   // length MAX_CRIB_LEN 
    const char *__restrict__ d_cipher, // length MAX_CRIB_LEN 
    const int *__restrict__ d_forward,
    const int *__restrict__ d_inverse,
    const bool *__restrict__ d_reversed,
    const int *__restrict__ d_delta,
    // DP table for the 65-length pass (we filled rows 19..64)
    const uint32_t *__restrict__ d_dp_p2,
    // CTLs table (TOTAL_CTL * CTL_LEN) to read ctl_id per survivor
    const char *__restrict__ d_ctls,
    // input: Phase-1 survivors (ctl_id already set by filter kernel)
    const GpuEntry *__restrict__ d_p1_survivors,
    int num_survivors,
    int start_step_p2,
    int end_len_p2,
    // output: Phase-2 survivors (extended sequences)
    GpuEntry *__restrict__ d_out_p2,
    int *__restrict__ d_out_p2_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_survivors)
        return;

    // constants and indices for a 65-char pass
    const int end_len = end_len_p2;          // 65 characters
    const int terminal_step = end_len - 1;   // 64 (no transition at 39 -> emission)
    const int last_transition = end_len - 2; // 63 (last transition index)
    const int start_step = start_step_p2;    // 19 (join point)
    const int steps_p1 = CRIB_LEN_P1 - 1;    // 19 transitions in phase-1

    // take this survivor
    const GpuEntry s1 = d_p1_survivors[idx];

    // decode initial cphP -> starting position at step 0
    int p_init[5];
    p_init[0] = s1.cphP[0] - 'A';
    p_init[1] = s1.cphP[1] - 'A';
    p_init[2] = s1.cphP[2] - 'A';
    p_init[3] = s1.cphP[3] - 'A';
    p_init[4] = s1.cphP[4] - 'A';

    // replay EXACTLY the 19 P1 transitions (indices 0..18) to land at position of step 19
    int p[5] = {p_init[0], p_init[1], p_init[2], p_init[3], p_init[4]};
#pragma unroll
    for (int step = 0; step < steps_p1; ++step)
    { // 0..18 inclusive
        const int base_bit = step * 5;
#pragma unroll
        for (int r = 0; r < 5; ++r)
        {
            int b = get_packed_bit(s1.bit_seq, base_bit + r);
            if (b)
            {
                int pr = (p[r] + d_delta[r]) % 26;
                if (pr < 0)
                    pr += 26;
                p[r] = pr;
            }
        }
    }
    // p[] is now rotor position at step 19 (the join point)
    int pos19 = p[0] + p[1] * 26 + p[2] * 26 * 26 + p[3] * 26 * 26 * 26 + p[4] * 26 * 26 * 26 * 26;

    // fetch starting DP mask at step 19 for this position
    const int base_step = start_step; // join point
    uint32_t bm0 = d_dp_p2[(start_step - base_step) * NUM_POS + pos19];
    if (!bm0)
        return; // cannot extend this survivor

    // decode CTL for this survivor to build active-pin masks for positions 19..65
    int ctl_idx = s1.ctl_id;
    if (ctl_idx < 0 || ctl_idx >= TOTAL_CTL)
        return; // safety

    const char *ctl = d_ctls + ctl_idx * CTL_LEN;
    int control_wi[5];
    bool control_rev[5];
    int control_pos[5]; // initialized from cphP as well
#pragma unroll
    for (int r = 0; r < 5; ++r)
    {
        control_wi[r] = ctl[r * 2] - '0';
        control_rev[r] = (ctl[r * 2 + 1] == 'R');
        control_pos[r] = s1.cphP[r] - 'A';
    }

    // advance CTL position from 0 up to 19, and then compute active masks for 19..64
    int active_mask_local[MAX_CRIB_LEN] = {0}; // mask at *position* index
    int ac_local[MAX_CRIB_LEN] = {0};          // popcount(active_mask) for transitions (19..63)
    {
        int cp[5] = {control_pos[0], control_pos[1], control_pos[2], control_pos[3], control_pos[4]};
        // bring CTL to position 19 (apply 19 ticks)
        for (int i = 0; i < start_step; ++i)
        {
            if (cp[2] == 14)
            {
                if (cp[3] == 14)
                    advance_rotor_dev_p1(control_rev[1], cp[1]);
                advance_rotor_dev_p1(control_rev[3], cp[3]);
            }
            advance_rotor_dev_p1(control_rev[2], cp[2]);
        }
        // compute masks for positions 19..64
        for (int pos_step = start_step; pos_step <= terminal_step; ++pos_step)
        {
            bool move_ctrl[10] = {false};
            for (int pin = 5; pin <= 8; ++pin)
            {
                int v = pin;
#pragma unroll
                for (int r = 4; r >= 0; --r)
                    v = rotor_r2l_dev(control_wi[r], control_rev[r], cp[r], v);
                move_ctrl[INDEX_IN[v]] = true;
            }
            int mask = 0;
#pragma unroll
            for (int pin = 0; pin < 10; ++pin)
                if (move_ctrl[pin])
                    mask |= (1 << pin);
            active_mask_local[pos_step] = mask;
            if (pos_step <= last_transition)
                ac_local[pos_step] = __popc(mask);

            // tick CTL to next position
            if (cp[2] == 14)
            {
                if (cp[3] == 14)
                    advance_rotor_dev_p1(control_rev[1], cp[1]);
                advance_rotor_dev_p1(control_rev[3], cp[3]);
            }
            advance_rotor_dev_p1(control_rev[2], cp[2]);
        }
    }

    // build equal-active-mask groups over transitions 19..63 (use position masks at same indices)
    uint64_t ag_masks[d_ag_masks_SLOTS];
    int num_groups = 0;
    bool visited[MAX_CRIB_LEN] = {false};
    for (int i = start_step; i <= last_transition && num_groups < d_ag_masks_SLOTS; ++i)
    {
        if (visited[i])
            continue;
        uint64_t msk = (1ULL << i);
        visited[i] = true;
        for (int j = i + 1; j <= last_transition; ++j)
        {
            if (!visited[j] && active_mask_local[i] == active_mask_local[j])
            {
                visited[j] = true;
                msk |= (1ULL << j);
            }
        }
        if (__popcll(msk) > 1)
            ag_masks[num_groups++] = msk;
    }

    // accumulator entry: start with the P1 packed bits copied through index 0..(19*5-1)
    GpuEntry acc = s1; // copy cph, cphP, ctl_id, and bit_seq prefix

    // DFS stack starting at step 19 and position pos19
    struct Frame
    {
        int step;
        int pos;
        uint32_t mask;
    };
    Frame stack_local[MAX_CRIB_LEN + 2];
    int depth = 0;
    stack_local[depth++] = {start_step, pos19, bm0};

    // equal-group “first value” tracking
    int group_value[d_ag_masks_SLOTS];
    int group_depth[d_ag_masks_SLOTS];
#pragma unroll
    for (int g = 0; g < d_ag_masks_SLOTS; ++g)
    {
        group_value[g] = -1;
        group_depth[g] = -1;
    }

    while (depth > 0)
    {
        Frame &top = stack_local[depth - 1];
        int step = top.step; // transition index: 19..64
        int pos = top.pos;
        uint32_t bm = top.mask;

        // emit when we reach step == 64 (terminal position, no transition)
        if (step == terminal_step)
        {
            int out_idx = atomicAdd(d_out_p2_count, 1);
            if (out_idx < MAX_ANSWERS_phaze2)
            {
                d_out_p2[out_idx] = acc;
            }
            depth--;
            // undo group seeds made at this depth (nothing to undo at terminal, but keep symmetry)
            for (int g = 0; g < num_groups; ++g)
                if (group_depth[g] == depth)
                {
                    group_value[g] = -1;
                    group_depth[g] = -1;
                }
            continue;
        }

        if (!bm)
        {
            depth--;
            for (int g = 0; g < num_groups; ++g)
                if (group_depth[g] == depth)
                {
                    group_value[g] = -1;
                    group_depth[g] = -1;
                }
            continue;
        }

        int m = __ffs(bm) - 1;    // pick a child move
        top.mask = bm & (bm - 1); // pop lsb

        // apply filters ONLY where a real transition exists (19..63)
        if (step <= last_transition)
        {
            int ones = move_ones(m);
            int active_cnt = ac_local[step];
            if (ones > active_cnt || (ones == 1 && active_cnt > 2))
            {
                continue;
            }
            int u5 = legal5_to_u5(m);
            bool group_ok = true;
            for (int g = 0; g < num_groups && group_ok; ++g)
            {
                if ((ag_masks[g] >> step) & 1ULL)
                {
                    if (group_value[g] == -1)
                    {
                        group_value[g] = u5;
                        group_depth[g] = depth;
                    }
                    else if (group_value[g] != u5)
                        group_ok = false;
                }
            }
            if (!group_ok)
                continue;
        }

        // write packed bits for this transition at indices starting from 19*5
        int bit_base = step * 5; // step in [19..63] is written; 39 never written
#pragma unroll
        for (int r = 0; r < 5; ++r)
            set_packed_bit(acc.bit_seq, bit_base + r, d_LEGAL[m * 5 + r] ? 1 : 0);

        // advance rotor position according to move m
        int tmp = pos;
        int p0 = tmp % 26;
        tmp /= 26;
        int p1 = tmp % 26;
        tmp /= 26;
        int p2 = tmp % 26;
        tmp /= 26;
        int p3 = tmp % 26;
        tmp /= 26;
        int p4 = tmp % 26;

        int np0 = d_LEGAL[m * 5 + 0] ? (p0 + d_delta[0] + 26) % 26 : p0;
        int np1 = d_LEGAL[m * 5 + 1] ? (p1 + d_delta[1] + 26) % 26 : p1;
        int np2 = d_LEGAL[m * 5 + 2] ? (p2 + d_delta[2] + 26) % 26 : p2;
        int np3 = d_LEGAL[m * 5 + 3] ? (p3 + d_delta[3] + 26) % 26 : p3;
        int np4 = d_LEGAL[m * 5 + 4] ? (p4 + d_delta[4] + 26) % 26 : p4;

        int next_pos = np0 + np1 * 26 + np2 * 26 * 26 + np3 * 26 * 26 * 26 + np4 * 26 * 26 * 26 * 26;

        // child mask for next transition index (step+1)
        uint32_t childMask = d_dp_p2[(step + 1 - base_step) * NUM_POS + next_pos];
        if (!childMask)
            continue;

        stack_local[depth++] = {step + 1, next_pos, childMask};
    }
}

// =======================
// NEW FILTER: COLLECT ALL MATCHES
// =======================

// Each block handles one cphP-group and iterates over all CTLs in a thread-strided loop.
//
__global__ void filter_collect_matches(
    const GpuEntry *d_entries_base, // d_out + base
    int N,                          // number of entries in this chunk
    int num_groups,
    const int *d_group_starts,  // RELATIVE (size = num_groups+1)
    const char *d_group_cphP,   // num_groups * CPH_P_LEN
    const char *d_ctls,         // TOTAL_CTL * CTL_LEN
    int steps,                  // (end_len-1)
    const int *d_move_ones,     // chunk-local: N * steps
    const int *d_num_mr,        // chunk-local: N
    const uint64_t *d_mr_masks, // chunk-local: N * steps
    GpuEntry *d_survivors,      // GLOBAL survivors buffer
    int *d_survivor_count)      // GLOBAL atomic counter
{
    int g = blockIdx.x;
    if (g >= num_groups)
        return;

    int start = d_group_starts[g];
    int end = d_group_starts[g + 1];
    int localN = end - start;
    if (localN <= 0)
        return;

    const char *cphP = d_group_cphP + g * CPH_P_LEN;

    // Each thread walks CTLs in stride
    for (int c = threadIdx.x; c < TOTAL_CTL; c += blockDim.x)
    {
        // Decode CTL -> wiring, reversed, positions derived from cphP
        const char *ctl = d_ctls + c * CTL_LEN;

        int control_wi[5];
        bool control_rev[5];
        int control_pos[5];
#pragma unroll
        for (int r = 0; r < 5; ++r)
        {
            control_wi[r] = ctl[r * 2] - '0';
            control_rev[r] = (ctl[r * 2 + 1] == 'R');
            control_pos[r] = cphP[r] - 'A';
        }

        // Precompute active mask over steps+1 characters
        int active_mask[MAX_CRIB_LEN];      // up to 65
        for (int i = 0; i < steps + 1; ++i) // steps+1 chars
        {
            bool move_ctrl[10] = {false};
            for (int p = 5; p <= 8; ++p)
            {
                int v = p;
#pragma unroll
                for (int r = 4; r >= 0; --r)
                    v = rotor_r2l_dev(control_wi[r], control_rev[r], control_pos[r], v);
                move_ctrl[INDEX_IN[v]] = true;
            }
            int mask = 0;
#pragma unroll
            for (int pin = 0; pin < 10; ++pin)
                if (move_ctrl[pin])
                    mask |= (1 << pin);
            active_mask[i] = mask;

            if (control_pos[2] == 14)
            {
                if (control_pos[3] == 14)
                    advance_rotor_dev_p1(control_rev[1], control_pos[1]);
                advance_rotor_dev_p1(control_rev[3], control_pos[3]);
            }
            advance_rotor_dev_p1(control_rev[2], control_pos[2]);
        }

        int ac[MAX_CRIB_LEN]; // steps
        for (int i = 0; i < steps; i++)
            ac[i] = __popc(active_mask[i]);

        // Build same-active-group masks (ag_masks) limited by d_ag_masks_SLOTS
        uint64_t ag_masks[d_ag_masks_SLOTS];
#pragma unroll
        for (int gg = 0; gg < d_ag_masks_SLOTS; gg++)
            ag_masks[gg] = 0ULL;

        int num_g = 0;
        bool visited[MAX_CRIB_LEN] = {false};
        for (int i = 0; i < steps && num_g < d_ag_masks_SLOTS; ++i)
        {
            if (visited[i])
                continue;
            uint64_t mask = 1ULL << i;
            visited[i] = true;
            for (int j = i + 1; j < steps; ++j)
            {
                if (!visited[j] && active_mask[i] == active_mask[j])
                {
                    mask |= (1ULL << j);
                    visited[j] = true;
                }
            }
            if (__popcll(mask) > 1)
                ag_masks[num_g++] = mask;
        }

        // Check each sequence in this group
        for (int seq = 0; seq < localN; ++seq)
        {
            int idx = (start + seq); // index within chunk
            bool seq_ok = true;

            // Move-ones quick check
            for (int step = 0; step < steps; ++step)
            {
                int mo = d_move_ones[idx * steps + step];
                int a = ac[step];
                if (mo > a || (mo == 1 && a > 2))  //this is two new filters.
                {
                    seq_ok = false;
                    break;
                }
            }
            if (!seq_ok)
                continue;

            // Repetition check
            bool rep_ok = (num_g == 0);
            if (!rep_ok)
            {
                int nm = d_num_mr[idx];
                if (nm > steps)
                    nm = steps;
                for (int gg = 0; gg < num_g && !rep_ok; ++gg)
                {
                    uint64_t ag_mask = ag_masks[gg];
                    for (int j = 0; j < nm && !rep_ok; ++j)
                    {
                        uint64_t mr = d_mr_masks[idx * steps + j];
                        if ((mr & ag_mask) == ag_mask)
                            rep_ok = true;
                    }
                }
            }
            if (!rep_ok)
                continue;

            // Emit survivor: copy entry & set ctl_id
            int out_idx = atomicAdd(d_survivor_count, 1);
            if (out_idx < MAX_SURVIVORS)
            {
                GpuEntry out = d_entries_base[start + seq];
                out.ctl_id = c; // record which CTL matched
                d_survivors[out_idx] = out;
            }
            else // else:  overflow 
            {
                printf("OVERFLOW HAPPENED!!!!");
            }
            
        }
    }
}

// -------------------------------
// FORWARD DECLARATIONS (kernels)
// -------------------------------
__global__ void collect_group_cphP(const GpuEntry *d_out, const int *d_group_starts, int num_groups, char *d_group_cphP)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= num_groups)
        return;
    int start = d_group_starts[g];
    const char *src = d_out[start].cphP;
    char *dst = d_group_cphP + g * CPH_P_LEN;
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
    dst[4] = src[4];
}

// process packed bits with runtime steps =====
__global__ void process_packed_bit_sequences_kernel(
    const GpuEntry *d_entries,
    int start_idx, int N,
    int steps,           // runtime: (end_len-1)
    int *d_move_ones,    // N * steps
    int *d_num_mr,       // N
    uint64_t *d_mr_masks // N * steps
)
{
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= N)
        return;

    const GpuEntry *e = &d_entries[start_idx + seq_idx];

    const int group_bits = 5;

    int group_val[MAX_CRIB_LEN];

    for (int s = 0; s < steps; ++s)
    {
        int val = 0, ones = 0;
        for (int b = 0; b < group_bits; ++b)
        {
            int bit_idx = s * group_bits + b;
            int w = bit_idx >> 5;
            int off = bit_idx & 31;
            uint32_t word = e->bit_seq[w];
            int bit = (word >> off) & 1u;
            val = (val << 1) | bit;
            ones += bit;
        }
        group_val[s] = val;

        d_move_ones[seq_idx * steps + s] = ones;
    }

    int count[32] = {0};
    for (int s = 0; s < steps; ++s)
        count[group_val[s]]++;

    int k = 0;
    for (int v = 0; v < 32; ++v)
    {
        if (count[v] > 1)
        {
            uint64_t mask = 0ULL;
            for (int s = 0; s < steps; ++s)
                if (group_val[s] == v)
                    mask |= (1ULL << s);
            d_mr_masks[seq_idx * steps + k] = mask;
            ++k;
        }
    }
    d_num_mr[seq_idx] = k;
    for (int rem = k; rem < steps; ++rem)
        d_mr_masks[seq_idx * steps + rem] = 0ULL;
}

// build DP table for 65-char pass but only rows 19..64; then extend P1 survivors on device
void phase2_extend_on_device(
    // fixed context
    const char *d_cph,
    const char *d_crib,
    const char *d_cipher,
    const int *d_forward,
    const int *d_inverse,
    const bool *d_reversed,
    const int *d_delta,
    const char *d_ctls,
    // input from P1
    GpuEntry *d_p1_survivors,
    int num_p1_survivors,
    // result
    GpuEntry **d_out_p2_ptr,
    int *h_p2_count_out)
{
    // allocate DP for P2 (65 x NUM_POS)
    const int start_step_p2 = CRIB_LEN_P1 - 1;      // 19
    const int end_len_p2 = MAX_CRIB_LEN;            // absolute end length (e.g. 65 or 85)
    const int rows_p2 = end_len_p2 - start_step_p2; // number of rows we actually need (e.g. 21 for 65)

    uint32_t *d_dp_p2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dp_p2, (size_t)rows_p2 * NUM_POS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_dp_p2, 0, (size_t)rows_p2 * NUM_POS * sizeof(uint32_t)));

    // fill rows 39 down to 19 inclusive
    {
        dim3 grid(GRID_X, 1, 1);
        dim3 block(BLOCK_X, BLOCK_Y, 1);
        for (int step = end_len_p2 - 1; step >= start_step_p2; --step)
        {
            dp_fill_kernel<<<grid, block>>>(
                d_cph, d_crib, d_cipher,
                d_forward, d_inverse, d_reversed, d_delta,
                d_dp_p2, step, end_len_p2, start_step_p2);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // output buffer
    GpuEntry *d_out_p2 = nullptr;
    int *d_out_p2_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_p2, sizeof(GpuEntry) * MAX_ANSWERS_phaze2));
    CUDA_CHECK(cudaMalloc(&d_out_p2_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out_p2_count, 0, sizeof(int)));

    // launch backtrack @ step 19 for each P1 survivor
    {
        const int tpb = 256;
        const int blocks = (num_p1_survivors + tpb - 1) / tpb;
        dp_backtrack_kernel_p2<<<blocks, tpb>>>(
            d_cph, d_crib, d_cipher,
            d_forward, d_inverse, d_reversed, d_delta,
            d_dp_p2,
            d_ctls,
            d_p1_survivors,
            num_p1_survivors,
            start_step_p2,
            end_len_p2,
            d_out_p2,
            d_out_p2_count);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_out_p2_count, sizeof(int), cudaMemcpyDeviceToHost));

    // return
    *d_out_p2_ptr = d_out_p2;
    *h_p2_count_out = h_count;

    // cleanup temps
    CUDA_CHECK(cudaFree(d_dp_p2));
    CUDA_CHECK(cudaFree(d_out_p2_count));
}

void generate_combinations(int start, int k, std::vector<int> &comb, std::vector<std::array<int, 5>> &out)
{
    if (k == 0)
    {
        std::array<int, 5> a;
        std::copy_n(comb.begin(), 5, a.begin());
        out.push_back(a);
        return;
    }
    for (int i = start; i <= 9; ++i)
    {
        comb.push_back(i);
        generate_combinations(i + 1, k - 1, comb, out);
        comb.pop_back();
    }
}

__device__ bool sigaba_check(
    const char *cph,  // e.g. "0R1N2N3N4R"
    const char *ctl,  // e.g. "5N6N7R8N9N"
    const char *cphP, // e.g. "ABCDE"
    const char *ctlP, // e.g. "ABCDE"
    const char *crib,
    const char *cipher,
    const uint8_t *index_map)
{
    // Shared memory for cipher rotor data (same for all threads in a block)
    __shared__ int sh_cipher_wi[5];
    __shared__ bool sh_cipher_rev[5];
    __shared__ int sh_initial_pos[5];

    // Populate shared memory with cipher rotor data (single thread per block)
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int r = 0; r < 5; ++r)
        {
            sh_cipher_wi[r] = cph[r * 2] - '0';
            sh_cipher_rev[r] = (cph[r * 2 + 1] == 'R');
            sh_initial_pos[r] = cphP[r] - 'A';
        }
    }
    __syncthreads();

    // Per-thread rotor positions
    int cipher_pos[5];
    int control_pos[5];
    for (int r = 0; r < 5; ++r)
    {
        cipher_pos[r] = sh_initial_pos[r];
        control_pos[r] = sh_initial_pos[r]; // Since ctlP = cphP
    }

    // Per-thread control rotor data
    int control_wi[5];
    bool control_rev[5];
    for (int r = 0; r < 5; ++r)
    {
        control_wi[r] = ctl[r * 2] - '0';
        control_rev[r] = (ctl[r * 2 + 1] == 'R');
    }

    char out[MAX_CRIB_LEN];

    for (int i = 0; i < MAX_CRIB_LEN; ++i)
    {
        int c = crib[i] - 'A';
        for (int r = 0; r < 5; ++r)
        {
            c = rotor_l2r_dev(sh_cipher_wi[r], sh_cipher_rev[r], cipher_pos[r], c);
        }
        out[i] = char(c + 'A');
        if (out[i] != cipher[i])
        {
            return false;
        }

        bool move[5] = {0};
#pragma unroll
        for (int p = 'F' - 'A'; p <= 'I' - 'A'; ++p)
        {
            int v = p;
            for (int r = 4; r >= 0; --r)
            {
                v = rotor_r2l_dev(control_wi[r], control_rev[r], control_pos[r], v);
            }
            int pin = INDEX_IN[v];
            int ip = index_map[pin];
            move[INDEX_OUT[ip] - 1] = true;
        }

        for (int r = 0; r < 5; ++r)
        {
            if (move[r])
                advance_rotor_dev_p1(sh_cipher_rev[r], cipher_pos[r]);
        }

        if (control_pos[2] == ('O' - 'A'))
        {
            if (control_pos[3] == ('O' - 'A'))
                advance_rotor_dev_p1(control_rev[1], control_pos[1]);
            advance_rotor_dev_p1(control_rev[3], control_pos[3]);
        }
        advance_rotor_dev_p1(control_rev[2], control_pos[2]);
    }

    // Match found, print result
    char local_cph[11];
    char local_ctl[11];
    char local_cphP[6];
    memcpy(local_cph, cph, 10);
    local_cph[10] = '\0';
    memcpy(local_ctl, ctl, 10);
    local_ctl[10] = '\0';
    memcpy(local_cphP, cphP, 5);
    local_cphP[5] = '\0';

    int idx_map_id = ((index_map - g_index_map) / 10);
    char idx_rotors[6];
    char idx_positions[6];
    const char *idx_data = d_idx_idxp + idx_map_id * 10;
    memcpy(idx_rotors, idx_data, 5);
    idx_rotors[5] = '\0';
    memcpy(idx_positions, idx_data + 5, 5);
    idx_positions[5] = '\0';

    printf("\n\033[1;32mSolution Found!!!!!  %s %s %s %s %s \033[0;32m\n\n", local_cph, local_ctl, local_cphP, idx_rotors, idx_positions);

    return true;
}

__global__ void sigaba_kernel(const char *cph, const char *cphP, const char *d_ctls, const uint8_t *d_index_map, const char *__restrict__ d_crib,
                              const char *__restrict__ d_cipher)
{
    int idx_map_id = blockIdx.x * blockDim.x + threadIdx.x;
    int ctl_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_map_id < 113400 && ctl_id < 1920)
    {
        const char *ctl = d_ctls + ctl_id * 10;
        const char *ctlP = cphP; // Since ctlP = cphP
        const uint8_t *index_map = d_index_map + idx_map_id * 10;
        sigaba_check(cph, ctl, cphP, ctlP, d_crib, d_cipher, index_map);
    }
}

// ----------  phase1 ----------
void phase1(const std::string &host_cph,
            const std::string &CRIB_SOURCE,
            const std::string &CIPHER_SOURCE,
            const std::string &digits,
            uint8_t *d_index_map,                     // preallocated device index map (shared)
            char *d_ctls,                             // preallocated device CTL list (shared)
            const std::vector<std::string> &all_ctls, // host CTL list (shared)
            const std::vector<char> &hostIdxIdxp      // hostBuf already copied to d_idx_idxp symbol by caller
)
{
    // Note: 
    // d_index_map/host_index_map/d_ctls/d_idx_idxp. Those are prepared by main.


    std::locale loc(std::cout.getloc(), new comma_numpunct);
    std::cout.imbue(loc);
    auto p1_start = std::chrono::high_resolution_clock::now();

    // Upload full crib/cipher (0..64)
    char h_crib[MAX_CRIB_LEN], h_cipher[MAX_CRIB_LEN];
    for (int i = 0; i < MAX_CRIB_LEN; ++i)
    {
        h_crib[i] = CRIB_SOURCE[i];
        h_cipher[i] = CIPHER_SOURCE[i];
    }

    // Rotor wiring host prep / memcpyToSymbol 
    static int host_WN[10][26], host_WR[10][26];
    for (int wi = 0; wi < 10; ++wi)
    {
        for (int c = 0; c < 26; ++c)
        {
            int o = d_ROTOR_WIRINGS[wi][c] - 'A';
            host_WN[wi][c] = o;
            host_WR[wi][o] = c;
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_ROTOR_WIRINGS_NORMAL, host_WN, sizeof(host_WN)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ROTOR_WIRINGS_REVERSE, host_WR, sizeof(host_WR)));


    // Device allocations that persist across both phases are done per-CPH here 
    char *d_cph = nullptr, *d_crib = nullptr, *d_cipher = nullptr; // note: d_ctls and d_index_map are passed in
    int *d_forward = nullptr, *d_inverse = nullptr, *d_delta = nullptr, *d_answer_count = nullptr;
    bool *d_reversed = nullptr;
    uint32_t *d_dp = nullptr;  // per-phase
    GpuEntry *d_out = nullptr; // Phase-1 raw backtracks

    CUDA_CHECK(cudaMalloc(&d_cph, CPH_LEN));
    CUDA_CHECK(cudaMalloc(&d_crib, MAX_CRIB_LEN));
    CUDA_CHECK(cudaMalloc(&d_cipher, MAX_CRIB_LEN));
    CUDA_CHECK(cudaMalloc(&d_forward, 5 * 26 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inverse, 5 * 26 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_reversed, 5 * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_delta, 5 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_answer_count, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_crib, h_crib, MAX_CRIB_LEN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cipher, h_cipher, MAX_CRIB_LEN, cudaMemcpyHostToDevice));

    

        auto p1_now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = p1_now - p1_start;
        printf(" CPH : %s  ", host_cph.c_str());
 

    // Wiring for this CPH
    int h_forward[5 * 26], h_inverse[5 * 26];
    bool h_reversed[5];
    int h_delta[5];
    for (int r = 0; r < 5; ++r)
    {
        int wi = host_cph[r * 2] - '0';
        h_reversed[r] = (host_cph[r * 2 + 1] == 'R');
        const char *rw = d_ROTOR_WIRINGS[wi];
        for (int j = 0; j < 26; ++j)
        {
            int o = rw[j] - 'A';
            h_forward[r * 26 + j] = o;
            h_inverse[r * 26 + o] = j;
        }
        h_delta[r] = (h_reversed[r] ? 1 : -1);
    }

    CUDA_CHECK(cudaMemcpy(d_cph, host_cph.data(), CPH_LEN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forward, h_forward, 5 * 26 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverse, h_inverse, 5 * 26 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reversed, h_reversed, 5 * sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta, h_delta, 5 * sizeof(int), cudaMemcpyHostToDevice));

    // Phase-1 DP buffer
    {
        const size_t DP_SIZE_P1 = (size_t)CRIB_LEN_P1 * (size_t)NUM_POS;
        CUDA_CHECK(cudaMalloc(&d_dp, DP_SIZE_P1 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_dp, 0, DP_SIZE_P1 * sizeof(uint32_t)));
    }

    // Raw backtracked answers
    CUDA_CHECK(cudaMalloc(&d_out, MAX_ANSWERS_phaze1 * sizeof(GpuEntry)));
    CUDA_CHECK(cudaMemset(d_answer_count, 0, sizeof(int)));

    // Fill DP for steps 19..0 (backward)
    {
        dim3 grid(GRID_X, 1, 1);
        dim3 block(BLOCK_X, BLOCK_Y, 1);
        for (int step = CRIB_LEN_P1 - 1; step >= 0; --step)
        {
            dp_fill_kernel<<<grid, block>>>(
                d_cph, d_crib, d_cipher,
                d_forward, d_inverse, d_reversed, d_delta,
                d_dp, step, CRIB_LEN_P1, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Backtrack all starts that survive at step 0
    {
        const int block_size = 128;
        const int num_blocks = (NUM_POS + block_size - 1) / block_size;
        dp_backtrack_kernel<<<num_blocks, block_size>>>(
            d_cph, d_crib, d_cipher,
            d_forward, d_inverse, d_reversed, d_delta,
            d_dp, d_out, d_answer_count,
            CRIB_LEN_P1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int h_answer_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_answer_count, d_answer_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_answer_count <= 0)
    {
        CUDA_CHECK(cudaFree(d_out));
        d_out = nullptr;
        CUDA_CHECK(cudaFree(d_dp));
        d_dp = nullptr;
    }

    // std::cout << "Answer count (Phase1 raw) : " << h_answer_count << "\n";
    std::cout << "P1_n=" << h_answer_count;


    // Sort by (cph, cphP) then group
    thrust::device_ptr<GpuEntry> d_begin = thrust::device_pointer_cast(d_out);
    thrust::device_ptr<GpuEntry> d_end = d_begin + h_answer_count;
    thrust::sort(d_begin, d_end, GpuEntryLess());

    thrust::device_vector<int> d_flags(h_answer_count);
    {
        const int tpb = 256;
        const int blocks = (h_answer_count + tpb - 1) / tpb;
        mark_group_starts<<<blocks, tpb>>>(d_out, h_answer_count, thrust::raw_pointer_cast(d_flags.data()));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    int num_groups = thrust::reduce(d_flags.begin(), d_flags.end(), 0, thrust::plus<int>());
    thrust::device_vector<int> d_group_starts(num_groups + 1);
    thrust::copy_if(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(h_answer_count),
        d_flags.begin(),
        d_group_starts.begin(),
        [] __host__ __device__(int f)
        { return f != 0; });
    d_group_starts[num_groups] = h_answer_count;

    // ---------------------------
    //  collect ALL matches
    // ---------------------------
    const int steps_p1 = CRIB_LEN_P1 - 1; // 19

    // Global survivors buffer & counter
    GpuEntry *d_survivors = nullptr;
    int *d_survivor_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_survivors, sizeof(GpuEntry) * MAX_SURVIVORS));
    CUDA_CHECK(cudaMalloc(&d_survivor_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_survivor_count, 0, sizeof(int)));

    auto t0 = std::chrono::high_resolution_clock::now();

    // Host copy of group starts for chunking
    thrust::host_vector<int> h_group_starts = d_group_starts;

    int g0 = 0;
    while (g0 < num_groups)
    {
        const int base = h_group_starts[g0];

        int g1 = g0 + 1;
        while (g1 < num_groups && (h_group_starts[g1] - base) <= CHUNK_P1)
            ++g1;
        if (g1 == g0 + 1 && (h_group_starts[g1] - base) > CHUNK_P1)
        {
            // single huge group: process alone
        }

        const int num_groups_chunk = g1 - g0;
        const int chunkN = h_group_starts[g1] - base;
        if (num_groups_chunk <= 0 || chunkN <= 0)
        {
            g0 = std::min(g1, num_groups);
            continue;
        }

        thrust::device_vector<int> d_group_starts_chunk(num_groups_chunk + 1);
        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(d_group_starts_chunk.data()),
            thrust::raw_pointer_cast(d_group_starts.data()) + g0,
            (size_t)(num_groups_chunk + 1) * sizeof(int),
            cudaMemcpyDeviceToDevice));

        thrust::transform(
            d_group_starts_chunk.begin(),
            d_group_starts_chunk.end(),
            thrust::make_constant_iterator(base),
            d_group_starts_chunk.begin(),
            thrust::minus<int>());

        CUDA_CHECK(cudaDeviceSynchronize());

        // cphP per-group
        char *d_group_cphP_chunk = nullptr;
        CUDA_CHECK(cudaMalloc(&d_group_cphP_chunk, num_groups_chunk * CPH_P_LEN));
        {
            const int tpb = 256;
            const int blocks = (num_groups_chunk + tpb - 1) / tpb;
            collect_group_cphP<<<blocks, tpb>>>(
                d_out + base,
                thrust::raw_pointer_cast(d_group_starts_chunk.data()),
                num_groups_chunk,
                d_group_cphP_chunk);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Per-seq temporaries for this chunk
        int *d_move_ones = nullptr, *d_num_mr = nullptr;
        uint64_t *d_mr_masks = nullptr;
        CUDA_CHECK(cudaMalloc(&d_move_ones, sizeof(int) * chunkN * steps_p1));
        CUDA_CHECK(cudaMalloc(&d_num_mr, sizeof(int) * chunkN));
        CUDA_CHECK(cudaMalloc(&d_mr_masks, sizeof(uint64_t) * chunkN * steps_p1));

        // Unpack & MR for this chunk
        {
            const int tpb = 128;
            const int blocks = (chunkN + tpb - 1) / tpb;
            process_packed_bit_sequences_kernel<<<blocks, tpb>>>(
                d_out,  // full array
                base,   // absolute base
                chunkN, // entries in this chunk
                steps_p1,
                d_move_ones, d_num_mr, d_mr_masks);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Collect ALL matches for this chunk , new filter 
        {
            const int threads = 128;
            filter_collect_matches<<<num_groups_chunk, threads>>>(
                d_out + base, // entries base
                chunkN,       // N
                num_groups_chunk,
                thrust::raw_pointer_cast(d_group_starts_chunk.data()),
                d_group_cphP_chunk,
                d_ctls,
                steps_p1,
                d_move_ones, d_num_mr, d_mr_masks,
                d_survivors,
                d_survivor_count);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaFree(d_group_cphP_chunk));
        CUDA_CHECK(cudaFree(d_move_ones));
        CUDA_CHECK(cudaFree(d_num_mr));
        CUDA_CHECK(cudaFree(d_mr_masks));

        g0 = g1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedF = t1 - t0;
    std::cout << "   P1_t=" << std::fixed << std::setprecision(3)
              << elapsedF.count() << std::defaultfloat;

    // Bring survivors back
    int h_survivor_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_survivor_count, d_survivor_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_survivor_count > MAX_SURVIVORS)
        h_survivor_count = MAX_SURVIVORS;

    std::cout << "   surv=" << h_survivor_count;



    // >>> NEW: Phase-2 (extend to 65) directly on device <<<
    GpuEntry *d_out_p2 = nullptr;
    int h_p2_count = 0;
    if (h_survivor_count > 0)
    {
        phase2_extend_on_device(
            d_cph, d_crib, d_cipher,
            d_forward, d_inverse, d_reversed, d_delta,
            d_ctls,
            d_survivors,
            h_survivor_count,
            &d_out_p2,
            &h_p2_count);
    }

    std::cout << "   P2_ans=" << h_p2_count;

    if (h_p2_count > 0)
    {
        std::vector<GpuEntry> survivors(h_p2_count);
        CUDA_CHECK(cudaMemcpy(survivors.data(), d_out_p2, h_p2_count * sizeof(GpuEntry), cudaMemcpyDeviceToHost));

        printf("\n Phase-2 survivors: %d\n", h_p2_count);

        // Build unique set of triples (cph, cphP, ctl)
        struct TripleKey
        {
            std::string cph, cphP, ctl;
            bool operator==(const TripleKey &o) const { return cph == o.cph && cphP == o.cphP && ctl == o.ctl; }
        };
        struct TripleHash
        {
            size_t operator()(TripleKey const &t) const noexcept { return std::hash<std::string>()(t.cph + '|' + t.cphP + '|' + t.ctl); }
        };

        std::unordered_set<TripleKey, TripleHash> uniqs;
        uniqs.reserve(h_p2_count);
        for (int i = 0; i < h_p2_count; ++i)
        {
            TripleKey key;
            key.cph.assign(survivors[i].cph, survivors[i].cph + CPH_LEN);
            key.cphP.assign(survivors[i].cphP, survivors[i].cphP + CPH_P_LEN);
            key.ctl = all_ctls[survivors[i].ctl_id];
            uniqs.insert(std::move(key));
        }

        printf("Phase-3: Unique triples found: %zu\n", uniqs.size());

        if (!uniqs.empty())
        {
            size_t ucount = uniqs.size();
            std::vector<char> h_cph(ucount * CPH_LEN);
            std::vector<char> h_cphP(ucount * CPH_P_LEN);
            std::vector<char> h_ctl(ucount * CTL_LEN);

            size_t idx = 0;
            for (const auto &t : uniqs)
            {
                memcpy(&h_cph[idx * CPH_LEN], t.cph.c_str(), CPH_LEN);
                memcpy(&h_cphP[idx * CPH_P_LEN], t.cphP.c_str(), CPH_P_LEN);
                memcpy(&h_ctl[idx * CTL_LEN], t.ctl.c_str(), CTL_LEN);
                ++idx;
            }

            // Device buffers for one triple at a time
            char *d_cph_single = nullptr, *d_cphP = nullptr, *d_surv_ctls = nullptr;
            CUDA_CHECK(cudaMalloc(&d_cph_single, CPH_LEN));
            CUDA_CHECK(cudaMalloc(&d_cphP, CPH_P_LEN));
            CUDA_CHECK(cudaMalloc(&d_surv_ctls, CTL_LEN));

            dim3 blockDim(16, 16);
            dim3 gridDim((113400 + blockDim.x - 1) / blockDim.x, (1920 + blockDim.y - 1) / blockDim.y);

            for (size_t i = 0; i < ucount; ++i)
            {
                CUDA_CHECK(cudaMemcpy(d_cph_single, &h_cph[i * CPH_LEN], CPH_LEN, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_cphP, &h_cphP[i * CPH_P_LEN], CPH_P_LEN, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_surv_ctls, &h_ctl[i * CTL_LEN], CTL_LEN, cudaMemcpyHostToDevice));

                // Launch sigaba kernel for this triple; kernel prints when solution found
                sigaba_kernel<<<gridDim, blockDim>>>(d_cph_single, d_cphP, d_ctls, d_index_map, d_crib, d_cipher);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            CUDA_CHECK(cudaFree(d_cph_single));
            CUDA_CHECK(cudaFree(d_cphP));
            CUDA_CHECK(cudaFree(d_surv_ctls));
        }
    }

    auto te = std::chrono::high_resolution_clock::now();
    elapsedF = te - t1;
    std::cout << "   P2_t=" << std::fixed << std::setprecision(3)
              << elapsedF.count() << std::defaultfloat;

    CUDA_CHECK(cudaFree(d_out_p2));
    // cleanup per-CPH device buffers from P1
    CUDA_CHECK(cudaFree(d_survivors));
    CUDA_CHECK(cudaFree(d_survivor_count));

    CUDA_CHECK(cudaFree(d_out));
    d_out = nullptr;
    CUDA_CHECK(cudaFree(d_dp));
    d_dp = nullptr;

    // free per-CPH allocations
    CUDA_CHECK(cudaFree(d_cph));
    CUDA_CHECK(cudaFree(d_crib));
    CUDA_CHECK(cudaFree(d_cipher));
    CUDA_CHECK(cudaFree(d_forward));
    CUDA_CHECK(cudaFree(d_inverse));
    CUDA_CHECK(cudaFree(d_reversed));
    CUDA_CHECK(cudaFree(d_delta));
    CUDA_CHECK(cudaFree(d_answer_count));

} // end phase1()

int main(int argc, char *argv[])
{
    auto phaze1_start = std::chrono::high_resolution_clock::now();
    if (argc != 4)
    {
        std::cerr << "Usage: ./sigaba_solver <GPU_ID> <Partition_id 0 to 252 > <CPH id 0 to 1919  or -1 for ALL>" << std::endl;
        return 1;
    }

    int deviceId = std::stoi(argv[1]);
    int pt_id = std::stoi(argv[2]);
    int cph_id = std::stoi(argv[3]); // -1 means all CPHs in this partition

    CUDA_CHECK(cudaSetDevice(deviceId));

    std::vector<std::array<int, 5>> parts;
    {
        std::vector<int> tmp;
        // generate_combinations: fill 'parts' 
        
        generate_combinations(0, 5, tmp, parts);
    }
    std::vector<std::string> partition_id;
    partition_id.reserve(parts.size());
    for (auto &arr : parts)
    {
        std::string key;
        key.reserve(5);
        for (int d : arr)
            key.push_back(char('0' + d));
        partition_id.push_back(key);
    }

    // Generate all CPHs for the partition
    auto All_CPHs = generate_all_cph(partition_id[pt_id]);

    // Read uniq_idx_idxp_87156.txt ONCE and upload to device symbol d_idx_idxp
    std::ifstream idxin("uniq_idx_idxp_87156.txt");
    assert(idxin && "Cannot open uniq_idx_idxp_87156.txt");
    std::vector<char> hostBuf;
    hostBuf.reserve(ALL_INDEX_NUM * 10);
    std::string line;
    while (std::getline(idxin, line))
    {
        assert(line.size() == 11 && line[5] == ',');
        for (int j = 0; j < 5; ++j)
            hostBuf.push_back(line[j]);
        for (int j = 6; j < 11; ++j)
            hostBuf.push_back(line[j]);
    }
    idxin.close();

    // copy hostBuf to device symbol once
    CUDA_CHECK(cudaMemcpyToSymbol(d_idx_idxp, hostBuf.data(), hostBuf.size()));

    // Build host_index_map once and upload to device (d_index_map)
    std::vector<uint8_t> host_index_map(ALL_INDEX_NUM * 10);
    for (int t = 0; t < ALL_INDEX_NUM; ++t)
    {
        int idxr[5], pos[5];
        for (int r = 0; r < 5; ++r)
        {
            idxr[r] = hostBuf[10 * t + r] - '0';
            pos[r] = hostBuf[10 * t + 5 + r] - '0';
        }
        for (int pin = 0; pin < 10; ++pin)
        {
            int ip = pin;
            for (int r = 0; r < 5; ++r)
            {
                int x = (ip + pos[r]) % 10;
                ip = (HOST_INDEX_WIRINGS[idxr[r]][x] - pos[r] + 10) % 10;
            }
            host_index_map[t * 10 + pin] = ip;
        }
    }
    uint8_t *d_index_map = nullptr;
    CUDA_CHECK(cudaMalloc(&d_index_map, host_index_map.size()));
    CUDA_CHECK(cudaMemcpy(d_index_map, host_index_map.data(), host_index_map.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(g_index_map, &d_index_map, sizeof(d_index_map)));

    // Prepare CTLs once and upload once
    std::string digits = partition_id[pt_id]; // used by generate_all_ctl
    std::vector<std::string> all_ctls = generate_all_ctl(digits);
    std::vector<char> flat_ctls(TOTAL_CTL * CTL_LEN);
    for (int i = 0; i < TOTAL_CTL; ++i)
        std::memcpy(&flat_ctls[i * CTL_LEN], all_ctls[i].data(), CTL_LEN);

    char *d_ctls = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ctls, flat_ctls.size()));
    CUDA_CHECK(cudaMemcpy(d_ctls, flat_ctls.data(), flat_ctls.size(), cudaMemcpyHostToDevice));

    // Set LASRY limits once
    double lasry_low = -1.99;
    double lasry_upp = -0.01;
    cudaMemcpyToSymbol(LASRY_LOWER, &lasry_low, sizeof(lasry_low));
    cudaMemcpyToSymbol(LASRY_UPPER, &lasry_upp, sizeof(lasry_upp));

    // Crib / cipher sources 
    std::string CRIB_SOURCE = "FROMZGENERALZMARKZWAYNEZCLARKZTOZGENERALZDWIGHTZDAVIDZEINSENHOWERZCHEFZOFZTHEZEUROPEANZALLIEDFORCESX";
    std::string CIPHER_SOURCE = "ATWHIARIQODJIUQKBYPHHWRNYUKEDOQDDGHYEYMRVPISEKHCHKFERLPQJETNCCXQQIUPQGCAEQQBOITWCLZJAWEYZATKNIRBXSGE";

    printf("CRIB_LEN = %d\n",MAX_CRIB_LEN);
    printf("CRIB   = %s\n",CRIB_SOURCE.c_str());
    printf("CIPHER = %s\n",CIPHER_SOURCE.c_str());
    // If user requested single CPH (non -1), call phase1 once using pre-uploaded d_index_map & d_ctls
    if (cph_id >= 0)
    {
        std::string CPH_input = All_CPHs[cph_id];
        phase1(CPH_input, CRIB_SOURCE, CIPHER_SOURCE, digits, d_index_map, d_ctls, all_ctls, hostBuf);
        auto phaze1_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = phaze1_end - phaze1_start;
        std::cout << "    " << elapsed.count() << " sec\n\n";
    }
    else
    {
        // Process ALL CPHs in this partition sequentially, reusing d_index_map & d_ctls set up above.
        // This avoids reading index file / regenerating CTLs / reuploading them per-CPH.
        for (int i = 0; i < (int)All_CPHs.size(); ++i)
        {
            std::string &CPH_input = All_CPHs[i];

            printf("%d::%04d/%04zu :", pt_id, i, All_CPHs.size());
            phase1(CPH_input, CRIB_SOURCE, CIPHER_SOURCE, digits, d_index_map, d_ctls, all_ctls, hostBuf);
            // Optionally check device free memory or early stop if needed
            auto phaze1_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = phaze1_end - phaze1_start;
            std::cout << "  Total=" << std::fixed << std::setprecision(3) << elapsed.count() << std::defaultfloat << " sec\n";
        }
    }

    // Cleanup shared device buffers
    CUDA_CHECK(cudaFree(d_ctls));
    CUDA_CHECK(cudaFree(d_index_map));

    return 0;
}


