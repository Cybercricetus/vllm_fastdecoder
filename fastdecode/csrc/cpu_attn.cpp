// FastDecode CPU kernels: partial decode attention over CPU-resident paged KV
// cache, plus the KV store op.
//
// CPU KV cache layout (per layer, separate K and V tensors):
//     [num_cpu_blocks, num_kv_heads, block_size, head_size]
// so the tokens of one (block, kv_head) pair are one contiguous run of
// block_size * head_size elements, which keeps the streaming reads sequential.
//
// decode_attn computes, for every decode request i and query head h,
//     o[i, h]   = softmax(q[i, h] . K_cpu[i]^T * scale) @ V_cpu[i]
//     lse[h, i] = logsumexp(q[i, h] . K_cpu[i]^T * scale)
// over the request's CPU-resident tokens only. The GPU-resident part is
// computed by FlashAttention and the two partial results are merged with the
// standard log-sum-exp rule, which is exact up to floating point rounding.
//
// fp16/bf16 values are widened to fp32 in registers (F16C / shift for bf16) and
// all accumulation is fp32, as in the FastDecode paper (Sec. 5.1).

#include <torch/extension.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#define FD_AVX2 1
#endif

namespace {

int g_num_threads = 0;  // 0 => OpenMP default

inline int num_threads() {
  return g_num_threads > 0 ? g_num_threads : omp_get_max_threads();
}

// ---------------------------------------------------------------- converts
template <typename T>
inline void to_float(const T* src, float* dst, int n);

template <>
inline void to_float<float>(const float* src, float* dst, int n) {
  std::memcpy(dst, src, sizeof(float) * n);
}

template <>
inline void to_float<at::Half>(const at::Half* src, float* dst, int n) {
  int i = 0;
#ifdef FD_AVX2
  for (; i + 8 <= n; i += 8) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
  }
#endif
  for (; i < n; ++i) dst[i] = static_cast<float>(src[i]);
}

template <>
inline void to_float<at::BFloat16>(const at::BFloat16* src, float* dst, int n) {
  int i = 0;
#ifdef FD_AVX2
  for (; i + 8 <= n; i += 8) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
  }
#endif
  for (; i < n; ++i) dst[i] = static_cast<float>(src[i]);
}

template <typename T>
inline void from_float(const float* src, T* dst, int n, float mul) {
  for (int i = 0; i < n; ++i) dst[i] = static_cast<T>(src[i] * mul);
}

// ---------------------------------------------------------------- math
inline float dot(const float* a, const float* b, int n) {
  int i = 0;
  float s = 0.f;
#ifdef FD_AVX2
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8),
                           acc1);
  }
  for (; i + 8 <= n; i += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  __m128 lo = _mm256_castps256_ps128(acc0);
  __m128 hi = _mm256_extractf128_ps(acc0, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  s = _mm_cvtss_f32(lo);
#endif
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

// y += a * x
inline void axpy(float a, const float* x, float* y, int n) {
  int i = 0;
#ifdef FD_AVX2
  const __m256 va = _mm256_set1_ps(a);
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i),
                                            _mm256_loadu_ps(y + i)));
  }
#endif
  for (; i < n; ++i) y[i] += a * x[i];
}

#ifdef FD_AVX2
// Cephes-style exp (same polynomial as avx_mathfun's exp256_ps), max rel err
// ~2e-7 on the clamped range. Inputs here are always <= 0 (x - max).
inline __m256 exp256_ps(__m256 x) {
  const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
  const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
  const __m256 c1 = _mm256_set1_ps(0.693359375f);
  const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);

  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);
  __m256 fx = _mm256_fmadd_ps(x, log2ef, half);
  fx = _mm256_floor_ps(fx);
  x = _mm256_fnmadd_ps(fx, c1, x);
  x = _mm256_fnmadd_ps(fx, c2, x);
  __m256 z = _mm256_mul_ps(x, x);

  __m256 y = _mm256_set1_ps(1.9875691500E-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, one);

  __m256i imm0 = _mm256_cvttps_epi32(fx);
  imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(imm0));
}
#endif

#ifdef FD_AVX2
// ------------------------------------------- register-blocked fast path
// Used when head_size % 64 == 0 (64, 128, 256). Avoids staging each K/V row
// through memory and amortizes horizontal sums over 8 tokens.
template <typename T>
inline __m256 load8(const T* p);

template <>
inline __m256 load8<float>(const float* p) {
  return _mm256_loadu_ps(p);
}

template <>
inline __m256 load8<at::Half>(const at::Half* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

template <>
inline __m256 load8<at::BFloat16>(const at::BFloat16* p) {
  __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

inline float hsum(__m256 a) {
  __m128 lo = _mm256_castps256_ps128(a);
  __m128 hi = _mm256_extractf128_ps(a, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

// returns [sum(a0), sum(a1), ..., sum(a7)]
inline __m256 hsum8x8(__m256 a0, __m256 a1, __m256 a2, __m256 a3, __m256 a4,
                      __m256 a5, __m256 a6, __m256 a7) {
  __m256 s01 = _mm256_hadd_ps(a0, a1);
  __m256 s23 = _mm256_hadd_ps(a2, a3);
  __m256 s45 = _mm256_hadd_ps(a4, a5);
  __m256 s67 = _mm256_hadd_ps(a6, a7);
  __m256 s0123 = _mm256_hadd_ps(s01, s23);
  __m256 s4567 = _mm256_hadd_ps(s45, s67);
  __m256 lo = _mm256_permute2f128_ps(s0123, s4567, 0x20);
  __m256 hi = _mm256_permute2f128_ps(s0123, s4567, 0x31);
  return _mm256_add_ps(lo, hi);
}

// out[o] = qg . k[o] for the ntok rows of one block (rows are D apart)
template <typename T>
inline void scores_block_avx2(const float* __restrict qg, const T* __restrict kb,
                              int ntok, int D, float* __restrict out) {
  const int NC = D / 8;
  int o = 0;
  for (; o + 8 <= ntok; o += 8) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0,
           a6 = a0, a7 = a0;
    const T* k = kb + static_cast<int64_t>(o) * D;
    for (int c = 0; c < NC; ++c) {
      const __m256 qc = _mm256_loadu_ps(qg + c * 8);
      const T* kc = k + c * 8;
      a0 = _mm256_fmadd_ps(qc, load8<T>(kc), a0);
      a1 = _mm256_fmadd_ps(qc, load8<T>(kc + D), a1);
      a2 = _mm256_fmadd_ps(qc, load8<T>(kc + 2 * D), a2);
      a3 = _mm256_fmadd_ps(qc, load8<T>(kc + 3 * D), a3);
      a4 = _mm256_fmadd_ps(qc, load8<T>(kc + 4 * D), a4);
      a5 = _mm256_fmadd_ps(qc, load8<T>(kc + 5 * D), a5);
      a6 = _mm256_fmadd_ps(qc, load8<T>(kc + 6 * D), a6);
      a7 = _mm256_fmadd_ps(qc, load8<T>(kc + 7 * D), a7);
    }
    _mm256_storeu_ps(out + o, hsum8x8(a0, a1, a2, a3, a4, a5, a6, a7));
  }
  for (; o < ntok; ++o) {
    __m256 a = _mm256_setzero_ps();
    const T* k = kb + static_cast<int64_t>(o) * D;
    for (int c = 0; c < NC; ++c) {
      a = _mm256_fmadd_ps(_mm256_loadu_ps(qg + c * 8), load8<T>(k + c * 8), a);
    }
    out[o] = hsum(a);
  }
}

// accg[:] += sum_o p[o] * v[o] for the ntok rows of one block; D % 64 == 0
template <typename T>
inline void pv_block_avx2(const float* __restrict p, const T* __restrict vb,
                          int ntok, int D, float* __restrict accg) {
  for (int d0 = 0; d0 < D; d0 += 64) {
    float* a = accg + d0;
    __m256 r0 = _mm256_loadu_ps(a), r1 = _mm256_loadu_ps(a + 8),
           r2 = _mm256_loadu_ps(a + 16), r3 = _mm256_loadu_ps(a + 24),
           r4 = _mm256_loadu_ps(a + 32), r5 = _mm256_loadu_ps(a + 40),
           r6 = _mm256_loadu_ps(a + 48), r7 = _mm256_loadu_ps(a + 56);
    for (int o = 0; o < ntok; ++o) {
      const __m256 pv = _mm256_set1_ps(p[o]);
      const T* v = vb + static_cast<int64_t>(o) * D + d0;
      r0 = _mm256_fmadd_ps(pv, load8<T>(v), r0);
      r1 = _mm256_fmadd_ps(pv, load8<T>(v + 8), r1);
      r2 = _mm256_fmadd_ps(pv, load8<T>(v + 16), r2);
      r3 = _mm256_fmadd_ps(pv, load8<T>(v + 24), r3);
      r4 = _mm256_fmadd_ps(pv, load8<T>(v + 32), r4);
      r5 = _mm256_fmadd_ps(pv, load8<T>(v + 40), r5);
      r6 = _mm256_fmadd_ps(pv, load8<T>(v + 48), r6);
      r7 = _mm256_fmadd_ps(pv, load8<T>(v + 56), r7);
    }
    _mm256_storeu_ps(a, r0);
    _mm256_storeu_ps(a + 8, r1);
    _mm256_storeu_ps(a + 16, r2);
    _mm256_storeu_ps(a + 24, r3);
    _mm256_storeu_ps(a + 32, r4);
    _mm256_storeu_ps(a + 40, r5);
    _mm256_storeu_ps(a + 48, r6);
    _mm256_storeu_ps(a + 56, r7);
  }
}
#endif  // FD_AVX2

// x[i] = exp(x[i] - m); returns sum
inline float exp_sub_sum(float* x, int n, float m) {
  int i = 0;
  float s = 0.f;
#ifdef FD_AVX2
  const __m256 vm = _mm256_set1_ps(m);
  __m256 acc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vm));
    _mm256_storeu_ps(x + i, e);
    acc = _mm256_add_ps(acc, e);
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  s = _mm_cvtss_f32(lo);
#endif
  for (; i < n; ++i) {
    x[i] = std::exp(x[i] - m);
    s += x[i];
  }
  return s;
}

// ---------------------------------------------------------------- kernels
template <typename T>
void decode_attn_impl(const T* __restrict q, const T* __restrict kc,
                      const T* __restrict vc, const int32_t* __restrict bt,
                      const int32_t* __restrict lens, int n, int H, int KVH,
                      int D, int BS, int W, float scale, T* __restrict out,
                      float* __restrict lse) {
  const int G = H / KVH;
  const int64_t head_stride = static_cast<int64_t>(BS) * D;
  const int64_t blk_stride = static_cast<int64_t>(KVH) * head_stride;
  const float neg_inf = -std::numeric_limits<float>::infinity();
#ifdef FD_AVX2
  const bool fast = (D % 64 == 0);
#endif

#pragma omp parallel num_threads(num_threads())
  {
    std::vector<float> qf(static_cast<size_t>(G) * D);
    std::vector<float> rowf(D);
    std::vector<float> acc(static_cast<size_t>(G) * D);
    std::vector<float> scores;
    std::vector<float> mx(G), sm(G);

#pragma omp for schedule(dynamic, 1) collapse(2)
    for (int i = 0; i < n; ++i) {
      for (int kh = 0; kh < KVH; ++kh) {
        const int len = lens[i];
        if (len <= 0) {
          for (int g = 0; g < G; ++g) {
            const int h = kh * G + g;
            T* o = out + (static_cast<int64_t>(i) * H + h) * D;
            for (int d = 0; d < D; ++d) o[d] = static_cast<T>(0.f);
            lse[static_cast<int64_t>(h) * n + i] = neg_inf;
          }
          continue;
        }
        // q (pre-scaled) for the G query heads sharing this kv head
        for (int g = 0; g < G; ++g) {
          const int h = kh * G + g;
          float* qg = qf.data() + static_cast<size_t>(g) * D;
          to_float<T>(q + (static_cast<int64_t>(i) * H + h) * D, qg, D);
          for (int d = 0; d < D; ++d) qg[d] *= scale;
        }
        scores.resize(static_cast<size_t>(G) * len);

        // pass 1: scores = q . k
        int t = 0;
        for (int b = 0; t < len; ++b) {
          const int64_t blk = bt[static_cast<int64_t>(i) * W + b];
          const T* kbase = kc + blk * blk_stride + kh * head_stride;
          const int ntok = std::min(BS, len - t);
#ifdef FD_AVX2
          if (fast) {
            for (int g = 0; g < G; ++g) {
              scores_block_avx2<T>(qf.data() + static_cast<size_t>(g) * D, kbase,
                                   ntok, D,
                                   scores.data() + static_cast<size_t>(g) * len + t);
            }
            t += ntok;
            continue;
          }
#endif
          for (int o = 0; o < ntok; ++o, ++t) {
            to_float<T>(kbase + static_cast<int64_t>(o) * D, rowf.data(), D);
            for (int g = 0; g < G; ++g) {
              scores[static_cast<size_t>(g) * len + t] =
                  dot(qf.data() + static_cast<size_t>(g) * D, rowf.data(), D);
            }
          }
        }
        // softmax numerators
        for (int g = 0; g < G; ++g) {
          float* sg = scores.data() + static_cast<size_t>(g) * len;
          float m = neg_inf;
          for (int j = 0; j < len; ++j) m = std::max(m, sg[j]);
          mx[g] = m;
          sm[g] = exp_sub_sum(sg, len, m);
        }
        // pass 2: acc = p @ v
        std::fill(acc.begin(), acc.end(), 0.f);
        t = 0;
        for (int b = 0; t < len; ++b) {
          const int64_t blk = bt[static_cast<int64_t>(i) * W + b];
          const T* vbase = vc + blk * blk_stride + kh * head_stride;
          const int ntok = std::min(BS, len - t);
#ifdef FD_AVX2
          if (fast) {
            for (int g = 0; g < G; ++g) {
              pv_block_avx2<T>(scores.data() + static_cast<size_t>(g) * len + t,
                               vbase, ntok, D,
                               acc.data() + static_cast<size_t>(g) * D);
            }
            t += ntok;
            continue;
          }
#endif
          for (int o = 0; o < ntok; ++o, ++t) {
            to_float<T>(vbase + static_cast<int64_t>(o) * D, rowf.data(), D);
            for (int g = 0; g < G; ++g) {
              axpy(scores[static_cast<size_t>(g) * len + t], rowf.data(),
                   acc.data() + static_cast<size_t>(g) * D, D);
            }
          }
        }
        for (int g = 0; g < G; ++g) {
          const int h = kh * G + g;
          from_float<T>(acc.data() + static_cast<size_t>(g) * D,
                        out + (static_cast<int64_t>(i) * H + h) * D, D,
                        1.f / sm[g]);
          lse[static_cast<int64_t>(h) * n + i] = mx[g] + std::log(sm[g]);
        }
      }
    }
  }
}

template <typename T>
void store_kv_impl(const T* __restrict k, const T* __restrict v,
                   const int64_t* __restrict slots, int m, int KVH, int D,
                   int BS, T* __restrict kc, T* __restrict vc) {
  const int64_t head_stride = static_cast<int64_t>(BS) * D;
  const int64_t blk_stride = static_cast<int64_t>(KVH) * head_stride;
#pragma omp parallel for num_threads(num_threads()) schedule(static)
  for (int j = 0; j < m; ++j) {
    const int64_t s = slots[j];
    const int64_t blk = s / BS;
    const int64_t off = s % BS;
    for (int kh = 0; kh < KVH; ++kh) {
      const int64_t dst = blk * blk_stride + kh * head_stride + off * D;
      const int64_t src = (static_cast<int64_t>(j) * KVH + kh) * D;
      std::memcpy(kc + dst, k + src, sizeof(T) * D);
      std::memcpy(vc + dst, v + src, sizeof(T) * D);
    }
  }
}

void check_cpu_contig(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}


// Dispatch over the dtypes the kernels support (fp16, bf16, fp32).
template <typename F>
void dispatch_dtype(at::ScalarType st, const char* name, F&& f) {
  switch (st) {
    case at::kHalf: f(at::Half{}); break;
    case at::kBFloat16: f(at::BFloat16{}); break;
    case at::kFloat: f(float{}); break;
    default: TORCH_CHECK(false, name, ": unsupported dtype ", st);
  }
}

}  // namespace

// q: [n, H, D]; k_cache/v_cache: [C, KVH, BS, D]; block_table: [n, W] int32
// (CPU-relative block ids, only the CPU-resident blocks, in sequence order);
// lens: [n] int32 (#CPU-resident tokens); out: [n, H, D] (same dtype as q);
// lse: [H, n] float32.
void decode_attn(const torch::Tensor& q, const torch::Tensor& k_cache,
                 const torch::Tensor& v_cache, const torch::Tensor& block_table,
                 const torch::Tensor& lens, double scale, torch::Tensor& out,
                 torch::Tensor& lse) {
  check_cpu_contig(q, "q");
  check_cpu_contig(k_cache, "k_cache");
  check_cpu_contig(v_cache, "v_cache");
  check_cpu_contig(block_table, "block_table");
  check_cpu_contig(lens, "lens");
  check_cpu_contig(out, "out");
  check_cpu_contig(lse, "lse");
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.sizes() == k_cache.sizes());
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 && lens.scalar_type() == torch::kInt32);
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32);
  TORCH_CHECK(q.scalar_type() == k_cache.scalar_type() && out.scalar_type() == q.scalar_type());
  const int n = q.size(0), H = q.size(1), D = q.size(2);
  const int KVH = k_cache.size(1), BS = k_cache.size(2);
  TORCH_CHECK(k_cache.size(3) == D, "head size mismatch");
  TORCH_CHECK(H % KVH == 0, "num_heads must be a multiple of num_kv_heads");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == n && lens.size(0) == n);
  TORCH_CHECK(out.sizes() == q.sizes() && lse.size(0) == H && lse.size(1) == n);
  const int W = block_table.size(1);
  if (n == 0) return;

  dispatch_dtype(q.scalar_type(), "decode_attn", [&](auto tag) {
        using scalar_t = decltype(tag);
        decode_attn_impl<scalar_t>(
            q.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(), block_table.data_ptr<int32_t>(),
            lens.data_ptr<int32_t>(), n, H, KVH, D, BS, W,
            static_cast<float>(scale), out.data_ptr<scalar_t>(),
            lse.data_ptr<float>());
      });
}

// k, v: [m, KVH, D]; slots: [m] int64 (CPU-relative slot ids)
void store_kv(const torch::Tensor& k, const torch::Tensor& v,
              const torch::Tensor& slots, torch::Tensor& k_cache,
              torch::Tensor& v_cache) {
  check_cpu_contig(k, "k");
  check_cpu_contig(v, "v");
  check_cpu_contig(slots, "slots");
  check_cpu_contig(k_cache, "k_cache");
  check_cpu_contig(v_cache, "v_cache");
  TORCH_CHECK(slots.scalar_type() == torch::kInt64);
  TORCH_CHECK(k.scalar_type() == k_cache.scalar_type() && v.scalar_type() == k.scalar_type());
  const int m = k.size(0);
  if (m == 0) return;
  const int KVH = k_cache.size(1), BS = k_cache.size(2), D = k_cache.size(3);
  TORCH_CHECK(k.size(1) == KVH && k.size(2) == D && v.sizes() == k.sizes());
  const int64_t capacity = k_cache.size(0) * static_cast<int64_t>(BS);
  const int64_t* s = slots.data_ptr<int64_t>();
  for (int j = 0; j < m; ++j) {
    TORCH_CHECK(s[j] >= 0 && s[j] < capacity, "CPU slot out of range: ", s[j]);
  }
  dispatch_dtype(k.scalar_type(), "store_kv", [&](auto tag) {
        using scalar_t = decltype(tag);
        store_kv_impl<scalar_t>(k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), s,
                                m, KVH, D, BS, k_cache.data_ptr<scalar_t>(),
                                v_cache.data_ptr<scalar_t>());
      });
}

void set_num_threads(int64_t n) { g_num_threads = static_cast<int>(n); }

int64_t get_num_threads() { return num_threads(); }

bool has_avx2() {
#ifdef FD_AVX2
  return true;
#else
  return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Neither kernel touches Python objects. Dropping the GIL lets the other S/R
  // micro-batch thread keep launching GPU work while they run.
  m.def("decode_attn", &decode_attn, "FastDecode CPU partial decode attention",
        py::call_guard<py::gil_scoped_release>());
  m.def("store_kv", &store_kv, "FastDecode CPU KV store",
        py::call_guard<py::gil_scoped_release>());
  m.def("set_num_threads", &set_num_threads);
  m.def("get_num_threads", &get_num_threads);
  m.def("has_avx2", &has_avx2);
}
