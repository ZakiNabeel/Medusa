/*
 * entropy_kernel.cl — Shannon Entropy via Parallel Reduction
 * ===========================================================
 * Computes H(P) = -sum(P(i) * log(P(i))) for each row of a
 * (num_nodes, vocab_size) probability matrix.
 *
 * Launch geometry:
 *   global_size = (num_nodes * LOCAL_SIZE,)
 *   local_size  = (LOCAL_SIZE,)          <- must be power-of-2
 *
 * One work-group  <-> one node row.
 * LOCAL_SIZE threads cooperatively reduce vocab_size elements.
 *
 * Memory layout:
 *   probs  [num_nodes * vocab_size]  float32  (row-major)
 *   output [num_nodes]               float32
 */

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 256          /* override at compile time if desired */
#endif

#define EPSILON 1e-10f          /* guards against log(0) */

/* -----------------------------------------------------------------
 * Warp-level (sub-group) reduction using shuffle — optional fast path
 * guarded by the SUB_GROUP_SIZE define which the host sets when the
 * device reports a non-zero preferred sub-group size.
 * ----------------------------------------------------------------- */
#ifdef SUB_GROUP_SIZE
inline float warp_reduce(float val) {
    for (int offset = SUB_GROUP_SIZE / 2; offset > 0; offset >>= 1)
        val += sub_group_shuffle_down(val, offset);
    return val;
}
#endif

/* -----------------------------------------------------------------
 * Main entropy kernel
 * ----------------------------------------------------------------- */
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
void compute_entropy(
    __global const float* restrict probs,   /* [num_nodes, vocab_size] */
    __global       float* restrict output,  /* [num_nodes]             */
    const int vocab_size
) {
    /* ---- Identify this work-item ---------------------------------- */
    const int node_id  = get_group_id(0);           /* one group = one node */
    const int lid      = get_local_id(0);            /* lane within group    */

    /* Pointer to the start of this node's probability row */
    __global const float* p = probs + (long)node_id * vocab_size;

    /* ---- Phase 1: each lane accumulates a partial sum ------------- */
    /*
     * Stride loop: lane `lid` covers indices lid, lid+LOCAL_SIZE,
     * lid+2*LOCAL_SIZE, … so every vocab element is processed exactly once
     * and the loop is fully coalesced.
     */
    float partial = 0.0f;
    for (int i = lid; i < vocab_size; i += LOCAL_SIZE) {
        float pi = p[i];
        /*
         * Fused conditional: when pi <= EPSILON the contribution is
         * effectively zero so we skip the log evaluation entirely.
         * This avoids branch divergence on zero-probability tokens that
         * are common after softmax at low temperature.
         */
        if (pi > EPSILON) {
            partial -= pi * native_log(pi);   /* native_log: ~4 ULP, ~2× faster */
        }
    }

    /* ---- Phase 2: parallel tree reduction in local (shared) memory  */
    __local float scratch[LOCAL_SIZE];
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    /*
     * Classic binary-tree reduction.
     * Unrolled for common LOCAL_SIZE values (256, 128, 64, 32, 16).
     * Each stage halves the active set; inactive lanes stay idle.
     */
#if LOCAL_SIZE >= 512
    if (lid < 256) scratch[lid] += scratch[lid + 256];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if LOCAL_SIZE >= 256
    if (lid < 128) scratch[lid] += scratch[lid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if LOCAL_SIZE >= 128
    if (lid < 64) scratch[lid] += scratch[lid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    /* Last 64 elements: still needs barriers on non-Nvidia devices */
    if (lid < 32) scratch[lid] += scratch[lid + 32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 16) scratch[lid] += scratch[lid + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <  8) scratch[lid] += scratch[lid +  8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <  4) scratch[lid] += scratch[lid +  4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <  2) scratch[lid] += scratch[lid +  2];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Lane 0 writes the final entropy for this node */
    if (lid == 0) {
        output[node_id] = scratch[0] + scratch[1];
    }
}

/* -----------------------------------------------------------------
 * Softmax + Entropy fused kernel  (optional, avoids a round-trip)
 * -----------------------------------------------------------------
 * Input:  raw logits  [num_nodes, vocab_size]
 * Output: entropy     [num_nodes]
 *
 * Uses two passes over local memory:
 *   Pass A — find max  (for numerical stability)
 *   Pass B — exp, sum, normalise, accumulate -p*log(p)
 *
 * The intermediate softmax is never materialised in global memory,
 * saving 15 × 50257 × 4 ≈ 3 MB of bandwidth per decoding step.
 * ----------------------------------------------------------------- */
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
void logits_to_entropy(
    __global const float* restrict logits,  /* [num_nodes, vocab_size] */
    __global       float* restrict output,  /* [num_nodes]             */
    const int vocab_size
) {
    const int node_id = get_group_id(0);
    const int lid     = get_local_id(0);
    __global const float* row = logits + (long)node_id * vocab_size;

    __local float scratch[LOCAL_SIZE];

    /* --- Pass A: parallel max reduction for numerical stability --- */
    float local_max = -INFINITY;
    for (int i = lid; i < vocab_size; i += LOCAL_SIZE)
        local_max = fmax(local_max, row[i]);

    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

#if LOCAL_SIZE >= 512
    if (lid < 256) scratch[lid] = fmax(scratch[lid], scratch[lid+256]);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if LOCAL_SIZE >= 256
    if (lid < 128) scratch[lid] = fmax(scratch[lid], scratch[lid+128]);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if LOCAL_SIZE >= 128
    if (lid <  64) scratch[lid] = fmax(scratch[lid], scratch[lid+64]);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    if (lid <  32) scratch[lid] = fmax(scratch[lid], scratch[lid+32]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <  16) scratch[lid] = fmax(scratch[lid], scratch[lid+16]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   8) scratch[lid] = fmax(scratch[lid], scratch[lid+8]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   4) scratch[lid] = fmax(scratch[lid], scratch[lid+4]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   2) scratch[lid] = fmax(scratch[lid], scratch[lid+2]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   1) scratch[0]   = fmax(scratch[0],   scratch[1]);
    barrier(CLK_LOCAL_MEM_FENCE);

    const float row_max = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* --- Pass B: sum of exp(x - max) ------------------------------ */
    float local_sum = 0.0f;
    for (int i = lid; i < vocab_size; i += LOCAL_SIZE)
        local_sum += native_exp(row[i] - row_max);

    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

#if LOCAL_SIZE >= 512
    if (lid < 256) scratch[lid] += scratch[lid+256]; barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if LOCAL_SIZE >= 256
    if (lid < 128) scratch[lid] += scratch[lid+128]; barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if LOCAL_SIZE >= 128
    if (lid <  64) scratch[lid] += scratch[lid+64];  barrier(CLK_LOCAL_MEM_FENCE);
#endif
    if (lid <  32) scratch[lid] += scratch[lid+32]; barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <  16) scratch[lid] += scratch[lid+16]; barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   8) scratch[lid] += scratch[lid+8];  barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   4) scratch[lid] += scratch[lid+4];  barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   2) scratch[lid] += scratch[lid+2];  barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   1) scratch[0]   += scratch[1];
    barrier(CLK_LOCAL_MEM_FENCE);

    const float inv_sum = 1.0f / scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* --- Pass C: compute -p*log(p) and reduce --------------------- */
    float partial_H = 0.0f;
    for (int i = lid; i < vocab_size; i += LOCAL_SIZE) {
        float pi = native_exp(row[i] - row_max) * inv_sum;
        if (pi > EPSILON)
            partial_H -= pi * native_log(pi);
    }

    scratch[lid] = partial_H;
    barrier(CLK_LOCAL_MEM_FENCE);

#if LOCAL_SIZE >= 512
    if (lid < 256) scratch[lid] += scratch[lid+256]; barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if LOCAL_SIZE >= 256
    if (lid < 128) scratch[lid] += scratch[lid+128]; barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if LOCAL_SIZE >= 128
    if (lid <  64) scratch[lid] += scratch[lid+64];  barrier(CLK_LOCAL_MEM_FENCE);
#endif
    if (lid <  32) scratch[lid] += scratch[lid+32]; barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <  16) scratch[lid] += scratch[lid+16]; barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   8) scratch[lid] += scratch[lid+8];  barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   4) scratch[lid] += scratch[lid+4];  barrier(CLK_LOCAL_MEM_FENCE);
    if (lid <   2) scratch[lid] += scratch[lid+2];  barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0)
        output[node_id] = scratch[0] + scratch[1];
}
