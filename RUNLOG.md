# Run Log

This document tracks each optimization run, showing settings, fitness function design, similarity scores, and observations on audio intelligibility.

---

## Run 1: Stock Voice Baseline
*   **Fitness Design**: Single-sentence cosine speaker-similarity score using Resemblyzer.
*   **Settings**: Stock voices loaded from local cache.
*   **Score (Similarity to Target)**:
    *   Top stock voice (`zm_yunxia`): **0.6272**
    *   Naive 50/50 blend of `zm_yunxia` + `af_nova`: **~0.6025**
*   **Auditory Inspection**: Clear and highly intelligible pronunciation (as they are stock voices), but did not sound closely like the target speaker's voice characteristics.
*   **Action**: Use the stock voice result (**0.6272**) as the baseline to beat. Proceed to mixture weight optimization.

---

## Run 2: Stage 1 - Convex Combination Search (Nelder-Mead)
*   **Fitness Design**: Average cosine speaker-similarity score evaluated over 4 different sentences to prevent sentence overfitting. Includes audio quality penalties for clipping and silence.
*   **Settings**: Optimization of softmax weights $w_i \ge 0, \sum w_i = 1$ over top 10 diverse stock voices using Nelder-Mead simplex optimization. Max iterations = 20.
*   **Top 10 voices ranked**:
    *   zm_yunxia (0.6272), af_nova (0.6133), hf_beta (0.6065), if_sara (0.6006)
    *   hm_omega (0.5833), af_heart (0.5750), jf_nezumi (0.5724), ff_siwis (0.5714)
    *   af_jessica (0.5658), hf_alpha (0.5651)
*   **Score**:
    *   Optimized Mixture (Stage 1 best): **0.6567**
*   **Auditory Inspection**: Excellent intelligibility, no robotic artifacts. Voice timbre shifted toward target speaker's vocal traits.
*   **Action**: Save `blend_opt.pt` and use it as the starting point for Stage 2 local perturbation.

---

## Run 3: Stage 2 - Adaptive Hybrid Perturbation Search (Final Submission)
*   **Fitness Design**: Four-sentence average speaker-similarity with hybrid perturbation strategy:
    *   Global: same 256-dim noise vector replicated across all 510 rows (structured shift)
    *   Local: independent row-level noise at 10% of global step magnitude
    *   Dimension importance weights estimated from sensitivity analysis before search begins
    *   Adaptive step decay: `step = 0.02 × 0.94^(iter-1)`
*   **Settings**: Stage 2 iterations = 30. Audio quality penalty = 0.0000 throughout (all candidates were natural and clear).
*   **Score**:
    *   Stage 1 starting point: **0.6567**
    *   Stage 2 final accepted fitness: **0.6811**
    *   **Verification across 5 sentences (unseen hold-out):**
        *   The quick brown fox jumps... : **0.6845**
        *   Please confirm your order... : **0.6802**
        *   I will call you back... : **0.6791**
        *   The beautiful sunrise... : **0.6728**
        *   A peaceful walk in the park... : **0.6734**
        *   **Average: 0.6780**
*   **Auditory Inspection**: Natural-sounding voice with recognizable speaker characteristics. Clear pronunciation without raspiness.
*   **Reference Audio**: Target speaker OGG file automatically preprocessed into split WAV segments, silence-trimmed, loudness-normalized, and outlier embeddings removed before computing the target embedding.
*   **Action**: Selected as the final submission tensor `voice.pt`.
