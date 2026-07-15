"""Search script for finding the style tensor that makes Kokoro voice sound like the target.

This script implements:
1. Target Embedding outlier removal and OGG preprocessing (delegated to similarity.py).
2. Multi-sentence evaluation (4 sentences during search).
3. Stage 1 Convex optimization with Nelder-Mead over softmax weights.
4. Caching evaluated style tensors to speed up searches.
5. Lightweight Audio Quality penalties (clipping, silence, abnormal RMS).
6. Dimension Importance Analysis before Stage 2.
7. Stage 2 search with:
   - Adaptive step-size scheduling (coarse-to-fine).
   - Candidate = best + global perturbation + small row-specific perturbation.
8. Structured JSON logging of every accepted/rejected candidate to optimization.log.
"""
import os
import argparse
import json
import numpy as np
import torch
from scipy.optimize import minimize

import synth
import similarity as sim

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your order number after the beep.",
    "I will call you back tomorrow at three thirty.",
    "The beautiful sunrise warmed the entire valley.",
    "A peaceful walk in the park refreshes the mind.",
]
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class EvaluationCache:
    """Caches evaluation fitness scores to avoid redundant synthesis."""
    def __init__(self):
        self.cache = {}

    def get(self, tensor):
        # Use rounded tensor bytes as key for quick cache lookup
        flat = tensor.detach().cpu().numpy().flatten()
        rounded = np.round(flat, 4)
        h = hash(rounded.tobytes())
        return self.cache.get(h, None)

    def set(self, tensor, fit, sim_score, penalty):
        flat = tensor.detach().cpu().numpy().flatten()
        rounded = np.round(flat, 4)
        h = hash(rounded.tobytes())
        self.cache[h] = (fit, sim_score, penalty)


class OptimizationLogger:
    """Logs detailed candidate evaluation data to optimization.log."""
    def __init__(self, log_path=None):
        if log_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.final_log_path = os.path.join(script_dir, "optimization.log")
        else:
            self.final_log_path = log_path
        self.temp_log_path = "/tmp/optimization.log"
        # Clean temp file on initialization
        with open(self.temp_log_path, "w") as f:
            f.write("# Voice Cloning Optimization log\n")

    def log(self, iteration, stage, fitness, similarity, penalty, step_size, accepted, best_score):
        entry = {
            "iteration": int(iteration),
            "stage": int(stage),
            "fitness": float(fitness),
            "similarity": float(similarity),
            "penalty": float(penalty),
            "step_size": float(step_size) if step_size is not None else None,
            "accepted": bool(accepted),
            "best_score": float(best_score)
        }
        with open(self.temp_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def finalize(self):
        import shutil
        try:
            shutil.copyfile(self.temp_log_path, self.final_log_path)
            print(f"Logged optimization history to: {self.final_log_path}")
        except Exception as e:
            print(f"Error copying log file: {e}")


def evaluate_quality(wav):
    """Audio quality metrics: clipping, silence, and abnormal RMS penalties."""
    if len(wav) == 0:
        return 0.5  # Heavy penalty for empty audio
    rms = np.sqrt(np.mean(wav**2))
    penalty = 0.0

    # 1. Silence penalty
    if rms < 0.001:
        penalty += 0.3

    # 2. Abnormal RMS energy penalty (speech typically ranges [0.05, 0.25])
    if rms < 0.01:
        penalty += 0.1
    elif rms > 0.4:
        penalty += 0.1

    # 3. Clipping penalty
    clipping_ratio = np.sum(np.abs(wav) > 0.98) / len(wav)
    if clipping_ratio > 0.02:
        penalty += 0.2 * clipping_ratio

    return penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", default="../reference", help="Reference target WAVs folder")
    ap.add_argument("--out", default="voice.pt", help="Output tensor file")
    ap.add_argument("--iters_blend", type=int, default=30, help="Stage 1 iterations")
    ap.add_argument("--iters_perturb", type=int, default=30, help="Stage 2 iterations")
    args = ap.parse_args()

    # Loggers and cache setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.out):
        args.out = os.path.join(script_dir, args.out)
    blend_opt_path = os.path.join(script_dir, "blend_opt.pt")

    logger = OptimizationLogger()
    cache = EvaluationCache()

    target = sim.target_embedding(args.reference_dir)
    voices = synth.stock_voices()

    # --- Stage 1: Convex Mixture Search ---
    print("Ranking stock voices...")
    scores = []
    for name, v in voices.items():
        wav = synth.synthesize(SENTENCES[0], v)
        s = sim.similarity_to_target(wav, target)
        scores.append((s, name))
    scores.sort(reverse=True)

    print("\nTop 10 voices:")
    for s, name in scores[:10]:
        print(f"  {name:20s} {s:.4f}")

    top_names = [name for s, name in scores[:10]]
    top_tensors = [voices[name] for name in top_names]

    x0 = np.zeros(len(top_names))
    best_sim = -1.0
    best_weights = None
    stage1_iter = 0

    def get_fitness(tensor, sentences):
        # Cache check
        cached = cache.get(tensor)
        if cached is not None:
            return cached

        sims = []
        quality_penalties = []
        for text in sentences:
            wav = synth.synthesize(text, tensor)
            sims.append(sim.similarity_to_target(wav, target))
            quality_penalties.append(evaluate_quality(wav))
        
        sim_val = np.mean(sims)
        penalty_val = np.mean(quality_penalties)
        fit = sim_val - penalty_val
        
        cache.set(tensor, fit, sim_val, penalty_val)
        return fit, sim_val, penalty_val

    def loss_function(x):
        nonlocal best_sim, best_weights, stage1_iter
        stage1_iter += 1
        w = softmax(x)
        blended = torch.zeros_like(top_tensors[0])
        for weight, tensor in zip(w, top_tensors):
            blended += weight * tensor

        # Evaluate over multiple diverse sentences (first 4 sentences)
        fit, sim_val, penalty_val = get_fitness(blended, SENTENCES[:4])

        accepted = False
        if fit > best_sim:
            best_sim = fit
            best_weights = w.copy()
            accepted = True
            print(f"[Stage 1] Iter {stage1_iter:2d} | New best fitness: {best_sim:.4f}")

        logger.log(stage1_iter, stage=1, fitness=fit, similarity=sim_val, penalty=penalty_val,
                   step_size=None, accepted=accepted, best_score=best_sim)
        return -fit

    print("\nStarting Stage 1 (Convex Optimization)...")
    minimize(loss_function, x0, method='Nelder-Mead', options={'maxiter': args.iters_blend, 'disp': False})

    print("\nStage 1 Finished! Best Stage 1 fitness:", best_sim)
    best_blend = torch.zeros_like(top_tensors[0])
    for weight, tensor in zip(best_weights, top_tensors):
        best_blend += weight * tensor

    torch.save(best_blend, blend_opt_path)
    print(f"Saved blended voice to: {blend_opt_path}")

    # --- Dimension Importance Analysis ---
    print("\nEstimating dimension importance...")
    dim_sensitivities = np.zeros(256)
    # Check sensitivity on the first sentence
    base_fit, _, _ = get_fitness(best_blend, [SENTENCES[0]])
    
    # Check every 4th dimension to make it fast
    for d in range(0, 256, 4):
        perturbed = best_blend.clone()
        perturbed[:, :, d] += 0.05
        fit, _, _ = get_fitness(perturbed, [SENTENCES[0]])
        sens = abs(fit - base_fit)
        for i in range(4):
            if d + i < 256:
                dim_sensitivities[d + i] = sens

    # Scale sensitivities to create bias weights
    max_sens = np.max(dim_sensitivities)
    if max_sens > 0:
        norm_sens = dim_sensitivities / max_sens
    else:
        norm_sens = np.ones(256)

    # Bias perturbation weights (range [0.1, 1.0])
    bias_weights = 0.1 + 0.9 * norm_sens
    bias_weights = torch.tensor(bias_weights, dtype=torch.float32)
    print("Dimension importance weights generated.")

    # --- Stage 2: Local Perturbation Search ---
    print("\nStarting Stage 2 (Local Perturbation Search)...")
    best = best_blend.clone()
    best_f, best_sim_val, best_pen_val = get_fitness(best, SENTENCES[:4])

    base_step = 0.02
    accepted_stage2 = 0

    for i in range(1, args.iters_perturb + 1):
        # 1. Adaptive step size decay
        step = base_step * (0.94 ** (i - 1))

        # Generate perturbation focused on biased dimensions
        noise_global = torch.randn(256) * bias_weights * step
        delta_global = noise_global.unsqueeze(0).unsqueeze(0).repeat(510, 1, 1)

        # 2. Local row-specific perturbation (magnitude 0.1 of global step)
        noise_local = torch.randn(510, 1, 256) * bias_weights * (0.1 * step)
        
        cand = best + delta_global + noise_local
        fit, sim_val, penalty_val = get_fitness(cand, SENTENCES[:4])

        accepted = False
        if fit > best_f:
            best, best_f, accepted = cand, fit, True
            best_sim_val, best_pen_val = sim_val, penalty_val
            accepted_stage2 += 1
            print(f"[Stage 2] Iter {i:2d} | Accepted #{accepted_stage2} | Fitness: {best_f:.6f} (Sim: {sim_val:.4f}, Pen: {penalty_val:.4f})")

        logger.log(i, stage=2, fitness=fit, similarity=sim_val, penalty=penalty_val,
                   step_size=step, accepted=accepted, best_score=best_f)

    torch.save(best, args.out)
    print(f"\nSaved final optimized voice to: {args.out}")

    # Evaluate final voice on all 5 sentences (robust validation)
    print("\nEvaluating final voice on all 5 sentences...")
    final_sims = []
    for text in SENTENCES:
        wav = synth.synthesize(text, best)
        final_sims.append(sim.similarity_to_target(wav, target))
    
    print(f"\nFinal Voice Scores:")
    for text, s in zip(SENTENCES, final_sims):
        print(f"  Sentence: {text[:25]}... | Similarity: {s:.4f}")
    print(f"Average Final Similarity: {np.mean(final_sims):.4f}")
    
    # Save log output to final location
    logger.finalize()


if __name__ == "__main__":
    main()
