"""
Baseline vs. Trained model evaluation.
Generates reward curves and comparison metrics.

Usage:
    python train/evaluate.py                          # baseline only
    python train/evaluate.py --model_path ./model     # compare with trained
"""
import json
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oceanus.models import OceanusEnv
from oceanus.adversary import AdversaryAgent
from oceanus.runner import OceanusRunner


def run_baseline_eval(n_episodes: int = 5, seed: int = 42) -> dict:
    print("\n[BASELINE] Running evaluation...")
    results = []
    for i in range(n_episodes):
        env = OceanusEnv(seed=seed + i, max_steps=100)
        adversary = AdversaryAgent(inject_interval=20, seed=seed + i)
        runner = OceanusRunner(env, adversary, use_mock=True, verbose=False)
        summary = runner.run_episode(episode_id=i)
        results.append(summary)
        print(
            f"  Episode {i}: reward={summary['total_reward']:.1f}, "
            f"bio={summary['biodiversity_final']:.1f}%, "
            f"cleaned={summary['total_cleaned']}"
        )
    return aggregate_results(results, label="Baseline")


def run_trained_eval(model_path: str, n_episodes: int = 5, seed: int = 42) -> dict:
    print(f"\n[TRAINED] Running evaluation from {model_path}...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        model.eval()

        def llm_fn(prompt: str) -> str:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

        results = []
        for i in range(n_episodes):
            env = OceanusEnv(seed=seed + i, max_steps=100)
            adversary = AdversaryAgent(inject_interval=20, seed=seed + i)
            runner = OceanusRunner(env, adversary, use_mock=False, llm_fn=llm_fn, verbose=False)
            summary = runner.run_episode(episode_id=i)
            results.append(summary)
            print(
                f"  Episode {i}: reward={summary['total_reward']:.1f}, "
                f"bio={summary['biodiversity_final']:.1f}%, "
                f"cleaned={summary['total_cleaned']}"
            )
        return aggregate_results(results, label="Trained")

    except Exception as e:
        print(f"  Could not load trained model: {e}")
        return {}


def aggregate_results(results: list, label: str) -> dict:
    rewards = [r["total_reward"] for r in results]
    bios = [r["biodiversity_final"] for r in results]
    cleaned = [r["total_cleaned"] for r in results]
    treaties = [1 if "Active" in r.get("treaty_status", "") else 0 for r in results]

    agg = {
        "label": label,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_biodiversity": float(np.mean(bios)),
        "mean_cleaned": float(np.mean(cleaned)),
        "treaty_success_rate": float(np.mean(treaties)),
        "raw": results,
    }

    print(f"\n  [{label}] Summary:")
    print(f"    Mean Reward:       {agg['mean_reward']:.2f} +/- {agg['std_reward']:.2f}")
    print(f"    Mean Biodiversity: {agg['mean_biodiversity']:.1f}%")
    print(f"    Mean Nets Cleaned: {agg['mean_cleaned']:.1f}")
    print(f"    Treaty Success:    {agg['treaty_success_rate']*100:.0f}%")
    return agg


def plot_comparison(baseline: dict, trained: dict, output_path: str = "reward_comparison.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Oceanus: Baseline vs. Trained Agent Performance", fontsize=14, fontweight="bold")

        metrics = [
            ("mean_reward", "Total Episode Reward", "Reward"),
            ("mean_biodiversity", "Final Biodiversity Index (%)", "%"),
            ("mean_cleaned", "Ghost Nets Cleaned", "Count"),
        ]
        colors = {"Baseline": "#e74c3c", "Trained": "#2ecc71"}

        for ax, (metric, title, ylabel) in zip(axes, metrics):
            b_val = baseline.get(metric, 0)
            t_val = trained.get(metric, 0)
            vals = [b_val, t_val]
            labels = [baseline.get("label", "Baseline"), trained.get("label", "Trained")]
            bar_colors = [colors.get(l, "#888") for l in labels]

            bars = ax.bar(labels, vals, color=bar_colors, width=0.5,
                          edgecolor="black", linewidth=0.8)

            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(max(vals, default=1)) * 0.02,
                    f"{val:.1f}", ha="center", va="bottom", fontweight="bold"
                )

            max_val = max(abs(v) for v in vals) if vals else 1
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel)
            # Safe ylim that handles negative values
            ax.set_ylim(min(0, min(vals)) * 1.2 - 1, max(vals) * 1.3 + 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n  Comparison plot saved to {output_path}")

    except ImportError:
        print("  matplotlib not available. Skipping plot.")


def plot_reward_curve(step_logs: list, label: str, output_path: str = "reward_curve.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not step_logs:
            return

        steps = [s["step"] for s in step_logs]
        rewards = [s["rewards"].get("__total__", 0) for s in step_logs]
        bios = [s["biodiversity"] for s in step_logs]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Oceanus Training Curves - {label}", fontsize=13, fontweight="bold")

        window = min(10, max(1, len(rewards)))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

        ax1.plot(steps[:len(smoothed)], smoothed, color="#3498db", linewidth=2,
                 label="Smoothed Reward")
        ax1.plot(steps, rewards, color="#3498db", alpha=0.2, linewidth=1)
        ax1.set_ylabel("Step Reward")
        ax1.legend()
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax2.plot(steps, bios, color="#2ecc71", linewidth=2, label="Biodiversity Index")
        ax2.axhline(y=75, color="#e74c3c", linestyle="--", alpha=0.7,
                    label="Recovery Threshold (75%)")
        ax2.set_ylabel("Biodiversity (%)")
        ax2.set_xlabel("Step")
        ax2.legend()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Reward curve saved to {output_path}")

    except ImportError:
        print("  matplotlib not available. Skipping plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline = run_baseline_eval(n_episodes=args.n_episodes, seed=args.seed)

    if args.model_path:
        trained = run_trained_eval(args.model_path, n_episodes=args.n_episodes, seed=args.seed)
        if trained:
            plot_comparison(baseline, trained)
    else:
        print("\n  Tip: Pass --model_path to compare against a trained model")

    # Plot reward curve from baseline
    if baseline.get("raw"):
        all_steps = []
        for ep in baseline["raw"]:
            all_steps.extend(ep.get("step_log", []))
        if all_steps:
            plot_reward_curve(all_steps, label="Baseline")
