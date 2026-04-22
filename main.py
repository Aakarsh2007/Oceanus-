"""
Oceanus entry point.
Run a demo episode with mock agents and print results.

Usage:
    python main.py                    # demo episode
    python main.py --episodes 5       # multiple episodes
    python main.py --verbose          # verbose output
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oceanus.models import OceanusEnv
from oceanus.adversary import AdversaryAgent
from oceanus.runner import OceanusRunner


def main():
    parser = argparse.ArgumentParser(description="Oceanus: Ghost-Gear Recovery Arena")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chaos_interval", type=int, default=20)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  🌊  OCEANUS: Multi-Layer Ghost-Gear Recovery & Treaty Arena")
    print("  Multi-Agent RL Environment | OpenEnv Compatible")
    print("="*65)

    env = OceanusEnv(seed=args.seed, max_steps=args.max_steps)
    adversary = AdversaryAgent(inject_interval=args.chaos_interval, seed=args.seed)
    runner = OceanusRunner(env, adversary, use_mock=True, verbose=args.verbose)

    all_rewards = []
    all_bios = []

    for ep in range(args.episodes):
        summary = runner.run_episode(episode_id=ep)
        all_rewards.append(summary["total_reward"])
        all_bios.append(summary["biodiversity_final"])

    if args.episodes > 1:
        import numpy as np
        print("\n" + "="*65)
        print(f"  MULTI-EPISODE SUMMARY ({args.episodes} episodes)")
        print(f"  Mean Reward:       {float(np.mean(all_rewards)):.2f} ± {float(np.std(all_rewards)):.2f}")
        print(f"  Mean Biodiversity: {float(np.mean(all_bios)):.1f}%")
        print(f"  Recovery Rate:     {sum(1 for b in all_bios if b >= 75) / len(all_bios) * 100:.0f}%")
        print("="*65)


if __name__ == "__main__":
    main()
