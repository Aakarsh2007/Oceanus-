"""
GRPO Training Script using HuggingFace TRL + Unsloth
Compatible with Google Colab T4 GPU.

Usage:
    python train/train_grpo.py --model Qwen/Qwen1.5-0.5B-Instruct --steps 500
"""
import argparse
import json
import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oceanus.models import OceanusEnv
from oceanus.adversary import AdversaryAgent


def get_args():
    parser = argparse.ArgumentParser(description="Train Oceanus agents with GRPO")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max_ep_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output_dir", type=str, default="./oceanus_grpo_model")
    parser.add_argument("--use_unsloth", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_training_samples(
    env: OceanusEnv, adversary: AdversaryAgent, n_rollouts: int = 10
) -> List[Dict]:
    """Generate (prompt, agent_id) pairs for GRPO training."""
    samples = []

    for rollout_idx in range(n_rollouts):
        obs_all = env.reset()
        done = False

        while not done:
            for agent_id, agent_obs in obs_all.items():
                samples.append({
                    "prompt": agent_obs["prompt"],
                    "agent_id": agent_id,
                    "step": agent_obs["observation"].get("step", 0),
                    "rollout": rollout_idx,
                })

            dummy_actions = {aid: '{"intent": "scan"}' for aid in obs_all}
            obs_all, rewards, done, info = env.step(dummy_actions)

            if adversary.should_inject(info["step"]):
                adversary.inject(env.state)

    return samples


def reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    GRPO reward function.
    Evaluates each LLM completion against the Oceanus reward model.
    """
    from oceanus.models import parse_asv_action, parse_policy_action

    rewards = []

    for completion, prompt in zip(completions, prompts):
        is_asv = "Autonomous Surface Vehicle" in prompt or "ASV-" in prompt

        r = 0.0
        if is_asv:
            action, valid = parse_asv_action(completion)
            if valid and action:
                intent = action.get("intent", "")
                if intent == "clean":
                    r = 3.0
                elif intent == "broadcast" and action.get("message"):
                    r = 1.0
                elif intent == "move" and action.get("direction") in (
                    "north", "south", "east", "west"
                ):
                    r = 0.2
                elif intent == "scan":
                    r = 0.1
                else:
                    r = -0.5
            else:
                r = -1.0
        else:
            action, valid = parse_policy_action(completion)
            if valid and action:
                intent = action.get("intent", "")
                if intent == "accept_treaty":
                    r = 8.0
                elif intent == "propose_treaty" and action.get("content"):
                    r = 4.0
                elif intent == "reply_email" and action.get("content"):
                    r = 2.0
                else:
                    r = -0.5
            else:
                r = -1.0

        rewards.append(r)

    return rewards


def train_with_trl(args):
    """Main training loop using TRL GRPOTrainer."""
    print(f"Loading model: {args.model}")

    try:
        if args.use_unsloth:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
            print("  Unsloth model loaded with 4-bit quantization")
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype="auto", device_map="auto"
            )
            print("  HuggingFace model loaded")

    except ImportError as e:
        print(f"  Import error: {e}")
        print("  Install with: pip install unsloth trl transformers")
        return None

    # Build training dataset
    print("Generating training rollouts...")
    env = OceanusEnv(seed=args.seed, max_steps=args.max_ep_steps)
    adversary = AdversaryAgent(inject_interval=20, seed=args.seed)
    samples = build_training_samples(env, adversary, n_rollouts=5)

    from datasets import Dataset
    dataset = Dataset.from_list([
        {"prompt": s["prompt"], "agent_id": s["agent_id"]}
        for s in samples
    ])
    print(f"  Dataset: {len(dataset)} samples")

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        max_steps=args.steps,
        logging_steps=10,
        save_steps=100,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        num_generations=4,
        max_new_tokens=256,
        temperature=0.9,
        beta=0.04,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print(f"Starting GRPO training for {args.steps} steps...")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Model saved to {args.output_dir}")

    return trainer


if __name__ == "__main__":
    args = get_args()
    train_with_trl(args)
