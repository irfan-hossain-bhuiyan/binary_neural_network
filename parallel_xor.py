"""Parallel XOR training across multiple GPUs / processes."""

import concurrent.futures
import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
import torch

from plot_utils import plot_weight_distribution
from train_xor import train_xor_main

# Add the directory containing this script to sys.path so multiprocessing 'spawn'
# can find the module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_train_xor_main_parallel():
    """Run several XOR training processes in parallel."""
    print("Running train_xor_main in parallel across different configs...")
    results = []

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(mp_context=ctx) as executor:
        future_to_r1 = {}
        for idx, reg in enumerate([0.5, 0.5, 0.5, 0.5]):
            dev_id = idx % num_gpus
            future_to_r1[
                executor.submit(train_xor_main, 200, dev_id, print_terminal=False, num_bits=16)
            ] = reg

        for future in concurrent.futures.as_completed(future_to_r1):
            reg = future_to_r1[future]
            try:
                res = future.result()
                results.append((reg, res))
                print(f"========== COMPLETED RUN WITH reg={reg} ==========")
            except Exception as e:
                print(f"========== FAILED RUN WITH reg={reg}: {e} ==========")

    if results:
        plt.figure(figsize=(10, 6))
        results.sort(key=lambda x: x[0])
        for idx, ckpt in results:
            errors = ckpt.avg_errors()
            plt.plot(
                range(1, len(errors) + 1),
                errors,
                label=f"odd_even = {idx}",
                linewidth=2,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Testing Error / Loss")
        plt.title("Effect of r1_scale Regularization on Training")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        for idx, ckpt in results:
            print(f"\n--- Weight Distribution for bias={idx} ---")
            plot_weight_distribution(ckpt.model)

    return results


if __name__ == "__main__":
    run_train_xor_main_parallel()
