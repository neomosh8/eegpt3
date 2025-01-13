import re
import numpy as np
import matplotlib.pyplot as plt

def parse_log_file(filepath):
    """
    Example line in your log file:
        3 train loss: 8.421755 lr: 1.0499e-05 | norm 6.2157
    Adjust the regex if your format differs.
    """
    iteration_list = []
    loss_list = []
    lr_list = []

    pattern = re.compile(
        r"^(\d+)\s+train loss:\s+([\d\.e\+\-]+)\s+lr:\s+([\d\.e\+\-]+)"
    )

    with open(filepath, "r") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                it = int(match.group(1))
                ls = float(match.group(2))
                lr = float(match.group(3))

                iteration_list.append(it)
                loss_list.append(ls)
                lr_list.append(lr)

    # Convert to numpy arrays for convenience
    return np.array(iteration_list), np.array(loss_list), np.array(lr_list)

def main():
    # 1. Parse log file
    iteration_list, loss_list, lr_list = parse_log_file("log.txt")  # change if needed

    # If there's fewer than 2 entries, we can't compute a ratio
    if len(loss_list) < 2:
        print("Not enough data to compute consecutive-loss ratios.")
        return

    # 2. Compute ratio r_i = loss[i+1] / loss[i] (length = N-1)
    ratios = loss_list[1:] / loss_list[:-1]

    # We'll align ratio[i] with iteration i
    # so ratio[i] is from iteration_list[i] to iteration_list[i+1].
    iteration_for_ratio = iteration_list[:-1]
    lr_for_ratio = lr_list[:-1]  # also length N-1

    # 3. Plot ratio vs. LR (log scale), color-coded by iteration i
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(
        lr_for_ratio,
        ratios,
        c=iteration_for_ratio,  # color by iteration index
        cmap="plasma",
        edgecolors='k',
        alpha=0.8
    )
    plt.xscale('log')  # LR on log scale
    cbar = plt.colorbar(sc)
    cbar.set_label("Iteration (i)")

    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Consecutive-Loss Ratio (loss[i+1] / loss[i])")
    plt.title("Consecutive-Loss Ratio vs. LR")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # 4. Two simple heuristics to find a candidate “optimal” LR:
    # --------------------------------------------------------------------

    # Heuristic #1: Fastest Drop LR
    #   => ratio is minimized, i.e. the biggest single-step % drop in loss
    min_ratio_idx = np.argmin(ratios)
    best_lr_fastest_drop = lr_for_ratio[min_ratio_idx]
    best_iter_fastest_drop = iteration_for_ratio[min_ratio_idx]
    print(f"[Heuristic #1] Fastest Drop -> Iteration {best_iter_fastest_drop}, ratio={ratios[min_ratio_idx]:.4f}, LR={best_lr_fastest_drop:.8g}")

    # Heuristic #2: Largest Stable LR
    #   => among ratio < 1.0 (loss actually decreased), pick the largest LR
    stable_mask = ratios < 1.0
    if np.any(stable_mask):
        # get LR among stable steps
        stable_lrs = lr_for_ratio[stable_mask]
        # pick the largest LR from those stable steps
        best_lr_stable = np.max(stable_lrs)
        print(f"[Heuristic #2] Largest Stable LR: {best_lr_stable:.8g}")
    else:
        print("[Heuristic #2] No stable steps found (all ratios >= 1.0) -> training might be diverging!")

if __name__ == "__main__":
    main()
