# plot_results.py
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_result_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    # Drop preallocated all-zero rows (common when saving a fixed-size array)
    df = df[(df.abs().sum(axis=1) > 0)].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def save_plot(x, y_list, labels, title, ylabel, out_path):
    plt.figure()
    for y, lab in zip(y_list, labels):
        plt.plot(x, y, label=lab)
    if len(labels) > 1:
        plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, help="Experiment folder that contains result.csv")
    ap.add_argument("--csv", default=None, help="Optional explicit path to result.csv")
    args = ap.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    csv_path = os.path.abspath(args.csv) if args.csv else os.path.join(exp_dir, "result.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find result.csv at: {csv_path}")

    df = read_result_csv(csv_path)
    if df.empty:
        raise RuntimeError("result.csv has no non-zero rows after cleaning. Nothing to plot.")

    # Columns (as written by your traintest.py)
    # 0: (mAP or acc), 1: AUC, 2: AvgPrecision, 3: AvgRecall, 4: d_prime,
    # 5: train_loss, 6: valid_loss, 7: (cum_mAP or cum_acc), 8: cum_AUC, 9: lr
    metric_name = "mAP_or_acc"
    cum_metric_name = "cum_mAP_or_cum_acc"

    epochs = np.arange(1, len(df) + 1)

    out_dir = os.path.join(exp_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    save_plot(
        epochs,
        [df[5].values, df[6].values],
        ["train_loss", "valid_loss"],
        "Train vs Validation Loss",
        "Loss",
        os.path.join(out_dir, "loss_curve.png"),
    )

    save_plot(
        epochs,
        [df[0].values, df[7].values],
        [metric_name, cum_metric_name],
        f"{metric_name} over Epochs",
        metric_name,
        os.path.join(out_dir, "metric_curve.png"),
    )

    save_plot(
        epochs,
        [df[1].values, df[8].values],
        ["AUC", "cum_AUC"],
        "AUC over Epochs",
        "AUC",
        os.path.join(out_dir, "auc_curve.png"),
    )

    save_plot(
        epochs,
        [df[9].values],
        ["lr"],
        "Learning Rate over Epochs",
        "Learning Rate",
        os.path.join(out_dir, "lr_curve.png"),
    )

    save_plot(
        epochs,
        [df[4].values],
        ["d_prime"],
        "d' over Epochs",
        "d'",
        os.path.join(out_dir, "dprime_curve.png"),
    )

    best_epoch = int(df[0].idxmax() + 1)
    best_val = float(df[0].max())

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"csv: {csv_path}\n")
        f.write(f"epochs_plotted: {len(df)}\n")
        f.write(f"best_epoch_by_col0: {best_epoch}\n")
        f.write(f"best_value_col0: {best_val:.6f}\n")

    print(f"Saved plots to: {out_dir}")
    print("Created: loss_curve.png, metric_curve.png, auc_curve.png, lr_curve.png, dprime_curve.png, summary.txt")


if __name__ == "__main__":
    main()