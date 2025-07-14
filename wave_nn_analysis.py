# wave_nn_analysis.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from torch.utils.data import DataLoader, Dataset

from wave_nn import ASTorqueModel, RPM_GRID

# ─────────────────────────────── CONFIG ────────────────────────────── #
MODEL_PARAMS = dict(hid=256, nhead=4, nlayer=8, topk=8, local_window=4, dropout=0.05)
DATASET_PATH = "wave data set.csv"
MODEL_PATH = "wave.nn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────── DATASET ─────────────────────────────── #
class TorqueDataset(Dataset):
    def __init__(self, csv_file: str):
        df = pd.read_csv(
            csv_file,
            header=None,
            names=["hdr1", "hdr2", "runner", "plenum", "rpm", "torque"],
        ).sort_values(["hdr1", "hdr2", "runner", "plenum", "rpm"])

        statics, curves = [], []
        for key, g in df.groupby(["hdr1", "hdr2", "runner", "plenum"]):
            statics.append(list(key))
            curves.append(np.interp(RPM_GRID, g.rpm, g.torque))

        S = np.asarray(statics, np.float32)
        T = np.stack(curves).astype(np.float32)

        self.Xs = torch.from_numpy((S - S.mean(0)) / (S.std(0) + 1e-8))
        self.Y_mean = T.mean()
        self.Y_std = T.std() + 1e-8
        self.Y = torch.from_numpy((T - self.Y_mean) / self.Y_std)

        self.S_mean, self.S_std = S.mean(0), S.std(0) + 1e-8
        self.rpm_norm = torch.from_numpy(
            (RPM_GRID - RPM_GRID.mean()) / (RPM_GRID.std() + 1e-8)
        ).float()
        self.S_raw = S    # retain for labelling

    def __len__(self): return len(self.Xs)
    def __getitem__(self, idx): return self.Xs[idx], self.Y[idx]

# ────────────────────────────── ANALYSIS ───────────────────────────── #
def analyze():
    # Load normalization stats
    S_mean = np.load("S_mean.npy")
    S_std = np.load("S_std.npy")
    Y_mean = np.load("Y_mean.npy").item()
    Y_std = np.load("Y_std.npy").item()

    # Load dataset
    ds = TorqueDataset(DATASET_PATH)
    n_train = int(0.8 * len(ds))
    # Use the same split as in training for fair residuals
    _, vl_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(42))
    ld_vl = DataLoader(vl_ds, batch_size=64)

    # Load model
    model = ASTorqueModel(**MODEL_PARAMS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    rpm_base = ds.rpm_norm.to(DEVICE).unsqueeze(0).unsqueeze(0)

    # ───────── Residual vs RPM + ACF + Ljung-Box test ──────────── #
    print("Calculating residuals...")
    residuals = []
    with torch.no_grad():
        for S, Y in ld_vl:
            S, Y = S.to(DEVICE), Y.to(DEVICE)
            r = rpm_base.expand(len(S), -1, -1)
            pred = model(r, S)
            # Unnormalize for analysis
            residuals.append((pred - Y).cpu() * Y_std)
    residuals = torch.cat(residuals).numpy()
    rpm_mean = residuals.mean(0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(RPM_GRID, rpm_mean)
    axes[0].set_title("Mean residual vs RPM")
    axes[0].set_xlabel("RPM")
    axes[0].set_ylabel("Residual (Nm)")

    plot_acf(rpm_mean, ax=axes[1], lags=20)
    axes[1].set_title("ACF of residuals")
    plt.tight_layout()
    plt.show()

    lb = acorr_ljungbox(rpm_mean, lags=[10], return_df=True)
    print("Ljung-Box lag-10 p-value:", lb["lb_pvalue"].iloc[0])

    # ───── Characteristic torque-curve figure (3 sub-plots) ────── #
    print("Plotting characteristic torque curves...")
    with torch.no_grad():
        S_ex, Y_ex = next(iter(ld_vl))
        S_ex, Y_ex = S_ex.to(DEVICE), Y_ex.to(DEVICE)
        pred_ex = model(rpm_base.expand(len(S_ex), -1, -1), S_ex).cpu()

    # pick three samples: min-runner, median-runner, max-runner for clarity
    runner_idx = 2  # runner length column index (hdr1,hdr2,runner,plenum)
    runner_vals = (S_ex.cpu() * torch.from_numpy(S_std) + torch.from_numpy(S_mean))[..., runner_idx]
    idx_small = runner_vals.argmin().item()
    idx_big   = runner_vals.argmax().item()
    idx_mid   = int(np.median(np.argsort(runner_vals)))

    chosen = [("Small", idx_small), ("Mid", idx_mid), ("Big", idx_big)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle("Characteristic torque curves: actual vs predicted")

    for ax, (tag, idx) in zip(axes, chosen):
        hdr1, hdr2, runner, plenum = (S_ex[idx].cpu() * torch.from_numpy(S_std) + torch.from_numpy(S_mean))[:4]
        lbl = f"{tag}: hdr1={hdr1:.1f}, hdr2={hdr2:.1f}, runner={runner:.1f}, plenum={plenum:.1f}"
        ax.plot(RPM_GRID, Y_ex[idx].cpu() * Y_std + Y_mean, label=f"{tag} actual")
        ax.plot(RPM_GRID, pred_ex[idx] * Y_std + Y_mean, "--", label=f"{tag} predicted")
        ax.set_ylabel("Torque (Nm)")
        ax.set_title(lbl)
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("RPM")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    analyze()
