import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, random_split

from wave_nn import ASTorqueModel, RPM_GRID

# ─────────────────────────────── CONFIG ────────────────────────────── #
MODEL_PARAMS = dict(hid=256, nhead=4, nlayer=8, topk=8, local_window=4, dropout=0.05)
TRAINING_PARAMS = dict(batch_size=64, epochs=30, lr=3e-4, weight_decay=1e-3, deriv_lambda=0.3)

DATASET_PATH = "wave data set.csv"
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


# ──────────────────────────── TRAIN LOOP ───────────────────────────── #
def train() -> None:
    ds = TorqueDataset(DATASET_PATH)
    n_train = int(0.8 * len(ds))
    tr_ds, vl_ds = random_split(ds, [n_train, len(ds) - n_train])
    ld_tr = DataLoader(tr_ds, batch_size=TRAINING_PARAMS["batch_size"], shuffle=True)
    ld_vl = DataLoader(vl_ds, batch_size=TRAINING_PARAMS["batch_size"])

    rpm_base = ds.rpm_norm.to(DEVICE).unsqueeze(0).unsqueeze(0)

    model = ASTorqueModel(**MODEL_PARAMS).to(DEVICE)
    opt = AdamW(model.parameters(), lr=TRAINING_PARAMS["lr"], weight_decay=TRAINING_PARAMS["weight_decay"])
    sched = OneCycleLR(opt, max_lr=TRAINING_PARAMS["lr"], epochs=TRAINING_PARAMS["epochs"],
                       steps_per_epoch=len(ld_tr), pct_start=0.3)

    tr_loss, vl_loss = [], []
    for ep in range(TRAINING_PARAMS["epochs"]):
        # ---------- Training ----------
        model.train()
        run = 0.0
        for S, Y in ld_tr:
            S, Y = S.to(DEVICE), Y.to(DEVICE)
            r = rpm_base.expand(len(S), -1, -1)

            S_aug = S + 0.01 * torch.randn_like(S)
            r_aug = r + 0.01 * torch.randn_like(r)

            pred = model(r_aug, S_aug)
            main = F.smooth_l1_loss(pred, Y)
            deriv = F.mse_loss(pred[:, 1:] - pred[:, :-1], Y[:, 1:] - Y[:, :-1])
            loss = main + TRAINING_PARAMS["deriv_lambda"] * deriv

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sched.step()
            run += loss.item() * len(S)
        tr_loss.append(run / n_train)

        # ---------- Validation ----------
        model.eval()
        run = 0.0
        with torch.no_grad():
            for S, Y in ld_vl:
                S, Y = S.to(DEVICE), Y.to(DEVICE)
                r = rpm_base.expand(len(S), -1, -1)
                pred = model(r, S)
                main = F.smooth_l1_loss(pred, Y)
                deriv = F.mse_loss(pred[:, 1:] - pred[:, :-1], Y[:, 1:] - Y[:, :-1])
                run += (main + TRAINING_PARAMS["deriv_lambda"] * deriv).item() * len(S)
        vl_loss.append(run / (len(ds) - n_train))

        print(f"Epoch {ep+1:02d}/{TRAINING_PARAMS['epochs']} "
              f"| L_tr {tr_loss[-1]:.4f} | L_vl {vl_loss[-1]:.4f}")

    # ─────────────────────── Save artefacts ─────────────────────── #
    torch.save(model.state_dict(), "wave.nn")
    np.save("S_mean.npy", ds.S_mean); np.save("S_std.npy", ds.S_std)
    np.save("Y_mean.npy", np.array(ds.Y_mean)); np.save("Y_std.npy", np.array(ds.Y_std))

    # ───────────────────── Learning-curve plot ───────────────────── #
    plt.figure()
    plt.plot(tr_loss, label="train"); plt.plot(vl_loss, label="val")
    plt.yscale("log"); plt.title("Smooth-L1 + λ·MSE(Δ) loss"); plt.legend(); plt.show()

    # ───────── Residual vs RPM + ACF + Ljung-Box test ──────────── #
    residuals = []
    model.eval()
    with torch.no_grad():
        for S, Y in ld_vl:
            S, Y = S.to(DEVICE), Y.to(DEVICE)
            r = rpm_base.expand(len(S), -1, -1)
            residuals.append((model(r, S) - Y).cpu() * ds.Y_std)
    residuals = torch.cat(residuals).numpy()
    rpm_mean = residuals.mean(0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(RPM_GRID, rpm_mean); axes[0].set_title("Mean residual vs RPM")
    plot_acf(rpm_mean, ax=axes[1], lags=20); axes[1].set_title("ACF")
    plt.tight_layout(); plt.show()

    lb = acorr_ljungbox(rpm_mean, lags=[10], return_df=True)
    print("Ljung-Box lag-10 p-value:", lb["lb_pvalue"].iloc[0])

    # ───── Characteristic torque-curve figure (3 sub-plots) ────── #
    model.eval()
    with torch.no_grad():
        S_ex, Y_ex = next(iter(ld_vl))
        S_ex, Y_ex = S_ex.to(DEVICE), Y_ex.to(DEVICE)
        pred_ex = model(rpm_base.expand(len(S_ex), -1, -1), S_ex).cpu()

    # pick three samples: min-runner, median-runner, max-runner for clarity
    runner_idx = ds.S_mean.shape[0] - 2  # runner length column index (hdr1,hdr2,runner,plenum)
    runner_vals = (S_ex.cpu() * ds.S_std + ds.S_mean)[..., runner_idx]
    idx_small = runner_vals.argmin().item()
    idx_big   = runner_vals.argmax().item()
    idx_mid   = int(np.median(np.argsort(runner_vals)))

    chosen = [("Small", idx_small), ("Mid", idx_mid), ("Big", idx_big)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))   # one figure, 3 rows[2]
    fig.suptitle("Characteristic torque curves: actual vs predicted")

    for ax, (tag, idx) in zip(axes, chosen):
        hdr1, hdr2, runner, plenum = (S_ex[idx].cpu() * ds.S_std + ds.S_mean)[:4]
        lbl = f"{tag}: hdr1={hdr1:.1f}, hdr2={hdr2:.1f}, runner={runner:.1f}, plenum={plenum:.1f}"
        ax.plot(RPM_GRID, Y_ex[idx].cpu() * ds.Y_std + ds.Y_mean, label=f"{tag} actual")
        ax.plot(RPM_GRID, pred_ex[idx] * ds.Y_std + ds.Y_mean, "--", label=f"{tag} predicted")
        ax.set_ylabel("Torque (Nm)")
        ax.set_title(lbl)
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("RPM")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    train()
