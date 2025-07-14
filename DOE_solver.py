import numpy as np
import torch
from tqdm import tqdm
from ffact import generate_factorial_table                   # external helper

from wave_nn import ASTorqueModel, RPM_GRID, STATIC_DIM  # shared module[2]

# -------------------------------------------------------------------- #
#                    NORMALISATION STATISTICS & DEVICE                 #
# -------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

S_mean = np.load("S_mean.npy")          # (4,)
S_std  = np.load("S_std.npy")

Y_mean = np.load("Y_mean.npy")          # could be scalar or 50-vector
Y_std  = np.load("Y_std.npy")

SEQ_LEN = len(RPM_GRID)

# Ensure broadcasting works regardless of stored shape
for arr_name, arr in (("Y_mean", Y_mean), ("Y_std", Y_std)):
    if arr.size not in (1, SEQ_LEN):
        print(f"[WARN] {arr_name}.npy has length {arr.size}; collapsing to scalar.")
        if arr_name == "Y_mean":
            Y_mean = np.array(float(arr.mean()))
        else:
            Y_std = np.array(float(arr.mean()))

rpm_norm_base = ((RPM_GRID - RPM_GRID.mean()) / RPM_GRID.std()).astype(np.float32)
rpm_norm_base = torch.tensor(rpm_norm_base, device=DEVICE).unsqueeze(0).unsqueeze(0)  # 1×1×50

# -------------------------------------------------------------------- #
#                       RESTORE TRAINED NETWORK                         #
# -------------------------------------------------------------------- #
model = ASTorqueModel().to(DEVICE)
model.load_state_dict(torch.load("wave.nn", map_location=DEVICE))
model.eval()

# -------------------------------------------------------------------- #
#                 DESIGN-OF-EXPERIMENT  (factorial table)              #
# -------------------------------------------------------------------- #
shlb, shub, shs = 10, 14, 0.5     # seconday header length
phlb, phub, phs = 400, 560, 5     # header length
irlb, irub, irs = 6, 15, 0.25     # intake runner length
pvlb, pvub, pvs = 1, 5, 0.5       # plenum volume

doe = np.asarray(
    generate_factorial_table(
        shlb, shub, shs,
        phlb, phub, phs,
        irlb, irub, irs,
        pvlb, pvub, pvs,
    )
)
n_cases = doe.shape[0]

rpm_plt       = RPM_GRID.copy()
torque_curves = np.zeros((len(rpm_plt), STATIC_DIM + 2, n_cases), dtype=np.float32)

# -------------------------------------------------------------------- #
#                        INFERENCE MAIN LOOP                           #
# -------------------------------------------------------------------- #
with tqdm(total=n_cases, desc="Solving DOE cases") as pbar:
    for i, params in enumerate(doe):
        # --- normalise static inputs -------------------------------- #
        stat_norm   = (params - S_mean) / S_std
        stat_tensor = torch.tensor(stat_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # --- predict torque curve ----------------------------------- #
        with torch.no_grad():
            pred_norm = model(rpm_norm_base, stat_tensor).squeeze(0).cpu().numpy()

        pred_torque = pred_norm * Y_std + Y_mean

        # --- build output block (static params | rpm | torque) ------- #
        block = np.hstack(
            [
                np.tile(params, (len(rpm_plt), 1)),  # 4 static features
                rpm_plt.reshape(-1, 1),              # rpm column
                pred_torque.reshape(-1, 1),          # torque column
            ]
        )
        torque_curves[:, :, i] = block
        pbar.update(1)

np.save("torque_curves.npy", torque_curves, allow_pickle=True)
print("Saved torque_curves.npy")
