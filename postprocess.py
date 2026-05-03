# For good and cool-looking plots and figures

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image
import os

# =========================================================================
# PATHS
# =========================================================================
PATH_PTH      = "./zPhase2_results1/pinn_model_phase2.pth"
PATH_BC       = "phase_2_boundary_data.csv"
PATH_INT      = "phase_2_interior_data.csv"
PATH_FEMM_IMG = "./zPhase2_results1/the_screenshot_femm.png"
OUTPUT_DIR    = "./results_coloridosssss"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for p in [PATH_PTH, PATH_BC, PATH_INT]:
    exists = os.path.exists(p)
    print(f"{'OK' if exists else 'FALTA'}: {os.path.abspath(p)}")
    if not exists:
        raise FileNotFoundError(f"No encuentro: {os.path.abspath(p)}")

# =========================================================================
# IMITATING FEMM COLORMAP EXACTLY, FOR |B| FIELD
# =========================================================================
FEMM_COLORS = [
    "#01ffff",  # < 7.920e-003 T
    "#25ffc3",  # 7.920e-003 : 1.584e-002
    "#45ff93",  # 1.584e-002 : 2.376e-002
    "#62ff6c",  # 2.376e-002 : 3.168e-002
    "#7bff4c",  # 3.168e-002 : 3.960e-002
    "#94ff33",  # 3.960e-002 : 4.752e-002
    "#abff1f",  # 4.752e-002 : 5.544e-002
    "#c2ff10",  # 5.544e-002 : 6.336e-002
    "#d9ff06",  # 6.336e-002 : 7.128e-002
    "#f2ff01",  # 7.128e-002 : 7.920e-002
    "#fff201",  # 7.920e-002 : 8.712e-002
    "#ffd906",  # 8.712e-002 : 9.504e-002
    "#ffc210",  # 9.504e-002 : 1.030e-001
    "#ffab1f",  # 1.030e-001 : 1.109e-001
    "#ff9433",  # 1.109e-001 : 1.188e-001
    "#ff7b4c",  # 1.188e-001 : 1.267e-001
    "#ff626c",  # 1.267e-001 : 1.346e-001
    "#ff4593",  # 1.346e-001 : 1.426e-001
    "#ff25c3",  # 1.426e-001 : 1.505e-001
    "#ff00ff",  # > 1.505e-001 T
]
FEMM_VMIN = 0.0
FEMM_VMAX = 0.1584  # máximo de la leyenda FEMM

femm_cmap = mcolors.LinearSegmentedColormap.from_list("femm_exact", FEMM_COLORS, N=512)

# =========================================================================
# MODEL
# =========================================================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        dims = [2] + [100]*6 + [1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

# =========================================================================
# SAVE MODEL AND DATA
# =========================================================================
print("\nCargando modelo...")
checkpoint = torch.load(PATH_PTH, map_location='cpu')
model = PINN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

params    = checkpoint['params']
R0, a     = params['R0'], params['a']
psi_min   = params['psi_min']
psi_max   = params['psi_max']
scale_psi = psi_max - psi_min
R_min, R_max = R0 - a, R0 + a
scale_R = 2.0 / (R_max - R_min)
scale_Z = 2.0 / (2 * a)
print(f"  R0={R0}, a={a}")

df_bc  = pd.read_csv(PATH_BC)
df_int = pd.read_csv(PATH_INT)

bc_R   = torch.tensor(df_bc['R'].values,      dtype=torch.float32)
bc_Z   = torch.tensor(df_bc['Z'].values,      dtype=torch.float32)
bc_psi = torch.tensor(df_bc['A_phi'].values,  dtype=torch.float32)
int_R  = torch.tensor(df_int['R'].values,     dtype=torch.float32)
int_Z  = torch.tensor(df_int['Z'].values,     dtype=torch.float32)
int_psi= torch.tensor(df_int['A_phi'].values, dtype=torch.float32)

def norm_input(R, Z):
    return (2*(R - R_min)/(R_max - R_min) - 1,
            2*(Z + a)/(2*a) - 1)

def denorm_psi(pn):
    return pn * scale_psi + psi_min

bc_Rn, bc_Zn = norm_input(bc_R, bc_Z)
bc_psi_n = (bc_psi - psi_min) / scale_psi
bc_pts_n = torch.stack([bc_Rn, bc_Zn], dim=1)

int_Rn, int_Zn = norm_input(int_R, int_Z)
int_psi_n = (int_psi - psi_min) / scale_psi
int_pts_n = torch.stack([int_Rn, int_Zn], dim=1)

# =========================================================================
# PREDICTION ON GRID
# =========================================================================
print("Calculando campo en malla 200x200...")
RR, ZZ = np.meshgrid(np.linspace(R_min, R_max, 200),
                      np.linspace(-a, a, 200))
mask = (RR - R0)**2 + ZZ**2 <= a**2

RRf = torch.tensor(RR.flatten(), dtype=torch.float32)
ZZf = torch.tensor(ZZ.flatten(), dtype=torch.float32)
RRn, ZZn = norm_input(RRf, ZZf)
X = torch.stack([RRn, ZZn], dim=1)
X.requires_grad_(True)

psi_norm_flat = model(X)
psi_pred = denorm_psi(psi_norm_flat).detach().numpy().reshape(RR.shape)
psi_pred[~mask] = np.nan
Aphi_pred = psi_pred / (2 * np.pi * RR)
Aphi_pred[~mask] = np.nan

grads = torch.autograd.grad(psi_norm_flat, X,
                             grad_outputs=torch.ones_like(psi_norm_flat),
                             create_graph=False)[0]
dpsi_dR = (scale_psi * scale_R * grads[:, 0]).detach().numpy().reshape(RR.shape)
dpsi_dZ = (scale_psi * scale_Z * grads[:, 1]).detach().numpy().reshape(RR.shape)

BR = -(1/(2*np.pi*RR)) * dpsi_dZ
BZ =  (1/(2*np.pi*RR)) * dpsi_dR
BR[~mask] = np.nan
BZ[~mask] = np.nan
Bmag = np.sqrt(np.nan_to_num(BR)**2 + np.nan_to_num(BZ)**2)
Bmag[~mask] = np.nan

B_min = np.nanmin(Bmag)
B_max = np.nanmax(Bmag)
print(f"  |B| PINN: [{B_min:.5f}, {B_max:.5f}] T")

tc = np.linspace(0, 2*np.pi, 300)

# =========================================================================
# PLOT 1: PINN fields con colormap of FEMM 
# =========================================================================
print("Figura 1: PINN fields con colormap FEMM...")

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

ax = axes[0]
cf = ax.contourf(RR, ZZ, Aphi_pred, 60, cmap='plasma')
plt.colorbar(cf, ax=ax, label=r'$A_\varphi = \psi/(2\pi R)$ (Wb/m)')
ax.plot(R0+a*np.cos(tc), a*np.sin(tc), 'w--', lw=1, alpha=.7)
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title(r'Vector potential $A_\varphi$', fontsize=12, fontweight='bold')
ax.set_aspect('equal')

ax = axes[1]
cf = ax.contourf(RR, ZZ, Bmag, 60, cmap=femm_cmap, vmin=FEMM_VMIN, vmax=FEMM_VMAX)
plt.colorbar(cf, ax=ax, label='|B| (T)')
ax.plot(R0+a*np.cos(tc), a*np.sin(tc), 'k--', lw=1.5, alpha=.8)
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title('|B| magnitude — FEMM colormap', fontsize=12, fontweight='bold')
ax.set_aspect('equal')

ax = axes[2]
cf = ax.contourf(RR, ZZ, Bmag, 60, cmap=femm_cmap, vmin=FEMM_VMIN, vmax=FEMM_VMAX)
plt.colorbar(cf, ax=ax, label='|B| (T)')
ax.streamplot(RR, ZZ, np.nan_to_num(BR,0), np.nan_to_num(BZ,0),
              color='white', density=2.0, linewidth=0.8)
ax.plot(R0+a*np.cos(tc), a*np.sin(tc), 'k--', lw=1.5, alpha=.8)
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Field lines B', fontsize=12, fontweight='bold')
ax.set_aspect('equal')

plt.suptitle('PINN — Magnetic field in the mini-Tokamak vacuum chamber',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_pinn_fields.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  02_pinn_fields.png OK")

# =========================================================================
# PLOT 2: FEMM vs PINN 
# =========================================================================
print("Figura 2: FEMM vs PINN comparación...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
if os.path.exists(PATH_FEMM_IMG):
    femm_img = np.array(Image.open(PATH_FEMM_IMG))
    ax.imshow(femm_img)
    ax.axis('off')
    ax.set_title("FEMM — Finite Element Ground Truth\nFull domain",
                 fontsize=11, fontweight='bold', pad=10)
else:
    ax.text(0.5, 0.5, "screenshot_femm.png no encontrado",
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='red')
    ax.axis('off')

ax = axes[1]
cf = ax.contourf(RR, ZZ, Bmag, 60, cmap=femm_cmap, vmin=FEMM_VMIN, vmax=FEMM_VMAX)
cbar = plt.colorbar(cf, ax=ax, label='|B| (T)')
ax.plot(R0+a*np.cos(tc), a*np.sin(tc), 'k--', lw=2, alpha=.9)
ax.set_xlabel('R (m)', fontsize=11)
ax.set_ylabel('Z (m)', fontsize=11)
ax.set_title("PINN — Predicted\nVacuum chamber only",
             fontsize=11, fontweight='bold', pad=10)
ax.set_aspect('equal')

plt.suptitle("FEMM (reference)  vs  PINN (this work)\n"
             "Identical colormap and scale · |B| in Tesla",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_femm_vs_pinn.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  06_femm_vs_pinn.png OK")

# =========================================================================
# PLOT 3: PINN vs FEMM on contour
# =========================================================================
print("Figura 3: comparación en contorno...")

with torch.no_grad():
    psi_pinn_bc = denorm_psi(model(bc_pts_n)).numpy()
psi_femm_bc = bc_psi.numpy()
err_rel_bc  = np.abs(psi_pinn_bc - psi_femm_bc) / np.abs(psi_femm_bc) * 100
theta_bc    = np.arctan2(bc_Z.numpy(), bc_R.numpy() - R0)
ixs         = np.argsort(theta_bc)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(np.degrees(theta_bc[ixs]), psi_femm_bc[ixs], 'b-',  lw=2, label='FEMM')
ax1.plot(np.degrees(theta_bc[ixs]), psi_pinn_bc[ixs], 'r--', lw=2, label='PINN')
ax1.set_xlabel(r'$\theta$ (°)'); ax1.set_ylabel(r'$\psi$ (Wb/m)')
ax1.set_title(r'PINN vs FEMM on chamber boundary ($\psi = 2\pi R A_\varphi$)',
              fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=.3)

ax2.plot(np.degrees(theta_bc[ixs]), err_rel_bc[ixs], color='#C0392B', lw=1.5)
ax2.set_xlabel(r'$\theta$ (°)')
ax2.set_ylabel('Relative error (%)')
ax2.set_title(f'Relative error — Mean: {err_rel_bc.mean():.4f}%   Max: {err_rel_bc.max():.4f}%',
              fontweight='bold')
ax2.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_pinn_vs_femm.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  03_pinn_vs_femm.png OK")

# =========================================================================
# PLOT 4: INTERIOR ERROR MAP
# =========================================================================
print("Figura 4: mapa de error interior...")

with torch.no_grad():
    psi_pinn_int = denorm_psi(model(int_pts_n)).numpy()
psi_femm_int = int_psi.numpy()
err_rel_int  = np.abs(psi_pinn_int - psi_femm_int) / np.abs(psi_femm_int) * 100

vmax_err = max(0.3, np.percentile(err_rel_int, 95))

fig, ax = plt.subplots(figsize=(8, 7))

sc = ax.scatter(int_R.numpy(), int_Z.numpy(),
                c=err_rel_int, cmap='Reds',
                s=14, vmin=0.02, vmax=vmax_err,
                edgecolors='none')
cbar = plt.colorbar(sc, ax=ax, label='Relative error (%)')

ax.plot(R0+a*np.cos(tc), a*np.sin(tc), 'k--', lw=1.5, alpha=.8)
ax.set_xlabel('R (m)')
ax.set_ylabel('Z (m)')
ax.set_title(f'PINN vs FEMM — Interior error map\n'
             f'Mean: {err_rel_int.mean():.4f}%   Max: {err_rel_int.max():.4f}%',
             fontweight='bold')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_interior_error.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  04_interior_error.png OK")

# =========================================================================
print()
print("=" * 55)
print("COMPLETADO")
print("=" * 55)
print(f"Error BC:       {err_rel_bc.mean():.4f}% medio  |  {err_rel_bc.max():.4f}% máx")
print(f"Error interior: {err_rel_int.mean():.4f}% medio  |  {err_rel_int.max():.4f}% máx")
print(f"|B| PINN: [{B_min:.5f}, {B_max:.5f}] T")
print(f"Figuras en: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 55)