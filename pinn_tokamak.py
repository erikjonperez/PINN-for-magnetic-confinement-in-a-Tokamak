import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

PATH_BOUNDARY = "boundary_data.csv"
PATH_INTERIOR = "interior_data.csv"
OUTPUT_DIR    = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("PINN - Mini Tokamak (toroidal + poloidales)")
print("=" * 60)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DATA READING
# Extract values of ψ from FEMM's CSV output.
# Note! The column is named A_phi but it actually contains ψ = 2πR·Aφ, not Aφ directly — this was the key discovery
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("1- DATA READING...")
df_bc  = pd.read_csv(PATH_BOUNDARY)
df_int = pd.read_csv(PATH_INTERIOR)
print(f"  Contorno: {len(df_bc)} puntos")
print(f"  Interior: {len(df_int)} puntos")
bc_R   = torch.tensor(df_bc['R'].values,     dtype=torch.float32) 
bc_Z   = torch.tensor(df_bc['Z'].values,     dtype=torch.float32) 
bc_psi = torch.tensor(df_bc['A_phi'].values, dtype=torch.float32) 
int_R   = torch.tensor(df_int['R'].values,     dtype=torch.float32)  
int_Z   = torch.tensor(df_int['Z'].values,     dtype=torch.float32)  
int_psi = torch.tensor(df_int['A_phi'].values, dtype=torch.float32)  
print(f"  ψ contorno: [{bc_psi.min():.6f}, {bc_psi.max():.6f}] Wb/m")
print(f"  ψ interior: [{int_psi.min():.6f}, {int_psi.max():.6f}] Wb/m")
print()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NORMALIZATION AND PARAMETERS
# Z ^
#   |
#   |
#   |
#   |_____________> R

# Transform R∈[0.33, 0.67] and Z∈[−0.17, 0.17] to the [−1, 1] range. Normalized values.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
R0 = 0.50   # major radius (m)
a  = 0.17   # minor radius (vacuum chamber)
R_min, R_max = R0 - a, R0 + a  
all_psi  = torch.cat([bc_psi, int_psi])  
psi_min  = all_psi.min().item()         
psi_max  = all_psi.max().item()          
scale_R  = 2.0 / (R_max - R_min)         
scale_Z  = 2.0 / (2 * a)               
scale_psi = psi_max - psi_min        
print(f"  Dominio PINN: R ∈ [{R_min:.2f}, {R_max:.2f}], Z ∈ [{-a:.2f}, {a:.2f}]")
print(f"  ψ rango: [{psi_min:.6f}, {psi_max:.6f}]")
print()
def norm_input(R, Z):
    return (2*(R - R_min)/(R_max - R_min) - 1,
            2*(Z + a)/(2*a) - 1)
def denorm_psi(pn):
    return pn * scale_psi + psi_min
bc_Rn, bc_Zn   = norm_input(bc_R, bc_Z)
bc_psi_n        = (bc_psi - psi_min) / scale_psi
bc_pts_n        = torch.stack([bc_Rn, bc_Zn], dim=1)
int_Rn, int_Zn  = norm_input(int_R, int_Z)
int_psi_n       = (int_psi - psi_min) / scale_psi
int_pts_n       = torch.stack([int_Rn, int_Zn], dim=1)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MODEL 
# Takes 2 values (R,Z) and outputs 1 value (ψ). The network has 6 hidden layers with 100 neurons each, using Tanh activation.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("2- CREATING MODEL...")

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
model = PINN()
print(f"  Parameterss: {sum(p.numel() for p in model.parameters()):,}")
print()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SAMPLING
# On each of the 20.000iterations, simple_pde is executed
# Then, with 'sample_bc' 100 out of 200 conour points are randomly sampled and the PINN prediction is compared to the FEMM value for those points
# Finally, with 'sample_data' 200 out of 1131 interior points are randomly
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sample_interior_pde(n):
    theta = 2 * np.pi * torch.rand(n)
    r = a * torch.sqrt(torch.rand(n))
    R = R0 + r * torch.cos(theta)
    Z = r * torch.sin(theta)
    Rn, Zn = norm_input(R, Z)
    pts = torch.stack([Rn, Zn], dim=1)
    pts.requires_grad_(True)
    return pts, R

def sample_bc(n):
    idx = torch.randint(0, len(bc_psi_n), (n,))
    return bc_pts_n[idx], bc_psi_n[idx]

def sample_data(n):
    idx = torch.randint(0, len(int_psi_n), (n,))
    return int_pts_n[idx], int_psi_n[idx]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# RESIDUAL PDE
# d²ψ/dR² - (1/R)dψ/dR + d²ψ/dZ² = 0
# First derivative: autograd computes ∂ψ/∂R and ∂ψ/∂Z in normalized coordinates. create_graph=True keeps the computational graph for second derivatives.
# Second derivative: we derive the first derivatives again to get ∂²ψ/∂R² and ∂²ψ/∂Z².
# Chain rule to go from normalized to real coordinates. The second derivatives have scale because the chain rule is applied twice.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def compute_pde_residual(model, pts, R_real):
    psi_norm = model(pts)

    grads = torch.autograd.grad(
        psi_norm, pts,
        grad_outputs=torch.ones_like(psi_norm),
        create_graph=True
    )[0]
    dpsi_dRn = grads[:, 0]
    dpsi_dZn = grads[:, 1]

    d2psi_dRn2 = torch.autograd.grad(
        dpsi_dRn, pts,
        grad_outputs=torch.ones_like(dpsi_dRn),
        create_graph=True
    )[0][:, 0]

    d2psi_dZn2 = torch.autograd.grad(
        dpsi_dZn, pts,
        grad_outputs=torch.ones_like(dpsi_dZn),
        create_graph=True
    )[0][:, 1]

    # Chain rule
    dpsi_dR   = scale_psi * scale_R   * dpsi_dRn
    d2psi_dR2 = scale_psi * scale_R**2 * d2psi_dRn2
    d2psi_dZ2 = scale_psi * scale_Z**2 * d2psi_dZn2

    # Residual
    residual = d2psi_dR2 - (1.0 / R_real) * dpsi_dR + d2psi_dZ2
    return residual

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# TRAINING 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("3-TRAINING...")
print()

lambda_pde  = 10.0
lambda_bc   =  5.0
lambda_data = 20.0
N_PDE, N_BC, N_DATA = 300, 100, 200

loss_hist, pde_hist, bc_hist, data_hist = [], [], [], []
t0 = time.time()

# ---- PHASE 1 ----
lr1, iters1 = 1e-3, 12000
print(f"Fase 1: lr={lr1}, {iters1} iters")
print("-" * 60)
optimizer = torch.optim.Adam(model.parameters(), lr=lr1)

for i in range(iters1):
    optimizer.zero_grad()

    pts_pde, R_real = sample_interior_pde(N_PDE)
    loss_pde = torch.mean(compute_pde_residual(model, pts_pde, R_real)**2)

    pts_bc, psi_bc = sample_bc(N_BC)
    loss_bc = torch.mean((model(pts_bc) - psi_bc)**2)

    pts_data, psi_data = sample_data(N_DATA)
    loss_data = torch.mean((model(pts_data) - psi_data)**2)

    loss = lambda_pde*loss_pde + lambda_bc*loss_bc + lambda_data*loss_data
    loss.backward()
    optimizer.step()

    loss_hist.append(loss.item())
    pde_hist.append(loss_pde.item())
    bc_hist.append(loss_bc.item())
    data_hist.append(loss_data.item())

    if i % 1000 == 0:
        t = time.time() - t0
        print(f"  Iter {i:5d}/20000 | Total: {loss.item():.3e} | "
              f"PDE: {loss_pde.item():.3e} | BC: {loss_bc.item():.3e} | "
              f"Data: {loss_data.item():.3e} | {t:.0f}s")

# ---- PHASE 2 ----
print()
lr2, iters2 = 5e-4, 8000
print(f"Fase 2: lr={lr2}, {iters2} iters (fine-tuning)")
print("-" * 60)
for pg in optimizer.param_groups:
    pg['lr'] = lr2

for i in range(iters2):
    optimizer.zero_grad()

    pts_pde, R_real = sample_interior_pde(N_PDE)
    loss_pde = torch.mean(compute_pde_residual(model, pts_pde, R_real)**2)

    pts_bc, psi_bc = sample_bc(N_BC)
    loss_bc = torch.mean((model(pts_bc) - psi_bc)**2)

    pts_data, psi_data = sample_data(N_DATA)
    loss_data = torch.mean((model(pts_data) - psi_data)**2)

    loss = lambda_pde*loss_pde + lambda_bc*loss_bc + lambda_data*loss_data
    loss.backward()
    optimizer.step()

    loss_hist.append(loss.item())
    pde_hist.append(loss_pde.item())
    bc_hist.append(loss_bc.item())
    data_hist.append(loss_data.item())

    if i % 1000 == 0:
        t = time.time() - t0
        ig = iters1 + i
        print(f"  Iter {ig:5d}/20000 | Total: {loss.item():.3e} | "
              f"PDE: {loss_pde.item():.3e} | BC: {loss_bc.item():.3e} | "
              f"Data: {loss_data.item():.3e} | {t:.0f}s")

t_total = time.time() - t0
print(f"\nCompletado en {t_total:.0f}s ({t_total/60:.1f} min)")
print(f"Loss final: {loss_hist[-1]:.3e}")
print()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SAVE THE MODEL
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("4- SAVIG THE MODEL...")
torch.save({
    'model_state_dict': model.state_dict(),
    'loss_hist': loss_hist, 'pde_hist': pde_hist,
    'bc_hist': bc_hist, 'data_hist': data_hist,
    'params': {'R0': R0, 'a': a, 'psi_min': psi_min, 'psi_max': psi_max}
}, os.path.join(OUTPUT_DIR, "pinn_model_phase2.pth"))
print("  pinn_model_phase2.pth guardado")
print()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PLOTS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("5- LOSS CURVE...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(loss_hist, lw=1)
ax1.set_yscale('log'); ax1.set_xlabel('Iter'); ax1.set_ylabel('Loss total')
ax1.grid(True, alpha=0.3)
ax1.axvline(iters1, color='red', ls='--', alpha=0.5, label=f'lr→{lr2}')
ax1.legend(); ax1.set_title('Loss total')

ax2.plot(pde_hist,  lw=1, label='PDE',  color='tab:blue')
ax2.plot(bc_hist,   lw=1, label='BC',   color='tab:orange')
ax2.plot(data_hist, lw=1, label='Data', color='tab:green')
ax2.set_yscale('log'); ax2.set_xlabel('Iter'); ax2.grid(True, alpha=0.3)
ax2.axvline(iters1, color='red', ls='--', alpha=0.5)
ax2.legend(); ax2.set_title('Loss por componente')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_loss_curves.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  01_loss_curves.png")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("6- PREDICTION ON GRID...")
model.eval()
Nr = Nz = 200
R_vals = np.linspace(R_min, R_max, Nr)
Z_vals = np.linspace(-a, a, Nz)
RR, ZZ = np.meshgrid(R_vals, Z_vals)
mask = (RR - R0)**2 + ZZ**2 <= a**2

RRf = torch.tensor(RR.flatten(), dtype=torch.float32)
ZZf = torch.tensor(ZZ.flatten(), dtype=torch.float32)
RRn, ZZn = norm_input(RRf, ZZf)
X = torch.stack([RRn, ZZn], dim=1)
X.requires_grad_(True)

psi_norm_flat = model(X)
psi_pred = denorm_psi(psi_norm_flat).detach().numpy().reshape(RR.shape) # ψ 
psi_pred[~mask] = np.nan

Aphi_pred = psi_pred / (2 * np.pi * RR)  # Aφ = ψ/(2πR)
Aphi_pred[~mask] = np.nan

grads = torch.autograd.grad(
    psi_norm_flat, X,
    grad_outputs=torch.ones_like(psi_norm_flat),
    create_graph=False
)[0]
dpsi_dRn = grads[:, 0].detach().numpy().reshape(RR.shape)
dpsi_dZn = grads[:, 1].detach().numpy().reshape(RR.shape)

dpsi_dR = scale_psi * scale_R * dpsi_dRn # chain rule
dpsi_dZ = scale_psi * scale_Z * dpsi_dZn

BR = -(1.0 / (2*np.pi*RR)) * dpsi_dZ  # BR = −(1/2πR)·∂ψ/∂Z
BZ =  (1.0 / (2*np.pi*RR)) * dpsi_dR  # BZ = (1/2πR)·∂ψ/∂R
BR[~mask] = np.nan; BZ[~mask] = np.nan
Bmag = np.sqrt(np.nan_to_num(BR)**2 + np.nan_to_num(BZ)**2)
Bmag[~mask] = np.nan

print(f"  ψ:   [{np.nanmin(psi_pred):.6f}, {np.nanmax(psi_pred):.6f}]")
print(f"  Aφ:  [{np.nanmin(Aphi_pred):.6f}, {np.nanmax(Aphi_pred):.6f}]")
print(f"  |B|: [{np.nanmin(Bmag):.6f}, {np.nanmax(Bmag):.6f}] T")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("7- RESULTS PLOTS...")
tc = np.linspace(0, 2*np.pi, 200)

# ---Figure 2: Aφ, |B|, field lines ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ax = axes[0]
cf = ax.contourf(RR, ZZ, Aphi_pred, 60, cmap='plasma')
plt.colorbar(cf, ax=ax, label=r'$A_\varphi$ (Wb/m)')
ax.plot(R0+a*np.cos(tc), a*np.sin(tc), 'w--', lw=1, alpha=.6)
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title(r'Potencial vector $A_\varphi = \psi/(2\pi R)$ (PINN)')
ax.set_aspect('equal')

ax = axes[1]
cf = ax.contourf(RR, ZZ, Bmag, 60, cmap='inferno')
plt.colorbar(cf, ax=ax, label='|B| (T)')
ax.plot(R0+a*np.cos(tc), a*np.sin(tc), 'w--', lw=1, alpha=.6)
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Magnitud campo |B| (PINN)')
ax.set_aspect('equal')

ax = axes[2]
cf = ax.contourf(RR, ZZ, Aphi_pred, 60, cmap='plasma')
plt.colorbar(cf, ax=ax, label=r'$A_\varphi$ (Wb/m)')
BR_plot = np.nan_to_num(BR, 0)
BZ_plot = np.nan_to_num(BZ, 0)
ax.streamplot(RR, ZZ, BR_plot, BZ_plot, color='white', density=2.0, linewidth=0.8)
ax.plot(R0+a*np.cos(tc), a*np.sin(tc), 'w--', lw=1, alpha=.6)
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Líneas de campo B (PINN)')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_pinn_fields.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  02_pinn_fields.png")

# ---Figure 3: PINN vs FEMM on the boundary ---
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
ax1.set_title(r'PINN vs FEMM en contorno ($\psi = 2\pi R A_\varphi$)')
ax1.legend(); ax1.grid(True, alpha=.3)

ax2.plot(np.degrees(theta_bc[ixs]), err_rel_bc[ixs], 'k-', lw=1.5)
ax2.set_xlabel(r'$\theta$ (°)'); ax2.set_ylabel('Error relativo (%)')
ax2.set_title('Error relativo en el contorno'); ax2.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_pinn_vs_femm.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  03_pinn_vs_femm.png")

# --- Figure 4: Interior error map ---
with torch.no_grad():
    psi_pinn_int = denorm_psi(model(int_pts_n)).numpy()
psi_femm_int = int_psi.numpy()
err_rel_int  = np.abs(psi_pinn_int - psi_femm_int) / np.abs(psi_femm_int) * 100

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(int_R.numpy(), int_Z.numpy(), c=err_rel_int, cmap='hot_r',
                s=8, vmin=0, vmax=max(1, np.percentile(err_rel_int, 95)))
plt.colorbar(sc, ax=ax, label='Error relativo (%)')
ax.plot(R0+a*np.cos(tc), a*np.sin(tc), 'b--', lw=1, alpha=.5)
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title(f'Error PINN vs FEMM interior (medio: {err_rel_int.mean():.3f}%)')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_interior_error.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  04_interior_error.png")

# --- Figure 5: BR and BZ profiles along the vertical axis (R=R0) ---
Z_line = np.linspace(-a*0.95, a*0.95, 300)
R_line = np.full_like(Z_line, R0)
Rln, Zln = norm_input(torch.tensor(R_line, dtype=torch.float32),
                       torch.tensor(Z_line, dtype=torch.float32))
Xl = torch.stack([Rln, Zln], dim=1); Xl.requires_grad_(True)
psi_l_n = model(Xl)
gl = torch.autograd.grad(psi_l_n, Xl,
                          grad_outputs=torch.ones_like(psi_l_n),
                          create_graph=False)[0]
dpsi_dR_l = (scale_psi * scale_R * gl[:,0]).detach().numpy()
dpsi_dZ_l = (scale_psi * scale_Z * gl[:,1]).detach().numpy()
BR_line = -(1/(2*np.pi*R_line)) * dpsi_dZ_l
BZ_line =  (1/(2*np.pi*R_line)) * dpsi_dR_l


# Comparison: Bz of FEMM along the vertical axis R=R0. 
int_on_axis = df_int[np.abs(df_int['R'] - R0) < 0.01].sort_values('Z')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(Z_line, BR_line, 'b-', lw=2, label='BR (PINN)')
ax1.plot(Z_line, BZ_line, 'r-', lw=2, label='BZ (PINN)')
ax1.axhline(0, color='gray', ls=':', lw=0.8)
ax1.axvline(0, color='gray', ls=':', lw=0.8)
ax1.set_xlabel('Z (m)'); ax1.set_ylabel('B (T)')
ax1.set_title(f'Perfil de B a lo largo de R=R₀={R0} m')
ax1.legend(); ax1.grid(True, alpha=.3)

# Bz of FEMM vs PINN along the vertical axis R=R0.
if len(int_on_axis) > 3:
    ax2.plot(int_on_axis['Z'], int_on_axis['Bz'], 'b--', lw=2, label='BZ FEMM')
ax2.plot(Z_line, BZ_line, 'r-', lw=2, label='BZ PINN')
ax2.axvline(0, color='gray', ls=':', lw=0.8)
ax2.set_xlabel('Z (m)'); ax2.set_ylabel('BZ (T)')
ax2.set_title(f'BZ a lo largo de R=R₀ — PINN vs FEMM')
ax2.legend(); ax2.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_axial_profile.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  05_axial_profile.png")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print()
print("=" * 60)
print("COMPLETADO - FASE 2")
print("=" * 60)
print(f"Tiempo total: {t_total:.0f}s ({t_total/60:.1f} min)")
print(f"Loss final:   {loss_hist[-1]:.3e}")
print(f"Error BC medio:       {err_rel_bc.mean():.4f}%")
print(f"Error interior medio: {err_rel_int.mean():.4f}%")
print(f"|B| rango: [{np.nanmin(Bmag):.4f}, {np.nanmax(Bmag):.4f}] T")
print(f"Archivos en '{OUTPUT_DIR}/'")
print("=" * 60)
