# Physics-Informed Neural Network for Magnetic Confinement in a Mini-Tokamak

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![FEMM](https://img.shields.io/badge/FEMM-4.2-lightgrey)](https://www.femm.info/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Physics-Informed Neural Network (PINN) that learns the magnetic flux function **ψ(R,Z)** inside the vacuum chamber of a mini-Tokamak, validated against FEMM finite-element simulations.

The network achieves **<0.04% boundary error** and **0.033% interior error** relative to the FEM reference, trained in **7.4 minutes** on a single RTX 3060 Ti GPU.

---

![PINN predicted fields](results/02_pinn_fields.png)
*From left to right: vector potential Aφ = ψ/(2πR), magnetic field magnitude |B|, and field lines inside the vacuum chamber.*

---

## Table of Contents

- [Overview](#overview)
- [Physics Background](#physics-background)
- [Model Geometry](#model-geometry)
- [Method](#method)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Mathematical Derivation](#mathematical-derivation)

---

## Overview

This project combines **finite-element simulation** (FEMM) with **physics-informed machine learning** (PyTorch) to study magnetic confinement in a simplified Tokamak geometry.

The setup consists of:
- A **toroidal coil** (N=100 turns, I=1000 A) — the main field source
- Two **poloidal coils** (N=50 turns, I=500 A each) — positioned above and below the torus, shaping the field

The PINN learns to predict the complete magnetic field at any point inside the vacuum chamber without solving the FEM problem at inference time, making it orders of magnitude faster once trained.

**Key features:**
- PDE residual computed via PyTorch autograd — no finite differences, no mesh
- Three-component loss: physics (PDE) + boundary data + interior data
- Full FEMM validation pipeline included
- Reproducible FEMM model via Lua script

---

## Physics Background

Under axisymmetry (∂/∂φ = 0), Maxwell's equations reduce to a single 2nd-order PDE for the magnetic flux function ψ(R,Z) in the vacuum region (J = 0):

$$\frac{\partial^2 \psi}{\partial R^2} - \frac{1}{R}\frac{\partial \psi}{\partial R} + \frac{\partial^2 \psi}{\partial Z^2} = 0$$

where ψ = 2πR·Aφ is the magnetic flux through a disk of radius R (Aφ is the azimuthal component of the vector potential A).

Once the PINN has learned ψ(R,Z), the magnetic field components are recovered analytically:

$$B_R = -\frac{1}{2\pi R}\frac{\partial \psi}{\partial Z}, \qquad B_Z = \frac{1}{2\pi R}\frac{\partial \psi}{\partial R}$$

For the full derivation from Maxwell's equations to the PDE, see [math_derivation.md](math_derivation.md).

> **Note on FEMM convention:** FEMM in axisymmetric mode returns the flux function ψ = 2πR·Aφ, not the vector potential Aφ directly. The PDE for ψ has a **minus** sign in the (1/R) term — opposite to the Aφ equation. Confusing these two leads to a ~5× error in Bz.

---

## Model Geometry

All dimensions in meters, centered at (R₀=0.50, Z=0):

| Region | Shape | R range | Z range | Material | Current |
|---|---|---|---|---|---|
| Vacuum chamber | Circle | r ≤ 0.17 m | — | Air | J = 0 |
| Toroidal coil | Annulus | 0.17–0.22 m | — | Copper | N=100, I=1000 A |
| Poloidal coil (top) | Square | 0.51–0.59 m | +0.36–+0.44 m | Copper | N=50, I=500 A |
| Poloidal coil (bottom) | Square | 0.51–0.59 m | −0.44–−0.36 m | Copper | N=50, I=500 A |
| Dirichlet domain | Rectangle | 0–1.20 m | ±0.80 m | Air | A = 0 (BC) |

The PINN operates inside the vacuum chamber disc (r ≤ 0.17 m from center).

---

## Method

### Network architecture

A fully-connected MLP with:
- **Input:** (R_norm, Z_norm) — coordinates normalized to [−1, 1]
- **Hidden layers:** 6 × 100 neurons with Tanh activation
- **Output:** ψ_norm — flux function normalized to [0, 1]
- **Parameters:** 50,901

Tanh is chosen over ReLU because the PDE requires second-order derivatives, which vanish for piecewise-linear activations.

### Loss function

$$\mathcal{L} = \lambda_{\text{pde}} \cdot \mathcal{L}_{\text{pde}} + \lambda_{\text{bc}} \cdot \mathcal{L}_{\text{bc}} + \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}}$$

| Term | Weight | Description |
|---|---|---|
| L_pde | λ=10 | PDE residual at random interior points (enforces physics) |
| L_bc | λ=5 | MSE vs FEMM data on the chamber boundary (200 points) |
| L_data | λ=20 | MSE vs FEMM data in the chamber interior (~1131 points) |

PDE derivatives are computed exactly via `torch.autograd.grad` with `create_graph=True`, enabling second-order differentiation through the network.

### Training

- **Phase 1:** 12,000 iterations, lr = 1×10⁻³ (Adam)
- **Phase 2:** 8,000 iterations, lr = 5×10⁻⁴ (fine-tuning)
- **Hardware:** Single RTX 3060 Ti GPU
- **Time:** ~7.4 minutes

---

## Results

| Metric | Value |
|---|---|
| Boundary error (mean) | 0.039 % |
| Interior error (mean) | 0.033 % |
| \|B\| range | 0.033 – 0.070 T |
| Training time | 7.4 min |
| Final loss | 9.76 × 10⁻⁶ |

![PINN vs FEMM boundary](results/03_pinn_vs_femm.png)
*PINN (red dashed) vs FEMM (blue solid) along the chamber boundary. Maximum relative error: 0.1%.*

![Interior error map](results/04_interior_error.png)
*Relative error across the entire vacuum chamber. Mean: 0.033%.*

The poloidal coils introduce a non-zero BR component (antisymmetric in Z) and a parabolic BZ profile along the axis — both correctly captured by the PINN purely through the boundary conditions extracted from FEMM.

---

## Repository Structure
