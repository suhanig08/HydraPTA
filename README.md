# HydraPTA

## Hydraulic Fracture Pressure Transient Analyzer

> **Tagline:** *From pressure response to fracture insight.*

---

## Overview

**HydraPTA** is a specialized petroleum engineering analytical application designed to simulate, diagnose, and characterize the transient pressure behavior of hydraulically fractured vertical wells. Built for technical interviews, placement demonstrations (SLB, BP, ExxonMobil, Chevron, Cairn), and reservoir engineering education, HydraPTA bridges the gap between theoretical diffusivity physics and practical well-test interpretation.

The application follows a complete, end-to-end petroleum engineering PTA workflow:

```text
Reservoir + Well + Fracture Properties
                ↓
       Pressure-Transient Model
                ↓
       Synthetic Pressure Data
                ↓
           ΔP Calculation
                ↓
       Pressure Derivative
                ↓
      Log-Log PTA Diagnostics
                ↓
       Flow-Regime Detection
                ↓
    Fracture Characterization
                ↓
      Parameter Sensitivity
                ↓
       Engineering Insights
```

---

## Engineering Objective

Demonstrate how hydraulic fracture geometry ($x_f$), conductivity ($k_f w_f$), and reservoir properties govern pressure drawdown and the logarithmic derivative response. The platform distinctly separates **forward analytical modeling** from **inverse parameter characterization**, performing genuine slope-based and non-linear regression without leaking ground-truth synthetic parameters.

---

## Petroleum Engineering Background & Methodology

### 1. Dimensionless Variables (Field Units)
- **Dimensionless Time**:
  $$t_{Dxf} = \frac{0.0002637 \cdot k \cdot t}{\phi \cdot \mu \cdot c_t \cdot x_f^2}$$
- **Dimensionless Pressure Drop**:
  $$P_D = \frac{k \cdot h \cdot \Delta P}{141.2 \cdot q \cdot B \cdot \mu}$$
- **Fracture Conductivity**:
  $$C_f = k_f \cdot w_f \quad [\text{mD}\cdot\text{ft}]$$
- **Dimensionless Fracture Conductivity**:
  $$C_{fD} = \frac{k_f \cdot w_f}{k \cdot x_f}$$
- **Dimensionless Wellbore Storage**:
  $$C_{Dxf} = \frac{0.8936 \cdot C}{\phi \cdot c_t \cdot h \cdot x_f^2}$$

### 2. Forward Analytical Solution in Laplace Domain
The pressure response of a finite/infinite-conductivity vertical fracture in an infinite-acting homogeneous reservoir is formulated in the Laplace domain ($s$) using the unified Green's function / trilinear formulation (Cinco-Ley & Meng 1988, Lee & Brockenbrough 1986, Ozkan & Raghavan 1991):
$$\bar{P}_{wD}(s) = \frac{\bar{P}_D(s) + S_f}{s [1 + C_{Dxf} s (\bar{P}_D(s) + S_f)]}$$
The Laplace-space solution is numerically inverted to the real-time domain using the **Gaver-Stehfest Numerical Inversion Algorithm** ($N = 10$).

### 3. Bourdet Logarithmic Pressure Derivative
To eliminate numerical discretization noise without masking true physical slope signatures, the Bourdet derivative is computed as:
$$\Delta P' = \frac{d\Delta P}{d\ln t} = t \frac{d\Delta P}{dt}$$
Local log-log slopes are evaluated continuously:
$$m(t) = \frac{d\log \Delta P'}{d\log t}$$

---

## Diagnostic Flow Regimes

| Flow Regime | Observed Slope ($m$) | Expected Slope | Physical Mechanism |
| :--- | :---: | :---: | :--- |
| **Wellbore Storage** | $\approx 1.0$ | $1.0$ | Fluid decompression/accumulation inside the wellbore casing. |
| **Bilinear Flow** | $\approx 0.25$ | $0.25$ ($1/4$) | Simultaneous linear flow inside finite-conductivity fracture ($C_{fD} < 10$) and matrix feeding. |
| **Formation Linear Flow** | $\approx 0.50$ | $0.50$ ($1/2$) | Linear flow perpendicular to fracture faces into high-conductivity conduit ($C_{fD} \ge 10$). |
| **Pseudoradial Flow** | $\approx 0.0$ | $0.0$ (Flat) | Late-time radial flow where pressure disturbance propagates far beyond fracture tips ($r_w' = x_f/2$). |

---

## Fracture Characterization (Inverse Interpretation)

HydraPTA performs inverse parameter estimation without hardcoding or reading true values:
1. **Linear Flow Regression**: $\Delta P$ vs $\sqrt{t} \implies m_{LF} \implies x_{f,\text{est}} = \frac{4.064 q B \mu}{m_{LF} h \sqrt{k \phi \mu c_t}}$.
2. **Bilinear Flow Regression**: $\Delta P$ vs $t^{1/4} \implies m_{BL} \implies (k_f w_f)_{\text{est}} = \left(\frac{44.1 q B \mu}{m_{BL} h (\phi \mu c_t k)^{1/4}}\right)^2$.
3. **Non-Linear Model Matching**: Levenberg-Marquardt / L-BFGS-B optimization on transient pressure and derivative.
4. **Synthetic-Data Validation**: Generates a comparison table with True Value, Estimated Value, Absolute Error, and Relative Error (%).

---

## Predefined Educational Demo Cases

1. **CASE 01 — High-Conductivity Fracture**:
   - $k = 5$ mD, $x_f = 500$ ft, $k_f = 10,000$ mD, $w_f = 0.02$ ft ($C_{fD} = 0.08$)
   - *Objective*: Formation linear flow (half-slope) and transition to pseudoradial behavior.
2. **CASE 02 — Finite-Conductivity Fracture**:
   - $k = 5$ mD, $x_f = 500$ ft, $k_f = 1,000$ mD, $w_f = 0.01$ ft ($C_f = 10$ mD-ft, $C_{fD} = 0.004$)
   - *Objective*: Internal fracture choke restriction and bilinear flow (quarter-slope).
3. **CASE 03 — Long-Fracture Case**:
   - $k = 5$ mD, $x_f = 1,000$ ft, $k_f = 10,000$ mD, $w_f = 0.02$ ft
   - *Objective*: Extended linear transient and delayed transition to radial flow.

---

## Technology Stack

- **Language**: Python 3.10+
- **Numerical Computation**: NumPy, SciPy (Bessel functions, quadrature, optimization, regression)
- **Data Structures**: Pandas, Dataclasses
- **Visualization**: Plotly Graph Objects & Express
- **Dashboard UI**: Streamlit
- **Testing & Quality**: Pytest, Jupyter Notebook

---

## Installation & Running

```bash
# Clone the repository
git clone https://github.com/your-username/HydraPTA.git
cd HydraPTA

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit Dashboard
streamlit run app.py
```

---

## Running Automated Tests

```bash
pytest tests/ -v
```

---

## Model Assumptions & Limitations

1. **Reservoir Geometry**: Infinite-acting, horizontal, homogeneous, and isotropic slab reservoir of constant thickness $h$ and constant petrophysical properties ($\phi, k, c_t$).
2. **Fluid Mechanics**: Single-phase, slightly compressible liquid with constant dynamic viscosity $\mu$ and constant formation volume factor $B$. Darcy flow is assumed (non-Darcy / Forchheimer effects inside the fracture are neglected).
3. **Fracture Geometry**: Single vertical planar hydraulic fracture extending symmetrically with half-length $x_f$ and fully penetrating the formation net pay thickness $h$.
4. **Educational Scope**: HydraPTA is an educational and analytical tool designed to demonstrate pressure-transient concepts. It does not replace commercial multiphase numerical reservoir simulators (e.g., Eclipse, CMG, Saphir).

---

## Future Work

- [ ] Custom CSV field data upload with pressure gauge noise filtering.
- [ ] Multi-rate superposition (Horner analysis, Agarwal equivalent time).
- [ ] Dual-porosity / naturally fractured reservoir extensions (Warren-Root).
- [ ] Multi-stage fractured horizontal well (MFHW) trilinear transient model.
