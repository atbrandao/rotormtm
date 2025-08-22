# RotorMTM - Gyroscopic Metastructure Rotordynamics

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)

A Python library for rotor analysis with gyroscopic metastructures, based on the theory of coupled resonators for vibration control in rotating systems.

## 🎯 Objective

**RotorMTM** (Rotor MetaStructure) implements an innovative approach for vibration control in rotating systems through the use of **gyroscopic metastructures**. The library enables:

- **Modal analysis** of rotors with coupled resonators
- **Vibration control** through selective frequency attenuation
- **Dynamic absorber design** for rotating machinery
- **Forced response analysis** with synchronous and asynchronous excitation
- **Parameter optimization** of resonators for maximum efficiency

## 🔬 Theoretical Foundation

### Gyroscopic Metastructures
The concept is based on coupling **gyroscopic resonators** to the main rotor, creating a metastructure capable of:

1. **Directionally selective attenuation**: Independent control of forward/backward motions
2. **Frequency bandgaps**: Creation of frequency ranges with low transmissibility
3. **Non-reciprocal effects**: Behavior dependent on rotation direction

### Mathematical Model
The system is governed by the equations:

```
[M]{ẍ} + ([C] + Ω[G]){ẋ} + [K]{x} = {F}
```

Where:
- `[M]`, `[C]`, `[K]`: Mass, damping, and stiffness matrices of the coupled system
- `[G]`: Gyroscopic matrix
- `Ω`: Rotation speed
- `{F}`: External force vector

## 📁 Repository Structure

```
RotorMTM/
├── rotor_mtm_lib/              # Main library
│   ├── rotor_mtm/              # System modules
│   │   ├── rotor_mtm.py        # Main RotorMTM class
│   │   ├── harmbal.py          # Nonlinear harmonic analysis
│   │   ├── results.py          # Result classes
│   │   └── __init__.py
│   └── setup.py                # Library installation
├── gui/                        # Graphical interfaces
│   ├── interface_rotor_step1.py # Interface: System configuration
│   ├── interface_rotor_step2.py # Interface: FRF analysis
│   └── README.md               # Interface documentation
├── examples/                   # Examples and case studies
│   ├── app_campbell.py         # Campbell diagrams
│   ├── generate_plots.py       # Plot generation
│   └── *.ipynb                # Jupyter notebooks
├── results/                    # Analysis results
├── scripts/                    # Auxiliary scripts
├── tools/                      # Analysis tools
├── Multistage Pump/           # Case study: Multistage pump
├── Turboexpander/             # Case study: Turboexpander
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites
```bash
pip install numpy scipy plotly streamlit ross-rotordynamics
```

### Library Installation
```bash
cd rotor_mtm_lib
pip install -e .
```

### Installation Verification
```python
from rotor_mtm.rotor_mtm import RotorMTM
from rotor_mtm.results import LinearResults
print("RotorMTM successfully installed!")
```

## 💡 Usage Example

### Basic Configuration
```python
import ross as rs
from rotor_mtm.rotor_mtm import RotorMTM
import numpy as np

# Create base rotor
rotor = rs.rotor_example()

# Configure resonators
n_pos = [5, 10, 15]  # Nodal positions
masses = [1.0, 1.0, 1.0]  # Resonator masses
Id_values = [1e-3, 1e-3, 1e-3]  # Diametral moments
Ip_values = [5e-4, 5e-4, 5e-4]  # Polar moments

# Create disk elements
resonators = []
for i, pos in enumerate(n_pos):
    disk = rs.DiskElement(
        n=pos, m=masses[i], 
        Id=Id_values[i], Ip=Ip_values[i],
        tag=f'Resonator_{i+1}'
    )
    resonators.append(disk)

# Build RotorMTM
k0 = 1e6  # Radial stiffness (N/m)
k1 = 1e3  # Rotational stiffness (N.m/rad)

metarotor = RotorMTM(
    rotor=rotor,
    n_pos=n_pos,
    dk_r=resonators,
    k0=k0,
    k1=k1,
    var=0.1,      # Mass variation
    var_k=0.1,    # Stiffness variation
    p_damp=1e-4   # Proportional damping
)
```

### Modal Analysis
```python
# Modal analysis for different speeds
speeds = np.linspace(0, 1000, 51)
modal_results = metarotor.run_analysis(
    sp_arr=speeds,
    n_modes=30,
    diff_analysis=True,
    heatmap=True
)

# Visualize Campbell diagrams
from rotor_mtm.rotor_mtm import plot_campbell, plot_diff_modal

fig_campbell = plot_campbell(
    modal_results['w'], 
    speeds
)

fig_diff = plot_diff_modal(
    modal_results['w'],
    modal_results['diff'],
    speeds,
    mode='abs'
)
```

### Forced Response Analysis (FRF)
```python
# Configure FRF analysis
speeds = np.linspace(100, 800, 100)
forces = np.ones_like(speeds)  # Unit force

# DOFs of interest
probe_dof = [0, 1, 4, 5]  # x, y of nodes 0 and 1
probe_names = ['Node_0_x', 'Node_0_y', 'Node_1_x', 'Node_1_y']

# Run analysis
linear_results = metarotor.calc_frf(
    sp_arr=speeds,
    f=forces,
    probe_dof=probe_dof,
    probe_names=probe_names,
    f_node=0,  # Excitation node
    rotor_solo=False
)

# Plot FRF
fig_forward, fig_backward = linear_results.plot_frf(
    dof=probe_names,
    whirl='both',
    amplitude_units='rms'
)
```

### Comparative Analysis
```python
# Compare with rotor without resonators
linear_results_solo = metarotor.calc_frf(
    sp_arr=speeds,
    f=forces,
    probe_dof=probe_dof[:2],  # Only rotor DOFs
    probe_names=probe_names[:2],
    f_node=0,
    rotor_solo=True  # Rotor without resonators
)

# Calculate attenuation efficiency
efficiency = np.abs(linear_results.rf['Node_0_x']) / np.abs(linear_results_solo.rf['Node_0_x'])
```

## 🖥️ Graphical Interface

RotorMTM includes graphical interfaces developed in Streamlit for ease of use:

### Step 1: System Configuration
```bash
streamlit run gui/interface_rotor_step1.py
```
- Rotor loading (file or example)
- Interactive resonator configuration
- System visualization
- RotorMTM construction and saving

### Step 2: FRF Analysis
```bash
streamlit run gui/interface_rotor_step2.py
```
- RotorMTM system loading
- FRF analysis configuration
- Results visualization
- Comparative analysis with solo rotor

## 📊 Case Studies

### 1. Multistage Pump (`Multistage Pump/`)
- Analysis of 5-stage centrifugal pump
- Resonator optimization for instability control
- Comparison with experimental data

### 2. Turboexpander (`Turboexpander/`)
- Turboexpander-compressor system
- High-speed vibration control
- Nonlinear effects analysis

### 3. Compressor (`Turboexpander Compressor/`)
- Industrial centrifugal compressor
- Optimized dynamic absorbers
- Field data validation

## 📈 Advanced Features

### Nonlinear Analysis
```python
# Create nonlinear system with cubic stiffness
nonlinear_system = metarotor.create_Sys_NL(
    x_eq0=(0.001, None),  # Radial equilibrium
    x_eq1=(None, None),   # Rotational equilibrium
    sp=500,               # Reference speed
    n_harm=10,            # Harmonics
    nu=1                  # Inter-harmonics
)

# Harmonic analysis
from rotor_mtm.harmbal import run_integration
nonlinear_results = run_integration(
    system=nonlinear_system,
    frequency_list=np.linspace(400, 600, 21),
    initial_conditions='auto'
)
```

### Rainbow Metastructures
```python
# Configure gradual variation of properties
metarotor_rainbow = RotorMTM(
    rotor=rotor,
    n_pos=n_pos,
    dk_r=resonators,
    k0=k0,
    k1=k1,
    var=0.3,      # 30% mass variation
    var_k=0.2,    # 20% stiffness variation
    exp_var=2     # Quadratic variation
)
```

### Parameter Optimization
```python
# Objective function: minimize response at critical frequency
def objective_function(params):
    k0, k1 = params
    temp_rotor = RotorMTM(rotor, n_pos, resonators, k0, k1)
    results = temp_rotor.calc_frf(critical_speeds, forces, probe_dof)
    return np.max(np.abs(results.rf['Node_0_x']))

# Optimization with scipy
from scipy.optimize import minimize
result = minimize(
    objective_function,
    x0=[1e6, 1e3],
    bounds=[(1e4, 1e8), (1e1, 1e5)],
    method='L-BFGS-B'
)
```

## 📚 Academic References

### Main Articles

1. **Brandão, A.T., et al. (2022)**  
   *"Gyroscopic metastructures for vibration control in rotating machinery"*  
   Journal of Sound and Vibration, 516, 116982.  
   DOI: [10.1016/j.jsv.2022.116982](https://doi.org/10.1016/j.jsv.2022.116982)

2. **Brandão, A.T., et al. (2025)**  
   *"Nonlinear dynamics of gyroscopic metastructures with bi-stable resonators"*  
   Nonlinear Dynamics (aceito para publicação)  
   DOI: [10.1007/s11071-025-11597-z](https://doi.org/10.1007/s11071-025-11597-z)

3. **Hussein, M.I., Leamy, M.J., Ruzzene, M. (2014)**  
   *"Dynamics of phononic materials and structures: Historical origins, recent progress, and future outlook"*  
   Applied Mechanics Reviews, 66(4), 040802.

### Foundational Theory

4. **Vangbo, M. (1998)**  
   *"An analytical analysis of a compressed bistable buckled beam"*  
   Sensors and Actuators A: Physical, 69(3), 212-216.

5. **Genta, G. (2005)**  
   *"Dynamics of Rotating Systems"*  
   Springer Science & Business Media.

6. **Lalanne, M., Ferraris, G. (1998)**  
   *"Rotordynamics Prediction in Engineering"*  
   John Wiley & Sons.

### Applications and Methods

7. **Yu, D., et al. (2006)**  
   *"Flexural vibration band gaps in Timoshenko beams with locally resonant structures"*  
   Journal of Applied Physics, 100(12), 124901.

8. **Ma, G., Sheng, P. (2016)**  
   *"Acoustic metamaterials: From local resonances to broad horizons"*  
   Science Advances, 2(2), e1501595.

## 🤝 Contributions

Contributions are welcome! Please:

1. **Fork** the repository
2. **Create** a branch for your feature (`git checkout -b feature/new-feature`)
3. **Commit** your changes (`git commit -am 'Add new feature'`)
4. **Push** to the branch (`git push origin feature/new-feature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow Python coding conventions (PEP 8)
- Add tests for new features
- Keep documentation updated
- Include usage examples when appropriate

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **André A. T. Brandão** - *Main development* - [GitHub](https://github.com/atbrandao)

## 📞 Contact

For questions, suggestions, or collaborations:
- **Email**: [alexandre.brandao@petrobras.com.br]
- **Issues**: [GitHub Issues](https://github.com/atbrandao/rotormtm/issues)

## 🙏 Acknowledgments

- **UnB / ENM - Group of System Dynamics (GDS)** - Research infrastructure
- **ROSS Community** - Base for rotordynamic development

---

*Developed with ❤️ for the rotordynamics community*
