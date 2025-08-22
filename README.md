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
├── scripts/                    # Auxiliary scripts
├── Turboexpander Compressor/   # Case study: Turboexpander
└── README.md                   # This file
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

## 📚 Academic References

### Main Articles

1. **Brandão, A.T., et al. (2022)**  
   *"Rainbow gyroscopic disk metastructures for broadband vibration attenuation in rotors"*  
   Journal of Sound and Vibration, 516, 116982.  
   DOI: [10.1016/j.jsv.2022.116982](https://doi.org/10.1016/j.jsv.2022.116982)

2. **Brandão, A.T., et al. (2025)**  
   *"Modally matched bistable disk metastructure for vibration attenuation in rotors: bandwidth widening and chaos"*  
   Nonlinear Dynamics
   DOI: [10.1007/s11071-025-11597-z](https://doi.org/10.1007/s11071-025-11597-z)

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## 👥 Authors

- **André A. T. Brandão** - *Main development* - [GitHub](https://github.com/atbrandao)

## 📞 Contact

For questions, suggestions, or collaborations:
- **Issues**: [GitHub Issues](https://github.com/atbrandao/rotormtm/issues)

## 🙏 Acknowledgments

- **UnB / ENM - Grupo de Dinâmica de Sistemas (GDS)** - Research infrastructure
- **ROSS Community** - Base for rotordynamic development

---

*Developed with ❤️ for the rotordynamics community*
