# RotorMTM Interface - GUI Folder

This folder contains the graphical interfaces developed in Streamlit for working with the RotorMTM system.

## Available Files

### Functional Interfaces
- **`interface_rotor_step1.py`** - Step 1 Interface: Rotor loading and resonator configuration
- **`interface_rotor_step2.py`** - Step 2 Interface: FRF analysis and results visualization

### Data Files
- **`rotor_mtm_3res.pkl`** - Example file with RotorMTM configured with 3 resonators
- **`rotor_system_3res.pkl`** - Additional example file for testing

## How to Use

### Folder Structure
```
gui/
├── interface_rotor_step1.py    # Step 1 Interface (working)
├── interface_rotor_step2.py    # Step 2 Interface (working)  
├── README.md                   # This documentation
├── rotor_mtm_3res.pkl         # RotorMTM example (3 resonators)
└── rotor_system_3res.pkl      # Additional example for testing
```

### Prerequisites
Make sure the `rotor_mtm` library is installed and accessible:
```python
from rotor_mtm.rotor_mtm import RotorMTM
from rotor_mtm.results import LinearResults
```

### Run the Interfaces

**Step 1 - System Configuration:**
```bash
streamlit run gui/interface_rotor_step1.py
```

**Step 2 - FRF Analysis:**
```bash
streamlit run gui/interface_rotor_step2.py
```

## Workflow

### Step 1: Configuration
1. **Rotor Loading**
   - Use ROSS example
   - Load from pickle file
   
2. **Resonator Configuration**
   - Define positions on rotor
   - Configure properties (mass, moments of inertia)
   
3. **RotorMTM Construction**
   - Define k0 and k1 parameters
   - Build RotorMTM object
   - Save .pkl file for use in Step 2

### Step 2: FRF Analysis
1. **RotorMTM Loading**
   - Load .pkl file from Step 1
   
2. **Analysis Configuration**
   - Define speed range
   - Configure excitation parameters
   - Select DOFs for analysis
   
3. **Execution and Visualization**
   - Execute FRF analysis
   - Visualize results (FRF, orbits)
   - Compare with solo rotor
   - Save results

## Main Features

### Step 1 Interface
- ✅ Rotor loading (example/file)
- ✅ Interactive resonator configuration
- ✅ 2D system visualization
- ✅ RotorMTM construction
- ✅ Configuration saving

### Step 2 Interface
- ✅ RotorMTM file loading
- ✅ Flexible FRF analysis configuration
- ✅ Parallel execution (with/without resonators)
- ✅ Advanced visualization (FRF, orbits)
- ✅ LinearResults saving

## Troubleshooting

### Import Error
If there's an error importing `rotor_mtm`, check:
- Library is installed: `pip install rotor_mtm`
- Library is in PYTHONPATH
- Library version is compatible

### File Not Recognized
If Step 2 doesn't recognize the file from Step 1:
- Check if file was generated correctly by `interface_rotor_step1.py`
- Verify file contains valid RotorMTM object
- Test with example file: `rotor_mtm_3res.pkl`

## Usage Examples

### Available Test File
Use the `rotor_mtm_3res.pkl` file as an example to test Step 2 without needing to configure a complete system.

### Typical Configurations
- **k0**: 1e6 N/m (radial stiffness)
- **k1**: 1e3 N.m/rad (rotational stiffness) 
- **Speed range**: 0-1000 rad/s
- **Analysis points**: 100 points
- **Excitation force**: 1 N

## Support

For problems or questions:
1. Check installation prerequisites
2. Consult error logs in terminal
3. Test with provided example files
4. Check library version compatibility
