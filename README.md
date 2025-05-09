# IMC-PINN-FE: A Physics-Informed Neural Network for Patient-Specific Left Ventricular Finite Element Modeling with Image Motion Consistency and Biomechanical Parameter Estimation

paper has been submitted to Computer Methods in Applied Mechanics and Engineering


## Project Structure
```
.
├── Pre-processing.ipynb     # Data preprocessing 
├── getStiffTmax.py         # Stiffness and maximum tension estimation
├── PINN_FE_V.py            # Volume-constrained PINN implementation
├── functions.py            # Utility functions and loss calculations
├── environment.yml         # Conda environment configuration
├── geo/                    # Geometry and mesh files
│   ├── image_shape/       # Tracked cardiac shapes
│   ├── POD_basis/         # Proper Orthogonal Decomposition basis
│   └── coefficients/      # POD coefficients
├── Output_CandT/          # Output for stiffness and tension
└── Output_PINNFE/         # Output for PINN-FE simulations
```

## Installation

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate IMC_PINN_FE
```

## Workflow

### 1. Pre-processing (Pre-processing.ipynb)
- Loads and processes cardiac MRI/echo data
- Pre-computes tracking results in `geo/image_shape/`
- Generates proper orthogonal decomposition (POD) basis
- Key outputs:
  - POD basis and coefficients in `geo/POD_basis/` and `geo/coefficients/`

### 2. Stiffness and Maximum Tension Estimation (getStiffTmax.py)
- Uses tracked shapes to estimate:
  - Tissue stiffness (C_strain)
  - Maximum active tension (Tmax)
- Implements physics-informed optimization
- Outputs results to `Output_CandT/`

### 3. Volume-constrained PINN (PINN_FE_V.py)
- Implements the main PINN model for cardiac mechanics
- Features:
  - Volume-constrained optimization
  - Diastolic and systolic phase modeling
  - Pressure-volume relationship prediction
- Outputs results to `Output_PINNFE/`
  - Pressure-volume curves
  - Loss history
  - Trained models
  - Displacement fields

## Key Parameters

### getStiffTmax.py
- `ED_pressure`: End-diastolic pressure
- `PS_pressure`: Peak systolic pressure
- `n_modesU`: Number of POD modes
- `t_total`: Total cardiac cycle duration
- `t0`: Time to peak tension

### PINN_FE_V.py
- `eval_mode`: Toggle between training and evaluation
- `v_normalization`: Volume normalization factor
- `t_normalization`: Time normalization factor
- `C_strain`: Tissue stiffness parameter
- `T_max`: Maximum active tension

## Usage

1. First, run the preprocessing:
```bash
jupyter notebook Pre-processing.ipynb
```

2. Estimate stiffness and maximum tension:
```bash
python getStiffTmax.py
```

3. Run the volume-constrained PINN:
```bash
python PINN_FE_V.py
```

## Output Files

### Output_CandT/
- `C_strain_Tmax.txt`: Estimated stiffness and maximum tension values
- `Trained_model.pth`: Trained model weights
- `Trained_model2.pth`: Secondary trained model weights

### Output_PINNFE/
- `PV.png`: Pressure-volume loop visualization
- `PV_data.txt`: Numerical pressure-volume data
- `Loss_log_dia.png`: Diastolic phase loss history
- `Loss_log_sys.png`: Systolic phase loss history
- `Trained_model_d.pth`: Diastolic phase model weights
- `Trained_model_s.pth`: Systolic phase model weights
- `disp/`: Displacement field visualizations

## Citation
This code has some functions from https://github.com/sbuoso/Cardio-PINN
