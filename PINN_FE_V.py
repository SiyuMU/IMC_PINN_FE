"""
Volume Constrained PINN - FE
===========================

This script implements a volume-constrained Physics-Informed Neural Network (PINN) for Finite Element (FE) cardiac modeling.

Author: Siyu MU
Date: 2025-05-08

Usage:
    python PINN_FE_V.py

"""

import random
import functions as fc
import torch
import os
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import deepxde.nn.pytorch as deepxde
import vtk
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d
from tqdm import tqdm
import copy

# Function to check if model parameters contain NaN or Inf
def check_model_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN or Inf detected in parameter: {name}")
            return False
    return True

def compute_increasing_slope_loss(volume, P, min_index, max_index, device):
    # === Additional Loss: Enforce Increasing Slopes ===
    # Ensure volume and P are 1D tensors
    volume_flat = volume.view(-1)
    P_flat = P.view(-1)
    
    # Clamp indices to valid range and ensure min_index allows for at least one preceding point
    length = volume_flat.shape[0]
    min_index = max(1, min_index)  # ensure at least one point before for slope computation
    max_index = min(length - 1, max_index)  # ensure max_index is within valid range
    
    # Select the subset from min_index to max_index
    P_subset = P_flat[min_index:max_index+1]
    volume_subset = volume_flat[min_index:max_index+1]
    
    # Compute differences in P and volume
    delta_P = P_subset[1:] - P_subset[:-1]
    # Add a small value to volume differences to prevent division by zero
    delta_V = volume_subset[1:] - volume_subset[:-1] + 1e-8
    
    # Compute slopes
    slopes = delta_P / delta_V
    
    # Compute differences between consecutive slopes
    if len(slopes) >= 2:
        delta_slopes = slopes[1:] - slopes[:-1]
        # Enforce increasing slopes: penalize negative slope differences
        loss_slope_increasing = torch.relu(-delta_slopes).mean()
    else:
        # If there are not enough points, set the loss to zero
        loss_slope_increasing = torch.tensor(0.0, device=device)
    
    return loss_slope_increasing

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def loss_increasing_P(P,min_index):
    """
    Calculate loss to ensure P increases monotonically from its minimum to the last value.
    Args:
        P (torch.Tensor): The tensor containing pressure values.

    Returns:
        torch.Tensor: The calculated loss.
    """
    
    # Extract the slice of P from min_index to the last index
    P_after_min = P[min_index:]
    
    # Calculate the difference between consecutive elements
    diff = P_after_min[1:] - P_after_min[:-1]
    
    # Ensure all differences are positive, penalize negative values
    loss = torch.sum(torch.relu(-diff))  # Negative differences are penalized
    
    return loss
#_____________________________________________________________________________
# SETTING UP (change the following parameters as needed)
# Set the device to GPU if available
set_seed(48) 
cuda_index = 1 
device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
map_location = device
output_folder_name = '/Output_PINNFE' # Output folder name

disp_match = True # if True, the volume is constrained to match the experimental data
C_strain =   94.21871304512024# if only estimate Tmax you need to define stiffness 
T_max = 126538.83695602417 # Maximum tension, Pa
ED_pressure = 11 # End-diastolic pressure
ED_pressure_max = 100 # Maximum pressure
n_modesU = 20 # Number of modes to be used in the POD basis
t_total = 800 # total time of the cardiac cycle, ms
t0 = 171 # time to peak tension, ms
Ca0 = 4.35 # Maximum intracellular calcium concentration
B = 4.75 # governs shape of peak isometric tension-sarcomere length relation, µm−1
l0 = 1.58 # sarcomere length at which no active tension develops
lr = 1.85 # Relaxed sarcomere length
m = 524.0 # slope of linear relation of relaxation duration and sarcomere length in ms um-1
b = -800.0 # time intercept of linear relation of relaxation duration and sarcomere length in ms
eval_mode = False # if True, the model is evaluated without training
v_normalization = 130 # scaling value for volume [mL]
t_normalization   = t_total # scaling value for t [ms]

#_____________________________________________________________________________
# Initialize the parameters
current_directory = os.getcwd()
out_folder = current_directory + output_folder_name
current_directory = current_directory + '/geo/' 
unloaded_shape_name = 'unloaded_geo.vtk'
pod_basis_filename = 'POD_basis/shape_modes.npy'
max_amplitude_filename = 'POD_basis/max_coefficients.npy'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# Define the material properties
Cfsn = np.array([
    29.9, 53.2, 53.2,
    53.2, 13.3, 26.6,
    53.2, 26.6, 13.3
], dtype=np.float32)

# Load stress parameters
stress_para = [Ca0, B, l0, lr, t0, m, b]

# Read the geometry
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(os.path.join(current_directory, unloaded_shape_name))
reader.ReadAllScalarsOn()
reader.ReadAllVectorsOn()
reader.Update()
geo_mesh = reader.GetOutput()

# Get the number of elements and nodes
n_el     = geo_mesh.GetNumberOfCells()
Els      = np.zeros((n_el,4),dtype=int)
for i in range(n_el):
    cell_type = geo_mesh.GetCellType(i)
    n_nodes_el   = geo_mesh.GetCell(i).GetPointIds().GetNumberOfIds()
    for n_sel in range(n_nodes_el):
        Els[i,n_sel] = int(geo_mesh.GetCell(i).GetPointId(n_sel))
Coords   =  vtk_to_numpy(geo_mesh.GetPoints().GetData())
num_nodes = Coords.shape[0]

# Get fsn, nodal volume
f  = vtk_to_numpy(geo_mesh.GetPointData().GetVectors('fiber vectors'))
s  = vtk_to_numpy(geo_mesh.GetPointData().GetVectors('sheet vectors'))
n  = vtk_to_numpy(geo_mesh.GetPointData().GetVectors('sheet normal vectors'))
nodal_volume = vtk_to_numpy(geo_mesh.GetPointData().GetScalars('Nodal_volume'))
nodal_area = vtk_to_numpy(geo_mesh.GetPointData().GetScalars('Nodal_area'))
label = vtk_to_numpy(geo_mesh.GetPointData().GetScalars('label'))

# Extract endocardial faces
faces_connectivity = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
Faces_Endo = []
start_faces = True
for kk in range(n_el):
    el_points = Els[kk, :]
    for jj in range(4):
        # Check if all nodes on a face belong to the endocardial label
        if all(label[int(v)] == 1 for v in el_points[faces_connectivity[jj]]):
            if start_faces:
                Faces_Endo = np.array(el_points[faces_connectivity[jj]], dtype=int).reshape(1, -1)
                start_faces = False
            else:
                Faces_Endo = np.concatenate((Faces_Endo, np.array(el_points[faces_connectivity[jj]], dtype=int).reshape(1, -1)), 0)

# unload chamber volume         
unloaded_chamber_volume = fc.compute_chamber_volume(nodal_area, Coords)

# volume truth
file_name = current_directory+'/tetra_vtks/chamber_volumes.txt'
values = []
with open(file_name, 'r') as file:
    for line in file:
        parts = line.split(':')
        if len(parts) == 2: 
            try:
                value = float(parts[1].strip())
                values.append(value)
            except ValueError:
                continue
volume_truth = np.array(values).reshape(-1)
volume_max_min = np.array([np.max(volume_truth), np.min(volume_truth)])

# Load the POD basis
VT = np.load(os.path.join(current_directory, pod_basis_filename))

# Extract the contribution of each direction (x, y, z)
Phix_s = VT[:n_modesU, 0::3].T  
Phiy_s = VT[:n_modesU, 1::3].T  
Phiz_s = VT[:n_modesU, 2::3].T 

# Compute the gradient of the deformation gradient tensor (POD)
dFdx, dFdy, dFdz = fc.GradientOperator_AvgBased(Coords, Els)
dFudx_s = dFdx.dot(Phix_s)
dFudy_s = dFdy.dot(Phix_s)
dFudz_s = dFdz.dot(Phix_s)

dFvdx_s = dFdx.dot(Phiy_s)
dFvdy_s = dFdy.dot(Phiy_s)
dFvdz_s = dFdz.dot(Phiy_s)

dFwdx_s = dFdx.dot(Phiz_s)
dFwdy_s = dFdy.dot(Phiz_s)
dFwdz_s = dFdz.dot(Phiz_s)

# Load the maximum amplitude (for normalization)
max_amplitude = np.load(os.path.join(current_directory, max_amplitude_filename))
max_amplitude = np.max(max_amplitude)

# Generate (volume,t) tuples for training
t_volume  = np.linspace(0.0,1.0,volume_truth.shape[0])
t_range = np.linspace(0.0,1.0,t_total)
volume_truth[-1] = volume_truth[0]
interp_func = interp1d(t_volume, volume_truth, kind='linear')
volume_range = interp_func(t_range) 
param_grid = []
for iv in range(volume_range.shape[0]): # allowed pressure values
    param_grid.append([volume_range[iv]/v_normalization,t_range[iv]])
param_grid = np.array(param_grid)


# Convert to torch tensors
tensor_f = torch.tensor(f, dtype=torch.float32, device=device)
tensor_s = torch.tensor(s, dtype=torch.float32, device=device)
tensor_n = torch.tensor(n, dtype=torch.float32, device=device)

tensor_nodal_volume = torch.tensor(nodal_volume, dtype=torch.float32, device=device).reshape(-1,1)

tensor_Phi_x = torch.tensor(Phix_s, dtype=torch.float32, device=device)
tensor_Phi_y = torch.tensor(Phiy_s, dtype=torch.float32, device=device)
tensor_Phi_z = torch.tensor(Phiz_s, dtype=torch.float32, device=device)
tensor_Phi = torch.stack((tensor_Phi_x, tensor_Phi_y, tensor_Phi_z), dim=2)

tensor_dFudx = torch.tensor(dFudx_s, dtype=torch.float32, device=device)
tensor_dFudy = torch.tensor(dFudy_s, dtype=torch.float32, device=device)
tensor_dFudz = torch.tensor(dFudz_s, dtype=torch.float32, device=device)

tensor_dFvdx = torch.tensor(dFvdx_s, dtype=torch.float32, device=device)
tensor_dFvdy = torch.tensor(dFvdy_s, dtype=torch.float32, device=device)
tensor_dFvdz = torch.tensor(dFvdz_s, dtype=torch.float32, device=device)

tensor_dFwdx = torch.tensor(dFwdx_s, dtype=torch.float32, device=device)
tensor_dFwdy = torch.tensor(dFwdy_s, dtype=torch.float32, device=device)
tensor_dFwdz = torch.tensor(dFwdz_s, dtype=torch.float32, device=device)

tensor_Cfsn = torch.tensor(Cfsn, dtype=torch.float32, device=device)
tensor_max_amplitude = torch.tensor(max_amplitude, dtype=torch.float32, device=device).reshape(1, 1)
tensor_unloaded_chamber_volume = torch.tensor(unloaded_chamber_volume, dtype=torch.float32, device=device).reshape(1, 1)
tensor_Coord = torch.tensor(Coords, dtype=torch.float32, device=device)
tensor_Faces_Endo = torch.tensor(Faces_Endo, dtype=torch.int32, device=device)
tensor_label = torch.tensor(label, dtype=torch.int32, device=device)
tensor_volume_max_min = torch.tensor(volume_max_min, dtype=torch.float32, device=device)
tensor_nodal_area = torch.tensor(nodal_area, dtype=torch.float32, device=device).reshape(-1,3)

input_parameter_tensor = torch.cat((tensor_f, tensor_s, tensor_n, tensor_nodal_volume, tensor_Phi_x, tensor_Phi_y, tensor_Phi_z, tensor_dFudx, tensor_dFudy, tensor_dFudz, tensor_dFvdx, tensor_dFvdy, tensor_dFvdz, tensor_dFwdx, tensor_dFwdy, tensor_dFwdz,tensor_nodal_area), dim=1)
input_tensor = torch.tensor(param_grid.reshape(-1,2), dtype=torch.float32, device=device)

#_____________________________________________________________________________
# diastole
# Define the neural network
# Define training parameters
epochs = 500  # Maximum training epochs
learn_rate = 0.0001  # Initial learning rate

print('Training the network...')
start_time = time.time()  # Start time

model = deepxde.fnn.FNN(
    [2, 128, 128, 128, 128,n_modesU+1],  # Layer sizes: input -> hidden layers -> output
    fc.cus_silu,                          # Activation function
    "Glorot normal",                      # Weight initializer
).to(device)

# eval mode
if eval_mode:
    epochs = 1
    model.eval()
    model.load_state_dict(torch.load('Output_PINNFE/Trained_model_d.pth', map_location=map_location, weights_only=True))

# Define the optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learn_rate,
    betas=(0.9, 0.98),
    weight_decay=1e-4,
    eps=1e-6
)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

loss_vector = np.array([])  # Initialize the loss vector

# Initialize variables to save the previous state
previous_state = copy.deepcopy(model.state_dict())
previous_optimizer_state = copy.deepcopy(optimizer.state_dict())

min_lr = 1e-6  # Minimum learning rate to prevent it from getting too small
print_FLAG = False
input_tensor_diastolic = input_tensor[torch.argmin(input_tensor[...,0:1]):].reshape(-1,2)
for epoch in tqdm(range(epochs), desc="Training Progress"):
    # Save the current state before training
    current_state = copy.deepcopy(model.state_dict())
    current_optimizer_state = copy.deepcopy(optimizer.state_dict())
    
    optimizer.zero_grad()
    output_tensor = model.forward(input_tensor_diastolic)
    
    # Calculate loss using your custom loss function
    loss, volume, P, disp = fc.cardioloss_FE_v_diastolic(
        input_tensor_diastolic, output_tensor, tensor_Coord, input_parameter_tensor, stress_para,
        C_strain, T_max, tensor_Cfsn, tensor_max_amplitude,
        t_normalization, tensor_unloaded_chamber_volume, device=device, Print=print_FLAG
    )
    print_FLAG = False

    # Calculate additional loss components
    loss_volume = torch.nn.functional.mse_loss(volume.reshape(-1), input_tensor_diastolic[...,0:1].reshape(-1)*v_normalization)
    loss_pressure_ED = torch.nn.functional.mse_loss(P[-1], torch.zeros_like(P[0])+ED_pressure)
    loss_pressure_0 = torch.nn.functional.mse_loss(P[0], torch.zeros_like(P[0])) 
    loss_diastolic_slope = compute_increasing_slope_loss(volume, P, 0, volume.shape[0], device)
    loss_diastolic_slope_increase = loss_increasing_P(P,0) # Alternative to the slope loss used in the paper.

    # Total loss with weighted components
    loss_total = loss + 1e2*(loss_pressure_ED + loss_pressure_0 + loss_volume) + 1e6*(loss_diastolic_slope + loss_diastolic_slope_increase)
    
    # Backward pass
    loss_total.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    
    # Optimizer step
    optimizer.step()
    
    # Scheduler step
    scheduler.step()
    
    # Append loss to the loss vector
    loss_vector = np.append(loss_vector, loss_total.item())
    
    # Check if loss is finite
    loss_is_finite = torch.isfinite(loss_total).all().item()
    params_are_finite = check_model_parameters(model)
    
    if not loss_is_finite or not params_are_finite:
        if epoch == 0 or epoch == 1:
            print("Loss is NaN or Inf at the first epoch. Re-initializing the model and optimizer.")
            # Re-initialize the model and optimizer
            # Define the neural network model
            model = deepxde.fnn.FNN(
                [2, 128, 128, 128, 128,n_modesU+1],  # Layer sizes: input -> hidden layers -> output
                'relu',#fc.cus_silu,                          # Activation function
                "Glorot normal",                      # Weight initializer
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learn_rate,
                betas=(0.9, 0.98),
                weight_decay=1e-4,
                eps=1e-6
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=700, gamma=0.2)
        else:
            print(f"NaN detected at epoch {epoch}. Reverting to previous model state and reduding the learning rate.")
            # Revert to previous state
            model.load_state_dict(previous_state)
            optimizer.load_state_dict(previous_optimizer_state)
            
            # Halve the learning rate
            for param_group in optimizer.param_groups:
                new_lr = param_group['lr'] * 0.8
                if new_lr < min_lr:
                    new_lr = min_lr
                    print(f"Learning rate is below minimum threshold. Setting to min_lr: {min_lr}")
                param_group['lr'] = new_lr
                print(f"Learning rate for param group set to {new_lr}")
            
            # Optionally, adjust the scheduler if necessary
            # Depending on the scheduler's state, you might need to reset or adjust it
            continue  # Skip the rest of this epoch
        
    else:
        # If no NaN, update the previous state
        previous_state = copy.deepcopy(current_state)
        previous_optimizer_state = copy.deepcopy(current_optimizer_state)
    
    # Record max value and index from volume
    max_value, max_index = torch.max(volume, dim=0)
    
    # Every 10 epochs, print logs and save loss plot
    if epoch % 10 == 0:
        print_FLAG = False
        # Print log
        tqdm.write(f"Epoch {epoch}: Loss_total={loss_total:.4f}, Loss={loss:.4f}, "
                   f"Loss_volume={loss_volume:.4f}, Loss_pressure_ED={loss_pressure_ED:.4f} , loss_pressure_0={loss_pressure_0:.4f}, loss_diastolic_slope={loss_diastolic_slope:.4f}, loss_diastolic_slope_increase={loss_diastolic_slope_increase:.4f}")
        # Plot and save loss (log scale)
        if epoch > 5:
            # Clean the loss vector to remove NaN and non-positive values for log scale
            clean_loss_vector = [loss for loss in loss_vector[5:epoch] if not np.isnan(loss) and loss > 0]
            if len(clean_loss_vector) > 0:
                plt.figure(facecolor='white') 
                plt.plot(clean_loss_vector, label='Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss (Log Scale)')
                plt.yscale('log')  # Set y-axis to log scale
                plt.grid(True, which="both", ls="--", linewidth=0.5)  # Add log grid
                
                ax = plt.gca()
                ax.set_facecolor('white') 
                ax.spines['bottom'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.spines['top'].set_color('none')
                ax.spines['right'].set_color('none')
                ax.xaxis.label.set_color('black')
                ax.yaxis.label.set_color('black')
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')
                
                plt.legend()
                plt.tight_layout()
                
                # Save plot
                plt.savefig(os.path.join(out_folder, 'Loss_log_dia.png'), dpi=400)
                plt.close()

# Optionally, save the final model
if not eval_mode:
    torch.save(model.state_dict(), os.path.join(out_folder, "Trained_model_d.pth"))

volume_d = volume.cpu().detach().numpy()
P_d = P.cpu().detach().numpy()
disp_d = disp.cpu().detach().numpy()

# _____________________________________________________________________________
# systole
# Define the neural network
# Define training parameters
epochs = 1000  # Maximum training epochs
learn_rate = 0.0007  # Initial learning rate

print('Training the network...')
start_time = time.time()  # Start time

model = deepxde.fnn.FNN(
    [2, 64, 64, 64, 64,n_modesU+1],  # Layer sizes: input -> hidden layers -> output
    fc.cus_silu,                          # Activation function
    "Glorot normal",                      # Weight initializer
).to(device)

# eval mode
if eval_mode:
    epochs = 1
    model.eval()
    model.load_state_dict(torch.load('Output_PINNFE/Trained_model_s.pth', map_location=map_location, weights_only=True))

# Define the optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learn_rate,
    betas=(0.9, 0.98),
    weight_decay=1e-4,
    eps=1e-6
)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


loss_vector = np.array([])  # Initialize the loss vector

# Initialize variables to save the previous state
previous_state = copy.deepcopy(model.state_dict())
previous_optimizer_state = copy.deepcopy(optimizer.state_dict())

min_lr = 1e-6  # Minimum learning rate to prevent it from getting too small
print_FLAG = False
input_tensor_systolic = input_tensor[0:torch.argmin(input_tensor[...,0:1])].reshape(-1,2)
for epoch in tqdm(range(epochs), desc="Training Progress"):
    # Save the current state before training
    current_state = copy.deepcopy(model.state_dict())
    current_optimizer_state = copy.deepcopy(optimizer.state_dict())
    
    optimizer.zero_grad()
    output_tensor = model.forward(input_tensor_systolic)
    
    # Calculate loss using your custom loss function
    loss, volume, P, disp = fc.cardioloss_FE_v_systolic(
        input_tensor_systolic, output_tensor, tensor_Coord, input_parameter_tensor, stress_para,
        C_strain, T_max, tensor_Cfsn, tensor_max_amplitude,
        t_normalization, tensor_unloaded_chamber_volume, device=device, Print=print_FLAG
    )
    print_FLAG = False

    # Calculate additional loss components
    loss_volume = torch.nn.functional.mse_loss(volume.reshape(-1), input_tensor_systolic[...,0:1].reshape(-1)*v_normalization)
    loss_pressure_ED = torch.nn.functional.mse_loss(P[0], torch.zeros_like(P[0])+ED_pressure)
    loss_max = torch.nn.functional.mse_loss(P.max(), torch.zeros_like(P.max())+ED_pressure_max)
    loss_pressure_0 = torch.nn.functional.mse_loss(P[-1], torch.zeros_like(P[0]))
    loss_systolic_slope = compute_increasing_slope_loss(volume, P, 0, volume.shape[0], device)

    # Total loss with weighted components
    loss_total = loss + 2*loss_max + 1e3*loss_volume + 1e2*loss_pressure_ED + 1e4*(loss_pressure_0+loss_systolic_slope)
    
    # Backward pass
    loss_total.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    
    # Optimizer step
    optimizer.step()
    
    # Scheduler step
    scheduler.step()
    
    # Append loss to the loss vector
    loss_vector = np.append(loss_vector, loss_total.item())
    
    # Check if loss is finite
    loss_is_finite = torch.isfinite(loss_total).all().item()
    params_are_finite = check_model_parameters(model)
    
    if not loss_is_finite or not params_are_finite:
        if epoch == 0 or epoch == 1:
            print("Loss is NaN or Inf at the first epoch. Re-initializing the model and optimizer.")
            # Re-initialize the model and optimizer
            # Define the neural network model
            model = deepxde.fnn.FNN(
                [2, 64, 64, 64, 64,n_modesU+1],  # Layer sizes: input -> hidden layers -> output
                'relu',#fc.cus_silu,                          # Activation function
                "Glorot normal",                      # Weight initializer
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learn_rate,
                betas=(0.9, 0.98),
                weight_decay=1e-4,
                eps=1e-6
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=700, gamma=0.2)
        else:
            print(f"NaN detected at epoch {epoch}. Reverting to previous model state and reduding the learning rate.")
            # Revert to previous state
            model.load_state_dict(previous_state)
            optimizer.load_state_dict(previous_optimizer_state)
            
            # Halve the learning rate
            for param_group in optimizer.param_groups:
                new_lr = param_group['lr'] * 0.8
                if new_lr < min_lr:
                    new_lr = min_lr
                    print(f"Learning rate is below minimum threshold. Setting to min_lr: {min_lr}")
                param_group['lr'] = new_lr
                print(f"Learning rate for param group set to {new_lr}")
            
            # Optionally, adjust the scheduler if necessary
            # Depending on the scheduler's state, you might need to reset or adjust it
            continue  # Skip the rest of this epoch
        
    else:
        # If no NaN, update the previous state
        previous_state = copy.deepcopy(current_state)
        previous_optimizer_state = copy.deepcopy(current_optimizer_state)
    
    # Record max value and index from volume
    max_value, max_index = torch.max(volume, dim=0)
    
    # Every 10 epochs, print logs and save loss plot
    if epoch % 10 == 0:
        print_FLAG = False
        # Print log
        tqdm.write(f"Epoch {epoch}: Loss_total={loss_total:.4f}, Loss={loss:.4f}, "
                   f"Loss_volume={loss_volume:.4f}, Loss_pressure_ED={loss_pressure_ED:.4f} , loss_pressure_0={loss_pressure_0:.4f}, loss_systolic_slope={loss_systolic_slope:.4f}, loss_max={loss_max:.4f}")
        # Plot and save loss (log scale)
        if epoch > 5:
            # Clean the loss vector to remove NaN and non-positive values for log scale
            clean_loss_vector = [loss for loss in loss_vector[5:epoch] if not np.isnan(loss) and loss > 0]
            if len(clean_loss_vector) > 0:
                plt.figure(facecolor='white') 
                plt.plot(clean_loss_vector, label='Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss (Log Scale)')
                plt.yscale('log')  # Set y-axis to log scale
                plt.grid(True, which="both", ls="--", linewidth=0.5)  # Add log grid
                
                ax = plt.gca()
                ax.set_facecolor('white') 
                ax.spines['bottom'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.spines['top'].set_color('none')
                ax.spines['right'].set_color('none')
                ax.xaxis.label.set_color('black')
                ax.yaxis.label.set_color('black')
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')
                
                plt.legend()
                plt.tight_layout()
                
                # Save plot
                plt.savefig(os.path.join(out_folder, 'Loss_log_sys.png'), dpi=400)
                plt.close()

# Optionally, save the final model
if not eval_mode:
    torch.save(model.state_dict(), os.path.join(out_folder, "Trained_model_s.pth"))

volume_s = volume.cpu().detach().numpy()
P_s = P.cpu().detach().numpy()
disp_s = disp.cpu().detach().numpy()
#_____________________________________________________________________________
# visualize the results
P = np.concatenate((P_s,P_d),axis=0)
volume = np.concatenate((volume_s,volume_d),axis=0)
disp = np.concatenate((disp_s,disp_d),axis=0)
P[-1] = P[0]
volume[-1] = volume[0]

# PV loop 
# save the data
data = np.column_stack((P, volume))
output_file = out_folder + "/PV_data.txt"
np.savetxt(output_file, data, fmt='%.6f', delimiter='\t', header='Pressure\tVolume', comments='')

plt.figure(facecolor='white') 
plt.plot(volume, P, 'b-', label='PINN')
plt.xlabel('Volume(mL)')
plt.ylabel('Pressure(mmHg)')
plt.title('PV loop')
ax = plt.gca()
ax.set_facecolor('white') 
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

plt.legend()
plt.tight_layout()
plt.savefig(out_folder + '/PV.png', dpi=400)
plt.close()

# Save VTK files with updated coordinates
fc.save_vtk_with_updated_coords(geo_mesh, disp, output_folder=out_folder+'/disp')















       



