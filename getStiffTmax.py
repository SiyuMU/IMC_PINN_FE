"""
Get Stiffness and Tmax using PINN
=================================

This script estimates cardiac tissue stiffness (C_strain) and maximum active tension (Tmax)
using Physics-Informed Neural Networks (PINN).

Author: Siyu MU
Date: 2025-05-08

Usage:
    python getStiffTmax.py

"""

import functions as fc
import torch
import os
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import deepxde.nn.pytorch as deepxde
import vtk
import time
from tqdm import tqdm
import copy

# Function to check if model parameters contain NaN or Inf
def check_model_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN or Inf detected in parameter: {name}")
            return False
    return True
#_____________________________________________________________________________
# SETTING UP (change the following parameters as needed)
# Set the device to GPU if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
output_folder_name = '/Output_CandT' # Output folder name

ED_pressure = 11 # End-diastolic pressure
PS_pressure = 125 # Maximum pressure
n_modesU = 20 # Number of modes to be used in the POD basis
t_total = 800 # total time of the cardiac cycle, ms
t0 = 171 # time to peak tension, ms
Ca0 = 4.35 # Maximum intracellular calcium concentration
B = 4.75 # governs shape of peak isometric tension-sarcomere length relation, µm−1
l0 = 1.58 # sarcomere length at which no active tension develops
lr = 1.85 # Relaxed sarcomere length
m = 524.0 # slope of linear relation of relaxation duration and sarcomere length in ms um-1
b = -800.0 # time intercept of linear relation of relaxation duration and sarcomere length in ms

C_strain_normalization = 100 # scaling value for C_strain
Tmax_normalization = 1e5 # scaling value for Tmax
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
min_index = np.argmin(volume_truth) # index of minimum volume


t_peak_pressure = int(np.ceil(t0/(t_total/(volume_truth.shape[0]-1)))) # assume t_peak approx. equal to t0, but normally, it delays a bit by experience

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

# Generate (V,t) tuples for training
input_ED = np.array([[volume_truth[0]/v_normalization, 0]]) # End-diastolic point
input_PS = np.array([[volume_truth[t_peak_pressure]/v_normalization, t_peak_pressure*(t_total/(volume_truth.shape[0]-1))/t_normalization]]) # Peak systolic point

# Load the coefficients
Amplitude = np.zeros((volume_truth.shape[0]-1, n_modesU))
count = 0
for i in range(volume_truth.shape[0]-1):
    Amplitude[count,...] = np.load(current_directory + 'coefficients/coefficients_{:02}'.format(i) + '.npy')
    count += 1
Amplitude = Amplitude/max_amplitude
A_ED = Amplitude[0:1,...] # End-diastolic coefficients
A_PS = Amplitude[t_peak_pressure:(t_peak_pressure+1),...] # Systolic coefficients

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
tensor_nodal_area = torch.tensor(nodal_area, dtype=torch.float32, device=device).reshape(-1,3)

input_parameter_tensor = torch.cat((tensor_f, tensor_s, tensor_n, tensor_nodal_volume, tensor_Phi_x, tensor_Phi_y, tensor_Phi_z, tensor_dFudx, tensor_dFudy, tensor_dFudz, tensor_dFvdx, tensor_dFvdy, tensor_dFvdz, tensor_dFwdx, tensor_dFwdy, tensor_dFwdz,tensor_nodal_area), dim=1)
tensor_input_ED = torch.tensor(input_ED, dtype=torch.float32, device=device)
tensor_input_PS = torch.tensor(input_PS, dtype=torch.float32, device=device)
tensor_A_ED = torch.tensor(A_ED, dtype=torch.float32, device=device)
tensor_A_PS = torch.tensor(A_PS, dtype=torch.float32, device=device)

#_____________________________________________________________________________
# C_Strain estimator
# Define the neural network
# Define training parameters
epochs = 1500  # Maximum training epochs
learn_rate = 0.001  # Initial learning rate
convergence_threshold = 1e-3  # Threshold for loss change to detect convergence
patience = 50  # Number of consecutive epochs to monitor for convergence

print('Training the C_strain network...')
start_time = time.time()  # Start time

# Define the neural network model
# input could be none, cuz it's a optimisation problem, but for simplicity, we use the input V
model = deepxde.fnn.FNN(
    [1, 16, 16, 16, 1],  # Layer sizes: input -> hidden layers -> output
    'relu',              # Activation function
    "Glorot normal",     # Weight initializer
).to(device)

# Define the optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learn_rate,
    betas=(0.9, 0.98),
    weight_decay=1e-4,
    eps=1e-6
)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

loss_vector = np.array([])  # Initialize the loss vector
previous_state = copy.deepcopy(model.state_dict())
previous_optimizer_state = copy.deepcopy(optimizer.state_dict())

min_lr = 1e-5  # Minimum learning rate to prevent it from getting too small
print_FLAG = False # Print flag

# Variables for convergence checking
previous_loss = float('inf')  # Initialize previous loss as infinity
no_improvement_epochs = 0  # Count epochs with small loss improvement

# Training loop
for epoch in tqdm(range(epochs), desc="Training Progress"):
    # Save the current state before training
    current_state = copy.deepcopy(model.state_dict())
    current_optimizer_state = copy.deepcopy(optimizer.state_dict())
    
    optimizer.zero_grad()
    output_ED = model.forward(tensor_input_ED[...,0:1])
    
    # Calculate loss using your custom loss function
    loss_PDE, volume = fc.cardioloss_FE_ED(
        tensor_A_ED, tensor_input_ED, output_ED, tensor_Coord, input_parameter_tensor, ED_pressure,
        tensor_Cfsn, tensor_max_amplitude, tensor_unloaded_chamber_volume, C_strain_normalization, n_modesU,device=device, Print=print_FLAG
    )

    # Calculate the total loss
    loss_total = loss_PDE

    # Backward pass
    loss_PDE.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    
    # Optimizer step
    optimizer.step()
    scheduler.step()
    
    # Append loss to the loss vector
    loss_vector = np.append(loss_vector, loss_total.item())

    # Check convergence: if loss change is smaller than threshold
    loss_change = abs(previous_loss - loss_total.item())
    if loss_change < convergence_threshold:
        no_improvement_epochs += 1
    else:
        no_improvement_epochs = 0  # Reset if improvement occurs
    
    if no_improvement_epochs >= patience:
        print(f"Loss converged at epoch {epoch}. Stopping early.")
        break  # Early stopping condition
    
    # Update previous loss for the next iteration
    previous_loss = loss_total.item()

    # Check for NaN or infinite loss
    loss_is_finite = torch.isfinite(loss_total).all().item()
    params_are_finite = check_model_parameters(model)
    
    if not loss_is_finite or not params_are_finite:
        print(f"NaN detected at epoch {epoch}. Reverting to previous model state and halving the learning rate.")
        model.load_state_dict(previous_state)
        optimizer.load_state_dict(previous_optimizer_state)
        for param_group in optimizer.param_groups:
            new_lr = max(param_group['lr'] * 0.5, min_lr)
            param_group['lr'] = new_lr
            print(f"Learning rate for param group set to {new_lr}")
        continue
        
    # Update the previous state
    previous_state = copy.deepcopy(current_state)
    previous_optimizer_state = copy.deepcopy(current_optimizer_state)

    # Print logs every 10 epochs
    if epoch % 100 == 0:
        print_FLAG = False
        tqdm.write(f"Epoch {epoch}: Loss={loss_total:.4f}")

# Save the trained model
torch.save(model.state_dict(), os.path.join(out_folder, "Trained_model.pth"))

end_time = time.time()  # End time
print(f"C_strain Training time: {end_time - start_time:.2f} seconds")
print("C_strain: ", output_ED[...,0].item() * C_strain_normalization)

C_strain = output_ED[...,0].item() * C_strain_normalization
#_____________________________________________________________________________
# Tmax estimator
# Define the neural network
# Define training parameters
epochs = 1000  # Maximum training epochs
learn_rate = 0.001  # Initial learning rate
convergence_threshold = 1e-3  # Threshold for loss change to detect convergence
patience = 50  # Number of consecutive epochs to monitor for convergence

print('Training the Tmax network...')
start_time = time.time()  # Start time

# Define the neural network model
# input could be none, cuz it's a optimisation problem, but for simplicity, we use the input V,t
model2 = deepxde.fnn.FNN(
    [2, 32, 32, 32, 1],  # Layer sizes: input -> hidden layers -> output
    'relu',              # Activation function
    "Glorot normal",     # Weight initializer
).to(device)

# model.load_state_dict(torch.load(os.path.join(out_folder, "Trained_model.pth")))
# model.eval()

# Define the optimizer
optimizer2 = torch.optim.AdamW(
    model2.parameters(),
    lr=learn_rate,
    betas=(0.9, 0.98),
    weight_decay=1e-4,
    eps=1e-6
)

# Define the learning rate scheduler
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=200, gamma=0.5)

loss_vector = np.array([])  # Initialize the loss vector

# Initialize variables to save the previous state
previous_state = copy.deepcopy(model2.state_dict())
previous_optimizer_state = copy.deepcopy(optimizer2.state_dict())

min_lr = 1e-5  # Minimum learning rate to prevent it from getting too small
print_FLAG = False # Print flag

# Variables for convergence checking
previous_loss = float('inf')  # Initialize previous loss as infinity
no_improvement_epochs = 0  # Count epochs with small loss improvement

for epoch in tqdm(range(epochs), desc="Training Progress"):
    # Save the current state before training
    current_state = copy.deepcopy(model2.state_dict())
    current_optimizer_state = copy.deepcopy(optimizer2.state_dict())
    
    optimizer2.zero_grad()
    output_PS = model2.forward(tensor_input_PS)

    # Check if output is negative
    if (output_PS < 0).any():
        print(f"Epoch {epoch}: Detected output < 0, resetting to previous state and retrying...")
        model2 = deepxde.fnn.FNN(
            [2, 16, 16, 16, 1],  # Layer sizes: input -> hidden layers -> output
            'relu',              # Activation function
            "Glorot normal",     # Weight initializer
        ).to(device)

        # Define the optimizer
        optimizer2 = torch.optim.AdamW(
            model2.parameters(),
            lr=learn_rate,
            betas=(0.9, 0.98),
            weight_decay=1e-4,
            eps=1e-6
        ) 
        continue 

    # Calculate loss using your custom loss function
    loss_PDE, volume = fc.cardioloss_FE_PS(tensor_A_PS, tensor_input_PS, output_PS, tensor_Coord, input_parameter_tensor, stress_para, C_strain, PS_pressure,
        tensor_Cfsn, tensor_max_amplitude, tensor_unloaded_chamber_volume, Tmax_normalization, t_normalization, n_modesU, device = device, Print=print_FLAG)

    # Calculate the total loss
    loss_total = loss_PDE

    # Backward pass
    loss_total.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=0.1)

    # Optimizer step
    optimizer2.step()
    scheduler2.step()
    
    # Append loss to the loss vector
    loss_vector = np.append(loss_vector, loss_total.item())

    # Check convergence: if loss change is smaller than threshold
    loss_change = abs(previous_loss - loss_total.item())
    if loss_change < convergence_threshold:
        no_improvement_epochs += 1
    else:
        no_improvement_epochs = 0  # Reset if improvement occurs
    
    if no_improvement_epochs >= patience:
        print(f"Loss converged at epoch {epoch}. Stopping early.")
        break  # Early stopping condition
    
    # Update previous loss for the next iteration
    previous_loss = loss_total.item()

    # Check for NaN or infinite loss
    loss_is_finite = torch.isfinite(loss_total).all().item()
    params_are_finite = check_model_parameters(model2)
    
    if not loss_is_finite or not params_are_finite:
        print(f"NaN detected at epoch {epoch}. Reverting to previous model state and halving the learning rate.")
        model2.load_state_dict(previous_state)
        optimizer2.load_state_dict(previous_optimizer_state)
        for param_group in optimizer2.param_groups:
            new_lr = max(param_group['lr'] * 0.5, min_lr)
            param_group['lr'] = new_lr
            print(f"Learning rate for param group set to {new_lr}")
        continue
        
    # Update the previous state
    previous_state = copy.deepcopy(current_state)
    previous_optimizer_state = copy.deepcopy(current_optimizer_state)

    # Print logs every 10 epochs
    if epoch % 100 == 0:
        print_FLAG = False
        tqdm.write(f"Epoch {epoch}: Loss={loss_total:.4f}")

# Save the trained model
torch.save(model2.state_dict(), os.path.join(out_folder, "Trained_model2.pth"))

end_time = time.time()  # End time
print(f"Tmax Training time: {end_time - start_time:.2f} seconds")
print("Tmax: ",output_PS[...,0].item()*Tmax_normalization)
Tmax = output_PS[...,0].item() * Tmax_normalization

#_____________________________________________________________________________
# Save the results
with open(os.path.join(out_folder, 'C_strain_Tmax.txt'), 'w') as file:
    file.write(f'C_strain: {C_strain}\n')
    file.write(f'Tmax: {Tmax}\n')



    
















       



