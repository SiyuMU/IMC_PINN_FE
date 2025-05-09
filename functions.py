import torch
import torch.nn as nn

def cus_silu(x):
    """Custom activation function: x * sigmoid(30 * x)"""
    return x * torch.sigmoid(30 * x)

def GradientOperator_AvgBased(Coords, Els, tolerance=1e-10):
    """
    Computes the gradient operator for each element in a mesh, handling degenerate elements with small variance.
    Args:
        - Coords: (N, 3) array of node coordinates.
        - Els: (M, 4) array of tetrahedral elements.
        - tolerance: Small tolerance to handle near-singular matrices.
    Returns:
        - dFdx, dFdy, dFdz: Gradient operators for x, y, z directions.

    # Claudio Mancinellia , Marco Livesub  and Enrico Puppoa (2019),  
    #     A Comparison of Methods for Gradient Field Estimation on Simplicial Meshes
    #     in Computers & Graphics (80), 37-50, doi.org/10.1016/j.cag.2019.03.005 
    """
    import numpy as np


    n_el = Els.shape[0]
    n_points = Coords.shape[0]

    dFcdx = np.zeros((n_el, n_points))
    dFcdy = np.zeros((n_el, n_points))
    dFcdz = np.zeros((n_el, n_points))

    Vol_el       = np.zeros((n_el,1))
    Nodal_volume = np.zeros((n_points,1))

    dFdx = np.zeros((n_points,n_points))
    dFdy = np.zeros((n_points,n_points))
    dFdz = np.zeros((n_points,n_points))

    for sel_el in range(n_el):
        AA = np.zeros((3, 3))
        AA[0, :] = Coords[Els[sel_el, 1]] - Coords[Els[sel_el, 0]]
        AA[1, :] = Coords[Els[sel_el, 2]] - Coords[Els[sel_el, 0]]
        AA[2, :] = Coords[Els[sel_el, 3]] - Coords[Els[sel_el, 0]]
        
        det_A = np.linalg.det(AA)

        if abs(det_A) < tolerance:
            # Handle degenerate case: allow small variance
            perturbation = np.random.normal(scale=1e-6, size=AA.shape)
            AA += perturbation
            det_A = np.linalg.det(AA)
            if abs(det_A) < tolerance:
                print(f"Skipping degenerate element {sel_el} after adjustment.")
                continue  # Skip if still degenerate

        invA = np.linalg.inv(AA)

        dFcdx[sel_el, Els[sel_el, 0]] = -np.sum(invA[0, :])
        dFcdx[sel_el, Els[sel_el, 1]] = invA[0, 0]
        dFcdx[sel_el, Els[sel_el, 2]] = invA[0, 1]
        dFcdx[sel_el, Els[sel_el, 3]] = invA[0, 2]

        dFcdy[sel_el, Els[sel_el, 0]] = -np.sum(invA[1, :])
        dFcdy[sel_el, Els[sel_el, 1]] = invA[1, 0]
        dFcdy[sel_el, Els[sel_el, 2]] = invA[1, 1]
        dFcdy[sel_el, Els[sel_el, 3]] = invA[1, 2]

        dFcdz[sel_el, Els[sel_el, 0]] = -np.sum(invA[2, :])
        dFcdz[sel_el, Els[sel_el, 1]] = invA[2, 0]
        dFcdz[sel_el, Els[sel_el, 2]] = invA[2, 1]
        dFcdz[sel_el, Els[sel_el, 3]] = invA[2, 2]

    # dFcdx, dFcdy,dFcdz are correct, validated with FEniCs and relative errors are around 10-5
    # only at apex they are around 1% (mesh distortion)

    # Volume weighted projection
    for i in range(n_el):
        Vol_el[i] = 1.0/6.0*abs((Coords[Els[i,3],:]-Coords[Els[i,0],:]).dot( np.cross(Coords[Els[i,2],:]-Coords[Els[i,0],:],Coords[Els[i,1],:]-Coords[Els[i,0],:])))
        for n_sel in Els[i]:
            Nodal_volume[n_sel] += Vol_el[i]/4.0

    for i in range(n_points):
        Els_per_node = np.where(Els == i)[0]

        for sel_el in Els_per_node: 
            dFdx[i,:] += dFcdx[sel_el,:].copy()*Vol_el[sel_el]/4.0/Nodal_volume[i]
            dFdy[i,:] += dFcdy[sel_el,:].copy()*Vol_el[sel_el]/4.0/Nodal_volume[i]
            dFdz[i,:] += dFcdz[sel_el,:].copy()*Vol_el[sel_el]/4.0/Nodal_volume[i]

    return dFdx, dFdy, dFdz

def compute_chamber_volume(Nodal_area, Coords):
    """
    Compute the chamber volume based on nodal areas and coordinates.
    """
    import numpy as np
    chamber_volume = np.sum(np.einsum('ij,ij->i', Nodal_area, Coords)) / 3.0
    return chamber_volume

def compute_chamber_volume_torch_with_area(Nodal_area: torch.Tensor, Coords: torch.Tensor) -> torch.Tensor:
    """
    Compute the chamber volume based on nodal areas and coordinates using PyTorch.

    Args:
        Nodal_area (torch.Tensor): Nodal areas, shape (batch, num_nodes, 3).
        Coords (torch.Tensor): Coordinates of nodes, shape (batch, num_nodes, 3).

    Returns:
        torch.Tensor: Chamber volume, shape (batch,).
    """
    # Compute the volume contribution from nodal areas and coordinates
    chamber_volume = torch.sum(torch.einsum('bni,bni->bn', Nodal_area, Coords), dim=-1) / 3.0
    return chamber_volume

def cardioloss_FE_ED(amplitude,input, output, Coord, input_parameter, ED_pressure,
        Cfsn, max_amplitude, unloaded_chamber_volume, C_strain_normalization, n_modes_U, device = 'cpu', Print=False):
    '''
    input: (N, 1) - time
    output: (N, n_modes_U+1) - C_strain and amplitudes
    Coord: (Coord, 3) - coordinates of the nodes
    input_parameter: (N, 10+n_modes_U*3) - nodal forces, nodal displacements, nodal areas, nodal volume, modes
    ED_pressure: (N, 1) - end-diastolic pressure
    Cfsn: (3,3) - Cfsn matrix
    max_amplitude: float - maximum amplitude
    t_normalization: float - time normalization factor
    unloaded_chamber_volume: float - unloaded chamber volume
    device: str - device to run the computation
    Print: bool - whether to print the results
    '''

    # Translate parameters
    f = input_parameter[...,0:3]
    s = input_parameter[...,3:6]
    n = input_parameter[...,6:9]
    fsnT = torch.stack([f, s, n], dim=1)

    Cfsn = Cfsn.reshape(3,3)
    nodal_volume = input_parameter[...,9:10]

    Phi_x = input_parameter[...,10:10+n_modes_U]
    Phi_y = input_parameter[...,10+n_modes_U:10+n_modes_U*2]
    Phi_z = input_parameter[...,10+n_modes_U*2:10+n_modes_U*3]
    dFudx = input_parameter[...,10+n_modes_U*3:10+n_modes_U*4]
    dFudy = input_parameter[...,10+n_modes_U*4:10+n_modes_U*5]
    dFudz = input_parameter[...,10+n_modes_U*5:10+n_modes_U*6]
    dFvdx = input_parameter[...,10+n_modes_U*6:10+n_modes_U*7]
    dFvdy = input_parameter[...,10+n_modes_U*7:10+n_modes_U*8]
    dFvdz = input_parameter[...,10+n_modes_U*8:10+n_modes_U*9]
    dFwdx = input_parameter[...,10+n_modes_U*9:10+n_modes_U*10]
    dFwdy = input_parameter[...,10+n_modes_U*10:10+n_modes_U*11]
    dFwdz = input_parameter[...,10+n_modes_U*11:10+n_modes_U*12]
    nodal_area = input_parameter[...,10+n_modes_U*12:10+n_modes_U*12+3]
    nodal_area = nodal_area.unsqueeze(0).repeat(input.shape[0],1,1)

    P = ED_pressure
    C_strain = output[...,0:1].reshape(-1)*C_strain_normalization # (N, 1)

    # Compute the amplitude
    a = max_amplitude*amplitude # (N, n_modes_U)

    # Nodal displacement
    ux = torch.matmul(a , Phi_x.T) # (N, Coord)
    uy = torch.matmul(a , Phi_y.T)
    uz = torch.matmul(a , Phi_z.T)
    
    New_Coord = Coord.repeat(ux.shape[0],1,1) + torch.stack([ux,uy,uz],dim=-1) # (N, Coord, 3)

    # Compute the deformation gradient tensor
    du_x = torch.matmul(a,dFudx.T) + 1 # (N, Coord)
    du_y = torch.matmul(a,dFudy.T)
    du_z = torch.matmul(a,dFudz.T)

    dv_x = torch.matmul(a,dFvdx.T)
    dv_y = torch.matmul(a,dFvdy.T) + 1 
    dv_z = torch.matmul(a,dFvdz.T)

    dw_x = torch.matmul(a,dFwdx.T)
    dw_y = torch.matmul(a,dFwdy.T)
    dw_z = torch.matmul(a,dFwdz.T) + 1
    F = torch.stack([
        torch.stack([du_x, du_y, du_z], dim=-1),  
        torch.stack([dv_x, dv_y, dv_z], dim=-1),  
        torch.stack([dw_x, dw_y, dw_z], dim=-1)   
    ], dim=-2)  # (N, Coord, 3, 3)
    F = torch.clamp(F, min=-10, max=10)
    # Compute Green-Lagrange strain tensor
    FTF = torch.matmul(torch.transpose(F,-1,-2), F) # (N, Coord, 3, 3)
    C = torch.matmul(torch.matmul(fsnT,FTF),torch.transpose(fsnT,-1,-2))
    E = 0.5 * (C - torch.eye(3).to(device)) # (N, Coord, 3, 3)

    # passive strain energy (Fung model)
    Q = torch.sum(torch.sum(Cfsn.repeat(E.shape[0],E.shape[1],1,1)*E**2,dim=-1,keepdim=False),dim=-1,keepdim=True) # (N, Coord, 1)
    Q = torch.clamp(Q,max=3)

    passive_energy_density =  C_strain / 133.32 * (torch.exp(Q) - 1) # (N, Coord, 1)
    passive_energy_density = passive_energy_density.squeeze(-1) # (N, Coord)
    passive_energy = torch.sum(passive_energy_density * nodal_volume.squeeze(-1)).reshape(-1)# (N, Coord)

    # pressure work done 
    # inverse, get new deformed nodal area 
    invF_00 =   dv_y * dw_z - dw_y * dv_z
    invF_10 = - dv_x * dw_z + dw_x * dv_z
    invF_20 =   dv_x * dw_y - dw_x * dv_y

    invF_01 = - du_y * dw_z + dw_y * du_z
    invF_11 =   du_x * dw_z - dw_x * du_z
    invF_21 = - du_x * dw_y + dw_x * du_y

    invF_02 =   du_y * dv_z - dv_y * du_z
    invF_12 = - du_x * dv_z + dv_x * du_z
    invF_22 =   du_x * dv_y - dv_x * du_y

    newNodal_areax = invF_00*nodal_area[...,0] + invF_10*nodal_area[...,1] + invF_20*nodal_area[...,2]
    newNodal_areay = invF_01*nodal_area[...,0] + invF_11*nodal_area[...,1] + invF_21*nodal_area[...,2]
    newNodal_areaz = invF_02*nodal_area[...,0] + invF_12*nodal_area[...,1] + invF_22*nodal_area[...,2]
    newNodal_area = torch.stack([newNodal_areax,newNodal_areay,newNodal_areaz],dim=-1)
    volumes = compute_chamber_volume_torch_with_area(newNodal_area, New_Coord)

    volume_change = volumes - unloaded_chamber_volume
    
    pressure_work_done = 1/3 * P * volume_change.reshape(-1,1) # 1/3 is a scale factor

    if Print:
        print('passive_energy:',passive_energy)
        print('pressure work done:',pressure_work_done)
        print('volume:',volumes)
        print('P',P)

    # Total energy
    Total_energy = torch.nn.functional.mse_loss(passive_energy-pressure_work_done.reshape(-1),torch.zeros_like(passive_energy))

    return Total_energy,volumes

def C_t(t, t0, tr):
    import math

    t0 = t0.expand_as(tr)  

    C_t_value = torch.where(
        t < t0,
        0.5 * (1 - torch.cos(math.pi * t / t0)),
        torch.where(
            t < t0 + tr,
            0.5 * (1 - torch.cos(math.pi * (t - t0 + tr) / tr)),
            torch.zeros_like(t)
        )
    )
    return C_t_value

def cardioloss_FE_PS(amplitude,input, output, Coord, input_parameter, stress_parameter, C_strain, PS_pressure,
        Cfsn, max_amplitude, unloaded_chamber_volume, Tmax_normalization, t_normalization, n_modes_U, device = 'cpu', Print=False):
    '''
    input: p,t # N,2
    output: amplitude of modes # N,n_modes_U
    input_parameter: f,s,n,nodal_volume,Phi_x,Phi_y,Phi_z,dFudx,dFudy,dFudz,dFvdx,dFvdy,dFvdz,dFwdx,dFwdy,dFwdz
    Cfsn: fung type material parameters
    max_amplitude: maximum amplitude of modes
    pressure_normalization: pressure normalization factor
    t_normalization: time normalization factor
    device: cpu or gpu
    '''
    import torch

    # Translate parameters
    Ca0 = stress_parameter[0]
    B = stress_parameter[1]
    l0 = stress_parameter[2]
    lr = stress_parameter[3]
    t0 = torch.tensor(stress_parameter[4],device=device)
    m = stress_parameter[5]
    b = stress_parameter[6]

    f = input_parameter[...,0:3]
    s = input_parameter[...,3:6]
    n = input_parameter[...,6:9]
    fsnT = torch.stack([f, s, n], dim=1)

    Cfsn = Cfsn.reshape(3,3)
    nodal_volume = input_parameter[...,9:10]

    Phi_x = input_parameter[...,10:10+n_modes_U]
    Phi_y = input_parameter[...,10+n_modes_U:10+n_modes_U*2]
    Phi_z = input_parameter[...,10+n_modes_U*2:10+n_modes_U*3]
    dFudx = input_parameter[...,10+n_modes_U*3:10+n_modes_U*4]
    dFudy = input_parameter[...,10+n_modes_U*4:10+n_modes_U*5]
    dFudz = input_parameter[...,10+n_modes_U*5:10+n_modes_U*6]
    dFvdx = input_parameter[...,10+n_modes_U*6:10+n_modes_U*7]
    dFvdy = input_parameter[...,10+n_modes_U*7:10+n_modes_U*8]
    dFvdz = input_parameter[...,10+n_modes_U*8:10+n_modes_U*9]
    dFwdx = input_parameter[...,10+n_modes_U*9:10+n_modes_U*10]
    dFwdy = input_parameter[...,10+n_modes_U*10:10+n_modes_U*11]
    dFwdz = input_parameter[...,10+n_modes_U*11:10+n_modes_U*12]
    nodal_area = input_parameter[...,10+n_modes_U*12:10+n_modes_U*12+3]
    nodal_area = nodal_area.unsqueeze(0).repeat(input.shape[0],1,1)

    # normalize the pressure,volume and time
    P = PS_pressure
    T_max = output[...,0:1].reshape(-1) * Tmax_normalization
    t = input[...,1:2] * t_normalization
    bias = 25
    # Compute the amplitude
    a = max_amplitude*amplitude # (N, n_modes_U)

    # Nodal displacement
    ux = torch.matmul(a , Phi_x.T) # (N, Coord)
    uy = torch.matmul(a , Phi_y.T)
    uz = torch.matmul(a , Phi_z.T)
    
    New_Coord = Coord.repeat(ux.shape[0],1,1) + torch.stack([ux,uy,uz],dim=-1) # (N, Coord, 3)

    # Compute the deformation gradient tensor
    du_x = torch.matmul(a,dFudx.T) + 1 # (N, Coord)
    du_y = torch.matmul(a,dFudy.T)
    du_z = torch.matmul(a,dFudz.T)

    dv_x = torch.matmul(a,dFvdx.T)
    dv_y = torch.matmul(a,dFvdy.T) + 1 
    dv_z = torch.matmul(a,dFvdz.T)

    dw_x = torch.matmul(a,dFwdx.T)
    dw_y = torch.matmul(a,dFwdy.T)
    dw_z = torch.matmul(a,dFwdz.T) + 1
    F = torch.stack([
        torch.stack([du_x, du_y, du_z], dim=-1),  
        torch.stack([dv_x, dv_y, dv_z], dim=-1),  
        torch.stack([dw_x, dw_y, dw_z], dim=-1)   
    ], dim=-2)  # (N, Coord, 3, 3)
    F = torch.clamp(F, min=-10, max=10)
    # Compute Green-Lagrange strain tensor
    FTF = torch.matmul(torch.transpose(F,-1,-2), F) # (N, Coord, 3, 3)
    C = torch.matmul(torch.matmul(fsnT,FTF),torch.transpose(fsnT,-1,-2))
    E = 0.5 * (C - torch.eye(3).to(device)) # (N, Coord, 3, 3)

    # passive strain energy (Fung model)
    Q = torch.sum(torch.sum(Cfsn.repeat(E.shape[0],E.shape[1],1,1)*E**2,dim=-1,keepdim=False),dim=-1,keepdim=True) # (N, Coord, 1)
    Q = torch.clamp(Q,max=3)

    passive_energy_density =   C_strain / 133.32 * (torch.exp(Q) - 1) # (N, Coord, 1)
    passive_energy_density = passive_energy_density.squeeze(-1) # (N, Coord)

    # Active strain energy
    val = torch.matmul(f.unsqueeze(-1).transpose(-1, -2), torch.matmul(FTF,f.unsqueeze(-1)))
    val = torch.clamp(val, min=1e-12)
    lamda = torch.sqrt(val)
    l=lamda*lr
    Eca50_d = torch.clamp(torch.exp(B*(l-l0))-1,min=1e-6)
    ECa50 = Ca0 / torch.sqrt(Eca50_d)
    tr = m*l+b
    Ct = C_t(t.repeat(1,tr.shape[1]).unsqueeze(-1).unsqueeze(-1),t0,tr)
    f = f.repeat(F.shape[0],1,1) # (N, Coord, 3)
    f_load = torch.matmul(F, f.unsqueeze(-1)).squeeze(-1)
    active_stress = (T_max / 133.32) * (Ca0**2 / (Ca0**2 + ECa50**2)) * Ct *torch.einsum('...i,...j->...ij', f, f_load)
    active_energy_density = 0.5 * torch.einsum('...ij,...ij->...', active_stress,E)
    active_energy_density = torch.clamp(active_energy_density,min=0)

    strain_energy = (passive_energy_density+active_energy_density)*nodal_volume.squeeze(-1) # (N, Coord)
    strain_energy = strain_energy.sum(dim=-1) # (N,)
    
    # pressure work done 
    # inverse, get new deformed nodal area 
    invF_00 =   dv_y * dw_z - dw_y * dv_z
    invF_10 = - dv_x * dw_z + dw_x * dv_z
    invF_20 =   dv_x * dw_y - dw_x * dv_y

    invF_01 = - du_y * dw_z + dw_y * du_z
    invF_11 =   du_x * dw_z - dw_x * du_z
    invF_21 = - du_x * dw_y + dw_x * du_y

    invF_02 =   du_y * dv_z - dv_y * du_z
    invF_12 = - du_x * dv_z + dv_x * du_z
    invF_22 =   du_x * dv_y - dv_x * du_y

    newNodal_areax = invF_00*nodal_area[...,0] + invF_10*nodal_area[...,1] + invF_20*nodal_area[...,2]
    newNodal_areay = invF_01*nodal_area[...,0] + invF_11*nodal_area[...,1] + invF_21*nodal_area[...,2]
    newNodal_areaz = invF_02*nodal_area[...,0] + invF_12*nodal_area[...,1] + invF_22*nodal_area[...,2]
    
    # volumes = compute_chamber_volume_torch(Face_endo, label, New_Coord)
    newNodal_area = torch.stack([newNodal_areax,newNodal_areay,newNodal_areaz],dim=-1)

    volumes = compute_chamber_volume_torch_with_area(newNodal_area, New_Coord)
    volume_change = volumes - unloaded_chamber_volume + bias # (N,) # bias to avoid negative volume

    pressure_work_done = 1/3 * P * volume_change.reshape(-1,1) # 1/3 is a scale factor

    if Print:
        print('passive_energy:',((passive_energy_density)*nodal_volume.squeeze(-1)).sum(dim=-1))
        print('strain_energy:',strain_energy)
        print('pressure work done:',pressure_work_done)
        print('T_max:',T_max)

    # Total energy
    Total_energy = torch.nn.functional.mse_loss(strain_energy-pressure_work_done.reshape(-1),torch.zeros_like(strain_energy))

    return Total_energy,volumes

def cardioloss_FE_v_diastolic(input,output,Coord,input_parameter,stress_parameter,C_strain,T_max,Cfsn,max_amplitude,t_normalization,unloaded_chamber_volume,device,Print=False):

    # Translate parameters
    Ca0 = stress_parameter[0]
    B = stress_parameter[1]
    l0 = stress_parameter[2]
    lr = stress_parameter[3]
    t0 = torch.tensor(stress_parameter[4],device=device)
    m = stress_parameter[5]
    b = stress_parameter[6]

    f = input_parameter[...,0:3]
    s = input_parameter[...,3:6]
    n = input_parameter[...,6:9]
    fsnT = torch.stack([f, s, n], dim=1)

    Cfsn = Cfsn.reshape(3,3)
    nodal_volume = input_parameter[...,9:10]
    n_modes_U = output.shape[1]-1

    Phi_x = input_parameter[...,10:10+n_modes_U]
    Phi_y = input_parameter[...,10+n_modes_U:10+n_modes_U*2]
    Phi_z = input_parameter[...,10+n_modes_U*2:10+n_modes_U*3]
    dFudx = input_parameter[...,10+n_modes_U*3:10+n_modes_U*4]
    dFudy = input_parameter[...,10+n_modes_U*4:10+n_modes_U*5]
    dFudz = input_parameter[...,10+n_modes_U*5:10+n_modes_U*6]
    dFvdx = input_parameter[...,10+n_modes_U*6:10+n_modes_U*7]
    dFvdy = input_parameter[...,10+n_modes_U*7:10+n_modes_U*8]
    dFvdz = input_parameter[...,10+n_modes_U*8:10+n_modes_U*9]
    dFwdx = input_parameter[...,10+n_modes_U*9:10+n_modes_U*10]
    dFwdy = input_parameter[...,10+n_modes_U*10:10+n_modes_U*11]
    dFwdz = input_parameter[...,10+n_modes_U*11:10+n_modes_U*12]
    nodal_area = input_parameter[...,10+n_modes_U*12:10+n_modes_U*12+3]
    nodal_area = nodal_area.unsqueeze(0).repeat(input.shape[0],1,1)

    # normalize the pressure,volume and time
    P = output[...,0:1] * 150
    # t = input[...,1:2] * t_normalization

    # Compute the amplitude
    a = max_amplitude*output[...,1:] # (N, n_modes_U)

    # Nodal displacement
    ux = torch.matmul(a , Phi_x.T) # (N, Coord)
    uy = torch.matmul(a , Phi_y.T)
    uz = torch.matmul(a , Phi_z.T)
    disp = torch.stack([ux,uy,uz],dim=-1) # (N, Coord, 3)
    
    New_Coord = Coord.repeat(ux.shape[0],1,1) + torch.stack([ux,uy,uz],dim=-1) # (N, Coord, 3)

    # Compute the deformation gradient tensor
    du_x = torch.matmul(a,dFudx.T) + 1 # (N, Coord)
    du_y = torch.matmul(a,dFudy.T)
    du_z = torch.matmul(a,dFudz.T)

    dv_x = torch.matmul(a,dFvdx.T)
    dv_y = torch.matmul(a,dFvdy.T) + 1 
    dv_z = torch.matmul(a,dFvdz.T)

    dw_x = torch.matmul(a,dFwdx.T)
    dw_y = torch.matmul(a,dFwdy.T)
    dw_z = torch.matmul(a,dFwdz.T) + 1
    F = torch.stack([
        torch.stack([du_x, du_y, du_z], dim=-1),  
        torch.stack([dv_x, dv_y, dv_z], dim=-1),  
        torch.stack([dw_x, dw_y, dw_z], dim=-1)   
    ], dim=-2)  # (N, Coord, 3, 3)
    F = torch.clamp(F, min=-10, max=10) # (N, Coord, 3, 3)

    # Compute Green-Lagrange strain tensor
    FTF = torch.matmul(torch.transpose(F,-1,-2), F) # (N, Coord, 3, 3)
    C = torch.matmul(torch.matmul(fsnT,FTF),torch.transpose(fsnT,-1,-2))
    E = 0.5 * (C - torch.eye(3).to(device)) # (N, Coord, 3, 3)

    # passive strain energy (Fung model)
    Q = torch.sum(torch.sum(Cfsn.repeat(E.shape[0],E.shape[1],1,1)*E**2,dim=-1,keepdim=False),dim=-1,keepdim=True) # (N, Coord, 1)
    Q = torch.clamp(Q,max=3)

    passive_energy_density =   C_strain / 133.32 * (torch.exp(Q) - 1) # (N, Coord, 1)
    passive_energy_density = passive_energy_density.squeeze(-1) # (N, Coord)
    
    # strain energy
    strain_energy = (passive_energy_density)*nodal_volume.squeeze(-1) # (N, Coord)
    strain_energy = strain_energy.sum(dim=-1) # (N,)

    # pressure work done 

    # inverse, get new deformed nodal area 
    invF_00 =   dv_y * dw_z - dw_y * dv_z
    invF_10 = - dv_x * dw_z + dw_x * dv_z
    invF_20 =   dv_x * dw_y - dw_x * dv_y

    invF_01 = - du_y * dw_z + dw_y * du_z
    invF_11 =   du_x * dw_z - dw_x * du_z
    invF_21 = - du_x * dw_y + dw_x * du_y

    invF_02 =   du_y * dv_z - dv_y * du_z
    invF_12 = - du_x * dv_z + dv_x * du_z
    invF_22 =   du_x * dv_y - dv_x * du_y

    newNodal_areax = invF_00*nodal_area[...,0] + invF_10*nodal_area[...,1] + invF_20*nodal_area[...,2]
    newNodal_areay = invF_01*nodal_area[...,0] + invF_11*nodal_area[...,1] + invF_21*nodal_area[...,2]
    newNodal_areaz = invF_02*nodal_area[...,0] + invF_12*nodal_area[...,1] + invF_22*nodal_area[...,2]
    
    newNodal_area = torch.stack([newNodal_areax,newNodal_areay,newNodal_areaz],dim=-1)

    volumes = compute_chamber_volume_torch_with_area(newNodal_area, New_Coord)
    volume_change = volumes - unloaded_chamber_volume  # (N,)
    pressure_work_done = 1/3 * P * volume_change.reshape(-1,1) # 1/3 is a scale factor

    if Print:
        print('volume:',volumes)
        print('P',P.reshape(-1))

    # Total energy
    Total_energy = torch.nn.functional.mse_loss(strain_energy-pressure_work_done.reshape(-1),torch.zeros_like(strain_energy))

    return Total_energy,volumes,P,disp

def cardioloss_FE_v_systolic(input,output,Coord,input_parameter,stress_parameter,C_strain,T_max,Cfsn,max_amplitude,t_normalization,unloaded_chamber_volume,device,Print=False):

    # Translate parameters
    Ca0 = stress_parameter[0]
    B = stress_parameter[1]
    l0 = stress_parameter[2]
    lr = stress_parameter[3]
    t0 = torch.tensor(stress_parameter[4],device=device)
    m = stress_parameter[5]
    b = stress_parameter[6]

    f = input_parameter[...,0:3]
    s = input_parameter[...,3:6]
    n = input_parameter[...,6:9]
    fsnT = torch.stack([f, s, n], dim=1)

    Cfsn = Cfsn.reshape(3,3)
    nodal_volume = input_parameter[...,9:10]
    n_modes_U = output.shape[1]-1
    bias = 35
    
    Phi_x = input_parameter[...,10:10+n_modes_U]
    Phi_y = input_parameter[...,10+n_modes_U:10+n_modes_U*2]
    Phi_z = input_parameter[...,10+n_modes_U*2:10+n_modes_U*3]
    dFudx = input_parameter[...,10+n_modes_U*3:10+n_modes_U*4]
    dFudy = input_parameter[...,10+n_modes_U*4:10+n_modes_U*5]
    dFudz = input_parameter[...,10+n_modes_U*5:10+n_modes_U*6]
    dFvdx = input_parameter[...,10+n_modes_U*6:10+n_modes_U*7]
    dFvdy = input_parameter[...,10+n_modes_U*7:10+n_modes_U*8]
    dFvdz = input_parameter[...,10+n_modes_U*8:10+n_modes_U*9]
    dFwdx = input_parameter[...,10+n_modes_U*9:10+n_modes_U*10]
    dFwdy = input_parameter[...,10+n_modes_U*10:10+n_modes_U*11]
    dFwdz = input_parameter[...,10+n_modes_U*11:10+n_modes_U*12]
    nodal_area = input_parameter[...,10+n_modes_U*12:10+n_modes_U*12+3]
    nodal_area = nodal_area.unsqueeze(0).repeat(input.shape[0],1,1)

    # normalize the pressure,volume and time
    P = output[...,0:1] * 150
    t = input[...,1:2] * t_normalization

    # Compute the amplitude
    a = max_amplitude*output[...,1:] # (N, n_modes_U)

    # Nodal displacement
    ux = torch.matmul(a , Phi_x.T) # (N, Coord)
    uy = torch.matmul(a , Phi_y.T)
    uz = torch.matmul(a , Phi_z.T)
    disp = torch.stack([ux,uy,uz],dim=-1) # (N, Coord, 3)
    
    New_Coord = Coord.repeat(ux.shape[0],1,1) + torch.stack([ux,uy,uz],dim=-1) # (N, Coord, 3)

    # Compute the deformation gradient tensor
    du_x = torch.matmul(a,dFudx.T) + 1 # (N, Coord)
    du_y = torch.matmul(a,dFudy.T)
    du_z = torch.matmul(a,dFudz.T)

    dv_x = torch.matmul(a,dFvdx.T)
    dv_y = torch.matmul(a,dFvdy.T) + 1 
    dv_z = torch.matmul(a,dFvdz.T)

    dw_x = torch.matmul(a,dFwdx.T)
    dw_y = torch.matmul(a,dFwdy.T)
    dw_z = torch.matmul(a,dFwdz.T) + 1
    F = torch.stack([
        torch.stack([du_x, du_y, du_z], dim=-1),  
        torch.stack([dv_x, dv_y, dv_z], dim=-1),  
        torch.stack([dw_x, dw_y, dw_z], dim=-1)   
    ], dim=-2)  # (N, Coord, 3, 3)
    F = torch.clamp(F, min=-10, max=10) # (N, Coord, 3, 3)

    # Compute Green-Lagrange strain tensor
    FTF = torch.matmul(torch.transpose(F,-1,-2), F) # (N, Coord, 3, 3)
    C = torch.matmul(torch.matmul(fsnT,FTF),torch.transpose(fsnT,-1,-2))
    E = 0.5 * (C - torch.eye(3).to(device)) # (N, Coord, 3, 3)

    # passive strain energy (Fung model)
    Q = torch.sum(torch.sum(Cfsn.repeat(E.shape[0],E.shape[1],1,1)*E**2,dim=-1,keepdim=False),dim=-1,keepdim=True) # (N, Coord, 1)
    Q = torch.clamp(Q,max=3)

    passive_energy_density =   C_strain / 133.32 * (torch.exp(Q) - 1) # (N, Coord, 1)
    passive_energy_density = passive_energy_density.squeeze(-1) # (N, Coord)
    
    # Active strain energy
    val = torch.matmul(f.unsqueeze(-1).transpose(-1, -2), torch.matmul(FTF,f.unsqueeze(-1)))
    val = torch.clamp(val, min=1e-12)
    lamda = torch.sqrt(val)

    l=lamda*lr
    Eca50_d = torch.clamp(torch.exp(B*(l-l0))-1,min=1e-6)
    ECa50 = Ca0 / torch.sqrt(Eca50_d)
    tr = m*l+b
    Ct = C_t(t.repeat(1,tr.shape[1]).unsqueeze(-1).unsqueeze(-1),t0,tr)
    f = f.repeat(F.shape[0],1,1) # (N, Coord, 3)
    f_load = torch.matmul(F, f.unsqueeze(-1)).squeeze(-1)
    active_stress = (T_max / 133.32) * (Ca0**2 / (Ca0**2 + ECa50**2)) * Ct *torch.einsum('...i,...j->...ij', f, f_load)
    active_energy_density = 0.5 * torch.einsum('...ij,...ij->...', active_stress,E)
    active_energy_density = torch.clamp(active_energy_density,min=0)

    # strain energy
    strain_energy = (passive_energy_density+active_energy_density)*nodal_volume.squeeze(-1) # (N, Coord)
    strain_energy = strain_energy.sum(dim=-1) # (N,)

    # pressure work done 

    # inverse, get new deformed nodal area 
    invF_00 =   dv_y * dw_z - dw_y * dv_z
    invF_10 = - dv_x * dw_z + dw_x * dv_z
    invF_20 =   dv_x * dw_y - dw_x * dv_y

    invF_01 = - du_y * dw_z + dw_y * du_z
    invF_11 =   du_x * dw_z - dw_x * du_z
    invF_21 = - du_x * dw_y + dw_x * du_y

    invF_02 =   du_y * dv_z - dv_y * du_z
    invF_12 = - du_x * dv_z + dv_x * du_z
    invF_22 =   du_x * dv_y - dv_x * du_y

    newNodal_areax = invF_00*nodal_area[...,0] + invF_10*nodal_area[...,1] + invF_20*nodal_area[...,2]
    newNodal_areay = invF_01*nodal_area[...,0] + invF_11*nodal_area[...,1] + invF_21*nodal_area[...,2]
    newNodal_areaz = invF_02*nodal_area[...,0] + invF_12*nodal_area[...,1] + invF_22*nodal_area[...,2]
    
    # volumes = compute_chamber_volume_torch(Face_endo, label, New_Coord)
    newNodal_area = torch.stack([newNodal_areax,newNodal_areay,newNodal_areaz],dim=-1)

    volumes = compute_chamber_volume_torch_with_area(newNodal_area, New_Coord)

    volume_change = volumes - unloaded_chamber_volume + bias # (N,) # bias to avoid negative volume
    pressure_work_done = 1/3 * P * volume_change.reshape(-1,1) # 1/3 is a scale factor

    if Print:
        print('volume:',volumes)
        print('P',P.reshape(-1))


    # Total energy
    Total_energy = torch.nn.functional.mse_loss(strain_energy-pressure_work_done.reshape(-1),torch.zeros_like(strain_energy))

    return Total_energy,volumes,P,disp


def save_vtk_with_updated_coords(geo_mesh, disp, output_folder):
    """
    Update mesh coordinates (geo_mesh.GetPoints() + disp) and save the entire mesh as VTK files.
    
    Parameters:
    - geo_mesh: A vtkPolyData or vtkUnstructuredGrid object representing the geometry.
    - disp: A numpy array of shape (n_iterations, num_points, 3) containing displacements.
    - output_folder: A string folder path for saving the output VTK files.
    """
    import vtk
    import numpy as np
    import os

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    num_iterations = disp.shape[0]
    num_points = geo_mesh.GetNumberOfPoints()

    # Ensure disp dimensions are compatible with geo_mesh
    if disp.shape[1] != num_points:
        raise ValueError("The number of points in geo_mesh and disp do not match.")

    # Extract original coordinates from geo_mesh
    original_coords = np.array([geo_mesh.GetPoint(i) for i in range(num_points)])

    for n in range(num_iterations):
        # Dynamically determine the type of geo_mesh
        if isinstance(geo_mesh, vtk.vtkPolyData):
            updated_mesh = vtk.vtkPolyData()
            writer = vtk.vtkXMLPolyDataWriter()
            file_extension = ".vtp"
        elif isinstance(geo_mesh, vtk.vtkUnstructuredGrid):
            updated_mesh = vtk.vtkUnstructuredGrid()
            writer = vtk.vtkXMLUnstructuredGridWriter()
            file_extension = ".vtu"
        else:
            raise TypeError("Unsupported mesh type. geo_mesh must be vtkPolyData or vtkUnstructuredGrid.")

        # Create a deep copy of the original mesh
        updated_mesh.DeepCopy(geo_mesh)

        # Update point coordinates by adding displacement
        points = vtk.vtkPoints()
        for i in range(num_points):
            updated_coord = original_coords[i] + disp[n, i]  # Add displacement to original coordinates
            if updated_coord[2]>0:
                updated_coord[2] = 0
            points.InsertNextPoint(updated_coord)

        # Set updated points back to the mesh
        updated_mesh.SetPoints(points)

        # Write the updated mesh to a VTK file
        output_filename = os.path.join(output_folder, f"t{n}{file_extension}")
        writer.SetFileName(output_filename)
        writer.SetInputData(updated_mesh)
        writer.Write()