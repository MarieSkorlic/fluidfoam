import numpy as np
import pandas as pd
#import fluidfoam
from fluidfoam import OpenFoamSimu

"""
This file is used to calculate profiles 
averaged over horizontal 'z' slices. 
"""

def get_mean_profiles(list_var, z, V, Cx, Cy):
    """
    Compute volume-weighted mean profiles along the z-axis.

    Inputs:
    - list_var: list of arrays to compute the mean (e.g., velocity components)
    - z: array of z-coordinates of each cell with readmesh of fluidfoam
    - V: array of cell volumes with writeCellVolumes of openfoam
    - Cx, Cy: coordinates with writeCellCentres of openfoam 

    Outputs:
    - unique_z: unique z values defining slices
    - mean_values: list of mean values for each variable per slice
    """
    df = pd.DataFrame({'z': np.round(z, 6), 'V': V})
    for i, var in enumerate(list_var):
        df[f'var_{i}'] = var * V

    df_sum = df.groupby('z', sort=True).sum().reset_index()
    unique_z = df_sum['z'].values
    V_tot_per_z = df_sum['V'].values  
    
    mean_values = []
    for i in range(len(list_var)) : 
        mean_values.append(df_sum[f'var_{i}'].values / V_tot_per_z )
    
    return unique_z, mean_values

def get_disp_stress(z ,Ui,Uj,V, rho = None) : 
    """
    Compute volume-weighted mean profiles of dispersive stress.

    Inputs:
    - z: array of z-coordinates of each cell with readmesh of fluidfoam
    - Ui,Uj : velocity used in the calculation, should be averaged in time 
    - V: array of cell volumes with writeCellVolumes of openfoam

    Outputs:
    - unique_z: unique z values defining slices
    - dispersive stress in [Pa]
    """

    if rho is None : 
        rho = 1000 #[kg/m3]
    #DataFrame Creation 
    df = pd.DataFrame({'z': np.round(z, 6),'Ui' : Ui*V , 'Uj' : Uj*V , 'UiUj' : Ui*Uj*V, 'V': V})
    
    df_sum = df.groupby('z', sort=True).sum().reset_index()
    unique_z = df_sum['z'].values
    V_tot_per_z = df_sum['V'].values

    UiUj_mean = df_sum['UiUj'] / V_tot_per_z
    Ui_mean = df_sum['Ui'] / V_tot_per_z
    Uj_mean = df_sum['Uj'] / V_tot_per_z

    disp_stress = rho * np.array((UiUj_mean - Ui_mean*Uj_mean))
    return unique_z , np.array(disp_stress)
    
def get_viscous_stress(z ,Ui,V, rho = None , nu = None) : 
    """
    Compute volume-weighted mean profiles of viscous stress.

    Inputs:
    - z: array of z-coordinates of each cell with readmesh of fluidfoam
    - Ui : velocity used in the calculation, should be averaged in time 
    - V: array of cell volumes with writeCellVolumes of openfoam

    Outputs:
    - unique_z: unique z values defining slices
    - viscous stress in [Pa]
    """

    if (rho is None) & (nu is None) : 
        rho = 1000 #[kg/m3]
        nu = 1e-6 #[m2/s]

    #DataFrame Creation 
    df = pd.DataFrame({'z': np.round(z, 6),'Ui' : Ui*V, 'V': V})
    
    df_sum = df.groupby('z', sort=True).sum().reset_index()
    unique_z = df_sum['z'].values
    V_tot_per_z = df_sum['V'].values

    Ui_mean = df_sum['Ui'] / V_tot_per_z
    grad_Ui = np.gradient(Ui_mean , unique_z)

    viscous_stress = rho * nu * grad_Ui
    return unique_z , viscous_stress

def get_reynolds_stress(z ,Tau,V, rho = None ) : 
    """
    Compute volume-weighted mean profiles of Reynolds stress.
    Reynolds Stress from RANS profiles computed with turbulent viscosity

    Inputs:
    - z: array of z-coordinates of each cell with readmesh of fluidfoam
    - Tau : Reynolds Stress given by openfoam : should be averaged in time 
    - V: array of cell volumes with writeCellVolumes of openfoam

    Outputs:
    - unique_z: unique z values defining slices
    - Reynolds stress in [Pa]
    """

    if (rho is None) : 
        rho = 1000 #[kg/m3]

    #DataFrame Creation 
    df = pd.DataFrame({'z': np.round(z, 6),'Tau' : Tau*V, 'V': V})
    
    df_sum = df.groupby('z', sort=True).sum().reset_index()
    unique_z = df_sum['z'].values
    V_tot_per_z = df_sum['V'].values

    Tau_mean = df_sum['Tau'] / V_tot_per_z

    reynolds_stress =   rho * np.array(Tau_mean)
    return unique_z , reynolds_stress

def get_dz_slice(z, V):
    """
    Compute the mean dz for each slice z.

    Inputs:
    - z: array of z-coordinates
    - V: array of cell volumes

    Outputs:
    - dz_slice: array of dz values per slice
    """

    df = pd.DataFrame({'z': np.round(z, 6), 'V': V})
    df_sum = df.groupby('z', sort=True).sum().reset_index()
    
    unique_z = df_sum['z'].values
    V_tot_per_z = df_sum['V'].values  

    # Compute the bottom cell layer thickness
    z_min = np.min(np.unique(z))
    cells_bottom = np.where(z == z_min)
    V_bottom = V[cells_bottom]
    dz_bottom = 2 * z_min 
    dS_bottom = V_bottom / dz_bottom
    S0 = np.sum(dS_bottom)

    # Compute dz per slice
    dz_slice = np.divide(V_tot_per_z,S0)
    return dz_slice

def get_u_star(z,Ui,V,rho = None , nu = None) : 
    """
    Compute u* from bed Shrear Stress based on viscous stress only

    Inputs:
    - z: array of z-coordinates
    - Ui : vertical veclocity : should be averaged in time
    - V: array of cell volumes

    Outputs:
    - u_star : friction velocity [m/s]
    """
    if (rho is None) & (nu is None) : 
        rho = 1000 #[kg/m3]
        nu = 1e-6 #[m2/s]

    #Get stress at the bottom taking only viscous stress contribution
    Taub = get_viscous_stress(z ,Ui,V, rho , nu)[1][0]

    u_star = np.sqrt(Taub / rho ) 
    return u_star


