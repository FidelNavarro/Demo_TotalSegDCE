import os
import numpy as np
from PIL import Image
import pydicom
from collections import defaultdict

import nibabel as nib
import scipy.ndimage as ndi
from tqdm import tqdm

import subprocess
from totalsegmentator.python_api import totalsegmentator
#---------------------------------------------------------
#Load 4d DICOM reconstructions 
#Notes:
    #Runs slow, too much data
    #Make sure all files are within the downloaded Drive folder
def load_dicom_4d(dicom_folder_path, voxel_dims=True, verbose=True): #Set voxel_dims to 'True' if voxel dimensions are required
    acquisition_groups = defaultdict(list)
    # Loop through each file in the DICOM folder
    for root, dirs, file_name in os.walk(dicom_folder_path):
        for file in tqdm(file_name, desc="Processing Files"):#file_name:
            file_path = os.path.join(root, file)
            if file_path.endswith('.dcm'):  # Ensure it's a DICOM file
                
                # Read DICOM file
                dicom_data = pydicom.dcmread(file_path)
                # Extract Acquisition Number 
                acquisition_number = dicom_data.AcquisitionNumber if hasattr(dicom_data, 'AcquisitionNumber') else None
                # Get pixel data (the 2D slice)
                pixel_array = dicom_data.pixel_array
                # Group slices by Acquisition Number
                acquisition_groups[acquisition_number].append(pixel_array)
            #print(int(acquisition_number))
    # Sort the acquisition numbers
    sorted_acquisition_numbers = sorted(acquisition_groups.keys())
    # Create a list of 3D volumes (each volume corresponds to an AcquisitionNumber)
    dicom_volumes = []
    for acquisition_number in sorted_acquisition_numbers:
        # Stack slices of same acquisition into 3D array (depth, height, width)
        volume = np.stack(acquisition_groups[int(acquisition_number)], axis=0)
        #print(acquisition_number, volume.shape)
        dicom_volumes.append(volume)
    # Stack 3D volumes into 4D array (AcquisitionNumber, depth, height, width)
    dicom_4d_array = np.stack(dicom_volumes, axis=0)

    voxel_dimensions = (*dicom_data.PixelSpacing, dicom_data.SliceThickness)
    if verbose:
        print(f"Shape of 4D DICOM array: {dicom_4d_array.shape}")
        print(f"With voxel dimension: {voxel_dimensions[0]} x {voxel_dimensions[1]} x {voxel_dimensions[2]} mm^3")

    if voxel_dims:
        return dicom_4d_array, voxel_dimensions
    else:
        return dicom_4d_array
#---------------------------------------------------------
#Check if folder exists 
def working_folder(path, folder_name):
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    return folder_path
#---------------------------------------------------------
#Run liver segmentations using TotalSegmentator
def run_totalsegmentator(dicom_4d_array, dims, output_folder, format='4D',
                         ha_vol=0, task="total_mr", return_mask=True,
                         quiet=True, verbose=True):
    #Get temporary nifti files of desired volumes:
    labels = 'labels'
    img = 'img'
    
    labels_folder = working_folder(output_folder, labels)
    img_folder = working_folder(output_folder, img)

    if format=='4D':
        nifti_path = save_volumes_as_nifti(dicom_4d_array, dims, img_folder, vol_format='3D_single', vol_3d=ha_vol,
                                           verbose=False, return_nifti=True)
    elif format=='3D':
        if len(dicom_4d_array.shape)==3:
            x, y, z = dicom_4d_array.shape
            dicom_4d_array = np.reshape(dicom_4d_array, (1, x, y, z ))
        
        nifti_path = save_volumes_as_nifti(dicom_4d_array, dims, img_folder, vol_format='3D_single', vol_3d=0,
                                           verbose=False, return_nifti=True)
    else:
        print('Format not supported!')
        
    if verbose:
        print("\n...<<Running TotalSegmentator>>...")
    
    totalsegmentator(nifti_path, labels_folder, task=task, roi_subset=['liver', 'aorta'], quiet=quiet, device='cpu')
    aorta_data = nib.load(os.path.join(labels_folder, 'aorta.nii.gz'))
    volume_aorta = aorta_data.get_fdata().transpose(1,0,2)
    volume_aorta = ndi.rotate(volume_aorta, 180, axes=(0,1), reshape=False)
    volume_aorta = np.transpose(volume_aorta, (2, 0, 1))

    if verbose:
        print("\n...<<Masks created>>...")

    volume_aorta[volume_aorta<1]=0
    volume_aorta[volume_aorta>1]=1

    if return_mask:
        return volume_aorta
#---------------------------------------------------------
def save_volumes_as_nifti(volumes, voxel_dim, output_folder, output_file='volume', flip_axes=[2,1,0], rotate=180, vol_format='3D', vol_3d=0,
                         verbose=True, return_nifti=False): 
    #flip_axes: is used to order the axes (Heparim data's order is 2,1,0)
    #rotate: use to rotate image
    #vol_format: '3D' or '4D', '3D' saves individual nifti files of each volume, '4D' saves one nifit file of all volumes
    #vol_3d = volume to save in vol_format='3D_single'
    os.makedirs(output_folder, exist_ok=True)
    #for time, Z, X, Y in volumes.shape: #(215, 72, 224, 224)
    affine = np.diag([voxel_dim[0], voxel_dim[1], voxel_dim[2], 1])
        # Stack slices into a 3D volume
    
    if vol_format == '3D':
        time, Z, X, Y = volumes.shape
        for i in tqdm(range(time), desc="...Processing Volumes..."):
            volume_data = volumes[i].transpose(flip_axes[0],flip_axes[1],flip_axes[2])
            volume_data = ndi.rotate(volume_data, rotate, axes=(0,1), reshape=False)
        
            nifti_img = nib.Nifti1Image(volume_data, affine)
            numb = int(i)
            output_path = os.path.join(output_folder, f"{output_file}_{numb}.nii.gz")
            nib.save(nifti_img, output_path)
        print(f"Saved {numb+1} NIFTI files to {output_folder}")
    elif vol_format == '3D_single':
        if vol_3d == None:
            #Z, X, Y = volumes.shape
            #print(f'Mask shape: {volumes.shape}')
            volume_data = volumes.transpose(flip_axes[0],flip_axes[1],flip_axes[2])
            volume_data = ndi.rotate(volume_data, rotate, axes=(0,1), reshape=False)
            
            nifti_img = nib.Nifti1Image(volume_data, affine=affine)
            #numb = int(vol_3d)
            output_path = os.path.join(output_folder, f"{output_file}.nii.gz")
            nib.save(nifti_img, output_path)
        else:
            #time, Z, X, Y = volumes.shape
            volume_data = volumes[vol_3d].transpose(flip_axes[0],flip_axes[1],flip_axes[2])
            volume_data = ndi.rotate(volume_data, rotate, axes=(0,1), reshape=False)
            
            nifti_img = nib.Nifti1Image(volume_data, affine=affine)
            numb = int(vol_3d)
            output_path = os.path.join(output_folder, f"{output_file}_slice{numb}.nii.gz")
            nib.save(nifti_img, output_path)
    else:
            print(f"{vol_format} format not supported.")
        
    #if verbose:    
            #print(f"...Saved 3D NIfTI file to {output_file}...")
    

    if return_nifti:
        return output_path