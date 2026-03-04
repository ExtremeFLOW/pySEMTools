#!/usr/bin/env python

import os
import sys
from typing import cast
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    # Read environment variables for configuration
    num_elements_per_section = os.environ.get("num_elements_per_section")
    pattern_factor = os.environ.get("pattern_factor")
    num_pattern_cross_sections = os.environ.get("num_pattern_cross_sections")

    # Check if any variable is missing
    if None in [num_elements_per_section, pattern_factor, num_pattern_cross_sections]:
        print("Error: Missing required environment variables.", file=sys.stderr)
        sys.exit(1)
        
    # Convert environment variables to integers
    num_elements_per_section = int(cast(str, num_elements_per_section))
    pattern_factor = int(cast(str, pattern_factor))
    num_pattern_cross_sections = int(cast(str, num_pattern_cross_sections))
else:
    num_elements_per_section = None
    pattern_factor = None
    num_pattern_cross_sections = None

# Broadcast the configuration variables to all ranks
num_elements_per_section = comm.bcast(num_elements_per_section, root=0)
pattern_factor = comm.bcast(pattern_factor, root=0)
num_pattern_cross_sections = comm.bcast(num_pattern_cross_sections, root=0)

# Compute essential parameters for element distribution.
elem_pattern_crosssections = num_elements_per_section * pattern_factor; # No. of elements in each pattern cross-section
total_elements = elem_pattern_crosssections * num_pattern_cross_sections # Total elements in the file.
elements_per_rank = total_elements // size  # No. of elements in each rank.
ranks_per_section = num_elements_per_section // elements_per_rank # No. of ranks per each cross-section
ranks_per_pattern_section = ranks_per_section * pattern_factor # No. of ranks per each pattern cross-section
leftover = total_elements % size  # Elements that don’t fit evenly in the rank (if any)

import numpy as np
from pysemtools.io.reorder_data import (
    generate_data, calculate_face_differences, calculate_face_averages, calculate_face_normals,
    face_mappings, compare_mappings, compare_methods, reduce_face_normals, directions_face_normals,
    reorder_assigned_data, correct_axis_signs
)

def validate_rank_size(total_elements: int, size: int, num_elements_per_section: int) -> bool:
    """
        Validates whether the total number of elements and sections can be evenly distributed among ranks.
        
        Args:
            total_elements (int): The total number of elements in the dataset.
            size (int): The number of ranks available for computation.
            num_elements_per_section (int): The number of elements in each section.

        Returns:
            bool: True if the distribution is valid, False otherwise.
        """
    # Criterion 1: elements_per_rank should be exactly divisible
    elements_per_rank = total_elements // size
    if total_elements % size != 0:
        print(f"Error: Criteria 1 Failed,  {total_elements} % {size} != 0 (elements_per_rank = {elements_per_rank})") if rank == 0 else None
        return False
    
    # Criterion 2: ranks_per_section should be exactly divisible
    ranks_per_section = num_elements_per_section // elements_per_rank
    if num_elements_per_section % elements_per_rank != 0:
        print(f"Error: Criteria 2 Failed, {num_elements_per_section} % {elements_per_rank} != 0 (ranks_per_section = {ranks_per_section})") if rank == 0 else None
        return False
    
    print(f"Both Criterias Passed: elements_per_rank = {elements_per_rank}, ranks_per_section = {ranks_per_section}") if rank == 0 else None
    return True

def valid_sizes(total_elements: int, num_elements_per_section: int, min_size: int=100, max_size: int=2000):
    """
    Finds valid rank sizes for distributing elements evenly across computational processes.

    Args:
        total_elements (int): Total number of elements.
        num_elements_per_section (int): Number of elements per section.
        min_size (int, optional): Minimum size of ranks to consider. Defaults to 150.
        max_size (int, optional): Maximum size of ranks to consider. Defaults to 2000.

    Returns:
        list: A list of valid sizes for distributing elements.
    """
    valid_sizes = []

    for size in range(min_size, max_size + 1):
        if total_elements % size == 0:
            elements_per_rank = total_elements // size
            if num_elements_per_section % elements_per_rank == 0:
                valid_sizes.append(size)

    if not valid_sizes:
        print("No valid sizes found. Consider adjusting your min/max limits.")
    
    print("Valid size options:", valid_sizes)
        
    return valid_sizes

if rank == 0:
    check = validate_rank_size(total_elements, size, num_elements_per_section)
    print(f"Validation passed! Running main computation with size {size}...\n")
else:
    check = None 
    
check = comm.bcast(check, root=0)

def compute_element_centers(data):
    """
    Computes the center of each element by averaging its spatial coordinates.

    Args:
        data (numpy.ndarray): The input element data array.

    Returns:
        numpy.ndarray: Array containing computed element centers.
    """
    num_elements = data.shape[0]
    flattened = data.reshape(num_elements, -1, 3)  # (num_elements, 512, 3)
    element_centers = np.mean(flattened, axis=1)       # (num_elements, 3)
    return element_centers

def compute_cross_section_centers(data, num_elements_per_section, prev_overlap=None):    
    """
    Computes the center of each cross-section by averaging the spatial coordinates of elements in a section.
    Handles any overlap between sections for proper continuity across ranks.

    Args:
        data (numpy.ndarray): The input element data array.
        num_elements_per_section (int): Number of elements per section.
        prev_overlap (numpy.ndarray, optional): Previous overlapping elements for continuity.

    Returns:
        tuple: A list of computed cross-section centers and remaining overlap elements for the next rank.
    """
    cross_section_centers = []
    all_data = []
    # Combine previous leftover and new data
    if prev_overlap is not None and len(prev_overlap) > 0:
        prev_overlap = np.array(prev_overlap)
        if prev_overlap.ndim == 1:
            prev_overlap = prev_overlap.reshape(1, -1)
        all_data = np.vstack((prev_overlap, data))
    else:
        all_data = data
            
    num_total_elements = all_data.shape[0]
    full_sections = num_total_elements // num_elements_per_section
    leftover_elements = num_total_elements % num_elements_per_section

    for i in range(full_sections):
        start_idx = i * num_elements_per_section
        end_idx = start_idx + num_elements_per_section
        cross_section = all_data[start_idx:end_idx]
        center = np.mean(cross_section, axis=0)
        cross_section_centers.append(center)

    # Prepare leftover for sending to next rank
    if leftover_elements > 0:
        leftover = all_data[-leftover_elements:]
        # print(f"For Rank {rank}: np.array(leftover_overlap) shape in the function is : {np.array(leftover).shape}")       
    else:
        leftover = None

    return cross_section_centers, leftover

def assign_section(rank, ranks_per_section, final_cross_section_centers):
    """
    Assigns a section number to the given rank and selects the corresponding cross-section center.

    Args:
        rank (int): The MPI rank of the process.
        ranks_per_section (int): Number of ranks per section.
        final_cross_section_centers (numpy.ndarray): Computed cross-section centers.

    Returns:
        tuple: Selected cross-section center and corresponding section number.
    """
    section_number = rank // ranks_per_section  # Group ranks dynamically
    cross_section_center_choosen = final_cross_section_centers[section_number]
    return cross_section_center_choosen

def calculate_vectors(cross_section_center_choosen, element_centers_local, flow_directions):
    """
    Computes the radial, theta, and z unit vectors for each element relative to its cross-section center.

    Args:
        cross_section_center_chosen (numpy.ndarray): Selected cross-section center.
        element_centers_local (numpy.ndarray): Local element centers.
        flow_directions (int): Index specifying the flow direction.

    Returns:
        tuple: Arrays containing radial, theta, and z vectors for each element.
    """

    element_centers_local = np.array(element_centers_local)
    # # Initialize vectors
    radial_vectors_local = np.zeros_like(element_centers_local)  # Shape: (num_elements, 3)
    theta_vectors_local = np.zeros_like(element_centers_local)  # Shape: (num_elements, 3)
    z_vectors_local = np.zeros_like(element_centers_local)       # Shape: (num_elements, 3) 
        
    # Assign Z-vector (flow direction unit vector)
    z_vectors_local[:, flow_directions] = 1
            
    # Compute radial vector (difference between element and cross-section center)
    element_centers_local = np.array(element_centers_local)   

    radial_vectors_local = element_centers_local - cross_section_center_choosen
    radial_vectors_local /= np.linalg.norm(radial_vectors_local, axis=1, keepdims=True)
        
    # Compute theta vector using the cross product (z_vector x radial_vector)
    theta_vectors_local = np.cross(z_vectors_local, radial_vectors_local)
    theta_vectors_local /= np.linalg.norm(theta_vectors_local, axis=1, keepdims=True)
        
    return radial_vectors_local, theta_vectors_local, z_vectors_local

def vector_mapping(A):
    """
    Maps computed vectors to reference directions by finding the best fit per element.

    Args:
        A (numpy.ndarray): Transformation matrix containing dot product values between computed vectors.

    Returns:
        list: A list of tuples representing mapped vector assignments (radial, theta, z).
    """
    num_elements = A.shape[0]
    mappings = []
    for elem in range(num_elements):
        A_copy = np.abs(A[elem])
        mapping = [""] * 3
        reference_vectors = ['radial', 'theta', 'z']
        assigned_rows = set()
        assigned_cols = set()
        for _ in range(3):
            max_index = np.unravel_index(np.argmax(A_copy, axis=None), A_copy.shape)
            row, col = max_index
            mapping[row] = reference_vectors[col]
            assigned_rows.add(row)
            assigned_cols.add(col)
            A_copy[row, :] = -np.inf
            A_copy[:, col] = -np.inf

        if len(assigned_rows) < 3 or len(assigned_cols) < 3:
            print(f"Warning: Not all vectors were assigned properly at element {elem}.")
        mappings.append(tuple(mapping))
    return mappings

def compare_vector_mappings(mappings_cy, ref_vector_mapping, idx_ranges):
    """
    Compares computed vector mappings with a reference mapping.

    Args:
        mappings_cy (list): Computed vector mappings for each element.
        ref_vector_mapping (tuple): Reference vector mapping.
        idx_ranges (list): Index ranges assigned to the current rank.

    Returns:
        None (Prints matching results for the rank).
    """
    num_elements = len(mappings_cy)
    match_count_cy = 0
    for i in range(num_elements):
        if mappings_cy[i][2] == ref_vector_mapping[2]:
            match_count_cy += 1
    # print(f"Total matches: {match_count_cy}")
    matching_percentage = (match_count_cy / num_elements) * 100
    start_idx, end_idx = idx_ranges[rank]
    print(f"Rank {rank} has operated between {start_idx} and {end_idx} has matching Percentage of mappings_cy: {matching_percentage:.2f}%")

def mappings() -> None:
    """
    Performs data processing and mapping calculations.
    
    Notation: A group of cross-sections, when repeated, forms a pattern cross-section. The first pattern cross-section 
    is designated as the reference pattern cross-section, which serves as the basis for comparisons.

    Does the following:
    - Reads input files and computes face differences, averages, and normal vectors.
    - Applies reordering and axis corrections.
    - Computes final mappings in cartesian coordinates after corrections.
    - Computes element centres and each cross-section center.
    - Calculates radial, theta, and z vectors for each element relative to the cross-section center.
    - Computes vector mappings in cylindrical coordinates using transformation matrices.

    Returns:
        None
    """

    fname_3d: str = "../../field0.f00000"

    fname_out: str = "../../fieldout0.f00000"

    assigned_data, fld_3d_r, msh_3d = generate_data(fname_3d, fname_out)

    # Methods for cal. the difference and normal vectors
    assigned_r_diff, assigned_s_diff, assigned_t_diff = calculate_face_differences(assigned_data)
    assigned_r_avg_diff, assigned_s_avg_diff, assigned_t_avg_diff = calculate_face_averages(assigned_data)
    assigned_normals = calculate_face_normals(assigned_data)

    # Output:
    # print(f"Rank {rank} has Face differences:", assigned_r_diff.shape, assigned_s_diff.shape, assigned_t_diff.shape)
    # print(f"Rank {rank} has Average differences:", assigned_r_avg_diff.shape, assigned_s_avg_diff.shape, assigned_t_avg_diff.shape)
    # print(f"Rank {rank} has Normals shape:", assigned_normals.shape)

    # Generate mappings for comparison
    mappings_fd = face_mappings(assigned_r_diff, assigned_s_diff, assigned_t_diff)
    mappings_fa = face_mappings(assigned_r_avg_diff, assigned_s_avg_diff, assigned_t_avg_diff)
    
    # Select reference mapping
    ref_mapping_fd = mappings_fd[0] if rank == 0 else None
    ref_mapping_fd = comm.bcast(ref_mapping_fd, root=0)
    
    # Compare mappings before reordering
    compare_mappings(mappings_fa, ref_mapping_fd, "mappings_fa_before_reorder")
    compare_mappings(mappings_fd, ref_mapping_fd, "mappings_fd_before_reorder")
    compare_methods(mappings_fd, mappings_fa)

    # Apply reordering of assigned data      
    assigned_data_new = reorder_assigned_data(assigned_data, mappings_fd, ref_mapping_fd)
    r_diff_new, s_diff_new, t_diff_new = calculate_face_differences(assigned_data_new)

    # Extract reference differences
    extract_reference_r = r_diff_new[0] if rank == 0 else None
    extract_reference_s = s_diff_new[0] if rank == 0 else None
    extract_reference_t = t_diff_new[0] if rank == 0 else None
        
    extract_reference_r = comm.bcast(extract_reference_r, root=0)
    extract_reference_s = comm.bcast(extract_reference_s, root=0)
    extract_reference_t = comm.bcast(extract_reference_t, root=0)
    
    # Apply corrections to axis signs
    assigned_data_final = correct_axis_signs(assigned_data_new, extract_reference_r, extract_reference_s, extract_reference_t, r_diff_new, s_diff_new, t_diff_new)

    # Compute final mappings
    r_diff_final, s_diff_final, t_diff_final = calculate_face_differences(assigned_data_final)
    final_mappings_fd = face_mappings(r_diff_final, s_diff_final, t_diff_final)
    r_avg_diff_final, s_avg_diff_final, t_avg_diff_final = calculate_face_averages(assigned_data_final)
    final_mappings_fa = face_mappings(r_avg_diff_final, s_avg_diff_final, t_avg_diff_final)
    
    # Compute normal vectors and directions
    assigned_normals = calculate_face_normals(assigned_data_final)
    averaged_normals =  reduce_face_normals(assigned_normals)
    final_mappings_fn, flow_directions = directions_face_normals(averaged_normals)
    flow_directions = np. array(flow_directions)

    # Compare mappings after reordering
    compare_mappings(final_mappings_fa, ref_mapping_fd, "mappings_fa_after_reorder")
    compare_mappings(final_mappings_fd, ref_mapping_fd, "mappings_fd_after_reorder")
    compare_methods(final_mappings_fd, final_mappings_fa)
    compare_mappings(final_mappings_fn, ref_mapping_fd, "mappings_fn_after_reorder")
    
    # Compute the starting and ending global indices for each rank.
    if rank == 0:
        idx_ranges = []
        start_idx = 0
        # Distribute leftovers among initial ranks
        for i in range(size):
            adjusted_chunk_size = elements_per_rank + (1 if i < leftover else 0)  # Distribute leftovers
            end_idx = start_idx + adjusted_chunk_size
            idx_ranges.append((start_idx, end_idx))
            start_idx = end_idx 
    else:
        idx_ranges = None
    idx_ranges = comm.bcast(idx_ranges, root=0)
    
    element_centers_local = compute_element_centers(assigned_data_final)
    element_centers_local = np.array(element_centers_local)
    
    # Compute Cross-Section Centers
    received_overlap = None
    if rank > 0:
        received_overlap = comm.recv(source=rank-1, tag=11)

    cross_section_centers, leftover_overlap = compute_cross_section_centers(
        element_centers_local, 
        num_elements_per_section,
        prev_overlap=received_overlap
    )

    if rank < size - 1:
        if leftover_overlap is not None and len(leftover_overlap) > 0:
            overlap_buffer = np.array(leftover_overlap).reshape(-1, 3)
        else:
            overlap_buffer = np.empty((0, 3))
        comm.send(overlap_buffer, dest=rank + 1, tag=11)

    # Gather all centers
    cross_section_centers = np.asarray(cross_section_centers, dtype=np.single)
    sendbuf = cross_section_centers.ravel() 
    sendcount = np.array(len(sendbuf))
    sendcounts = comm.gather(sendcount, root=0)
    if rank == 0:
        total_count = sum(sendcounts)
        recvbuf = np.empty(total_count, dtype=np.single)
        displacements = np.insert(np.cumsum(sendcounts[:-1]), 0, 0)
    else:
        recvbuf = None
        displacements = None

    comm.Gatherv(sendbuf=sendbuf,
                recvbuf=(recvbuf, sendcounts, displacements, MPI.FLOAT),
                root=0)
    
    if rank == 0:
        final_cross_section_centers = recvbuf.reshape(-1, 3)
        print(f"Final Cross-Section Centers has shape: {final_cross_section_centers.shape}")
    else:
        final_cross_section_centers = None

    final_cross_section_centers = comm.bcast(final_cross_section_centers, root=0)

    # Assign cross-section center based on the rank
    cross_section_center_choosen = assign_section(rank, ranks_per_section, final_cross_section_centers)

    # Calculate radial, theta, and z vectors for each element relative to the cross-section center
    radial_vectors, theta_vectors, z_vectors = calculate_vectors(cross_section_center_choosen, element_centers_local, flow_directions)

    # Ensure each variable is a NumPy array
    assigned_radial_vectors = np.asarray(radial_vectors)
    assigned_theta_vectors = np.asarray(theta_vectors)
    assigned_z_vectors = np.asarray(z_vectors)
    
    assigned_r_diff_final = np.asarray(r_diff_final)
    assigned_s_diff_final = np.asarray(s_diff_final)
    assigned_t_diff_final = np.asarray(t_diff_final)

    num_elements = assigned_radial_vectors.shape[0]

    # Prepare the transformation matrix A for vector mapping
    A = np.array([
        [
            [np.dot(assigned_r_diff_final[i], assigned_radial_vectors[i]), np.dot(assigned_r_diff_final[i], assigned_theta_vectors[i]), np.dot(assigned_r_diff_final[i], assigned_z_vectors[i])],
            [np.dot(assigned_s_diff_final[i], assigned_radial_vectors[i]), np.dot(assigned_s_diff_final[i], assigned_theta_vectors[i]), np.dot(assigned_s_diff_final[i], assigned_z_vectors[i])],
            [np.dot(assigned_t_diff_final[i], assigned_radial_vectors[i]), np.dot(assigned_t_diff_final[i], assigned_theta_vectors[i]), np.dot(assigned_t_diff_final[i], assigned_z_vectors[i])]
        ] for i in range(num_elements)
    ])

    # Calculate vector mappings using the transformation matrix A
    mappings_cy = vector_mapping(A)

    ref_vector_mapping = mappings_cy[0] if rank == 0 else None        
    ref_vector_mapping = comm.bcast(ref_vector_mapping, root=0)

    print(f"Mappings of cy after reordering")
    compare_vector_mappings(mappings_cy, ref_vector_mapping, idx_ranges)

if check:
    mappings()
elif rank ==0:
    print(f"Error:: Invalid size: {size}")
    suggested_sizes  = valid_sizes(total_elements, num_elements_per_section)