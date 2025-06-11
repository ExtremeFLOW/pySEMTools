#!/usr/bin/env python

import os
import sys
sys.path.append("/home/hpc/iwst/iwst115h/forked_git/pySEMTools")
from typing import cast
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    num_elements_per_section = os.environ.get("num_elements_per_section")
    pattern_factor = os.environ.get("pattern_factor")
    num_pattern_cross_sections = os.environ.get("num_pattern_cross_sections")

    # Check if any variable is missing
    if None in [num_elements_per_section, pattern_factor, num_pattern_cross_sections]:
        print("Error: Missing required environment variables.", file=sys.stderr)
        sys.exit(1)

    num_elements_per_section = int(cast(str, num_elements_per_section))
    pattern_factor = int(cast(str, pattern_factor))
    num_pattern_cross_sections = int(cast(str, num_pattern_cross_sections))
else:
    num_elements_per_section = None
    pattern_factor = None
    num_pattern_cross_sections = None

num_elements_per_section = comm.bcast(num_elements_per_section, root=0)
pattern_factor = comm.bcast(pattern_factor, root=0)
num_pattern_cross_sections = comm.bcast(num_pattern_cross_sections, root=0)

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

def mappings() -> None:
    # fname_3d = "/home/woody/iwst/iwst115h/Work/Checkpoint_files/field0.f00024"
    fname_3d: str = "/home/woody/iwst/iwst115h/Work/Checkpoint_files/field0.f00000"
    
    # fname_out = "/home/woody/iwst/iwst115h/Work/writefiles/unstruct_new/field_out_0.f00024"
    fname_out: str = "/home/woody/iwst/iwst115h/Work/writefiles/field/fieldout0.f00000"

    assigned_data, fld_3d_r, msh_3d = generate_data(fname_3d, fname_out)

    # Methods for cal. the difference and normal vectors
    assigned_r_diff, assigned_s_diff, assigned_t_diff = calculate_face_differences(assigned_data)
    assigned_r_avg_diff, assigned_s_avg_diff, assigned_t_avg_diff = calculate_face_averages(assigned_data)
    assigned_normals = calculate_face_normals(assigned_data)

    # Output:
    # print(f"Rank {rank} has Face differences:", assigned_r_diff.shape, assigned_s_diff.shape, assigned_t_diff.shape)
    # print(f"Rank {rank} has Average differences:", assigned_r_avg_diff.shape, assigned_s_avg_diff.shape, assigned_t_avg_diff.shape)
    # print(f"Rank {rank} has Normals shape:", assigned_normals.shape)

    # Method-1: Face_differences
    mappings_fd = face_mappings(assigned_r_diff, assigned_s_diff, assigned_t_diff)
    # Method-2: Face_averages
    mappings_fa = face_mappings(assigned_r_avg_diff, assigned_s_avg_diff, assigned_t_avg_diff)
    

    if rank == 0:
        ref_mapping_fd = mappings_fd[0]
    else:
        ref_mapping_fd = None
        
    ref_mapping_fd = comm.bcast(ref_mapping_fd, root=0)
    
    # Compare all methods:
    compare_mappings(mappings_fa, ref_mapping_fd, "mappings_fa_before_reorder")
    compare_mappings(mappings_fd, ref_mapping_fd, "mappings_fd_before_reorder")
    compare_methods(mappings_fd, mappings_fa)

    # Reordering of subset        
    assigned_data_new = reorder_assigned_data(assigned_data, mappings_fd, ref_mapping_fd)
    r_diff_new, s_diff_new, t_diff_new = calculate_face_differences(assigned_data_new)

    if rank == 0:
        extract_reference_r = r_diff_new[0]
        extract_reference_s = s_diff_new[0]
        extract_reference_t = t_diff_new[0]
    else:
        extract_reference_r = None
        extract_reference_s = None
        extract_reference_t = None
        
    extract_reference_r = comm.bcast(extract_reference_r, root=0)
    extract_reference_s = comm.bcast(extract_reference_s, root=0)
    extract_reference_t = comm.bcast(extract_reference_t, root=0)
    
    assigned_data_final = correct_axis_signs(assigned_data_new, extract_reference_r, extract_reference_s, extract_reference_t, r_diff_new, s_diff_new, t_diff_new)

    r_diff_final, s_diff_final, t_diff_final = calculate_face_differences(assigned_data_final)
    final_mappings_fd = face_mappings(r_diff_final, s_diff_final, t_diff_final)

    r_avg_diff_final, s_avg_diff_final, t_avg_diff_final = calculate_face_averages(assigned_data_final)
    final_mappings_fa = face_mappings(r_avg_diff_final, s_avg_diff_final, t_avg_diff_final)
    
    assigned_normals = calculate_face_normals(assigned_data_final)
    averaged_normals =  reduce_face_normals(assigned_normals)
    final_mappings_fn, flow_directions = directions_face_normals(averaged_normals)
    flow_directions = np. array(flow_directions)

    # Compare all methods:
    compare_mappings(final_mappings_fa, ref_mapping_fd, "mappings_fa_after_reorder")
    compare_mappings(final_mappings_fd, ref_mapping_fd, "mappings_fd_after_reorder")
    compare_methods(final_mappings_fd, final_mappings_fa)
    compare_mappings(final_mappings_fn, ref_mapping_fd, "mappings_fn_after_reorder")
    
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
            
    def compute_element_centers(data):
        num_elements = data.shape[0]
        flattened = data.reshape(num_elements, -1, 3)  # (num_elements, 512, 3)
        element_centers = np.mean(flattened, axis=1)       # (num_elements, 3)
        return element_centers
    
    def compute_cross_section_centers(data, num_elements_per_section, prev_overlap=None):    
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

    def assign_section(rank, ranks_per_section, final_cross_section_centers):
        section_number = rank // ranks_per_section  # Group ranks dynamically
        cross_section_center_choosen = final_cross_section_centers[section_number]
        return cross_section_center_choosen, section_number

    cross_section_center_choosen, section_number = assign_section(rank, ranks_per_section, final_cross_section_centers)                                               

    def calculate_vectors(cross_section_center_choosen, element_centers_local, flow_directions):

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

    radial_vectors, theta_vectors, z_vectors = calculate_vectors(cross_section_center_choosen, element_centers_local, flow_directions)

    assigned_radial_vectors = np.array(radial_vectors)
    assigned_theta_vectors = np.array(theta_vectors)
    assigned_z_vectors = np.array(z_vectors)
    
    assigned_r_diff_final = np.array(r_diff_final)
    assigned_s_diff_final = np.array(s_diff_final)
    assigned_t_diff_final = np.array(t_diff_final)

    num_elements = assigned_radial_vectors.shape[0]

    A = np.array([
        [
            [np.dot(assigned_r_diff_final[i], assigned_radial_vectors[i]), np.dot(assigned_r_diff_final[i], assigned_theta_vectors[i]), np.dot(assigned_r_diff_final[i], assigned_z_vectors[i])],
            [np.dot(assigned_s_diff_final[i], assigned_radial_vectors[i]), np.dot(assigned_s_diff_final[i], assigned_theta_vectors[i]), np.dot(assigned_s_diff_final[i], assigned_z_vectors[i])],
            [np.dot(assigned_t_diff_final[i], assigned_radial_vectors[i]), np.dot(assigned_t_diff_final[i], assigned_theta_vectors[i]), np.dot(assigned_t_diff_final[i], assigned_z_vectors[i])]
        ] for i in range(num_elements)
    ])

    def vector_mapping(A):
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

    mappings_cy = vector_mapping(A)
    
    if rank == 0:
        ref_vector_mapping = mappings_cy[0]
    else:
        ref_vector_mapping = None
        
    ref_vector_mapping = comm.bcast(ref_vector_mapping, root=0)

    def compare_vector_mappings(mappings_cy, ref_vector_mapping, idx_ranges):
        num_elements = len(mappings_cy)
        match_count_cy = 0
        for i in range(num_elements):
            if mappings_cy[i][2] == ref_vector_mapping[2]:
                match_count_cy += 1
        # print(f"Total matches: {match_count_cy}")
        matching_percentage = (match_count_cy / num_elements) * 100
        start_idx, end_idx = idx_ranges[rank]
        print(f"Rank {rank} has operated between {start_idx} and {end_idx} has matching Percentage of mappings_cy: {matching_percentage:.2f}%")

    print(f"Mappings of cy after reordering")
    compare_vector_mappings(mappings_cy, ref_vector_mapping, idx_ranges)

if check:
    mappings()
elif rank ==0:
    print(f"Error:: Invalid size: {size}")
    suggested_sizes  = valid_sizes(total_elements, num_elements_per_section)