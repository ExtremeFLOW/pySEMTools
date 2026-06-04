#!/usr/bin/env python

import os
import sys
from typing import cast
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    # Read environment variables for configuration as input parameters
    # These variables are expected to be set in the environment before running the script.
    num_elements_per_section = os.environ.get("num_elements_per_section")
    pattern_factor = os.environ.get("pattern_factor")
    num_pattern_cross_sections = os.environ.get("num_pattern_cross_sections")
    field_name = os.environ.get("field_name")
    
    # Check if any variable is missing and exit with an error message
    if None in [num_elements_per_section, pattern_factor, num_pattern_cross_sections, field_name]:
        print("Error: Missing required environment variables.", file=sys.stderr)
        sys.exit(1)

    # Convert environment variables to integers
    num_elements_per_section = int(cast(str, num_elements_per_section))
    pattern_factor = int(cast(str, pattern_factor))
    num_pattern_cross_sections = int(cast(str, num_pattern_cross_sections))
    field_name = cast(str, field_name)
else:
    num_elements_per_section = None
    pattern_factor = None
    num_pattern_cross_sections = None
    field_name = None

# Broadcast the configuration variables to all ranks
num_elements_per_section = comm.bcast(num_elements_per_section, root=0)
pattern_factor = comm.bcast(pattern_factor, root=0)
num_pattern_cross_sections = comm.bcast(num_pattern_cross_sections, root=0)
field_name = comm.bcast(field_name, root=0)

import numpy as np
from pysemtools.io.reorder_data import (
    generate_data, calculate_face_differences, calculate_face_averages, calculate_face_normals,
    face_mappings, compare_mappings, compare_methods, reduce_face_normals, directions_face_normals,
    reorder_assigned_data, correct_axis_signs
)
from pysemtools.io.ppymech.neksuite import pynekwrite
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.datatypes.msh_partitioning import MeshPartitioner

elem_pattern_cross_sections = num_elements_per_section * pattern_factor; # No. of elements in each pattern cross-section
total_elements = elem_pattern_cross_sections * num_pattern_cross_sections # Total elements in the file.
elements_per_rank = total_elements // size  # No. of elements in each rank.
ranks_per_section = num_elements_per_section // elements_per_rank # No. of ranks per each cross-section
ranks_per_pattern_section = ranks_per_section * pattern_factor # No. of ranks per each pattern cross-section
leftover_total = total_elements % size  # Elements that don’t fit evenly in the rank (if any)
chunk_size_section = elem_pattern_cross_sections // size  # No. of elements each rank holds in the section.
leftover_section = elem_pattern_cross_sections % size  # Elements that don’t fit evenly in the section (if any)

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

def find_valid_sizes(total_elements: int, num_elements_per_section: int, min_size=150, max_size=2000):
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

def truncate_reduce(data, decimals):
    """
    Applies controlled truncation to refine numerical precision for selected axes.

    Args:
        data (numpy.ndarray): The input data array.
        decimals (int): The number of decimal places to retain.

    Returns:
        numpy.ndarray: Data with adjusted precision for the first two axes.
    """
    trial = data[:, :, :, :, 0:2]
    factor = 10.0 ** decimals
    return np.floor(trial * factor) / factor

def extracted_fields(fld_3d_r):
    """
    Extracts all key fields from the "fname" dynamically.

    Args:
        fld_3d_r (FieldRegistry): Input data containing various field keys.

    Returns:
        dict: Extracted field mappings including velocity, pressure, temperature, and any all other
        additional fields.
    """
    extracted = {}
    extra_field_counter = 1 
    for key in fld_3d_r.fields.keys():
        length = len(fld_3d_r.fields[key])
        if key == 'vel':
            extracted['u'] = fld_3d_r.fields[key][0] if length > 0 else None
            extracted['v'] = fld_3d_r.fields[key][1] if length > 1 else None
            extracted['w'] = fld_3d_r.fields[key][2] if length > 2 else None   
        elif key == 'pres':
            extracted['pres'] = fld_3d_r.fields[key][0] if length > 0 else None
        elif key == 'temp':
            extracted['temp'] = fld_3d_r.fields[key][0] if length > 0 else None
        else:  # To dynamically assign unknown fields with "s1", "s2", ...
            for index in range(length): 
                extracted[f"s{extra_field_counter}"] = fld_3d_r.fields[key][index]
                extra_field_counter += 1
    return extracted

def get_rank_for(z_max_elem_index):
    """
    Identifies the rank responsible for storing a specific global element index.

    This function:
    - Determines which rank holds the last element in `elem_pattern_cross_sections`.
    - Computes the rank ID based on how elements are distributed across ranks.
    - Ensures boundary conditions are handled correctly.

    Args:
        z_max_elem_index (int): The global index of the element to locate.

    Returns:
        int: The rank ID that owns the specified element.
    """
    if z_max_elem_index < 0:
        return 0
    elif z_max_elem_index >= total_elements:
        return size - 1
    else:
        return (z_max_elem_index // elements_per_rank)-1

def element_grouping() -> None:
    """
    Groups elements, processes field data, and manages distributed computation across MPI ranks.

    Notation: A group of cross-sections, when repeated, forms a pattern cross-section. The first pattern cross-section 
    is designated as the reference pattern cross-section, which serves as the basis for comparisons.

    1. Data Loading & Processing:
        - Reads input files and computes face differences, averages, and normal vectors.
        - Applies reordering and axis corrections.

    2. Element Grouping:
        - Determines a reference pattern cross-section and identifies repeating structures.
        - Groups elements based on shape consistency, ensuring alignment across computational ranks.

    3. Coordinate Collection & Redistribution:
        - Retrieves coordinate data from the reference pattern cross-section that is distributed across multiple ranks.
        - Ensures efficient redistribution of coordinates to all ranks for consistent processing and averaging.

    4. Field Extraction & Inter-Rank Data Exchange:
        - Extracts relevant fields dynamically for structured storage and processing.
        - Facilitates communication between ranks, sending and receiving required element data based on groups.
        - Ensures each rank has access to necessary computational fields.

    5. Group Averaging & Field Registry Management:
        - Computes field averages across pattern cross-sections.
        - Adds averaged fields to the registry.

    6. Final Data Writing:
        - Constructs a subdomain mesh for processed data.
        - Writes computed fields to output files for continued analysis.

    Args:
        None

    Returns:
        None (Processes, distributes, and saves field data)
    """
    fname_3d = "../../field0.f00024"

    fname_out = "../../field_out_0.f00024"    
    
    # Load data
    assigned_data, fld_3d_r, msh_3d = generate_data(fname_3d, fname_out)

    # Methods for cal. the difference and normal vectors
    assigned_r_diff, assigned_s_diff, assigned_t_diff = calculate_face_differences(assigned_data)
    assigned_r_avg_diff, assigned_s_avg_diff, assigned_t_avg_diff = calculate_face_averages(assigned_data)
    assigned_normals = calculate_face_normals(assigned_data)

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
        idx_ranges_section =[]
        start_idx = 0
        start_idx_section = 0
        for i in range(size):
            adjusted_chunk_size = elements_per_rank + (1 if i < leftover_total else 0)  
            adjusted_chunk_size_section = chunk_size_section + (1 if i < leftover_section else 0)
            end_idx = start_idx + adjusted_chunk_size
            end_idx_section = start_idx_section + adjusted_chunk_size_section
            idx_ranges.append((start_idx, end_idx))
            idx_ranges_section.append((start_idx_section, end_idx_section))
            start_idx = end_idx 
            start_idx_section = end_idx_section
    else:
        idx_ranges = None
        idx_ranges_section = None
    idx_ranges = comm.bcast(idx_ranges, root=0)
    idx_ranges_section = comm.bcast(idx_ranges_section, root=0)

    # Determine the "full rank list" which has list of ranks in one reference pattern cross-section.
    full_rank_list = list(range(ranks_per_pattern_section))        
    ndim1, ndim2, ndim3, ndim4 = assigned_data_final.shape[1:]

    if rank in full_rank_list:
        assigned_data_final_array  = np.empty((elements_per_rank, ndim1, ndim2, ndim3, ndim4))
    else:
        assigned_data_final_array = None
    
    bundle = {i: [] for i in range(elements_per_rank)}

    # Data transmission among ranks: exchanging only relevant sections for comparison
    # The full_rank_list holds ranks in the reference pattern cross-section. 
    # Elements in later ranks align with earlier ranks due to repeating patterns in the cross-section.
    # Example: full_rank_list has [0,1,2,3,4,5] then, Rank 6 aligns with Rank 0, Rank 7 aligns with Rank 1, etc.
    # This ensures structured comparisons and grouping of elements with identical x, y coordinates.
    
    if rank in full_rank_list:
        start_index, end_index = idx_ranges[0]
        if assigned_data_final_array is not None:
            assigned_data_final_array[start_index:end_index] = assigned_data_final
            data_to_send = assigned_data_final_array[start_index:end_index]
        else: 
            data_to_send = None
        target_ranks = [rank + ranks_per_pattern_section * i for i in range(1, int(num_pattern_cross_sections) if num_pattern_cross_sections is not None else 0)]
        valid_targets = [target for target in target_ranks if target < size]
            
        for target in valid_targets:
            comm.Send(data_to_send, dest=target, tag=rank)
    
    elif rank not in full_rank_list:
        start_index, end_index = idx_ranges[rank]
        size_to_receive = end_index - start_index
        temp_buffer = np.empty((size_to_receive, ndim1, ndim2, ndim3, ndim4))
        source_rank = rank % ranks_per_pattern_section
        
        comm.Recv(temp_buffer, source=source_rank, tag=source_rank)
            
        rounded_data = truncate_reduce(assigned_data_final, decimals=6)
            
        rounded_data_received = truncate_reduce(temp_buffer, decimals=6)
            
        # Compare with local rounded_data
        if np.allclose(rounded_data, rounded_data_received, atol=1e-4):
            for i in range(elements_per_rank):
                source_global_idx = source_rank * elements_per_rank + i
                target_global_idx = rank * elements_per_rank + i
                if source_global_idx not in bundle:
                    bundle[source_global_idx] = []
                bundle[source_global_idx].append(target_global_idx)
        else:
            print(f"[NO MATCH] Rank {rank} did not match with {source_rank}")
    
    # Gather and broadcast final grouped data        
    all_bundled_data = comm.gather(bundle, root=0)
    bundled_array = None
    if rank == 0:
        bundle_dict = {}

        for unit_bundle in all_bundled_data:
            for src_idx, targets in unit_bundle.items():
                if src_idx not in bundle_dict:
                    bundle_dict[src_idx] = []
                bundle_dict[src_idx].extend(targets)

        final_bundle = [[src_idx] + sorted(set(targets)) for src_idx, targets in sorted(bundle_dict.items())]
        bundled_array = np.array(final_bundle, dtype=int)

        # print("\nFinal grouping of elements:")
        # for row in bundle_array:
        #     print(row)
            
    bundled_array = comm.bcast(bundled_array, root=0)    
    comm.Barrier()
    
    # Determine which rank owns `z_max_elem_index`  
    z_max_elem_rank = get_rank_for(elem_pattern_cross_sections)
    print(f"z_max_elem_rank:", z_max_elem_rank) if rank == 0 else None
    
    # Determine the local index of `z_max_elem` for the rank that owns it and max value of "z"
    if rank == z_max_elem_rank:
        z_max_elem: int = (elem_pattern_cross_sections // (rank + 1)) - 1
        print(f"Rank {rank} has local index of z_max_elem:",z_max_elem)
        z_max_elem_value =np.max(assigned_data_final[z_max_elem][:,:,:,2])
        print(f"Rank {rank} has z_max_elem_value",z_max_elem_value)
    else:
        z_max_elem_value = None

    z_max_elem_value = comm.bcast(z_max_elem_value, root=z_max_elem_rank)
    
    # Extract all relevant fields
    fields = extracted_fields(fld_3d_r)
    
    fields_mentioned = []
    if field_name == 'vel':
        fields_mentioned = ['u', 'v', 'w']
    elif field_name == 'pres':
        fields_mentioned = ['pres']
    elif field_name == 'temp':
        fields_mentioned = ['temp']
    elif field_name == 'default':
        fields_mentioned = []
        for key in fld_3d_r.fields.keys():
            if len(fld_3d_r.fields[key]) != 0:
                fields_mentioned.append(key)
        if 'vel' in fields_mentioned:
            fields_mentioned.remove('vel')
            fields_mentioned = ['u', 'v', 'w'] + fields_mentioned 
        if 'scal' in fields_mentioned:
            fields_mentioned.remove('scal')
            length_scal = len(fld_3d_r.fields['scal'])
            scal_fields = [f's{i+1}' for i in range(length_scal)] 
            fields_mentioned = fields_mentioned + scal_fields
        print(f"Rank {rank} has default field_mentioned:", fields_mentioned) if rank == 0 else None
    
    # Determine which groups, each rank is responsible for
    if rank == 0:
        owned_groups_split = []
        for i in range(size):
            start, end = idx_ranges_section[i]
            owned_groups_split.append(bundled_array[start:end])
    else:
        owned_groups_split = None

    owned_groups = comm.scatter(owned_groups_split, root=0)

    # Determine which elements belonging to which rank
    if rank == 0:
        element_rank_mapping = [(gid, gid // elements_per_rank) for gid in range(total_elements)]
    else:
        element_rank_mapping = None
    element_rank_mapping = comm.bcast(element_rank_mapping, root=0)
    element_to_rank = dict(element_rank_mapping) 
    
    # Create a mapping from global element indices to local indices within each rank 
    if rank == 0:
        global_to_local_all = {
            i: {gid: local_idx for local_idx, gid in enumerate(range(i * elements_per_rank, (i + 1) * elements_per_rank))}
            for i in range(size)
        }
    else:
        global_to_local_all = None
    global_to_local_all = comm.bcast(global_to_local_all, root=0)
    
    # Get list of ranks from which the current rank needs to get data
    recv_requests = {}
    for group in owned_groups:
        for gid in group:
            owner = element_to_rank[gid]
            if owner == rank:
                continue
            recv_requests.setdefault(owner, []).append(gid)
            
    all_recv_requests = comm.allgather(recv_requests)
    
    # Get list of ranks to which the current rank needs to send data    
    send_requests = {}
    for r, req_dict in enumerate(all_recv_requests):
        if rank in req_dict:
            send_requests[r] = req_dict[rank]
    
    # Send data to the ranks that requested it
    for dst, gids in send_requests.items():
        local_map = global_to_local_all[rank]
        local_idxs = [local_map[gid] for gid in gids]
        data_to_send = {field: fields[field][local_idxs] for field in fields_mentioned}
        comm.send((gids, data_to_send), dest=dst, tag=3)
    
    # Receive data from ranks that sent requests
    received_data = {}
    for src, gids in recv_requests.items():
        recv_gids, data = comm.recv(source=src, tag=3)
        received_data[src] = (recv_gids, data)

    # Calculate group averages for each field dynamically
    group_averages = []
    for group in owned_groups:
        group_data = {field: [] for field in fields_mentioned}
        for gid in group:
            owner = element_to_rank[gid]
            if owner == rank:
                local_idx = global_to_local_all[rank][gid]
                for field in fields_mentioned:
                    group_data[field].append(fields[field][local_idx])
            else:
                recv_gids, data = received_data[owner]
                idx = recv_gids.index(gid)
                for field in fields_mentioned:
                    group_data[field].append(data[field][idx])
        group_data = {field: np.stack(value, axis=0) for field, value in group_data.items()}
        avg = {field: np.mean(value, axis=0) for field, value in group_data.items()}
        group_averages.append(avg)

    # Prepare the field variables to be added to the FieldRegistry
    field_keys = {
        'vel': ['u', 'v', 'w'],
        'pres': ['pres'],
        'temp': ['temp'],
        'default': fields_mentioned
    }
    
    # Initialize all average fields to None
    for key in ['u', 'v', 'w', 'pres', 'temp']:
        globals().setdefault(f"avg_{key}_all", None)

    if field_name in field_keys:
        for key in field_keys[field_name]:
            globals()[f"avg_{key}_all"] = None
    
    # Assign average values to the field variables
    if field_name in field_keys:
        for key in field_keys[field_name]:
            if field_name == 'default' and key not in fields_mentioned:
                continue
            globals()[f"avg_{key}_all"] = np.array([avg[key] for avg in group_averages], dtype=np.single)

    print(f"Adding field using Field registry and coordinates also") if rank == 0 else None
    
    # Initializes the field registry
    fld = FieldRegistry(comm)  
    
    # Choose a condition that you want the subdomain to satisfy
    condition1 = msh_3d.z < z_max_elem_value
    conidtion2 = msh_3d.z > 0.0
    cond = condition1 & conidtion2
    
    # Initialize the mesh partitioner with the given condition
    mp = MeshPartitioner(comm, msh=msh_3d, conditions=[cond])
    
    # Create the properly partitioned sub mesh and field
    partitioned_mesh = mp.create_partitioned_mesh(msh_3d, partitioning_algorithm="load_balanced_linear", create_conectivity=False)
    partitioned_field = mp.create_partitioned_field(fld, partitioning_algorithm="load_balanced_linear")

    print(f"fld.fields are", partitioned_field.fields.keys()) if rank == 0 else None

    if rank == 0:
        for key in partitioned_field.fields.keys():
            print(f'Current field "fld" {key} has {len(partitioned_field.fields[key])} fields')

        for key in fld_3d_r.fields.keys():
            print(f'Earlier field "fld_3d" {key} has {len(fld_3d_r.fields[key])} fields')


    print(f"Rank {comm.rank} Fields before adding:", partitioned_field.registry.keys()) if rank == 0 else None
    
    # Checks for matching field names and adds fields dynamically
    if field_name in field_keys:
        for key in field_keys[field_name]:
            avg_var = globals().get(f"avg_{key}_all", None)
            if avg_var is not None:
                partitioned_field.add_field(comm, field_name=key, field=avg_var, dtype=np.single)
                if rank == 0:
                    print(f'Field {key} added to registry and fields directory in pos {partitioned_field.registry_pos[key]}')

    print(f"Rank {comm.rank} Fields after adding:", partitioned_field.registry.keys()) if rank == 0 else None
        
    # Write the data in a subdomain
    fout = "../../fieldout_field.f00024"    
    pynekwrite(fout, comm, msh=partitioned_mesh, fld=partitioned_field, write_mesh=True)
    print("Rank{rank} has done adding fields successfully and saved.") if rank == 0 else None

if check:    
    element_grouping()    
elif rank ==0:
    print(f"Error:: Invalid size: {size}")
    suggested_sizes  = find_valid_sizes(total_elements, num_elements_per_section)