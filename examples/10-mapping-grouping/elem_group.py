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
    field_name = os.environ.get("field_name")
    
    if None in [num_elements_per_section, pattern_factor, num_pattern_cross_sections, field_name]:
        print("Error: Missing required environment variables.", file=sys.stderr)
        sys.exit(1)

    num_elements_per_section = int(cast(str, num_elements_per_section))
    pattern_factor = int(cast(str, pattern_factor))
    num_pattern_cross_sections = int(cast(str, num_pattern_cross_sections))
    field_name = cast(str, field_name)
else:
    num_elements_per_section = None
    pattern_factor = None
    num_pattern_cross_sections = None
    field_name = None

num_elements_per_section = comm.bcast(num_elements_per_section, root=0)
pattern_factor = comm.bcast(pattern_factor, root=0)
num_pattern_cross_sections = comm.bcast(num_pattern_cross_sections, root=0)
field_name = comm.bcast(field_name, root=0)

import numpy as np
from pysemtools.io.reorder_data_test import (
    generate_data, calculate_face_differences, calculate_face_averages, calculate_face_normals,
    face_mappings, compare_mappings, compare_methods, reduce_face_normals, directions_face_normals,
    reorder_assigned_data, correct_axis_signs
)
from pysemtools.datatypes.msh import Mesh
from pysemtools.io.ppymech.neksuite import pynekwrite
from pysemtools.datatypes.field import FieldRegistry

elem_pattern_crosssections = num_elements_per_section * pattern_factor; # No. of elements in each pattern cross-section
total_elements = elem_pattern_crosssections * num_pattern_cross_sections # Total elements in the file.
elements_per_rank = total_elements // size  # No. of elements in each rank.
ranks_per_section = num_elements_per_section // elements_per_rank # No. of ranks per each cross-section
ranks_per_pattern_section = ranks_per_section * pattern_factor # No. of ranks per each pattern cross-section
leftover_total = total_elements % size  # Elements that don’t fit evenly in the rank (if any)
chunk_size_section = elem_pattern_crosssections // size
leftover_section = elem_pattern_crosssections % size  # Elements that don’t fit evenly in the rank (if any)

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

def find_valid_sizes(total_elements: int, num_elements_per_section: int, min_size=150, max_size=2000):

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
    trial = data[:, :, :, :, 0:2]
    factor = 10.0 ** decimals
    return np.floor(trial * factor) / factor

def extracted_fields(fld_3d_r):
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

def element_grouping() -> None:
    
    # fname_3d = "/home/woody/iwst/iwst115h/Work/Checkpoint_files/field0.f00024"
    fname_3d = "/home/woody/iwst/iwst115h/Work/Checkpoint_files/field0.f00000"
    
    # fname_out = "/home/woody/iwst/iwst115h/Work/writefiles/unstruct_new/field_out_0.f00024"
    fname_out = "/home/woody/iwst/iwst115h/Work/writefiles/field/fieldout0.f00000"
    
    assigned_data, fld_3d_r, msh_3d = generate_data(fname_3d, fname_out)

    # Methods for cal. the difference and normal vectors
    assigned_r_diff, assigned_s_diff, assigned_t_diff = calculate_face_differences(assigned_data)
    assigned_r_avg_diff, assigned_s_avg_diff, assigned_t_avg_diff = calculate_face_averages(assigned_data)
    assigned_normals = calculate_face_normals(assigned_data)

    # Method-1: Face_differences
    mappings_fd = face_mappings(assigned_r_diff, assigned_s_diff, assigned_t_diff)
    # Method-2: Face_averages
    mappings_fa = face_mappings(assigned_r_avg_diff, assigned_s_avg_diff, assigned_t_avg_diff)
    

    if rank == 0:
        ref_mapping_fd = mappings_fd[0]
    else:
        ref_mapping_fd = None
        
    ref_mapping_fd = comm.bcast(ref_mapping_fd, root=0)
    
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

    full_rank_list = list(range(ranks_per_pattern_section))        
    ndim1, ndim2, ndim3, ndim4 = assigned_data_final.shape[1:]

    if rank in full_rank_list:
        assigned_data_final_array  = np.empty((elements_per_rank, ndim1, ndim2, ndim3, ndim4))
    else:
        assigned_data_final_array = None
    
    bundle = {i: [] for i in range(elements_per_rank)}

    #send data to the ranks
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
    x_coords = np.empty((elem_pattern_crosssections, ndim1, ndim2, ndim3))
    y_coords = np.empty((elem_pattern_crosssections, ndim1, ndim2, ndim3))
    z_coords = np.empty((elem_pattern_crosssections, ndim1, ndim2, ndim3))
    
    if rank == 0:
        gathered_data_x_coords = np.empty((elem_pattern_crosssections, ndim1, ndim2, ndim3))
        print(f"gathered_data_x_coords shape and same for all other cooords: {gathered_data_x_coords.shape}")
        gathered_data_y_coords = np.empty((elem_pattern_crosssections, ndim1, ndim2, ndim3))
        gathered_data_z_coords = np.empty((elem_pattern_crosssections, ndim1, ndim2, ndim3))
        print(f"full_rank_list: {full_rank_list}")
    else:
        gathered_data_x_coords, gathered_data_y_coords, gathered_data_z_coords = None, None, None

    if rank in full_rank_list:
        start_index, end_index = idx_ranges[0]
        x_coords[start_index:end_index] = msh_3d.x[start_index:end_index]
        y_coords[start_index:end_index] = msh_3d.y[start_index:end_index]
        z_coords[start_index:end_index] = msh_3d.z[start_index:end_index]
        data_to_send_x_coords = x_coords[start_index:end_index]
        data_to_send_y_coords = y_coords[start_index:end_index]
        data_to_send_z_coords = z_coords[start_index:end_index]
        
        if rank != 0: # Non-root ranks send their data        
            comm.Send(data_to_send_x_coords, dest=0, tag=0)
            comm.Send(data_to_send_y_coords, dest=0, tag=1)
            comm.Send(data_to_send_z_coords, dest=0, tag=2)
        else: # Root already has its data, just copy it to the right place
            start_index, end_index = idx_ranges[0]
            gathered_data_x_coords[start_index:end_index] = data_to_send_x_coords
            gathered_data_y_coords[start_index:end_index] = data_to_send_y_coords
            gathered_data_z_coords[start_index:end_index] = data_to_send_z_coords

    if rank == 0:
        for sender_rank in full_rank_list:
            if sender_rank != 0:  # Skip self
                start_index, end_index = idx_ranges[sender_rank]
                size_to_receive = end_index - start_index
                
                temp_buffer_x = np.empty((size_to_receive, ndim1, ndim2, ndim3))
                temp_buffer_y = np.empty((size_to_receive, ndim1, ndim2, ndim3))
                temp_buffer_z = np.empty((size_to_receive, ndim1, ndim2, ndim3))
                
                comm.Recv(temp_buffer_x, source=sender_rank, tag=0)
                comm.Recv(temp_buffer_y, source=sender_rank, tag=1)
                comm.Recv(temp_buffer_z, source=sender_rank, tag=2)
                
                gathered_data_x_coords[start_index:end_index] = temp_buffer_x
                gathered_data_y_coords[start_index:end_index] = temp_buffer_y
                gathered_data_z_coords[start_index:end_index] = temp_buffer_z
                
        print(f"Root {rank} has gathered all data with shape: x: {gathered_data_x_coords.shape}, y: {gathered_data_y_coords.shape}, z: {gathered_data_z_coords.shape}")      
    
    size_list = list(range(size))
    if rank == 0:
        flat_x = gathered_data_x_coords.reshape(len(gathered_data_x_coords), -1)
        flat_y = gathered_data_y_coords.reshape(len(gathered_data_y_coords), -1)
        flat_z = gathered_data_z_coords.reshape(len(gathered_data_z_coords), -1)

        # Preparing the data to send for each rank
        sendcounts = []
        displs = []
        flattened_chunks_x, flattened_chunks_y, flattened_chunks_z = [], [], []

        offset = 0
        for i in range(size):
            start, end = idx_ranges_section[i]
            count = end - start
            sendcounts.append(count * ndim1 * ndim2 * ndim3)
            displs.append(offset)
            offset += count * ndim1 * ndim2 * ndim3
            flattened_chunks_x.append(flat_x[start:end].reshape(-1))
            flattened_chunks_y.append(flat_y[start:end].reshape(-1))
            flattened_chunks_z.append(flat_z[start:end].reshape(-1))

        sendbuf_x = np.concatenate(flattened_chunks_x)
        sendbuf_y = np.concatenate(flattened_chunks_y)
        sendbuf_z = np.concatenate(flattened_chunks_z)
    else:
        sendcounts = None
        displs = None
        sendbuf_x = sendbuf_y = sendbuf_z = None
        
    recv_count = (idx_ranges_section[rank][1] - idx_ranges_section[rank][0]) * ndim1 * ndim2 * ndim3

    recv_x = np.empty(recv_count, dtype=np.float64)
    recv_y = np.empty(recv_count, dtype=np.float64)
    recv_z = np.empty(recv_count, dtype=np.float64)
    
    comm.Scatterv([sendbuf_x, sendcounts, displs, MPI.DOUBLE], recv_x, root=0)
    comm.Scatterv([sendbuf_y, sendcounts, displs, MPI.DOUBLE], recv_y, root=0)
    comm.Scatterv([sendbuf_z, sendcounts, displs, MPI.DOUBLE], recv_z, root=0)
    
    local_count = idx_ranges_section[rank][1] - idx_ranges_section[rank][0]
    recv_x = recv_x.reshape((local_count, ndim1, ndim2, ndim3))
    recv_y = recv_y.reshape((local_count, ndim1, ndim2, ndim3))
    recv_z = recv_z.reshape((local_count, ndim1, ndim2, ndim3))

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

    if rank == 0:
        owned_groups_split = []
        for i in range(size):
            start, end = idx_ranges_section[i]
            owned_groups_split.append(bundled_array[start:end])
    else:
        owned_groups_split = None

    owned_groups = comm.scatter(owned_groups_split, root=0)
    
    if rank == 0:
        element_rank_mapping = [(gid, gid // elements_per_rank) for gid in range(total_elements)]
    else:
        element_rank_mapping = None
    element_rank_mapping = comm.bcast(element_rank_mapping, root=0)
    element_to_rank = dict(element_rank_mapping) 
    
    if rank == 0:
        global_to_local_all = {
            i: {gid: local_idx for local_idx, gid in enumerate(range(i * elements_per_rank, (i + 1) * elements_per_rank))}
            for i in range(size)
        }
    else:
        global_to_local_all = None
    global_to_local_all = comm.bcast(global_to_local_all, root=0)
    
    recv_requests = {}
    for group in owned_groups:
        for gid in group:
            owner = element_to_rank[gid]
            if owner == rank:
                continue
            recv_requests.setdefault(owner, []).append(gid)
            
    all_recv_requests = comm.allgather(recv_requests)
    
    send_requests = {}
    for r, req_dict in enumerate(all_recv_requests):
        if rank in req_dict:
            send_requests[r] = req_dict[rank]
            
    for dst, gids in send_requests.items():
        local_map = global_to_local_all[rank]
        local_idxs = [local_map[gid] for gid in gids]
        data_to_send = {field: fields[field][local_idxs] for field in fields_mentioned}
        comm.send((gids, data_to_send), dest=dst, tag=3)
        
    received_data = {}
    for src, gids in recv_requests.items():
        recv_gids, data = comm.recv(source=src, tag=3)
        received_data[src] = (recv_gids, data)

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

    field_keys = {
        'vel': ['u', 'v', 'w'],
        'pres': ['pres'],
        'temp': ['temp'],
        'default': fields_mentioned
    }

    for key in ['u', 'v', 'w', 'pres', 'temp']:
        globals().setdefault(f"avg_{key}_all", None)

    if field_name in field_keys:
        for key in field_keys[field_name]:
            globals()[f"avg_{key}_all"] = None
            
    if field_name in field_keys:
        for key in field_keys[field_name]:
            if field_name == 'default' and key not in fields_mentioned:
                continue
            globals()[f"avg_{key}_all"] = np.array([avg[key] for avg in group_averages], dtype=np.single)

    print(f"Adding field using Field registry and coordinates also") if rank == 0 else None

    fld = FieldRegistry(comm)
    #Subdomain creation
    msh_3d_sub = Mesh(comm, x=recv_x, y=recv_y, z=recv_z, create_connectivity=False)

    print(f"fld.fields are", fld.fields.keys()) if rank == 0 else None

    if rank == 0:
        for key in fld.fields.keys():
            print(f'Current field "fld" {key} has {len(fld.fields[key])} fields')

        for key in fld_3d_r.fields.keys():
            print(f'Earlier field "fld_3d" {key} has {len(fld_3d_r.fields[key])} fields')


    print(f"Rank {comm.rank} Fields before adding:", fld.registry.keys()) if rank == 0 else None

    if field_name in field_keys:
        for key in field_keys[field_name]:
            avg_var = globals().get(f"avg_{key}_all", None)
            if avg_var is not None:
                fld.add_field(comm, field_name=key, field=avg_var, dtype=np.single)
                if rank == 0:
                    print(f'Field {key} added to registry and fields directory in pos {fld.registry_pos[key]}')

    print(f"Rank {comm.rank} Fields after adding:", fld.registry.keys()) if rank == 0 else None
        
    # Write the data in a subdomain
    fout = "/home/woody/iwst/iwst115h/Work/writefiles/field/fieldout_test_field.f00001"
    # fout = "/home/woody/iwst/iwst115h/Work/writefiles/unstruct_new/fieldout_test_field.f00024"
    pynekwrite(fout, comm, msh=msh_3d_sub, fld=fld, write_mesh=True, wdsz=4) 
    print("Rank{rank} has done adding fields successfully and saved.") if rank == 0 else None

if check:    
    element_grouping()    
elif rank ==0:
    print(f"Error:: Invalid size: {size}")
    suggested_sizes  = find_valid_sizes(total_elements, num_elements_per_section)