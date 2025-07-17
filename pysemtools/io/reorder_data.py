from mpi4py import MPI
import numpy as np
from ..datatypes.msh import Mesh
from ..datatypes.field import FieldRegistry
from .ppymech.neksuite import pynekread
from .wrappers import read_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def generate_data(fname_3d: str, fname_out: str):
    """
    Loads and processes 3D field data.

    This function:
    - Reads mesh and field data from `fname_3d`.
    - Extracts global coordinates of elements.
    - Prepares structured mesh and field registry for further computations.

    Args:
        fname_3d (str): Path to the input 3D field file.
        fname_out (str): Path to the output processed field file.

    Returns:
        assigned_data (np.ndarray): Global coordinates of elements.
        fld_3d_r (FieldRegistry): Registry containing field mappings.
        msh_3d (Mesh): Mesh object.
    """

    msh_3d = Mesh(comm, create_connectivity=False)
    pynekread(fname_3d, comm, data_dtype=np.single, msh=msh_3d)

    fld_3d_r = FieldRegistry(comm)
    pynekread(fname_3d, comm, data_dtype=np.single, fld=fld_3d_r)

    pynekwrite(fname_out, comm, msh=msh_3d, fld=fld_3d, write_mesh=True, wdsz=4)

    data = read_data(comm, fname_out, ["x", "y", "z", "scal_0"], dtype = np.single)

    # Extract global coordinates of elements
    x_global, y_global, z_global = data["x"], data["y"], data["z"]
    size_data = x_global.shape[0]

    subset_x, subset_y, subset_z = (arr[:size_data] for arr in (x_global, y_global, z_global))
    
    assigned_data = np.stack((subset_x, subset_y, subset_z), axis=-1)
    
    return assigned_data, fld_3d_r, msh_3d

# Methods for cal. the difference and normal vectors
def calculate_face_differences(assigned_data):
    """
    Computes face difference vectors for each element.

    The function calculates:
    - r_diff: Difference between front and back faces.
    - s_diff: Difference between top and bottom faces.
    - t_diff: Difference between left and right faces.

    Args:
        assigned_data (np.ndarray): Array containing element coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Normalized difference vectors for r, s, and t faces.
    """
    r_diff = np.mean(assigned_data[:, :, :, 0, :] - assigned_data[:, :, :, -1, :], axis=(1, 2))  # Front vs. Back, w/o reduction, shape (N,8,8,3) 
    r_diff = r_diff/np.linalg.norm(r_diff, axis=1, keepdims=True)
    s_diff = np.mean(assigned_data[:, :, 0, :, :] - assigned_data[:, :, -1, :, :], axis=(1, 2))  # Top vs. Bottom, w/o reduction, shape (N,8,8,3)
    s_diff = s_diff/np.linalg.norm(s_diff, axis=1, keepdims=True)
    t_diff = np.mean(assigned_data[:, 0, :, :, :] - assigned_data[:, -1, :, :, :], axis=(1, 2))  # Left vs. Right, w/o reduction, shape (N,8,8,3)
    t_diff = t_diff/np.linalg.norm(t_diff, axis=1, keepdims=True)
    return r_diff, s_diff, t_diff

def calculate_face_normals(assigned_data):
    """
    Computes normal vectors for each face of an element.

    Extracts:
    - Front and back face normals.
    - Left and right face normals.
    - Top and bottom face normals.

    Args:
        assigned_data (np.ndarray): Array containing element coordinates.

    Returns:
        np.ndarray: Array containing normal vectors for each face (Ex: shape [num_elements, 6, 3]).
    """
    normals = np.zeros((assigned_data.shape[0], 6, 3))  # 6 faces per element

    mid = (assigned_data.shape[1]) // 2
    offset = 1 if mid % 2 == 0 else 2

    # Normals for the r (front/back)
    front_edges1 = assigned_data[:, mid, mid, 0, :] - assigned_data[:, mid + offset, mid + offset, 0, :]
    front_edges2 = assigned_data[:, mid, mid + offset, 0, :] - assigned_data[:, mid + offset, mid, 0, :]

    back_edges1 = assigned_data[:, mid, mid, -1, :] - assigned_data[:, mid + offset, mid + offset, -1, :]
    back_edges2 = assigned_data[:, mid, mid + offset, -1, :] - assigned_data[:, mid + offset, mid, -1, :]

    normals[:, 0, :] = np.cross(front_edges1, front_edges2)
    normals[:, 1, :] = np.cross(back_edges1, back_edges2)

    # Normals for the t (left/right)
    left_edges1 = assigned_data[:, 0, mid, mid, :] - assigned_data[:, 0, mid + offset, mid + offset, :]
    left_edges2 = assigned_data[:, 0, mid, mid + offset, :] - assigned_data[:, 0, mid + offset, mid, :]

    right_edges1 = assigned_data[:, -1, mid, mid, :] - assigned_data[:, -1, mid + offset, mid + offset, :]
    right_edges2 = assigned_data[:, -1, mid, mid + offset, :] - assigned_data[:, -1, mid + offset, mid, :]

    normals[:, 4, :] = np.cross(left_edges1, left_edges2)
    normals[:, 5, :] = np.cross(right_edges1, right_edges2)

    # Normals for the s (top/bottom)
    top_edges1 = assigned_data[:, mid, -1, mid, :] - assigned_data[:, mid + offset, -1, mid + offset, :]
    top_edges2 = assigned_data[:, mid, -1, mid + offset, :] - assigned_data[:, mid + offset, -1, mid, :]

    bottom_edges1 = assigned_data[:, mid, 0, mid, :] - assigned_data[:, mid + offset, 0, mid + offset, :]
    bottom_edges2 = assigned_data[:, mid, 0, mid + offset, :] - assigned_data[:, mid + offset, 0, mid, :]

    normals[:, 2, :] = np.cross(top_edges1, top_edges2)
    normals[:, 3, :] = np.cross(bottom_edges1, bottom_edges2)

    return normals

def calculate_face_averages(assigned_data):
    """
    Similar to calculate_face_differences, just for re-verification. It computes the average difference vectors for each face.

    The function calculates:
    - r_avg_diff: Averaged difference between front and back faces.
    - s_avg_diff: Averaged difference between top and bottom faces.
    - t_avg_diff: Averaged difference between left and right faces.

    Args:
        assigned_data (np.ndarray): Array containing element coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Normalized average difference vectors for r, s, and t faces.
    """
    r_avg_diff = np.mean(assigned_data[:, :, :, 0, :], axis=(1, 2)) - np.mean(assigned_data[:, :, :, -1, :], axis=(1, 2)) # Front vs. Back
    r_avg_diff = r_avg_diff/np.linalg.norm(r_avg_diff, axis=1, keepdims=True)
    s_avg_diff = np.mean(assigned_data[:, :, 0, :, :], axis=(1, 2)) - np.mean(assigned_data[:, :, -1, :, :], axis=(1, 2)) # Top vs. Bottom
    s_avg_diff = s_avg_diff/np.linalg.norm(s_avg_diff, axis=1, keepdims=True)
    t_avg_diff = np.mean(assigned_data[:, 0, :, :, :], axis=(1, 2)) - np.mean(assigned_data[:, -1, :, :, :], axis=(1, 2)) # Left vs. Right
    t_avg_diff = t_avg_diff/np.linalg.norm(t_avg_diff, axis=1, keepdims=True)
    return r_avg_diff, s_avg_diff, t_avg_diff

# Method: 1
def face_mappings(cal_r, cal_s, cal_t):
    """
    Determines face mappings based on directional differences.

    This function:
    - Assigns maps local coordinate axes (r,s,t) to global coordinate axes (X,Y,Z).
    - Determines mappings based on the dominant direction per face.
    - Ensures unique assignments to avoid duplication.

    Args:
        cal_r (np.ndarray): Difference vectors for r faces.
        cal_s (np.ndarray): Difference vectors for s faces.
        cal_t (np.ndarray): Difference vectors for t faces.

    Returns:
        List[Tuple[str, str, str]]: Face mappings for each element.
    """
    num_elements = cal_r.shape[0]
    mappings = []

    for elem in range(num_elements):
         # Combine into a matrix for looping
        reduced_data = np.array([np.abs(cal_r[elem]), np.abs(cal_s[elem]), np.abs(cal_t[elem])])

        mapping = [""] * 3
        directions = ['x', 'y', 'z']
        assigned = [False] * 3 # Track assigned directions

        # Find the direction with the largest diff for each r, s, and t in the above-combined matrix
        for i in range(3):
            max_dir = np.argmax(reduced_data[i]) # here we get index of max value

            # avoid repetition of looking in the same index again and again
            while max_dir < len(directions) and assigned[max_dir]:
                reduced_data[i, max_dir] = -np.inf  # Invalidate this
                max_dir = np.argmax(reduced_data[i])

            if max_dir < len(directions):
                mapping[i] = directions[max_dir]
                assigned[max_dir] = True
            else:
                print(f"Error: max_dir ({max_dir}) out of range for directions list.")

        if not all(assigned):
            print(f"Unassigned directions in element {elem}, please check data.")
        else:
            mappings.append(tuple(mapping))
    return mappings

# Compare all methods:
def compare_mappings(mappings, ref_mapping, method: str):
    """
    Compares computed face mappings against a reference mapping.

    Args:
        mappings (List[Tuple[str, str, str]]): Computed face mappings.
        ref_mapping (Tuple[str, str, str]): Reference mapping for comparison.
        method (str): Name of the method for logging results.

    Returns:
        None: Prints the percentage of matching mappings.
    """
    num_elements = len(mappings)
    match_count = 0
    for i in range(num_elements):
        if mappings[i] == ref_mapping:
            match_count += 1
            # print(f"Element {i} matches the reference mapping.")
        else:
            # print(f"Mismatch at Element {i}: {mappings_fa[i]}")
            continue
                
    # print(f"Rank {rank} has total matches: {match_count}")
    matching_percentage = (match_count / num_elements) * 100
    # print(f"Rank {rank} has matching percentage of {method} w.r.t ref_mapping_fd {ref_mapping}: {matching_percentage:.2f}%")

def compare_methods(mappings_fd, mappings_fa):
    """
    Compares face difference-based mappings and face average-based mappings.

    Args:
        mappings_fd (List[Tuple[str, str, str]]): Face difference-based mappings.
        mappings_fa (List[Tuple[str, str, str]]): Face average-based mappings.

    Returns:
        None: Prints the percentage of matching mappings.
    """
    num_elements = len(mappings_fd)
    match_count =0  
    for i in range(num_elements): 
        if mappings_fd[i] == mappings_fa[i]:
            match_count += 1
        else: 
            continue
            # print(" Mismatch found.")
            # print(f" Face Diff Mapping: Element {i}: r -> {mappings_fd[i][0]}, s -> {mappings_fd[i][1]}, t -> {mappings_fd[i][2]}") 
            # print(f" Face Avg Mapping: Element {i}: r -> {mappings_fa[i][0]}, s -> {mappings_fa[i][1]}, t -> {mappings_fa[i][2]}") 
                
    matching_percentage = (match_count / num_elements) * 100
    # print(f"Rank {rank} has matching Percentage b/w mappings_fd & mappings_fa: {matching_percentage:.2f}%")
    
# Method-2: Face Normals
def align_normals(normals, normals_pair, direction_type):
    """
    Aligns normal vectors for paired faces.

    - Ensures consistency in direction for paired faces.
    - Uses a reference normal based on `direction_type` for alignment.

    Args:
        normals (np.ndarray): Array of normal vectors (shape: [num_elements, 6, 3]).
        normals_pair (np.ndarray): Array of paired face normals.
        direction_type (str): Face orientation type ('front_back', 'left_right', or 'top_bottom').

    Returns:
        np.ndarray: Adjusted paired normal vectors ensuring correct orientation.
    """
    # reference normal based on direction_type
    if direction_type == "front_back":
        reference_normal = normals[0, 0, :]
    elif direction_type == "left_right":
        reference_normal = normals[0, 4, :]  
    elif direction_type == "top_bottom":
        reference_normal = normals[0, 2, :] 
    else:
        raise ValueError("Invalid direction_type. Must be one of 'front_back', 'left_right', 'top_bottom'.")

    # align paired face normals within each element
    dot_products_pair = np.einsum('ij,ij->i', normals_pair[:, 0, :], normals_pair[:, 1, :])    
    # flip normals
    flip_mask_pair = dot_products_pair < 0
    normals_pair[flip_mask_pair, 1, :] *= -1

    # align with the reference normal
    dot_products_ref_pair = np.einsum('j,ij->i', reference_normal, normals_pair[:, 0, :])    
    # flip normals
    flip_mask_ref_pair = dot_products_ref_pair < 0
    normals_pair[flip_mask_ref_pair, 0, :] *= -1
    normals_pair[flip_mask_ref_pair, 1, :] *= -1
    return normals_pair

def reduce_face_normals(normals):
    """
    Computes reduced normals for face pairs by ensuring consistency and averaging.

    This function:
    - Separates face normals into paired groups (front/back, left/right, top/bottom).
    - Aligns normals to maintain consistent orientation.
    - Normalizes the face normal vectors.
    - Computes averaged normals for each face pair.

    Args:
        normals (np.ndarray): Array of normal vectors (shape: [num_elements, 6, 3]).

    Returns:
        np.ndarray: Averaged and aligned normals for each element (shape: [num_elements, 3, 3]).
    """
    # Separate normals by face pairs
    front_back_normals = normals[:, [0, 1], :] #r
    left_right_normals = normals[:, [4, 5], :] #t
    top_bottom_normals = normals[:, [2, 3], :] #s

    # consistent orientation
    front_back_normals = align_normals(normals, front_back_normals, "front_back") 
    left_right_normals = align_normals(normals, left_right_normals, "left_right") 
    top_bottom_normals = align_normals(normals, top_bottom_normals, "top_bottom")

    # Normalize the face normals
    front_back_normals = front_back_normals / np.linalg.norm(front_back_normals, axis=2, keepdims=True)
    left_right_normals = left_right_normals / np.linalg.norm(left_right_normals, axis=2, keepdims=True)
    top_bottom_normals = top_bottom_normals / np.linalg.norm(top_bottom_normals, axis=2, keepdims=True)

    # Compute the averages
    averaged_front_back = np.mean(front_back_normals, axis=1)
    averaged_left_right = np.mean(left_right_normals, axis=1)
    averaged_top_bottom = np.mean(top_bottom_normals, axis=1)
    # Stack the averaged normals
    averaged_normals = np.stack([averaged_front_back, averaged_top_bottom, averaged_left_right,], axis=1)
     
    return averaged_normals
    
def directions_face_normals(averaged_normals):
    """
    Determines face normal mappings and flow directions.

    This function:
    - Assigns maps local coordinate axes (r,s,t) to global coordinate axes (X,Y,Z) based on dominant normal directions.
    - Ensures unique directional assignment per element.
    - Identifies the principal flow direction for each element.

    Args:
        averaged_normals (np.ndarray): Array of averaged face normals (shape: [num_elements, 3, 3]).

    Returns:
        Tuple[List[Tuple[str, str, str]], List[int]]: Face mappings and associated flow directions.
    """
    num_elements = averaged_normals.shape[0]
    mappings = []
    flow_directions = []

    for elem in range(num_elements):
        face_normals = np.abs(averaged_normals[elem])  # Shape: (3, 3)
        mapping = [""] * 3
        directions = ['x', 'y', 'z']        
        assigned = [False] * 3
        flow_direction_set = False
            
        for i in range(3):  # Iterate over the three pairs of normals
            max_dir = np.argmax(face_normals[i])  # here we get index of max value
                
         # Ensure unique assignment
            while max_dir < len(directions) and assigned[max_dir]:
                face_normals[i][max_dir] = -np.inf  # Invalidate this direction
                max_dir = np.argmax(face_normals[i])

            if max_dir < len(directions):
                mapping[i] = directions[max_dir]
                assigned[max_dir] = True

            if assigned[2] and not flow_direction_set:
                flow_direction = i 
                flow_direction_set = True 
                # to prevent overwriting 
                flow_directions.append(flow_direction)
                    
        if not all(assigned):
            print(f"Unassigned directions in element {elem}, please check data.")
        else:
            mappings.append(tuple(mapping))
                
    return mappings, flow_directions

# Reordering of subset
def reorder_assigned_data(assigned_data, mappings_fd, ref_mapping_fd):
    """
    Reorders assigned data to match a reference mapping.

    This function:
    - Identifies mismatches between computed mappings and the reference mapping.
    - Applies permutation swaps to correct element order.
    - Ensures consistency in face assignments across all elements.

    Args:
        assigned_data (np.ndarray): Array of element data.
        mappings_fd (List[Tuple[str, str, str]]): Computed face difference mappings.
        ref_mapping_fd (Tuple[str, str, str]): Reference mapping from rank 0.

    Returns:
        np.ndarray: Reordered element data matching the reference mapping.
    """
    reference_mapping = ref_mapping_fd  # take reference mapping from rank 0->first element
    num_elements = assigned_data.shape[0]
    dimension = assigned_data.shape[1]

    for elem in range(num_elements):
        if mappings_fd[elem] != reference_mapping:
            k = np.copy(assigned_data[elem])

            # Get the current order and reference order
            current_order = mappings_fd[elem]
            ref_order = reference_mapping

            # Find permutation needed to match reference order
            order_map = {axis: idx for idx, axis in enumerate(current_order)}
            ref_indices = [order_map[axis] for axis in ref_order]

            # Apply swaps based on required permutation
            if ref_indices == [1, 0, 2]:  # Swap first two axes
                for i in range(dimension):
                    assigned_data[elem, :, i, :, :] = k[:, :, i, :]      # s_diff = r_diff                    
                    assigned_data[elem, :, :, i, :] = k[:, i, :, :]      # r_diff = s_diff
                assigned_data[elem] = np.flip(assigned_data[elem], axis=2)

            elif ref_indices == [2, 0, 1]:  # Full cycle shift
                for i in range(dimension):
                    assigned_data[elem, :, :, i, :] = k[:, i, :, :]      # r_diff = s_diff
                    assigned_data[elem, :, i, :, :] = k[i, :, :, :]      # s_diff = t_diff
                    assigned_data[elem, i, :, :, :] = k[:, :, i, :]      # t_diff = r_diff

            elif ref_indices == [0, 2, 1]:  # Swap last two axes
                for i in range(dimension):
                    assigned_data[elem, :, i, :, :] = k[i, :, :, :]      # s_diff = t_diff
                    assigned_data[elem, i, :, :, :] = k[:, i, :, :]      # t_diff = s_diff
                
            elif ref_indices == [1, 2, 0]:  # Full reverse cycle
                for i in range(dimension):
                    assigned_data[elem, :, :, i, :] = k[i, :, :, :]      # r_diff = t_diff
                    assigned_data[elem, :, i, :, :] = k[:, :, i, :]      # s_diff = r_diff
                    assigned_data[elem, i, :, :, :] = k[:, i, :, :]      # t_diff = s_diff

            elif ref_indices == [2, 1, 0]:  # Swap first and last axes
                for i in range(dimension):
                    assigned_data[elem, :, :, i, :] = k[i, :, :, :]      # r_diff = t_diff
                    assigned_data[elem, i, :, :, :] = k[:, :, i, :]      # t_diff = s_diff
            elif ref_indices == [0, 1, 2]:
                continue
    return assigned_data

def correct_axis_signs(assigned_data_new, extract_reference_r, extract_reference_s, extract_reference_t, r_diff_new, s_diff_new, t_diff_new):
    """
    Adjusts the axis signs based on a reference element.

    This function:
    - Computes sign consistency using dot products.
    - Flips axes if mismatched with the reference element.
    - Ensures all elements conform to the correct axis orientation.

    Args:
        assigned_data_new (np.ndarray): Reordered element data.
        extract_reference_r (np.ndarray): Reference r-axis direction.
        extract_reference_s (np.ndarray): Reference s-axis direction.
        extract_reference_t (np.ndarray): Reference t-axis direction.
        r_diff_new (np.ndarray): Updated r-axis difference vectors.
        s_diff_new (np.ndarray): Updated s-axis difference vectors.
        t_diff_new (np.ndarray): Updated t-axis difference vectors.

    Returns:
        np.ndarray: Corrected element data with aligned axes.
    """
    assigned_data_final = np.copy(assigned_data_new)

    num_elements = assigned_data_final.shape[0]
    for elem in range(num_elements):
        r_sign = np.sign(np.dot(r_diff_new[elem], extract_reference_r))
        s_sign = np.sign(np.dot(s_diff_new[elem], extract_reference_s))
        t_sign = np.sign(np.dot(t_diff_new[elem], extract_reference_t))

        # Correcting signs based on reference element
        if r_sign < 0:
            assigned_data_final[elem] = np.flip(assigned_data_final[elem], axis=2)  # Flip r-axis
        if s_sign < 0:
            assigned_data_final[elem] = np.flip(assigned_data_final[elem], axis=1)  # Flip s-axis
        if t_sign < 0:
            assigned_data_final[elem] = np.flip(assigned_data_final[elem], axis=0)  # Flip t-axis

    return assigned_data_final