import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def assign_rgb_colors(cluster_centers, mapping_style='orthorhombic'):
    """Assigns RGB numerical values to cluster centers based on proximity to standard poles."""
    if mapping_style == 'cubic':
        reference_poles = {
            '~[100]': {'vector': normalize([[1.0, 0.0, 0.0]], norm='l2')[0], 'rgb': [1.0, 0.0, 0.0]}, 
            '~[110]': {'vector': normalize([[1.0, 1.0, 0.0]], norm='l2')[0], 'rgb': [0.0, 1.0, 0.0]}, 
            '~[111]': {'vector': normalize([[1.0, 1.0, 1.0]], norm='l2')[0], 'rgb': [0.0, 0.0, 1.0]}  
        }
    else:
        reference_poles = {
            '~[100]': {'vector': np.array([1.0, 0.0, 0.0]), 'rgb': [1.0, 0.0, 0.0]}, 
            '~[010]': {'vector': np.array([0.0, 1.0, 0.0]), 'rgb': [0.0, 1.0, 0.0]}, 
            '~[001]': {'vector': np.array([0.0, 0.0, 1.0]), 'rgb': [0.0, 0.0, 1.0]}  
        }

    cluster_rgbs = []
    cluster_names = []
    
    for center in cluster_centers:
        best_rgb = [0.5, 0.5, 0.5] 
        best_name = 'Unknown'
        highest_similarity = -1
        
        for name, data in reference_poles.items():
            similarity = np.dot(center, data['vector'])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_rgb = data['rgb']
                best_name = name

        cluster_rgbs.append(best_rgb)
        cluster_names.append(best_name)
        
    return cluster_rgbs, cluster_names


def map_3d_orientations(hkl_map_3d, num_clusters=3, crystal_system='orthorhombic', color_style='orthorhombic', threshold=0.01):
    """
    Accepts 3D array of shape (Y, X, 3) representing orientation vectors.
    Filters noise, folds vectors by symmetry, runs K-Means clustering, and assigns RGB colors.
    """
    hkl_map_3d = np.array(hkl_map_3d)
    Y_height, X_width, _ = hkl_map_3d.shape
    total_pixels = Y_height * X_width
    
    # Flatten the 3D grid into a 2D list of shape (N, 3)
    flat_indices = hkl_map_3d.reshape(-1, 3)
    
    # 1. Threshold Masking
    magnitudes = np.linalg.norm(flat_indices, axis=1)
    valid_mask = magnitudes >= threshold
    valid_indices = flat_indices[valid_mask]
    
    if len(valid_indices) == 0:
        return {
            "rgb_image": np.zeros((Y_height, X_width, 3)),
            "labels_grid": np.full((Y_height, X_width), -1),
            "cluster_rgbs": [], "cluster_names": [], "centers": [], "threshold": threshold
        }

    # 2. Symmetry Folding
    if crystal_system.lower() == 'cubic':
        folded_indices = np.abs(valid_indices)
        folded_indices = np.sort(folded_indices, axis=1)[:, ::-1]
    elif crystal_system.lower() == 'tetragonal':
        folded_indices = np.abs(valid_indices)
        hk_sorted = np.sort(folded_indices[:, :2], axis=1)[:, ::-1]
        folded_indices = np.column_stack((hk_sorted, folded_indices[:, 2]))
    elif crystal_system.lower() == 'orthorhombic':
        folded_indices = np.abs(valid_indices)
    else:
        raise ValueError("Unsupported crystal system.")

    # 3. Normalize and Cluster
    normalized_vectors = normalize(folded_indices, norm='l2', axis=1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    valid_labels = kmeans.fit_predict(normalized_vectors)
    final_centers = normalize(kmeans.cluster_centers_, norm='l2', axis=1)
    
    rgbs, names = assign_rgb_colors(final_centers, mapping_style=color_style)
    
    # 4. Reconstruct the Full Image
    flat_pixel_colors = np.zeros((total_pixels, 3)) 
    flat_labels = np.full(total_pixels, -1)         
    
    valid_pixel_colors = np.array([rgbs[label] for label in valid_labels])
    flat_pixel_colors[valid_mask] = valid_pixel_colors
    flat_labels[valid_mask] = valid_labels
    
    rgb_image = flat_pixel_colors.reshape(Y_height, X_width, 3)
    labels_grid = flat_labels.reshape(Y_height, X_width)
    
    return {
        "rgb_image": rgb_image,
        "labels_grid": labels_grid,
        "cluster_rgbs": rgbs,
        "cluster_names": names,
        "centers": final_centers,
        "threshold": threshold
    }


def acom_orientation_summary(pkl_path, crystal_system='cubic', num_clusters=3, threshold=0.1):
    """
    Loads ACOM result pickle file and clusters orientations to provide a Markdown summary report.
    """
    if not os.path.exists(pkl_path):
        return f"# ACOM Orientation Summary\n\n*Error: ACOM result file not found at `{pkl_path}`.*"
        
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
        
    report = []
    report.append("# ACOM Dominant Orientation Summary Report\n")
    report.append(f"- **Result File**: `{os.path.basename(pkl_path)}`")
    report.append(f"- **Crystal System**: `{crystal_system}`")
    report.append(f"- **Number of Clusters**: {num_clusters}")
    report.append(f"- **Intensity Threshold**: {threshold}\n")
    
    for key in result.keys():
        if "image_orientation" in key:
            dataset_name = key.replace("_image_orientation", "")
            report.append(f"## Dataset: `{dataset_name}`")
            
            for str_key in result[key].keys():
                dominant_orientation = result[key][str_key]
                report.append(f"### CIF Phase Structure: `{str_key}`")
                
                # dominant_orientation shape is (Y, X, 3, 2)
                # index 0 is out-of-plane, index 1 is in-plane
                out_of_plane = dominant_orientation[:, :, :, 0]
                in_plane = dominant_orientation[:, :, :, 1]
                
                for dir_name, dir_data in [("In-Plane", in_plane), ("Out-of-Plane", out_of_plane)]:
                    res = map_3d_orientations(dir_data, num_clusters=num_clusters, crystal_system=crystal_system, color_style=crystal_system, threshold=threshold)
                    
                    report.append(f"#### {dir_name} Direction:")
                    centers = res["centers"]
                    names = res["cluster_names"]
                    labels_grid = res["labels_grid"]
                    total_pixels = labels_grid.size
                    
                    if len(centers) == 0:
                        report.append("- *No pixels passed the intensity threshold.*")
                        continue
                        
                    for c_idx in range(len(centers)):
                        pct = (np.sum(labels_grid == c_idx) / total_pixels) * 100
                        c_vec = centers[c_idx]
                        c_vec_str = f"[{c_vec[0]:.2f}, {c_vec[1]:.2f}, {c_vec[2]:.2f}]"
                        report.append(f"- **Cluster {c_idx+1}**: Pole `{names[c_idx]}` (Center Vector: `{c_vec_str}`) - **{pct:.1f}%** area")
                    
                    unindexed_pixels = np.sum(labels_grid == -1)
                    if unindexed_pixels > 0:
                        unindexed_pct = (unindexed_pixels / total_pixels) * 100
                        report.append(f"- **Unindexed (Noise/Void)**: **{unindexed_pct:.1f}%**")
                    report.append("")
            report.append("---")
            
    return "\n".join(report)
