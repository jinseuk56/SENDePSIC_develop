# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import the pre-defined clustering and mapping helper functions from sendepsic.acom
from sendepsic.acom import map_3d_orientations

def plot_orientation_image(results):
    """Plots the final RGB image map, including black unindexed pixels."""
    rgb_image = results["rgb_image"]
    labels_grid = results["labels_grid"]
    
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    im = ax.imshow(rgb_image, origin='upper')
    
    legend_elements = []
    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
    
    # Plot the valid clusters
    if len(results["centers"]) > 0:
        for i in range(len(results["centers"])):
            c_rgb = results["cluster_rgbs"][i]
            c_name = results["cluster_names"][i]
            # Calculate percentage relative to the WHOLE image
            pct = (np.sum(labels_grid == i) / total_pixels) * 100
            legend_elements.append(Patch(
                facecolor=c_rgb, edgecolor='black', 
                label=f'{c_name} ({pct:.1f}%) ({results["centers"][i][0]:.2f}, {results["centers"][i][1]:.2f}, {results["centers"][i][2]:.2f})'
            ))
            
    # Calculate and plot the black (unindexed) pixels
    unindexed_pixels = np.sum(labels_grid == -1)
    if unindexed_pixels > 0:
        unindexed_pct = (unindexed_pixels / total_pixels) * 100
        threshold_val = results["threshold"]
        legend_elements.append(Patch(
            facecolor='black', edgecolor='gray', 
            label=f'Unindexed (<{threshold_val}) ({unindexed_pct:.1f}%)'
        ))
        
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Dominant Planes")
    ax.set_xlabel('X Scan (Pixels)')
    ax.set_ylabel('Y Scan (Pixels)')
    ax.set_title('ACOM Crystal Orientation Map')
    
    plt.tight_layout()
    plt.show()

# %%
# Load the ACOM result
with open('ACOM_result.pkl', "rb") as f:
    result = pickle.load(f)

# %%
str_key = ''
crystal_system = "cubic"
for key in result.keys():
    if "orientation_maps" in key:
        print(key)
        corr = np.mean(result[key][str_key].corr)
        print("correlation value: %.2f"%corr)
    if "image_orientation" in key:
        dominant_orientation = result[key][str_key]
        out_of_plane = dominant_orientation[:, :, :, 0]
        in_plane = dominant_orientation[:, :, :, 1]

        print("Dominant in-plane planes")
        # Process the data using imported function from sendepsic
        results = map_3d_orientations(
            in_plane, 
            num_clusters=3, 
            crystal_system=crystal_system, 
            color_style=crystal_system,
            threshold=0.1
        )

        # Plot it!
        plot_orientation_image(results)

        print("Dominant out-of-plane planes")
        # Process the data using imported function from sendepsic
        results = map_3d_orientations(
            out_of_plane, 
            num_clusters=3, 
            crystal_system=crystal_system, 
            color_style=crystal_system,
            threshold=0.1
        )

        # Plot it!
        plot_orientation_image(results)
