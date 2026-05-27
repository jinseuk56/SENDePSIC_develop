# %%
import glob
import os
import hyperspy.api as hs
import matplotlib.pyplot as plt

# Import from the installed SENDePSIC package
from sendepsic import phase_analysis

# %%
cif_dir_path = ''  # Path to the directory containing structure files (.cif)
str_path = glob.glob(os.path.join(cif_dir_path, "*.cif"), recursive=True)
str_path.sort()
print("Found structures:")
print(*str_path, sep='\n')

target_structures = None
profile_path = ''  # Path to the profile data (e.g., loading vectors, averaged RMP, or RVP data)
print(f"Loading profiles from: {profile_path}")

profile = hs.load(profile_path)
print(profile)
print(profile.axes_manager)

num_sub = int(profile.axes_manager[0].name)
num_comp = int(profile.axes_manager[0].units)

profile_pixel_size = profile.axes_manager[1].scale
print(f"pixel = {profile_pixel_size} 1/Å")
profile = profile.data
print(f"Profile data shape: {profile.shape}")

crop_start, crop_end = 10, 63  # Crop values used in the radial profile analysis
prominence_profile = 0.00000001  # Small value threshold for detecting peaks in the profile data
color_rep = ["white", "black", "red", "green", "blue", "orange", "purple", "yellow", "lime", "cyan"]
max_num_peaks = 3  # Number of profile peaks to consider

profile_step_list = [0.0095, 0.00953]  # step size(s) of the profile data
xrd_step_list = [0.0001, 0.00001]      # Step size(s) for the simulated XRD pattern
xrd_broadening_list = [0.009, 0.01, 0.011, 0.012, 0.013]  # Broadening value(s) for the simulated XRD
xrd_prominence_list = [0.005]          # Prominence value(s) for peak finding

# %%
# Parameter initialization
run_analysis = phase_analysis()
run_analysis.phase_matching_initialization(
    str_path=str_path,
    target_structures=target_structures,
    profile=profile,
    num_comp=num_comp,
    num_sub=num_sub,
    crop=[crop_start, crop_end],
    color_rep=color_rep
)

# %%
# Plot the radial profile data by loading vector and find peaks
run_analysis.profile_comparison(profile_k_step=profile_pixel_size, max_num_peaks=max_num_peaks)

# %%
# Simulate an XRD pattern for a given structure file and compare it with the profile data
run_analysis.simple_comparsion(
    cif_adr=str_path[0], 
    profile_k_step=profile_pixel_size,
    max_num_peaks=max_num_peaks, 
    xrd_k_step=0.00001, 
    broadening=0.012, 
    xrd_peak_prominence=0.0001,
    ylim=None
)

# %%
# Calculate the Mean of Minimum Absolute Differences (MMAD) between nearest peak positions
run_analysis.MMAD_matching(
    profile_step_list=profile_step_list, 
    max_num_peaks=max_num_peaks, 
    xrd_step_list=xrd_step_list, 
    broadening_list=xrd_broadening_list, 
    xrd_peak_prominence_list=xrd_prominence_list
)

# %%
# 1. Match using Difference method (largest difference between 1st and 2nd rank)
run_analysis.best_match_diff()

# %%
# 2. Match using Frequency method (most frequent 1st rank across all parameter sets)
run_analysis.best_match_frequency()

# %%
# 3. Match using Median method (lowest median MMAD score)
run_analysis.best_match_median()

# %%
# Compare all matching results
run_analysis.compare_all_methods()

# %%
# Spatial distribution analysis and neighborhood phase mapping
pkl_adr = "/~/~.pkl"
print(f"Loading pickled RPA summary from: {pkl_adr}")

run_analysis.neighbor_phase_initialization(pkl_adr=pkl_adr)
run_analysis.area_analysis()

# %%
# Identify the nearest neighbors for each cluster by centroid distance calculations
global_hist_df, global_prox_df, sub_index_histograms, sub_index_proximities = run_analysis.closest_neighbor_analysis(
    prox_neighbors=3, plot_result=True
)

# Print/Display global trackers
print("Global Neighborhood Count:")
print(global_hist_df)
print("\nGlobal Proximity Percentages:")
print(global_prox_df)
