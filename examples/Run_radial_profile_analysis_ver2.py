# %% [markdown]
# # Atomic structure analysis with radial (azimuthal) mean & variance profile data
# ### Jinseok Ryu (jinseuk56@gmail.com)
# Developed for the ePSIC data processing workflow.
# Required Python packages: scipy, numpy, matplotlib, py4DSTEM, hyperspy, scikit-learn, shapely, sendepsic

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.signal import find_peaks

# Import from the installed SENDePSIC package
from sendepsic import radial_profile_analysis

# %%
# Configure data loading
base_dir = '/dls/e02/data/2025/mgXXXXX-X'
subfolders = ['']  # subfolder names you want to load and compare, e.g., ['sub1', 'sub2']
final_dir = None   # (optional) folder name where the data is stored in

profile_length = 360  # limit the profile size
num_load = 2          # limit the number of data for every subfolder (select files randomly)

include_key = []  # keyword (datetime) for screening (to only include the specified data)
exclude_key = []  # keyword (datetime) for screening (to exclude poor quality data)

# Instantiate analysis object
run_analysis = radial_profile_analysis(
    base_dir, subfolders, 
    profile_length, num_load, final_dir,
    include_key, exclude_key,
    simult_edx=False, roll_axis=True, verbose=False
)

# %%
plt.close("all")
# Transformation quality check (center beam alignment)
run_analysis.center_beam_alignment_check(crop=[0, -1, 0, -1], visual_title=True, title_font_size=10)

# %%
plt.close("all")
# Intensity integration image (BF + DF)
run_analysis.intensity_integration_image(visual_title=True, title_font_size=10)

# %%
# Setup structure files for simulating diffraction patterns (XRD) (optional)
str_path = []  # structure paths to compare, e.g., ['path1', 'path2']

# Specify the scattering vector range (used in NMF decomposition and plotting)
from_unit = 0.2  # unit: 1/angstrom
to_unit = 0.6    # unit: 1/angstrom
run_analysis.basic_setup(str_path, from_unit, to_unit, broadening=0.01, fill_width=0.005)

# %%
plt.close("all")
# Sum of radial variance and average profiles
run_analysis.sum_radial_profile(str_name=[], profile_type="variance", visual_legend=False, individual_visual=False)

# %% [markdown]
# # NMF decomposition

# %%
# Optimize the number of loading vectors (NMF)
# Performs calculations in-memory to avoid redundant disk-reads
error_list = []
comp_list = []
num_comp_list = np.arange(2, 15, 1)

# Run initial decomposition
run_analysis.NMF_decompose(
    num_comp_list[0], profile_type="variance", 
    scale_crop=True, rescale_SI=False, max_normalize=False, rescale_0to1=True, 
    verbose=False, coeff_map_type="relative"
)
error_list.append(run_analysis.run_SI.DR.reconstruction_err_)
comp_list.append(run_analysis.run_SI.DR.components_)

# Iteratively evaluate components
for num_comp in num_comp_list[1:]:
    print(f'Evaluating NMF with components: {num_comp}')
    run_analysis.run_SI.ini_DR(method="nmf", num_comp=num_comp, result_visual=False, intensity_range="relative")
    error_list.append(run_analysis.run_SI.DR.reconstruction_err_)
    comp_list.append(run_analysis.run_SI.DR.components_)

# Plot the errors and slope gradient to identify optimal number of components
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=100)
ax[0].plot(num_comp_list, error_list, 'k-', marker='*')
ax[0].set_xlabel("Number of loading vectors")
ax[0].set_ylabel("Reconstruction Error")
ax[0].set_title("NMF Reconstruction Error")

slope = np.gradient(error_list)
ax[1].plot(num_comp_list, error_list, 'r-', marker='*')
ax[1].set_xlabel("Number of loading vectors")
ax[1].set_ylabel("Reconstruction Error", color='r')

ax_twin = ax[1].twinx()
ax_twin.plot(num_comp_list, slope, 'b-', marker='o')
ax_twin.set_xticks(num_comp_list)
ax_twin.set_ylabel("Gradient", color='b')
ax_twin.set_title("Error Gradient analysis")

fig.tight_layout()
plt.show()

# %%
plt.close("all")
# Run NMF decomposition with the chosen optimal number of components
num_comp = 8
run_analysis.NMF_decompose(
    num_comp, profile_type="variance", scale_crop=True, rescale_SI=False,
    max_normalize=False, rescale_0to1=True, verbose=False, coeff_map_type="relative"
)

# %%
plt.close("all")
# Plot NMF loading vectors and corresponding coefficient maps
run_analysis.NMF_result(lv_show=None, visual_title=True, title_font_size=8)

# %%
plt.close("all")
# Find pixels with high coefficients and compute their average profiles
by_nmf_lv = run_analysis.NMF_comparison(str_name=[], percentile_threshold=90, visual_individual=False)

# %%
# Find and plot peaks in the averaged profiles
plt.close("all")
fill_width = 0.001
prominence_profile = 0.0005

fig_lv, ax_lv = plt.subplots(figsize=(7, 4.5), dpi=100)
for l, line in enumerate(by_nmf_lv):
    line = line[run_analysis.range_ind[0]:run_analysis.range_ind[1]].copy()
    peaks = find_peaks(line, prominence=prominence_profile)[0]
    peaks = peaks * run_analysis.pixel_size_inv_Ang
    peaks = peaks + run_analysis.from_
    
    tmp_ax, = ax_lv.plot(run_analysis.x_axis, line, c=run_analysis.color_rep[l+1], label=f'LV {l+1}')
    shadow_effect = path_effects.withStroke(linewidth=3, foreground='gray')
    tmp_ax.set_path_effects([shadow_effect])

    for ip, peak in enumerate(peaks):
        if run_analysis.from_ <= peak <= run_analysis.to_:
            print(f"LV {l+1} Peak: {peak:.4f}")
            ax_lv.axvline(peak, ls=':', lw=1.5, c=run_analysis.color_rep[l+1], alpha=0.5)
            ax_txt = ax_lv.text(peak, np.max(line) * 0.7, f"{peak:.3f}", c=run_analysis.color_rep[l+1], fontsize=7)
            shadow_effect_txt = path_effects.withStroke(linewidth=1, foreground='gray')
            ax_txt.set_path_effects([shadow_effect_txt])           

ax_lv.set_xlabel("1/Å")
ax_lv.set_ylabel("Normalized Intensity")
ax_lv.set_title("Peaks in Average Radial Profiles")
ax_lv.legend(loc='upper right')
fig_lv.tight_layout()
plt.show()
