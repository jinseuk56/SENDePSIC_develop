# %%
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM
import hyperspy.api as hs

gpu = True  # True only if a NVIDIA gpu is available and CUDA+Cupy is installed

# %%
dir_path = ''
data_adr = glob.glob(os.path.join(dir_path, '*/*_corrected_scaled.hspy'), recursive=True)
data_adr.sort()
print("Found datasets:")
print(*data_adr, sep="\n")

cif_paths = [""]

detect_params = {
    'corrPower': 1.0,
    'sigma': 0,
    'edgeBoundary': 2,
    'minRelativeIntensity': 0,
    'minAbsoluteIntensity': 0.1,
    'minPeakSpacing': 5,
    'subpixel': 'poly',
    'upsample_factor': 4,
    'maxNumPeaks': 20,
    'CUDA': gpu
}

width = 300  # 515
top, left = 110, 140  # 0, 0
crop_ind = [top, top+width, left, left+width] 

# %%
result = {}
for adr in data_adr:
    dp_data = hs.load(adr)
    print(f"\nProcessing: {adr}")
    print(dp_data)

    dp_shape = dp_data.data.shape[2:]
    vir_y, vir_x = dp_data.data.shape[:2]

    datacube = dp_data.data[:vir_y, :vir_x, crop_ind[0]:crop_ind[1], crop_ind[2]:crop_ind[3]]
    datacube = py4DSTEM.DataCube(datacube)

    # Diffraction space calibration
    datacube.calibration.set_Q_pixel_size(dp_data.axes_manager[2].scale)
    datacube.calibration.set_Q_pixel_units('A^-1')

    # Real space calibration
    datacube.calibration.set_R_pixel_size(dp_data.axes_manager[0].scale)
    datacube.calibration.set_R_pixel_units('nm')

    # Try making a synthetic probe
    syn_probe_rad = 3
    syn_probe_width = 3
    probe_kernel_pre = 0.5
    probe_kernel_post = 5

    syn_probe = py4DSTEM.braggvectors.probe.Probe.generate_synthetic_probe(
        syn_probe_rad, syn_probe_width, (datacube.data.shape[-1], datacube.data.shape[-1])
    )

    # Construct a probe template to use as a kernel for correlation disk detection
    syn_probe_kernel = syn_probe.get_kernel(
        mode='sigmoid',
        radii=(syn_probe_rad * probe_kernel_pre, syn_probe_rad * probe_kernel_post),
        bilinear=True,
    )

    bragg_peaks = datacube.find_Bragg_disks(
        template=syn_probe_kernel,
        **detect_params,
    )

    qxy_origins = bragg_peaks.measure_origin()
    qx0_fit, qy0_fit, qx0_residuals, qy0_residuals = bragg_peaks.fit_origin(
        plot=False,
        figsize=(4, 4)
    )

    k_max = 0.7  # the scattering vector range [0, k_max] will be considered
    orientation_maps = {}
    image_orientations = {}
    for i, cif_path in enumerate(cif_paths):
        crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(cif_path)
        crystal.calculate_structure_factors(k_max)

        crystal.orientation_plan(
            zone_axis_range=np.array([[0,1,1], [1,1,1]]),
            angle_step_zone_axis=1.0,
            angle_step_in_plane=1.0,
            accel_voltage=300e3,
            CUDA=gpu,
        )

        orientation_map = crystal.match_orientations(
            bragg_peaks,
            num_matches_return=1,
            min_number_peaks=3,
        )

        images_orientation = crystal.plot_orientation_maps(
            orientation_map,
            orientation_ind=0,
            corr_range=np.array([0, 5]),
            camera_dist=10,
            show_axes=False,
        )
        orientation_maps[os.path.basename(cif_path).split(".")[0]] = orientation_map
        image_orientations[os.path.basename(cif_path).split(".")[0]] = images_orientation

    result[os.path.basename(adr).split(".")[0]+"_orientation_maps"] = orientation_maps
    result[os.path.basename(adr).split(".")[0]+"_image_orientation"] = image_orientations

# %%
# Save the ACOM result
with open('ACOM_result.pkl', 'wb') as f:
    pickle.dump(result, f)
