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
data_adr = ''
print(f"Data Address: {data_adr}")

cif_paths = [""]

dp_data = hs.load(data_adr)
print(dp_data)
print(dp_data.axes_manager)

# %%
dp_shape = dp_data.data.shape[1:]
num_img = dp_data.data.shape[0]
print(f"Number of images: {num_img}")
grid_size = int(np.sqrt(num_img))
vir_y, vir_x = grid_size, grid_size
width = 300  # 515
top, left = 110, 140  # 0, 0
crop_ind = [top, top+width, left, left+width]  # Limit the diffraction region

datacube = np.asarray(dp_data.data[:vir_y*vir_x]).reshape(vir_y, vir_x, dp_shape[0], dp_shape[1])
datacube = datacube[:, :, crop_ind[0]:crop_ind[1], crop_ind[2]:crop_ind[3]]
datacube = py4DSTEM.DataCube(datacube)
print(datacube)

# Diffraction space calibration
datacube.calibration.set_Q_pixel_size(dp_data.axes_manager[2].scale)
datacube.calibration.set_Q_pixel_units('A^-1')

# Real space calibration (1 pixel = 1 unit area)
datacube.calibration.set_R_pixel_size(1)
datacube.calibration.set_R_pixel_units('area')
print(datacube.calibration)

plt.close("all")
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

# Plot the probe kernel
py4DSTEM.visualize.show_kernel(
    syn_probe.kernel, 
    R=20,
    L=20, 
    W=1,
    figsize=(8, 4),
)

plt.close("all")
detect_params = {
    'corrPower': 1.0,
    'sigma': 0,
    'edgeBoundary': 2,
    'minRelativeIntensity': 0,
    'minAbsoluteIntensity': 0.5,
    'minPeakSpacing': 10,
    'subpixel': 'poly',
    'upsample_factor': 4,
    'maxNumPeaks': 20,
    'CUDA': gpu,
}

bragg_peaks = datacube.find_Bragg_disks(
    template=syn_probe_kernel,
    **detect_params,
)

qxy_origins = bragg_peaks.measure_origin()
qx0_fit, qy0_fit, qx0_residuals, qy0_residuals = bragg_peaks.fit_origin(figsize=(4, 4))

# %%
# Verify Bragg disk detection parameters on a sample grid
detect_params_check = {
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

if vir_y >= 10:
    rys = np.random.randint(0, vir_y, size=10)
    rxs = np.random.randint(0, vir_x, size=10)
else:
    rys = np.random.randint(0, vir_y, size=vir_y)
    rxs = np.random.randint(0, vir_x, size=vir_x)    

yv, xv = np.meshgrid(rys, rxs)
yv = yv.flatten()
xv = xv.flatten()

disks_selected = datacube.find_Bragg_disks(
    data=(yv, xv),
    template=syn_probe_kernel,
    **detect_params_check,
)

py4DSTEM.visualize.show_image_grid(
    get_ar=lambda i: datacube.data[yv[i], xv[i], :, :],
    H=vir_y, 
    W=vir_x,
    axsize=(5, 5),
    intensity_range='absolute',
    vmin=0,
    vmax=5,
    get_x=lambda i: disks_selected[i].data['qx'],
    get_y=lambda i: disks_selected[i].data['qy'],
    open_circles=True,
    scale=350,
)

# %%
k_max = 0.7
result = {}
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
        CUDA=gpu
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

result[os.path.basename(data_adr).split(".")[0]+"_orientation_maps"] = orientation_maps
result[os.path.basename(data_adr).split(".")[0]+"_image_orientation"] = image_orientations

# %%
# Save the ACOM result
with open('ACOM_result.pkl', 'wb') as f:
    pickle.dump(result, f)
