# %%
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

edx_adr = ''
lv_edx = hs.load(edx_adr)
print(lv_edx)
print(lv_edx.axes_manager)

num_sub = eval(lv_edx.axes_manager[0].name)
num_comp = eval(lv_edx.axes_manager[0].units)

# %% [markdown]
# ## Quantification of a single spectrum (e.g. Component 7)

# %%
lv = 7
gauss_sigma = 1
filtered = gaussian_filter(lv_edx.data[lv-1], gauss_sigma)
filtered = filtered / np.max(filtered)
filtered *= 2048

one_edx = hs.signals.Signal1D(filtered)
one_edx.axes_manager[0].scale = lv_edx.axes_manager[1].scale
one_edx.axes_manager[0].name = lv_edx.axes_manager[1].name
one_edx.axes_manager[0].offset = -0.2
one_edx.axes_manager[0].units = lv_edx.axes_manager[1].units
one_edx.set_signal_type("EDS_TEM")
one_edx.set_microscope_parameters(beam_energy=300)

# %%
plt.close("all")
crop_edx_1 = one_edx.isig[0.2:6.0]
crop_edx_1.set_elements(['N', 'Br', 'Pb', 'I', 'Cs', 'O', 'Si', 'Cu', 'C'])

# Plot elements crop
crop_edx_1.plot(True)

# %%
# Create and fit EDS model
m = crop_edx_1.create_model(auto_background=False)
m.multifit(bounded=True)
m.plot(True)

# %%
sI = m.get_lines_intensity(["N_Ka","Cs_La","Pb_Ma","Br_La","I_La"])
print("\nLines Intensity:")
print(*sI, sep="\n")

kfactors = [1.834,  # N_Ka
            1.962,  # Cs_La
            1.76,   # Pb_Ma
            1.645,  # Br_La
            1.914]  # I_La

composition = crop_edx_1.quantification(method="CL", intensities=sI, factors=kfactors, plot_result=True)

comp_list = []
for comp in composition:
    comp_list.append([comp.metadata['Sample']['xray_lines'][0], comp.data[0]])

comp_list = np.asarray(comp_list)
print("\nQuantification Results:")
print(*comp_list[:, 0], sep="\t")
print(*comp_list[:, 1], sep="\t")

# %% [markdown]
# # Batch Quantification over all Components and Sub-indices

# %%
gauss_sigma = 0
element_list = ['N', 'Br', 'Pb', 'I', 'Cs', 'O', 'Si', 'Cu', 'C']
edge_list = ["Cs_La","Pb_Ma","Br_La","I_La"]
kfactors = [1.962,  # Cs_La
            1.76,   # Pb_Ma
            1.645,  # Br_La
            1.914]  # I_La

k = 0
for sub_index in range(num_sub):
    total_list = []
    print(f"\n=========================================\nSub Index {sub_index+1}")
    for lv in range(num_comp):
        filtered = gaussian_filter(lv_edx.data[k], gauss_sigma)
        filtered = filtered / np.max(filtered)
        filtered *= 2048
        
        one_edx = hs.signals.Signal1D(filtered)
        one_edx.axes_manager[0].scale = lv_edx.axes_manager[1].scale
        one_edx.axes_manager[0].name = lv_edx.axes_manager[1].name
        one_edx.axes_manager[0].offset = -0.2
        one_edx.axes_manager[0].units = lv_edx.axes_manager[1].units
        one_edx.set_signal_type("EDS_TEM")
        one_edx.set_microscope_parameters(beam_energy=300)
        
        crop_edx_1 = one_edx.isig[0.2:6.0]
        crop_edx_1.set_elements(element_list)
        
        m = crop_edx_1.create_model(auto_background=False)
        m.multifit(bounded=True)
        m.plot(True)
        plt.show()
        
        sI = m.get_lines_intensity(edge_list)
        composition = crop_edx_1.quantification(method="CL", intensities=sI, factors=kfactors, plot_result=False)

        comp_list = []
        for comp in composition:
            comp_list.append([comp.metadata['Sample']['xray_lines'][0], comp.data[0]])

        comp_list = np.asarray(comp_list)
        total_list.append(comp_list)
        k += 1
        
    total_list = np.asarray(total_list)
    print(f"\nSummary for Sub Index {sub_index+1}:")
    print(*total_list[0, :, 0], sep="\t")
    for lv in range(num_comp):
        print(*total_list[lv, :, 1], sep='\t')
