import os
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import tifffile
from scipy.signal import find_peaks
import py4DSTEM
from shapely.geometry import Point, LineString, Polygon
import shapely

def data_load_3d(adr, crop=None, rescale=True, DM_file=True, rebin_256=False, verbose=True):
    """
    load a spectrum image
    """
    storage = []
    shape = []
    for i, ad in enumerate(adr):
        if DM_file:
            if crop:
                temp = hs.load(ad)
                temp = temp.isig[crop[0]:crop[1]]
                if rebin_256 and temp.data.shape[1] > 250:
                    temp = temp.rebin(scale=(2,2,1))
                temp = temp.data
                if rescale:
                    temp = temp/np.max(temp)
            else:
                temp = hs.load(ad).data
                if rescale:
                    temp = temp/np.max(temp)
        else:
            if crop:
                temp = tifffile.imread(ad)
                temp = temp[:, :, crop[0]:crop[1]]
                if rescale:
                    temp = temp/np.max(temp)
            else:
                temp = tifffile.imread(ad)
                if rescale:
                    temp = temp/np.max(temp)               

        if verbose:
            print(ad)
            print(temp.shape)
        shape.append(temp.shape)
        storage.append(temp)       
    
    shape = np.asarray(shape)
    return storage, shape

def data_load_4d(adr, rescale=False, rebin_256=False, verbose=True):
    storage = []
    shape = []   
    for i, ad in enumerate(adr):
        tmp = hs.load(ad)
        if rebin_256 and tmp.data.shape[1] > 250:
            tmp = tmp.rebin(scale=(2,2,1,1))
        tmp = tmp.data
        if rescale:
            tmp = tmp / np.max(tmp)
        if len(tmp.shape) == 3:
            try:
                tmp = tmp.reshape(int(tmp.shape[0]**(1/2)), int(tmp.shape[0]**(1/2)), tmp.shape[1], tmp.shape[2])
                print("The scanning shape is automatically corrected")
            except:
                print("The input data is not 4-dimensional")
                print("Please confirm that all options are correct")

        if verbose:
            print(ad)
            print(tmp.shape)
        shape.append(list(tmp.shape))
        storage.append(tmp)
    
    shape = np.asarray(shape)
    return storage, shape

def zero_one_rescale(spectrum):
    """
    normalize one spectrum from 0.0 to 1.0
    """
    spectrum = spectrum.clip(min=0.0)
    min_val = np.min(spectrum)
    rescaled = spectrum - min_val
    if np.max(rescaled) != 0:
        rescaled = rescaled / np.max(rescaled)
    return rescaled

def binning_SI(si, bin_y, bin_x, str_y, str_x, offset, depth, rescale=True):
    """
    re-bin a spectrum image
    """
    si = np.asarray(si)
    rows = range(0, si.shape[0]-bin_y+1, str_y)
    cols = range(0, si.shape[1]-bin_x+1, str_x)
    new_shape = (len(rows), len(cols))
    
    binned = []
    for i in rows:
        for j in cols:
            temp_sum = np.mean(si[i:i+bin_y, j:j+bin_x, offset:(offset+depth)], axis=(0, 1))
            if rescale:
                binned.append(zero_one_rescale(temp_sum))
            else:
                binned.append(temp_sum)
            
    binned = np.asarray(binned).reshape(new_shape[0], new_shape[1], depth)
    return binned

def radial_indices(shape, radial_range, center=None):
    y, x = np.indices(shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    
    r = np.hypot(y - center[0], x - center[1])
    ri = np.ones(r.shape)
    
    if len(np.unique(radial_range)) > 1:
        ri[np.where(r <= radial_range[0])] = 0
        ri[np.where(r > radial_range[1])] = 0
    else:
        r = np.round(r)
        ri[np.where(r != round(radial_range[0]))] = 0
    return ri

def flattening(fdata, flat_option="box", crop_dist=None, c_pos=None, save_path=None):
    fdata_shape = fdata.shape
    if flat_option == "box":
        if crop_dist:     
            box_size = np.array([crop_dist, crop_dist])
            h_si = np.floor(c_pos[0]-box_size[0]).astype(int)
            h_fi = np.ceil(c_pos[0]+box_size[0]).astype(int)
            w_si = np.floor(c_pos[1]-box_size[1]).astype(int)
            w_fi = np.ceil(c_pos[1]+box_size[1]).astype(int)

            tmp = fdata[:, :, h_si:h_fi, w_si:w_fi]
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
            ax.imshow(np.log(np.mean(tmp, axis=(0, 1))), cmap="viridis")
            ax.axis("off")
            if save_path is not None:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
            
            tmp = tmp.reshape(fdata_shape[0], fdata_shape[1], -1)
            return tmp
        else:
            tmp = fdata.reshape(fdata_shape[0], fdata_shape[1], -1)
            return tmp
    elif flat_option == "radial":
        if len(crop_dist) != 3:
            print("Warning! 'crop_dist' must be a list containing 3 elements")
        tmp = circle_flatten(fdata, crop_dist, c_pos)
        return tmp
    else:
        print("Warning! Wrong option ('flat_option')")
        return

def circle_flatten(f_stack, radial_range, c_pos):
    k_indx = []
    k_indy = []
    for r in range(radial_range[0], radial_range[1], radial_range[2]):
        tmp_k, tmp_a = indices_at_r(f_stack.shape[2:], r, c_pos)
        k_indx.extend(tmp_k[0].tolist())
        k_indy.extend(tmp_k[1].tolist())
    k_indx = np.asarray(k_indx)
    k_indy = np.asarray(k_indy)
    flat_data = f_stack[:, :, k_indy, k_indx]
    return flat_data

def indices_at_r(shape, radius, center=None):
    y, x = np.indices(shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(y - center[0], x - center[1])
    r = np.around(r)
    ri = np.where(r == radius)
    
    angle_arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            angle_arr[i, j] = np.angle(complex(x[i, j]-center[1], y[i, j]-center[0]), deg=True)
            
    angle_arr = angle_arr + 180
    angle_arr = np.around(angle_arr)
    ai = np.argsort(angle_arr[ri])
    r_sort = (ri[1][ai], ri[0][ai])
    a_sort = np.sort(angle_arr[ri])
    return r_sort, a_sort

def reshape_coeff(coeffs, new_shape):
    """
    reshape a coefficient matrix to restore the original scanning shapes.
    """
    coeff_reshape = []
    coeffs = coeffs.copy()
    for i in range(len(new_shape)):
        temp = coeffs[:int(new_shape[i, 0]*new_shape[i, 1]), :]
        coeffs = np.delete(coeffs, range(int(new_shape[i, 0]*new_shape[i, 1])), axis=0)
        temp = np.reshape(temp, (new_shape[i, 0], new_shape[i, 1], -1))
        coeff_reshape.append(temp)
    return coeff_reshape

def label_arrangement(label_arr, new_shape):
    """
    reshape a clustering result obtained by performing OPTICS
    """
    label_sort = np.unique(label_arr)
    num_label = len(label_sort)
    hist, edge = np.histogram(label_arr, bins=num_label)
    label_reshape = reshape_coeff(label_arr.reshape(-1, 1), new_shape)
    
    selected = []
    for i in range(num_label):
        temp = []
        for j in range(len(label_reshape)):
            img_temp = np.zeros_like(label_reshape[j])
            img_temp[np.where(label_reshape[j] == label_sort[i])] = 1.0
            temp.append(img_temp)
        selected.append(temp)    
    return label_reshape, selected, hist

def profile_peak(line, k_step, crop=None, peak_prominence=1E-10, max_num_peaks=3):
    if crop == None:
        crop_start = 0
        crop_end = len(line)
        print("the full profile is used")
    else:
        crop_start = crop[0]
        crop_end = crop[-1]
        print(f"crop range: {crop_start} ({crop_start*k_step}) - {crop_end} ({crop_end*k_step}) ")
    
    line = line[crop_start:crop_end].copy()
    line /= np.max(line)
    peaks = find_peaks(line, prominence=peak_prominence)
    peaks, properties = peaks[0], peaks[1]['prominences']
    # print(properties)
    peak_argsort = np.argsort(properties)    
    peak_ind = peaks[:]
    peak_values = line[peak_ind]
    peak_values = peak_values[peak_argsort]
    peaks = peaks * k_step
    peaks = peaks + crop_start * k_step
    # print(peaks)
    peaks = peaks[peak_argsort][-max_num_peaks:]
    return peaks

def simul_xrd_peak(adr, k_range, broadening, peak_prominence):
    basename = os.path.basename(adr).split('.')[0]
    try:
        crystal_cif = py4DSTEM.process.diffraction.Crystal.from_CIF(adr, conventional_standard_structure=False)
    except:
        crystal_cif = py4DSTEM.process.diffraction.Crystal.from_CIF(adr, conventional_standard_structure=True)
    crystal_cif.calculate_structure_factors(k_range[-1])

    from_cif = py4DSTEM.process.diffraction.utils.calc_1D_profile(
        k_range,
        crystal_cif.g_vec_leng,
        crystal_cif.struct_factors_int,
        k_broadening=broadening,
        int_scale=1.0,
        normalize_intensity=True)

    peaks = find_peaks(from_cif, height=None, 
                        width=None, 
                        threshold=None, 
                        distance=None, 
                        prominence=peak_prominence)[0]
    peaks = peaks * (k_range[1]-k_range[0]) + k_range[0]
    return basename, from_cif, peaks

def mmad_score(df_data, experimental_cols, target_structures=None):
    if target_structures != None:
        valid_targets_in_new_data = [s for s in target_structures if s in df_data.columns]
        missing_targets_in_new_data = [s for s in target_structures if s not in df_data.columns]
    else:
        valid_targets_in_new_data = [s for s in df_data.columns]
        missing_targets_in_new_data = []

    if missing_targets_in_new_data:
        print(f"Warning: The following target structures were not found in the new data columns: {missing_targets_in_new_data}")
    if not valid_targets_in_new_data:
        raise ValueError("Error: None of the specified target structures are present in the new data. "
                         f"Specified targets: {target_structures}. "
                         f"Available structures: {[col for col in df_data.columns if col not in experimental_cols]}")

    results_mmad_subset = {}
    for lv_col_name in experimental_cols:
        if lv_col_name not in df_data.columns:
            print(f"Warning: LV column {lv_col_name} not found in the new data. Skipping this LV.")
            results_mmad_subset[lv_col_name] = []
            continue

        lv_peaks_series = df_data[lv_col_name].dropna()
        if lv_peaks_series.empty:
            print(f"LV column {lv_col_name} has no experimental peaks. Skipping.")
            results_mmad_subset[lv_col_name] = [{'simulation': s, 'mmad_score': float('inf'), 'lv_peaks_matched': 0} for s in valid_targets_in_new_data]
            results_mmad_subset[lv_col_name] = sorted(results_mmad_subset[lv_col_name], key=lambda x: x['mmad_score'])
            continue

        lv_peaks_np = lv_peaks_series.values.reshape(-1, 1)
        num_lv_peaks = len(lv_peaks_np)
        scores_for_lv = []

        for sim_col_name in valid_targets_in_new_data:
            sim_peaks_series = df_data[sim_col_name].dropna()
            mmad_score_val = float('inf')
            if not sim_peaks_series.empty:
                sim_peaks_np = sim_peaks_series.values.reshape(-1, 1)
                mmad_score_val = 0
                for peak in lv_peaks_np:
                    abs_diff = np.abs(sim_peaks_np - peak)
                    mmad_score_val += np.min(abs_diff)

            scores_for_lv.append({
                'simulation': sim_col_name, 
                'mmad_score': mmad_score_val, 
                'lv_peaks_matched': num_lv_peaks
            })

        sorted_scores = sorted(scores_for_lv, key=lambda x: x['mmad_score'])
        results_mmad_subset[lv_col_name] = sorted_scores
    return results_mmad_subset

# Concave hull
# original code: https://github.com/M-Lin-DM/Concave-Hulls
# modified by J. Ryu
class ConcaveHull(object):
    def __init__(self, points, k, use_gpu=False):
        if hasattr(points, 'device') and points.device != 'cpu':  # cupy array
            self.data_set = points
            use_gpu = True
        elif isinstance(points, np.ndarray):
            self.data_set = points
        elif isinstance(points, list):
            self.data_set = np.array(points)
        else:
            raise ValueError('Please provide an [N,2] array or a list of lists.')

        if use_gpu:
            try:
                import cupy as cp
                self.xp = cp
                self.data_set_cpu = cp.asnumpy(self.data_set)
                self.data_set_cpu = np.unique(self.data_set_cpu, axis=0)
                self.data_set = cp.asarray(self.data_set_cpu)
            except Exception:
                import warnings
                warnings.warn("CuPy not available or failed to load. Falling back to CPU/NumPy.")
                self.xp = np
                self.data_set = np.unique(self.data_set, axis=0)
                self.data_set_cpu = self.data_set
                use_gpu = False
        else:
            self.xp = np
            self.data_set = np.unique(self.data_set, axis=0)
            self.data_set_cpu = self.data_set

        self.indices = self.xp.ones(self.data_set.shape[0], dtype=bool)
        self.k = k
        self.use_gpu = use_gpu
        self.pts_geom = shapely.points(self.data_set_cpu)

    @staticmethod
    def dist_pt_to_group(a, b):
        if hasattr(a, 'device') and a.device != 'cpu':
            import cupy as cp
            xp = cp
        else:
            xp = np
        return xp.linalg.norm(a - b, axis=1)

    @staticmethod
    def get_lowest_latitude_index(points):
        if hasattr(points, 'device') and points.device != 'cpu':
            import cupy as cp
            xp = cp
        else:
            xp = np
        indices = xp.argsort(points[:, 1])
        return int(indices[0])

    @staticmethod
    def norm_array(v):
        if hasattr(v, 'device') and v.device != 'cpu':
            import cupy as cp
            xp = cp
        else:
            xp = np
        norms = xp.linalg.norm(v, axis=1, keepdims=True)
        return xp.divide(v, norms)

    @staticmethod
    def norm(v):
        if hasattr(v, 'device') and v.device != 'cpu':
            import cupy as cp
            xp = cp
        else:
            xp = np
        norms = xp.linalg.norm(v)
        return v / norms

    def get_k_nearest(self, ix, k):
        ixs = self.indices
        base_indices = self.xp.arange(len(ixs))[ixs]
        distances = self.dist_pt_to_group(self.data_set[ixs, :], self.data_set[ix, :])
        sorted_indices = self.xp.argsort(distances)
        kk = min(k, len(sorted_indices))
        k_nearest = sorted_indices[:kk]
        return base_indices[k_nearest]

    def clockwise_angles(self, last, ix, ixs, first):
        if first == 1:
            last_norm = self.xp.array([-1.0, 0.0])
        elif first == 0:
            last_norm = self.norm(self.data_set[last, :] - self.data_set[ix, :])
        
        ixs_norm = self.norm_array(self.data_set[ixs, :] - self.data_set[ix, :])
        
        dot_vals = self.xp.clip(self.xp.dot(ixs_norm, last_norm), -1.0, 1.0)
        thetas = self.xp.arccos(dot_vals)
        z_comps = last_norm[0] * ixs_norm[:, 1] - last_norm[1] * ixs_norm[:, 0]
        ang = self.xp.where(z_comps <= 0, thetas, 2 * self.xp.pi - thetas)
        return self.xp.squeeze(ang)

    def recurse_calculate(self):
        recurse = ConcaveHull(self.data_set, self.k + 1, use_gpu=self.use_gpu)
        if recurse.k >= self.data_set.shape[0]:
            print(" max k reached, at k={0}".format(recurse.k))
            return None
        return recurse.calculate()

    def calculate(self):
        if self.data_set.shape[0] < 3:
            return None
        if self.data_set.shape[0] == 3:
            return self.data_set_cpu

        kk = min(self.k, self.data_set.shape[0])
        first_point = self.get_lowest_latitude_index(self.data_set)
        current_point = first_point

        hull = np.reshape(self.data_set_cpu[first_point, :], (1, 2))
        test_hull = hull
        self.indices[first_point] = False

        step = 2
        stop = 2 + kk

        while ((current_point != first_point) or (step == 2)) and self.xp.sum(self.indices) > 0:
            if step == stop:
                self.indices[first_point] = True
            knn = self.get_k_nearest(current_point, kk)

            if step == 2:
                angles = self.clockwise_angles(1, current_point, knn, 1)
            else:
                angles = self.clockwise_angles(last_point, current_point, knn, 0)

            if self.use_gpu:
                candidates = self.xp.argsort(-angles).get()
                knn_cpu = knn.get()
            else:
                candidates = np.argsort(-angles)
                knn_cpu = knn

            i = 0
            invalid_hull = True

            while invalid_hull and i < len(candidates):
                candidate = candidates[i]
                next_point = np.reshape(self.data_set_cpu[knn_cpu[candidate], :], (1, 2))
                test_hull = np.append(hull, next_point, axis=0)
                line = LineString(test_hull)
                invalid_hull = not line.is_simple
                i += 1

            if invalid_hull:
                return self.recurse_calculate()

            last_point = current_point
            current_point = int(knn_cpu[candidate])
            hull = test_hull
            self.indices[current_point] = False
            step += 1

        poly = Polygon(hull)
        in_poly = shapely.intersects(poly, self.pts_geom)
        if np.all(in_poly):
            return hull
        else:
            return self.recurse_calculate()
