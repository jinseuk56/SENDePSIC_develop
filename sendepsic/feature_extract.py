import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
from sklearn.decomposition import NMF, PCA

from .utils import data_load_3d, zero_one_rescale, reshape_coeff

class feature_extract():
    def __init__(self, adr, dat_dim, dat_unit, cr_range=None, dat_scale=1, rescale=True, DM_file=True, verbose=True, rebin_256=False, data_storage=None):
        # We only support dat_dim = 3 profile analysis
        self.dat_dim = 3
        
        colors_yellows = [(1, 1, 1), (1, 1, 0.9), (1, 1, 0.7), (1, 1, 0.5), (0.9, 0.9, 0.3), (0.8, 0.8, 0.1)]
        cmap_yellows = mcolors.LinearSegmentedColormap.from_list("Yellows", colors_yellows)
        colors_cyans = [(1, 1, 1), (0.95, 1, 1), (0.8, 1, 1), (0.5, 0.9, 1), (0.3, 0.8, 0.95), (0.1, 0.6, 0.8)]
        cmap_cyans = mcolors.LinearSegmentedColormap.from_list("Cyans", colors_cyans)
        colors_limes = [(1, 1, 1), (0.95, 1, 0.95), (0.9, 1, 0.8), (0.7, 1, 0.5), (0.5, 0.9, 0.3), (0.3, 0.7, 0.1)]
        cmap_limes = mcolors.LinearSegmentedColormap.from_list("Limes", colors_limes)
        colors_magenta = [(1, 1, 1), (1, 0.95, 1), (1, 0.8, 1), (1, 0.5, 1), (0.95, 0.3, 0.95), (0.8, 0.1, 0.8)]
        cmap_magenta = mcolors.LinearSegmentedColormap.from_list("Magenta", colors_magenta)

        # create a customized colorbar
        self.color_rep = ["black", "red", "green", "blue", "orange", "purple", "yellow", "lime", 
                    "cyan", "magenta", "lightgray", "peru", "springgreen", "deepskyblue", 
                    "hotpink", "darkgray"]
        self.custom_cmap = mcolors.ListedColormap(self.color_rep)
        bounds = np.arange(-1, len(self.color_rep))
        self.norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(self.color_rep))

        self.cm_rep = ["gray", "Reds", "Greens", "Blues", "Oranges", "Purples", cmap_yellows, cmap_limes, cmap_cyans, cmap_magenta]  

        self.file_adr = adr
        self.dat_unit = dat_unit
        self.cr_range = cr_range
        
        if cr_range:
            self.dat_dim_range = np.arange(cr_range[0], cr_range[1], cr_range[2]) * dat_scale
            self.num_dim = len(self.dat_dim_range)
        
        if data_storage is not None:
            self.num_img = len(data_storage)
            processed_storage = []
            processed_shape = []
            for temp in data_storage:
                if hasattr(temp, "data"):
                    # HyperSpy Signal1D / Signal2D
                    if cr_range is not None:
                        temp_sig = temp.isig[cr_range[0]:cr_range[1]]
                    else:
                        temp_sig = temp
                    # Apply optional rebinning if requested (for 3D datasets)
                    if rebin_256 and temp_sig.data.shape[1] > 250:
                        temp_sig = temp_sig.rebin(scale=(2, 2, 1))
                    temp_arr = temp_sig.data.copy()
                else:
                    # Raw numpy array
                    temp_arr = temp.copy()
                    if cr_range is not None:
                        temp_arr = temp_arr[:, :, cr_range[0]:cr_range[1]]
                
                if rescale:
                    temp_arr = temp_arr / np.max(temp_arr)
                
                processed_storage.append(temp_arr)
                processed_shape.append(temp_arr.shape)
            
            self.data_storage = processed_storage
            self.data_shape = np.asarray(processed_shape)
        else:
            self.num_img = len(adr) if adr else 0
            self.data_storage, self.data_shape = data_load_3d(adr, cr_range, rescale, DM_file, rebin_256, verbose)
            
        self.original_data_shape = self.data_shape.copy()
        
        if len(self.dat_dim_range) > self.original_data_shape[0, 2]:
            difference = len(self.dat_dim_range) - self.original_data_shape[0, 2]
            self.dat_dim_range = self.dat_dim_range[:-difference]
            self.num_dim = len(self.dat_dim_range)
            if verbose:
                print("Data shape")
                print(self.original_data_shape)
                print("Spectrum length: %d"%self.num_dim)
        elif len(self.dat_dim_range) < self.original_data_shape[0, 2]:
            difference = self.original_data_shape[0, 2] - len(self.dat_dim_range)
            self.dat_dim_range = np.arange(cr_range[0], cr_range[1]+difference*cr_range[2], cr_range[2]) * dat_scale
            self.num_dim = len(self.dat_dim_range)
            if verbose:
                print("Data shape")
                print(self.original_data_shape)
                print("Spectrum length: %d"%self.num_dim)
        else:
            if verbose:
                print("Data shape")
                print(self.original_data_shape)
                print("Spectrum length: %d"%self.num_dim)
        self._default_figure_save_path = None



    def set_figure_save_path(self, path):
        """Set a default base path for saving all figures.
        
        When set, all visualization methods will automatically save figures
        to disk using this as the base path prefix, without needing to
        specify save_path on each call. Individual method calls can still
        override this with their own save_path argument.
        
        Parameters
        ----------
        path : str or None
            Base file path prefix (e.g. '/results/sample_A'). Extension is
            optional; .png is used by default. Pass None to revert to
            notebook display mode.
        """
        self._default_figure_save_path = path
        if path is not None:
            print(f"Default figure save path set to: {path}")
        else:
            print("Default figure save path cleared. Figures will display in notebook.")
    def make_input(self, min_val=0.0, max_normalize=True, rescale_0to1=False, log_scale=False):
        dataset_flat = []
        for i in range(self.num_img):
            dataset_flat.extend(self.data_storage[i].clip(min=min_val).reshape(-1, self.num_dim).tolist())
        dataset_flat = np.asarray(dataset_flat)
        print(dataset_flat.shape)
            
        if log_scale:
            dataset_flat[np.where(dataset_flat==0.0)] = 1.0
            dataset_flat = np.log(dataset_flat)
            print(np.min(dataset_flat), np.max(dataset_flat))
            
        if max_normalize:
            dataset_flat = dataset_flat / np.max(dataset_flat, axis=1)[:, np.newaxis]
            dataset_flat = np.nan_to_num(dataset_flat)
            print(np.min(dataset_flat), np.max(dataset_flat))
            
        if rescale_0to1:
            for i in range(len(dataset_flat)):
                dataset_flat[i] = zero_one_rescale(dataset_flat[i])
                
        dataset_flat = dataset_flat.clip(min=min_val)
        print(np.min(dataset_flat), np.max(dataset_flat))
        self.total_num = len(dataset_flat)
        self.dataset_flat = dataset_flat
        self.ri = np.random.choice(self.total_num, self.total_num, replace=False)

        self.dataset_input = dataset_flat[self.ri]
        self.dataset_input = self.dataset_input.astype(np.float32)


    def ini_DR(self, method="nmf", num_comp=5, result_visual=True, intensity_range="absolute", tolerance=1E-4, max_iteration=2000, save_path=None):
        _eff_save = save_path if save_path is not None else self._default_figure_save_path
        self.DR_num_comp = num_comp
        if method=="nmf":
            self.DR = NMF(n_components=num_comp, init="nndsvda", solver="mu", max_iter=max_iteration, verbose=result_visual, tol=tolerance)
            self.DR_coeffs = self.DR.fit_transform(self.dataset_input)
            self.DR_comp_vectors = self.DR.components_
        elif method=="pca":
            self.DR = PCA(n_components=num_comp, whiten=False, 
                     random_state=np.random.randint(100), svd_solver="auto")
            self.DR_coeffs = self.DR.fit_transform(self.dataset_input)
            self.DR_comp_vectors = self.DR.components_
        else:
            print(method+" not supported")
            return
        
        coeffs = np.zeros_like(self.DR_coeffs)
        coeffs[self.ri] = self.DR_coeffs.copy()
        self.DR_coeffs = coeffs
        self.coeffs_reshape = reshape_coeff(self.DR_coeffs, self.data_shape)
        
        if result_visual or _eff_save is not None:
            import os
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
            for i in range(self.DR_num_comp):
                tmp_ax, = ax.plot(self.dat_dim_range, self.DR_comp_vectors[i], "-", c=self.color_rep[i+1], label="loading vector %d"%(i+1))
                shadow_effect = path_effects.withStroke(linewidth=3, foreground='gray')
                tmp_ax.set_path_effects([shadow_effect])
                                    
            ax.legend(fontsize="large")
            ax.set_xlabel(self.dat_unit, fontsize=10)
            ax.tick_params(axis="x", labelsize=10)
            fig.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig.savefig(f"{base}_loading_vectors{ext}", bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
            
            if intensity_range == "relative":
                for i in range(self.num_img):
                    fig, ax = plt.subplots(1, self.DR_num_comp, figsize=(5*self.DR_num_comp, 5), dpi=100)
                    for j in range(self.DR_num_comp):
                        tmp = ax[j].imshow(self.coeffs_reshape[i][:, :, j], cmap="inferno")
                        ax[j].set_title("loading vector %d map"%(j+1), fontsize=10)
                        ax[j].axis("off")
                        fig.colorbar(tmp, cax=fig.add_axes([0.92, 0.15, 0.04, 0.7]))
                    fig.suptitle(self.file_adr[i] if self.file_adr else f"Image {i+1}")
                    if _eff_save is not None:
                        base, ext = os.path.splitext(_eff_save)
                        if not ext:
                            ext = '.png'
                        fig.savefig(f"{base}_intensity_map_{i}{ext}", bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()
            else:               
                min_val = np.min(coeffs)
                max_val = np.max(coeffs)
                for i in range(self.num_img):
                    fig, ax = plt.subplots(1, self.DR_num_comp, figsize=(5*self.DR_num_comp, 5), dpi=100)
                    for j in range(self.DR_num_comp):
                        tmp = ax[j].imshow(self.coeffs_reshape[i][:, :, j], vmin=min_val, vmax=max_val, cmap="inferno")
                        ax[j].set_title("loading vector %d map"%(j+1), fontsize=10)
                        ax[j].axis("off")
                        fig.colorbar(tmp, cax=fig.add_axes([0.92, 0.15, 0.04, 0.7]))
                    fig.suptitle(self.file_adr[i] if self.file_adr else f"Image {i+1}")
                    if _eff_save is not None:
                        base, ext = os.path.splitext(_eff_save)
                        if not ext:
                            ext = '.png'
                        fig.savefig(f"{base}_intensity_map_{i}{ext}", bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()
