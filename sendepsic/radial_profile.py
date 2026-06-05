import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.path as mpath
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, binary_fill_holes
import hyperspy.api as hs
import tifffile
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist, squareform
import shapely
from shapely.geometry import Polygon, Point

# Import package modules
from .utils import (
    data_load_3d, data_load_4d, zero_one_rescale, binning_SI,
    radial_indices, flattening, circle_flatten, indices_at_r,
    reshape_coeff, label_arrangement, profile_peak, simul_xrd_peak,
    mmad_score, ConcaveHull
)
from .feature_extract import feature_extract

class radial_profile_analysis():
    def __init__(self, base_dir, subfolders, profile_length, num_load, final_dir=None,
                 include_key=None, exclude_key=None, simult_edx=False, rebin_256=False, roll_axis=True, 
                 verbose=True, zernike=False, use_gpu=False, boundary_method="custom", fill_method="shapely", concave_ratio=0.2):
        
        now = time.localtime()
        self.formatted = time.strftime("%Y%m%d_%H%M%S", now)
        print(f"Formatted local time: {self.formatted}")

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

        self.cm_rep = ["gray", "Reds", "Greens", "Blues", "Oranges", "Purples", cmap_yellows, cmap_limes, cmap_cyans, cmap_magenta]  
        

        edx_split = []
        edx_avg_split = []
        radial_var_split = []
        radial_var_sum_split = []
        pixel_size_split = []
        loaded_data_path = []
        loaded_edx_path = []
        
        print(final_dir)
        
        for i, sub in enumerate(subfolders):
            if final_dir == None or final_dir == [] or final_dir == '': 
                file_adrs = glob.glob(base_dir+'/'+sub+'/*/*_azimuthal_var.hspy', recursive=True)
                if file_adrs == []:
                    print("Please make sure that the base directory and subfolder name are correct.")
                    return
                        
            else:             
                file_adrs = glob.glob(base_dir+'/'+sub+'/*/%s/*_azimuthal_var.hspy'%final_dir, recursive=True)
                if file_adrs == []:
                    print("Please make sure that the base directory and subfolder name are correct.")
                    return
                
            if simult_edx:
                edx_adrs = glob.glob(base_dir+'/'+sub+'/EDX/*.rpl', recursive=True)
                edx_adrs.sort()
                if edx_adrs == []:
                    print("Please make sure that the base directory and subfolder name are correct.")
                    return                
            
            file_adrs.sort()

            if include_key == []:
                key_list = []
                edx_adr = []
                keyword_ = list(exclude_key)
                for adr in file_adrs:
                    check = []
                    for key in keyword_:
                        if key in adr:
                            check.append(1)
                    if check == []:
                        key_list.append(adr)
                        if simult_edx:
                            adr = adr.replace('\\', '/') # for Windows OS
                            datetime = adr.split('/')[-1][:15]
                            for adr in edx_adrs:
                                if datetime in adr:
                                    edx_adr.append(adr)
                            
                        
                print(len(key_list))
                key_list = np.asarray(key_list)
                edx_adr = np.asarray(edx_adr)
            
                if len(key_list) > num_load:
                    ri = np.random.choice(len(key_list), num_load, replace=False)
                    file_adr_ = key_list[ri]
                    if simult_edx:
                        # print(key_list)
                        # print(edx_adr)
                        edx_adr_ = edx_adr[ri]
                else:
                    file_adr_ = key_list
                    if simult_edx:
                        edx_adr_ = edx_adr
        
            else:
                key_list = []
                edx_adr = []
                keyword_ = list(exclude_key)
                for adr in file_adrs:
                    for key in include_key:
                        if key in adr:
                            check = []
                            for key in keyword_:
                                if key in adr:
                                    check.append(1)
                            if check == []:
                                key_list.append(adr)
                                if simult_edx:
                                    adr = adr.replace('\\', '/') # for Windows OS
                                    datetime = adr.split('/')[-1][:15]
                                    for adr in edx_adrs:
                                        if datetime in adr:
                                            edx_adr.append(adr)
                
                print(len(key_list))
                edx_adr = np.asarray(edx_adr)
                key_list = np.asarray(key_list)
                
                if len(key_list) > num_load:
                    ri = np.random.choice(len(key_list), num_load, replace=False)
                    file_adr_ = key_list[ri]
                    if simult_edx:
                        # print(edx_adr)
                        edx_adr_ = edx_adr[ri]
                else:
                    file_adr_ = key_list
                    if simult_edx:
                        edx_adr_ = edx_adr                  
                    
            print("number of data in subfolder '%s'"%sub)
            #print(*file_adr_, sep='\n')
            print(len(file_adr_))
            file_adr_.sort()
            try:
                edx_adr_.sort()
            except:
                print('No EDX files')
            # for f_adr, e_adr in zip(file_adr_, edx_adr_):
            #     print(f_adr.split('/')[-1], e_adr.split('/')[-1])

            edx_tmp_list = []
            edx_avg_list = []
            radial_var_list = []
            avg_radial_var_list = []
            file_adr = []
            pixel_size_list = []
            scan_step_list = []
            edx_adr = []
            for e, adr in enumerate(file_adr_):
                data = hs.load(adr)
                print('original profile size = ', data.data.shape[-1])
                        
                if rebin_256:
                    if data.data.shape[1] > 250:
                        data = data.rebin(scale=(2,2,1))
                        
                file_adr.append(adr)
                data.data = data.data[:, :, :profile_length]
                local_radial_var_sum = data.mean()
                pixel_size_inv_Ang = local_radial_var_sum.axes_manager[-1].scale

                if simult_edx:                 
                    edx_data = hs.load(edx_adr_[e]).data
                    edx_adr.append(edx_adr_[e])
                    if roll_axis:
                        edx_data = np.rollaxis(edx_data, 0, 3)[:data.data.shape[0], :data.data.shape[1]]
                    edx_data = hs.signals.Signal1D(edx_data)
                    
                    if rebin_256:
                        if edx_data.data.shape[1] > 250:
                            edx_data = edx_data.rebin(scale=(2,2,1))        
                            
                    if data.data.shape[:2] != edx_data.data.shape[:2]:
                        print("The scan shapes are different between 4DSTEM and EDX")
                        print(adr)
                        print(edx_adr_[e])
                        return

                if verbose:
                    print("radial profile data information")
                    print(adr)
                    print(data)
                    print(data.axes_manager)
                    if simult_edx:
                        print("EDX data information")
                        print(edx_adr_[e])
                        print(edx_data)
                        print(edx_data.axes_manager)                   

                radial_var_list.append(data)
                avg_radial_var_list.append(local_radial_var_sum.data)
                pixel_size_list.append(pixel_size_inv_Ang)
                if simult_edx:
                    edx_tmp_list.append(edx_data)
                    edx_avg_list.append(edx_data.mean().data)
                    
        
            avg_radial_var_list = np.asarray(avg_radial_var_list)
            radial_var_split.append(radial_var_list)
            radial_var_sum_split.append(avg_radial_var_list)
            pixel_size_split.append(pixel_size_list)
        
            loaded_data_path.append(file_adr)

            if simult_edx:
                edx_avg_list = np.asarray(edx_avg_list)
                edx_split.append(edx_tmp_list)
                edx_avg_split.append(edx_avg_list)
                loaded_edx_path.append(edx_adr)

        # mean profile data load
        loaded_data_mean_path = []
        radial_avg_split = []
        radial_avg_sum_split = []
        for i, sub in enumerate(subfolders):
            radial_avg_list = []
            radial_avg_sum_list = []
            loaded_data_mean = []
            for adr in loaded_data_path[i]:
                dir_path = os.path.dirname(adr)
                data_name = os.path.basename(adr).split("_")
                data_name = data_name[0]+'_'+data_name[1]
                
                try:
                    adr_ = dir_path+"/"+data_name+"_azimuthal_mean.hspy"
                    data = hs.load(adr_)

                except:
                    print('There is no mean profile data, so it will be replaced with variance profile data')
                    adr_ = dir_path+"/"+data_name+"_variance.hspy"
                    data = hs.load(adr_)
                        
                if rebin_256:
                    if data.data.shape[1] > 250:
                        data = data.rebin(scale=(2,2,1))
                        
                loaded_data_mean.append(adr_)
                data.data = data.data[:, :, :profile_length]
                local_radial_avg_sum = data.mean()
                radial_avg_list.append(data)
                radial_avg_sum_list.append(local_radial_avg_sum.data)

            loaded_data_mean_path.append(loaded_data_mean)
            radial_avg_split.append(radial_avg_list)
            radial_avg_sum_split.append(radial_avg_sum_list)

        # aligned center beam image load
        BF_disc_align = []
        for i, sub in enumerate(subfolders):
            BF_disc_list = []
            for adr in loaded_data_path[i]:
                dir_path = os.path.dirname(adr)
                data_name = os.path.basename(adr).split("_")
                data_name = data_name[0]+'_'+data_name[1]

                adr_ = dir_path+"/"+data_name+"_azimuthal_data_centre.png"
                data = plt.imread(adr_) 

                BF_disc_list.append(data)
            BF_disc_align.append(BF_disc_list)
            
        # load zernike
        if zernike:
            loaded_data_zernike_path = []
            zernike_split = []
            zernike_sum_split = []
            for i, sub in enumerate(subfolders):
                zernike_list = []
                zernike_sum_list = []
                loaded_data_zernike = []
                for adr in loaded_data_path[i]:
                    dir_path = os.path.dirname(adr)
                    data_name = os.path.basename(adr).split("_")
                    data_name = data_name[0]+'_'+data_name[1]

                    adr_ = dir_path+"/"+data_name+"_zernike.hspy"
                    data = hs.load(adr_)
                    local_zernike_sum = data.mean()
                    
                    self.zernike_length = data.data.shape[2]

                    if rebin_256:
                        if data.data.shape[1] > 250:
                            data = data.rebin(scale=(2,2,1))

                    loaded_data_zernike.append(adr_)
                    zernike_list.append(data)
                    zernike_sum_list.append(local_zernike_sum.data)

                loaded_data_zernike_path.append(loaded_data_zernike)
                zernike_split.append(zernike_list)
                zernike_sum_split.append(zernike_sum_list)          
        

        self.zernike = zernike
        if zernike:
            self.loaded_data_zernike_path = loaded_data_zernike_path
            self.zernike_split = zernike_split
            self.zernike_sum_split = zernike_sum_split
            
        self.pixel_size_inv_Ang = pixel_size_split[0][0]
        self.base_dir = base_dir
        self.subfolders = subfolders
        self.profile_length = profile_length
        self.num_load = num_load
        self.radial_var_split = radial_var_split
        self.radial_var_sum_split = radial_var_sum_split
        self.radial_avg_split = radial_avg_split
        self.radial_avg_sum_split = radial_avg_sum_split
        self.pixel_size_split = pixel_size_split
        self.edx_split = edx_split
        self.loaded_data_path = loaded_data_path
        self.loaded_data_mean_path = loaded_data_mean_path
        self.BF_disc_align = BF_disc_align
        self.simult_edx = simult_edx
        if simult_edx:
            self.loaded_edx_path = loaded_edx_path
            
        self.rebin_256 = rebin_256
        self.final_dir = final_dir
        self.include_key = include_key
        self.exclude_key = exclude_key
        self.roll_axis = roll_axis
        self.saved_files = []
        self.crop = [0, -1, 0, -1]
        self.use_gpu = use_gpu
        self.boundary_method = boundary_method
        self.fill_method = fill_method
        self.concave_ratio = concave_ratio
        self._default_figure_save_path = None
        
        print("data loaded.")


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


    def print_colormaps(self, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'print_colormaps') if self._default_figure_save_path is not None else None)
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        fig, ax = plt.subplots(len(self.cm_rep)-1, 1, figsize=(6, 8), dpi=100)
        for i, axs in enumerate(ax.flat):
            axs.imshow(gradient, aspect='auto', cmap=self.cm_rep[i+1])
            axs.set_axis_off()
        fig.tight_layout()
        if _eff_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
            fig.savefig(_eff_save, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


    def center_beam_alignment_check(self, crop=[0, -1, 0, -1], visual_title=True, title_font_size=10, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'center_beam_alignment_check') if self._default_figure_save_path is not None else None)

        self.crop = crop
        top, bottom, left, right = self.crop
        
        for i in range(len(self.subfolders)):
            num_img = len(self.BF_disc_align[i])
            print(num_img)
            grid_size = int(np.around(np.sqrt(num_img)))
            if num_img == 1:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                ax = np.array([ax])
            elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
            else:
                fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
            for j in range(num_img):
                ax.flat[j].imshow(self.BF_disc_align[i][j][top:bottom, left:right])
                if visual_title:
                    ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
            
            for a in ax.flat:
                a.axis('off')
            fig.suptitle(self.subfolders[i]+' BF disc align result')
            plt.subplots_adjust(hspace=0.1, wspace=0.1)
            if visual_title:
                fig.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig.savefig(f"{base}_sub_{i}{ext}", bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()


    def intensity_integration_image(self, visual_title=True, title_font_size=10, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'intensity_integration_image') if self._default_figure_save_path is not None else None)
    
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_avg_split[i])
            grid_size = int(np.around(np.sqrt(num_img)))
            if num_img == 1:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                ax = np.array([ax])
            elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
            else:
                fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
            for j in range(num_img):
                sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                ax.flat[j].imshow(sum_map, cmap='inferno')
                if visual_title:
                    ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
            
            for a in ax.flat:
                a.axis('off')             
            fig.suptitle(self.subfolders[i]+' sum of intensities map')
            plt.subplots_adjust(hspace=0.1, wspace=0.1)
            if visual_title:
                fig.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig.savefig(f"{base}_sub_{i}{ext}", bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()


    def basic_setup(self, str_path, from_unit, to_unit, broadening=0.01, 
                    fill_width=0.1, height=None, width=None, threshold=None, 
                    distance=None, prominence=0.001, visual=True, visual_legend=True, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'basic_setup') if self._default_figure_save_path is not None else None)
        print("Original scattering vector range = [%.6f, %.6f]"%(0, self.profile_length*self.pixel_size_inv_Ang))
        
        self.str_path = str_path
        self.from_unit = from_unit
        self.to_unit = to_unit

        self.from_ind = int(np.around(from_unit/self.pixel_size_inv_Ang))
        self.to_ind = int(np.around(to_unit/self.pixel_size_inv_Ang))
        self.from_ = self.pixel_size_inv_Ang*self.from_ind
        self.to_ = self.pixel_size_inv_Ang*self.to_ind
        self.x_axis = np.linspace(self.from_, self.to_-self.pixel_size_inv_Ang, self.to_ind-self.from_ind)
        print("Selected scattering vector range = [%.6f, %.6f]"%(self.x_axis.min(), self.x_axis.max()))
        print('Reciprocal pixel size : %.6f (original), %.6f (present)'%(self.pixel_size_inv_Ang, self.x_axis[1]-self.x_axis[0]))

        self.range_ind = [self.from_ind, self.to_ind]
        print('Selected scattering vector index range = [%d, %d]'%(self.range_ind[0], self.range_ind[1]))
        
        if str_path != []:
            int_sf = {}
            peak_sf = {}
            for adr in self.str_path:
                str_name = adr.split('/')[-1].split('.')[0]
                
                crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(adr)
                crystal.calculate_structure_factors(self.to_)
            
                int_sf[str_name] = py4DSTEM.process.diffraction.utils.calc_1D_profile(
                                            self.x_axis,
                                            crystal.g_vec_leng,
                                            crystal.struct_factors_int,
                                            k_broadening=broadening,
                                            int_scale=1.0,
                                            normalize_intensity=True)
                
                peaks = find_peaks(int_sf[str_name], 
                                   height=height, 
                                   width=width, 
                                   threshold=threshold, 
                                   distance=distance, 
                                   prominence=prominence)[0]
                
                peaks = peaks * self.pixel_size_inv_Ang
                peaks = peaks + self.from_

                peak_sf[str_name] = peaks

                if visual or _eff_save is not None:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
                    ax.plot(self.x_axis, int_sf[str_name], 'k-', label=str_name)
                    if visual_legend:
                        ax.legend(loc='right')
    
                    for j, peak in enumerate(peaks):
                        if peak >= self.from_ and peak <= self.to_:
                            print(peak)
                            ax.axvline(peak, ls=':', lw=1.5, c='r')
                            ax.fill_between([peak-fill_width, peak+fill_width], y1=np.max(int_sf[str_name]), y2=np.min(int_sf[str_name]), alpha=0.5, color='orange')
                            ax.text(peak, 1.0, "%d"%(j+1))
                    
                    fig.tight_layout()
                    if _eff_save is not None:
                        os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                        base, ext = os.path.splitext(_eff_save)
                        if not ext:
                            ext = '.png'
                        fig.savefig(f"{base}_{str_name}{ext}", bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()
            
            if visual or _eff_save is not None:
                int_sf["sum_of_all"] = np.sum(list(int_sf.values()), axis=0)
                fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
                ax.plot(self.x_axis, int_sf["sum_of_all"], 'k-', label="sum of all")
                ax.legend(loc='right')
                fig.tight_layout()
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig.savefig(f"{base}_sum_of_all{ext}", bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
    
            self.int_sf = int_sf
            self.peak_sf = peak_sf
    
            print('structure name')
            print(*int_sf.keys(), sep="\n")
    

    def sum_radial_profile(self, str_name=None, profile_type="variance", 
                           visual_legend=True, visual_title=True, title_font_size=10,
                           axis_off=True, individual_visual=True, save_path=None):            
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'sum_radial_profile') if self._default_figure_save_path is not None else None)
        
        fig_tot, ax_tot = plt.subplots(2, 1, figsize=(8, 12), dpi=100)
        
        base, ext = ("", "")
        if _eff_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
            base, ext = os.path.splitext(_eff_save)
            if not ext:
                ext = '.png'

        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_sum_split[i])
            grid_size = int(np.around(np.sqrt(num_img)))

            total_sp = []

            if individual_visual:
                if num_img == 1:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                    ax = np.array([ax])
                elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)


                fig_sub, ax_sub = plt.subplots(2, 1, figsize=(8, 12), dpi=100)
                ax_sub_twin = ax_sub[1].twinx()

            for j, sp in enumerate(self.radial_var_sum_split[i]):
                if profile_type == "variance":
                    tmp_sp = sp[self.range_ind[0]:self.range_ind[1]]
                elif profile_type == "mean":
                    tmp_sp = self.radial_avg_sum_split[i][j][self.range_ind[0]:self.range_ind[1]]
                else:
                    print("Warning! wrong profile type!")
                    return
                
                total_sp.append(tmp_sp)

                if individual_visual:
                    if profile_type == "variance":
                        ax.flat[j].plot(self.x_axis, tmp_sp, 'k-', label="var_sum")
                    elif profile_type == "mean":
                        ax.flat[j].plot(self.x_axis, tmp_sp, 'k-', label="mean_sum")
                    else:
                        print("Warning! wrong profile type!")
                        return                          
                    
                    if visual_title:
                        ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                    if visual_legend:
                        ax.flat[j].legend(loc='upper right')               
                
                    ax_twin = ax.flat[j].twinx()
                    if profile_type == "variance":
                        tmp_ap = self.radial_avg_sum_split[i][j][self.range_ind[0]:self.range_ind[1]]
                        ax_twin.plot(self.x_axis, tmp_ap, 'r:', label="mean_sum")
                    elif profile_type == "mean":
                        tmp_ap = self.radial_var_sum_split[i][j][self.range_ind[0]:self.range_ind[1]]
                        ax_twin.plot(self.x_axis, tmp_ap, 'r:', label="var_sum")
                    else:
                        print("Warning! wrong profile type!")
                        return    
                    if visual_legend:
                        ax_twin.legend(loc='right')
                    
                    if axis_off:
                        ax.flat[j].tick_params(axis="y", labelsize=0, color='white')
                        ax_twin.tick_params(axis="y", labelsize=0, color='white')
                
                    ax_sub[1].plot(self.x_axis, tmp_sp/np.max(tmp_sp), label=self.subfolders[i]+"_"+os.path.basename(self.loaded_data_path[i][j])[:15])
                    ax_sub[1].set_title("max-normalized")
                    ax_sub[0].plot(self.x_axis, tmp_sp, label=self.subfolders[i]+"_"+os.path.basename(self.loaded_data_path[i][j])[:15])
                    ax_sub[0].set_title("without normalization")
                
                    if str_name != None and str_name != []:
                        for key in str_name:
                            ax_sub_twin.plot(self.x_axis, self.int_sf[key], label=key, linestyle=":")
                        ax_sub_twin.legend(loc="right")

            if individual_visual:
                if visual_legend:
                    ax_sub[0].legend(loc="upper right")
                    ax_sub[1].legend(loc="upper right")
                fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(self.from_, self.to_))
                fig.tight_layout()

                if profile_type == "variance":
                    fig_sub.suptitle("mean of radial variance profiles - scattering vector range %.3f-%.3f (1/Å)"%(self.from_, self.to_))
                else:
                    fig_sub.suptitle("mean of radial mean profiles - scattering vector range %.3f-%.3f (1/Å)"%(self.from_, self.to_))
                fig_sub.tight_layout()
                
                if _eff_save is not None:
                    fig.savefig(f"{base}_sub_{i}_individual{ext}", bbox_inches='tight')
                    fig_sub.savefig(f"{base}_sub_{i}_summary{ext}", bbox_inches='tight')
                    plt.close(fig)
                    plt.close(fig_sub)

            mean_tot = np.mean(total_sp, axis=0)
            ax_tot[0].plot(self.x_axis, mean_tot, label=self.subfolders[i])
            ax_tot[1].plot(self.x_axis, mean_tot/np.max(mean_tot), label=self.subfolders[i]+" (max-normalized)")
            ax_tot[0].legend(loc="upper right")            
            ax_tot[1].legend(loc="upper right")
            
        if profile_type == "variance":
            fig_tot.suptitle("Compare the mean of radial variance profiles between subfolders")
        else:
            fig_tot.suptitle("Compare the mean of radial mean profiles between subfolders")
        fig_tot.tight_layout()
        if _eff_save is not None:
            fig_tot.savefig(f"{base}_total{ext}", bbox_inches='tight')
            plt.close(fig_tot)
        else:
            plt.show()


    def NMF_decompose(self, num_comp, scale_crop=True, rescale_SI=False, max_normalize=False, rescale_0to1=True, profile_type="variance", verbose=True, tolerance=1E-4, coeff_map_type="absolute"):
        
        self.num_comp = num_comp
        self.NMF_profile_type = profile_type
        self.nmf_params = {
            "scale_crop": scale_crop,
            "rescale_SI": rescale_SI,
            "max_normalize": max_normalize,
            "rescale_0to1": rescale_0to1,
            "tolerance": tolerance,
            "coeff_map_type": coeff_map_type
        }
        # NMF - load data
        flat_adr = []
        preloaded_data = []
        if profile_type == "variance":
            for adrs in self.loaded_data_path:
                flat_adr.extend(adrs)
            for sublist in self.radial_var_split:
                preloaded_data.extend(sublist)
        elif profile_type == "mean":
            for adrs in self.loaded_data_mean_path:
                flat_adr.extend(adrs)
            for sublist in self.radial_avg_split:
                preloaded_data.extend(sublist)
        elif profile_type == "zernike":
            for adrs in self.loaded_data_zernike_path:
                flat_adr.extend(adrs)            
            for sublist in self.zernike_split:
                preloaded_data.extend(sublist)
        else:
            print("Warning! wrong profile type!")
            return
        
        if profile_type == "zernike":
            self.run_SI = feature_extract(flat_adr, dat_dim=3, dat_unit='index j', cr_range=[0, self.zernike_length, 1], 
                                    dat_scale=1, rescale=rescale_SI, DM_file=True, verbose=verbose, rebin_256=self.rebin_256, data_storage=preloaded_data)
        else:
            if scale_crop:
                self.run_SI = feature_extract(flat_adr, dat_dim=3, dat_unit='1/Å', cr_range=[self.from_, self.to_, self.pixel_size_inv_Ang], 
                                        dat_scale=1, rescale=rescale_SI, DM_file=True, verbose=verbose, rebin_256=self.rebin_256, data_storage=preloaded_data)
            else:
                self.run_SI = feature_extract(flat_adr, dat_dim=3, dat_unit='1/Å', cr_range=[self.from_ind, self.to_ind, 1], 
                                        dat_scale=self.pixel_size_inv_Ang, rescale=rescale_SI, DM_file=True, verbose=verbose, rebin_256=self.rebin_256, data_storage=preloaded_data)           

        # NMF - prepare the input dataset
        self.run_SI.make_input(min_val=0.0, max_normalize=max_normalize, rescale_0to1=rescale_0to1)

        # NMF - NMF decomposition and visualization
        self.run_SI.ini_DR(method="nmf", num_comp=num_comp, result_visual=verbose, intensity_range=coeff_map_type, tolerance=tolerance)
        self.nmf_reconstruction_err = getattr(self.run_SI.DR, 'reconstruction_err_', None)


    def NMF_result(self, lv_show=None, transparency_percentile=100, visual_title=True, title_font_size=10, save_path=None):
        
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'NMF_result') if self._default_figure_save_path is not None else None)
        base, ext = ("", "")
        if _eff_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
            base, ext = os.path.splitext(_eff_save)
            if not ext:
                ext = '.png'

        # Loading vectors
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        for lv in range(self.num_comp):
            if self.NMF_profile_type == "zernike":
                tmp_ax, = ax.plot(self.run_SI.DR_comp_vectors[lv], self.color_rep[lv+1], label="lv %d"%(lv+1))
            else:
                tmp_ax, = ax.plot(self.x_axis, self.run_SI.DR_comp_vectors[lv], self.color_rep[lv+1], label="lv %d"%(lv+1))

            shadow_effect = path_effects.withStroke(linewidth=3, foreground='gray')
            tmp_ax.set_path_effects([shadow_effect])
                
        # ax.set_facecolor("lightgray")
        ax.legend(loc='upper right')
        fig.tight_layout()
        if _eff_save is not None:
            fig.savefig(f"{base}_loading_vectors{ext}", bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        # All coefficient maps in one image plot

        if lv_show == None or lv_show == []:
            if self.num_comp <= len(self.cm_rep)-1:
                num_comp = self.num_comp
                k = 0
                for i in range(len(self.subfolders)):
                    num_img = len(self.radial_var_split[i])
                    grid_size = int(np.around(np.sqrt(num_img)))
                    if num_img == 1:
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                        ax = np.array([ax])
                    elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                        fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
                    else:
                        fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
                    for j in range(num_img):
                        sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                        ax.flat[j].imshow(sum_map, cmap='gray')
                        for lv in range(num_comp):
                            transparency = self.run_SI.coeffs_reshape[k][:, :, lv]/np.max(self.run_SI.coeffs_reshape[k][:, :, lv].flatten())
                            transparency = np.nan_to_num(transparency)
                            if transparency_percentile != 100:
                                transparency[transparency>=np.percentile(self.run_SI.coeffs_reshape[k][:, :, lv].flatten(), transparency_percentile)] = 1.0
                            ax.flat[j].imshow(self.run_SI.coeffs_reshape[k][:, :, lv], 
                                              cmap=self.cm_rep[lv+1], 
                                              alpha=transparency)
                        if visual_title:
                            ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                        k += 1
                    for a in ax.flat:
                        a.axis('off')
                    fig.suptitle(self.subfolders[i])
                    plt.subplots_adjust(hspace=0.1, wspace=0.1)
                    if visual_title:
                        fig.tight_layout()
                    if _eff_save is not None:
                        fig.savefig(f"{base}_coeff_maps_sub_{i}{ext}", bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()
    
            else:
                print("#############################################################################################")
                print("####################################     Caution!      ######################################")
                print("#############################################################################################")
                print('The number of loading vectors exceeds the number of the preset colormaps.')
                print(self.cm_rep[1:])
                print('So, it will show the coefficient maps for only loading vector 1-%d'%(len(self.cm_rep)-1))
                num_comp = len(self.cm_rep)-1
                k = 0
                for i in range(len(self.subfolders)):
                    num_img = len(self.radial_var_split[i])
                    grid_size = int(np.around(np.sqrt(num_img)))
                    if num_img == 1:
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                        ax = np.array([ax])
                    elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                        fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
                    else:
                        fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
                    for j in range(num_img):
                        sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                        ax.flat[j].imshow(sum_map, cmap='gray')
                        for lv in range(num_comp):
                            transparency = self.run_SI.coeffs_reshape[k][:, :, lv]/np.max(self.run_SI.coeffs_reshape[k][:, :, lv].flatten())
                            transparency = np.nan_to_num(transparency)
                            if transparency_percentile != 100:
                                transparency[transparency>=np.percentile(self.run_SI.coeffs_reshape[k][:, :, lv].flatten(), transparency_percentile)] = 1.0
                            ax.flat[j].imshow(self.run_SI.coeffs_reshape[k][:, :, lv], 
                                              cmap=self.cm_rep[lv+1], 
                                              alpha=transparency)
                        if visual_title:
                            ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                        k += 1
                    for a in ax.flat:
                        a.axis('off')
                    fig.suptitle(self.subfolders[i])
                    plt.subplots_adjust(hspace=0.1, wspace=0.1)
                    if visual_title:
                        fig.tight_layout()
                    if _eff_save is not None:
                        fig.savefig(f"{base}_coeff_maps_sub_{i}{ext}", bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()

        elif lv_show != None and lv_show != []:
            print("#############################################################################################")
            print("####################################     Caution!      ######################################")
            print("#############################################################################################")
            num_comp = self.num_comp
            k = 0
            for i in range(len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if num_img == 1:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                    ax = np.array([ax])
                elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
                for j in range(num_img):
                    sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                    ax.flat[j].imshow(sum_map, cmap='gray')
                    for c, lvs in enumerate(lv_show):
                        lv = lvs-1
                        if i == 0 and j == 0:
                            print("Color map of loading vector %d = %s"%(lvs, self.cm_rep[c+1]))
                        transparency = self.run_SI.coeffs_reshape[k][:, :, lv]/np.max(self.run_SI.coeffs_reshape[k][:, :, lv].flatten())
                        transparency = np.nan_to_num(transparency)
                        if transparency_percentile != 100:
                            transparency[transparency>=np.percentile(self.run_SI.coeffs_reshape[k][:, :, lv].flatten(), transparency_percentile)] = 1.0
                        ax.flat[j].imshow(self.run_SI.coeffs_reshape[k][:, :, lv], 
                                          cmap=self.cm_rep[c+1], 
                                          alpha=transparency)
                    if visual_title:
                        ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                    k += 1
                for a in ax.flat:
                    a.axis('off')
                fig.suptitle(self.subfolders[i])
                plt.subplots_adjust(hspace=0.1, wspace=0.1)
                if visual_title:
                    fig.tight_layout()
                if _eff_save is not None:
                    fig.savefig(f"{base}_coeff_maps_sub_{i}{ext}", bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()            


    def NMF_comparison(self, str_name=None, percentile_threshold=90, ref_variance=0.7, 
                       visual_title=True, title_font_size=10, axis_off=True, visual_individual=True, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'NMF_comparison') if self._default_figure_save_path is not None else None)
        self.percentile_threshold = percentile_threshold
        # Show the pixels with high coefficients for each loading vector and the averaged profiles for the mask region
        coeff_split = []
        thresh_coeff_split = []
        lv_mean_split = []
        lv_line = []
        high_coeff_area_split = []

        for lv in range(self.num_comp):
            if self.NMF_profile_type == 'zernike':
                lv_tot = np.zeros(self.zernike_length)
            else:
                lv_tot = np.zeros(self.profile_length)
            total_num = 0
            coeff_lv = []
            thresh_coeff_lv = []
            high_coeff_area_lv = []
            lv_mean_lv = []
            fig_lv, ax_lv = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
            if self.NMF_profile_type == 'zernike':
                tmp_ax, = ax_lv[0].plot(self.run_SI.DR_comp_vectors[lv], self.color_rep[lv+1])
            else:
                tmp_ax, = ax_lv[0].plot(self.x_axis, self.run_SI.DR_comp_vectors[lv], self.color_rep[lv+1])

            shadow_effect = path_effects.withStroke(linewidth=3, foreground='gray')
            tmp_ax.set_path_effects([shadow_effect])

            ax_twin = ax_lv[0].twinx()
            if str_name != None and str_name != []:
                for key in str_name:
                    ax_twin.plot(self.x_axis, self.int_sf[key], label=key, linestyle=":")
                ax_twin.legend(loc="right")
            # ax_lv[0].set_facecolor("lightgray")
            fig_lv.suptitle("Loading vector %d"%(lv+1))
        
            thresh = np.percentile(self.run_SI.DR_coeffs[:, lv], percentile_threshold)
            print("threshold coefficient value for loading vector %d = %f"%(lv+1, thresh))

            k=0
            for i in range(len(self.subfolders)):
                if self.NMF_profile_type == 'zernike':
                    lv_sub = np.zeros(self.zernike_length)
                else:
                    lv_sub = np.zeros(self.profile_length)
                sub_num = 0
                coeff = []
                thresh_coeff = []
                high_coeff_area = []
                lv_mean = []
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                fig_sub_tot, ax_sub_tot = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
                if visual_individual:
                    if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                        fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12), dpi=100)
                    else:
                        fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10), dpi=100)
                    
                for j in range(num_img):
                    coeff.append(self.run_SI.coeffs_reshape[k][:, :, lv])
                    coeff_map = self.run_SI.coeffs_reshape[k][:, :, lv].copy()
                    coeff_map[coeff_map<=thresh] = 0
                    coeff_map[coeff_map>thresh] = 1
                    thresh_coeff.append(coeff_map)
                    area = self.radial_var_split[i][j].axes_manager[0].scale**2 * np.sum(coeff_map)
                    high_coeff_area.append(area)

                    if visual_individual:
                        ax.flat[j*2].imshow(coeff_map, cmap='gray')
                        if visual_title:
                            ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                        ax.flat[j*2].axis("off")
                    if len(np.where(coeff_map==1)[0]) != 0:
                        tmp_num = len(np.where(coeff_map==1)[0])
                        sub_num += tmp_num
                        total_num += tmp_num
                        if self.NMF_profile_type == "variance":
                            coeff_rv = np.sum(self.radial_var_split[i][j].data[np.where(coeff_map==1)], axis=0)
                            coeff_mean = np.mean(self.radial_var_split[i][j].data[np.where(coeff_map==1)], axis=0)
                        elif self.NMF_profile_type == "mean":
                            coeff_rv = np.sum(self.radial_avg_split[i][j].data[np.where(coeff_map==1)], axis=0)
                            coeff_mean = np.mean(self.radial_avg_split[i][j].data[np.where(coeff_map==1)], axis=0)
                        elif self.NMF_profile_type == "zernike":
                            coeff_rv = np.sum(self.zernike_split[i][j].data[np.where(coeff_map==1)], axis=0)
                            coeff_mean = np.mean(self.zernike_split[i][j].data[np.where(coeff_map==1)], axis=0)
                        else:
                            print("This does not support the current profile type %s"%self.NMF_profile_type)
                            
                        lv_tot += coeff_rv
                        lv_sub += coeff_rv

                        lv_mean.append(coeff_mean)
                        
                        if visual_individual:
                            if self.NMF_profile_type == "zernike":
                                ax.flat[j*2+1].plot(coeff_mean, 'k-')
                            else:
                                ax.flat[j*2+1].plot(self.x_axis, coeff_mean[self.range_ind[0]:self.range_ind[1]], 'k-')
                            if axis_off:
                                ax.flat[j*2+1].tick_params(axis="y", labelsize=0, color='white')
                            #ax.flat[j*2+1].set_ylim(0.0, ref_variance*1.5)
                            #ax.flat[j*2+1].hlines(y=ref_variance, xmin=self.x_axis.min(), xmax=self.x_axis.max(), color="k", linestyle=":", alpha=0.5)
                            #ax.flat[j*2+1].plot(self.x_axis, self.radial_var_sum_split[i][j][self.range_ind[0]:self.range_ind[1]], 'k:', alpha=0.5)
                            ax_lv_contri = ax.flat[j*2+1].twinx()
                            for lva in range(self.num_comp):
                                mean_coeff = np.mean(self.run_SI.coeffs_reshape[k][:, :, lva][np.where(coeff_map==1)])
                                if self.NMF_profile_type == "zernike":
                                    ax_lv_contri.plot(self.run_SI.DR_comp_vectors[lva]*mean_coeff, self.color_rep[lva+1], alpha=0.7)
                                else:
                                    ax_lv_contri.plot(self.x_axis, self.run_SI.DR_comp_vectors[lva]*mean_coeff, self.color_rep[lva+1], alpha=0.7)
                            if visual_title:
                                ax.flat[j*2+1].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                            ax.flat[j*2+1].set_facecolor("lightgray")
                    else:
                        if self.NMF_profile_type == "zernike":
                            lv_mean.append(np.zeros(self.zernike_length))
                        else:
                            lv_mean.append(np.zeros(self.profile_length))
                        
                    k+=1
                    
                if sub_num != 0:
                    lv_sub /= sub_num
                    
                if self.NMF_profile_type == "zernike":
                    ax_sub_tot.plot(lv_sub, 'k-')
                else:
                    ax_sub_tot.plot(self.x_axis, lv_sub[self.range_ind[0]:self.range_ind[1]], 'k-')
                ax_sub_tot.set_title("sum of profiles for all threshold maps - subfolder by subfolder")
                ax_sub_twin = ax_sub_tot.twinx()
                if str_name != None and str_name != []:
                    for key in str_name:
                        ax_sub_twin.plot(self.x_axis, self.int_sf[key], label=key, linestyle=":")
                    ax_sub_twin.legend(loc="right")
                fig_sub_tot.tight_layout()
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig_sub_tot.savefig(f"{base}_lv_{lv+1}_sub_{i}_total{ext}", bbox_inches='tight')
                    plt.close(fig_sub_tot)

                if self.NMF_profile_type == "zernike":
                    tmp_ax, = ax_lv[2].plot(lv_sub, c=self.color_rep[i+1], label=self.subfolders[i])
                else:
                    tmp_ax, = ax_lv[2].plot(self.x_axis, lv_sub[self.range_ind[0]:self.range_ind[1]], c=self.color_rep[i+1], label=self.subfolders[i])

                shadow_effect = path_effects.withStroke(linewidth=3, foreground='gray')
                tmp_ax.set_path_effects([shadow_effect])                
                
                if visual_individual:
                    fig.suptitle(self.subfolders[i]+' threshold coefficient map for loading vector %d'%(lv+1))
                    fig.tight_layout()
                    if _eff_save is not None:
                        os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                        base, ext = os.path.splitext(_eff_save)
                        if not ext:
                            ext = '.png'
                        fig.savefig(f"{base}_lv_{lv+1}_sub_{i}{ext}", bbox_inches='tight')
                        plt.close(fig)
                    else:
                        pass  # fig_sub_tot shown below or keep open
                    
                coeff_lv.append(coeff)
                thresh_coeff_lv.append(thresh_coeff)
                high_coeff_area_lv.append(high_coeff_area)
                lv_mean_lv.append(lv_mean)
                
            coeff_split.append(coeff_lv)
            thresh_coeff_split.append(thresh_coeff_lv)
            high_coeff_area_split.append(high_coeff_area_lv)
            lv_mean_split.append(lv_mean_lv)
            
            if total_num != 0:
                lv_tot /= total_num
            lv_line.append(lv_tot)
            if self.NMF_profile_type == "zernike":
                ax_lv[1].plot(lv_tot, 'k-')
            else:
                ax_lv[1].plot(self.x_axis, lv_tot[self.range_ind[0]:self.range_ind[1]], 'k-')
            ax_lv[1].set_title("sum of profiles for all threshold maps - loading vector %d"%(lv+1))
            ax_lv[2].legend()
            fig_lv.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig_lv.savefig(f"{base}_lv_{lv+1}{ext}", bbox_inches='tight')
                plt.close(fig_lv)

        fig_tot, ax_tot = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        for l, line in enumerate(lv_line):
            if self.NMF_profile_type == "zernike":
                ax_tot.plot(line, c=self.color_rep[l+1], label='lv %d'%(l+1))
            else:
                ax_tot.plot(self.x_axis, line[self.range_ind[0]:self.range_ind[1]], c=self.color_rep[l+1], label='lv %d'%(l+1))
                
        ax_tot.legend()
        fig_tot.suptitle("Compare the mean of radial profiles between loading vectors")
        fig_tot.tight_layout()
        if _eff_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
            base, ext = os.path.splitext(_eff_save)
            if not ext:
                ext = '.png'
            fig_tot.savefig(f"{base}_total{ext}", bbox_inches='tight')
            plt.close(fig_tot)
        else:
            plt.show()
            
        self.coeff_split = coeff_split
        self.thresh_coeff_split = thresh_coeff_split
        self.high_coeff_area_split = high_coeff_area_split
        self.lv_mean_split = lv_mean_split

        return lv_line


    def high_coeff_area_comparison(self, save_path=None):

        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'high_coeff_area_comparison') if self._default_figure_save_path is not None else None)
        lv_coeff_area_mean = []
        lv_coeff_area_std = []
        for lv in range(self.num_comp):
            lv_coeff_area_sub_mean = []
            lv_coeff_area_sub_std = []
            for i in range(len(self.subfolders)):
                lv_coeff_area_sub_mean.append(np.mean(self.high_coeff_area_split[lv][i]))
                lv_coeff_area_sub_std.append(np.std(self.high_coeff_area_split[lv][i]))
            lv_coeff_area_mean.append(lv_coeff_area_sub_mean)
            lv_coeff_area_std.append(lv_coeff_area_sub_std)
            
        for lv in range(self.num_comp):
            fig, ax = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
            ax.plot(self.subfolders, lv_coeff_area_mean[lv], 'k-')
            ax.scatter(self.subfolders, lv_coeff_area_mean[lv], c='r', marker="*")
            ax.errorbar(self.subfolders, lv_coeff_area_mean[lv], yerr=lv_coeff_area_std[lv], capsize=5, c='b')
            fig.suptitle("Effective high coeffcieint area of loading vector %d by subfolder"%(lv+1))
            # plt.subplots_adjust(hspace=0.02, wspace=0.02)
            fig.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig.savefig(f"{base}_lv_{lv+1}{ext}", bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

        self.lv_coeff_area_mean = lv_coeff_area_mean
        self.lv_coeff_area_std = lv_coeff_area_std


    def NMF_summary_save(self, save=False, also_dp=False, log_scale_dp=True, also_tiff=False, fill_width=0.01, prominence_lv=0.001, prominence_profile=0.001, figure_save_path=None):
        _eff_save = figure_save_path if figure_save_path is not None else (os.path.join(self._default_figure_save_path, 'NMF_summary_save') if self._default_figure_save_path is not None else None)
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_split[i])
            for j in range(num_img):
                if also_dp:
                    dataset = hs.load(self.loaded_data_path[i][j][:-18]+'corrected_scaled.hspy')
                    if self.rebin_256:
                        if dataset.data.shape[1] > 250:
                            dataset = dataset.rebin(scale = (2,2,1,1))             
                
                save_path = os.path.dirname(self.loaded_data_path[i][j]) # able to change the base directory for saving
                print("save directory: ", save_path)
                data_name = os.path.basename(self.loaded_data_path[i][j]).split("_")
                data_name = data_name[0]+'_'+data_name[1]
                print("save prefix: ", data_name)
                top, bottom, left, right = self.crop
                fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
                ax[0].imshow(self.BF_disc_align[i][j][top:bottom, left:right], cmap='inferno')
                ax[0].set_title("Aligned BF disc")
                ax[0].axis("off")

                sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                ax[1].imshow(sum_map, cmap='inferno')
                ax[1].set_title("Intensity map")
                ax[1].axis("off")                 
        
                rv = self.radial_var_sum_split[i][j]
                ax[2].plot(self.x_axis, rv[self.range_ind[0]:self.range_ind[1]], 'k-', label="var_sum")
                ax[2].set_title("Sum of radial variance/mean profiles")
                ax[2].legend(loc='upper right')
                
                ra = self.radial_avg_sum_split[i][j]
                ax_twin = ax[2].twinx()
                ax_twin.plot(self.x_axis, ra[self.range_ind[0]:self.range_ind[1]], 'r:', label="mean_sum")
                ax_twin.legend(loc='right')

                fig.suptitle(self.subfolders[i]+" - "+os.path.basename(self.loaded_data_path[i][j])[:15])
                fig.tight_layout()
                fig.savefig(save_path+'/'+data_name+"_summary.png")
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig.savefig(f"{base}_summary_{i}_{j}{ext}", bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()

                if save:
                    sum_map = hs.signals.Signal2D(sum_map)
                    sum_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                    sum_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                    sum_map.save(save_path+'/'+data_name+"_intensity_map.hspy", overwrite=True)
                    rv = hs.signals.Signal1D(rv)
                    rv.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                    rv.save(save_path+'/'+data_name+"_mean_radial_variance_profile.hspy", overwrite=True)
                    ra = hs.signals.Signal1D(ra)
                    ra.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                    ra.save(save_path+'/'+data_name+"_mean_radial_mean_profile.hspy", overwrite=True)
                    if also_tiff:
                        tifffile.imwrite(save_path+'/'+data_name+"_intensity_map.tif", sum_map.data)
                        tifffile.imwrite(save_path+'/'+data_name+"_mean_radial_variance_profile.tif", rv.data)
                        tifffile.imwrite(save_path+'/'+data_name+"_mean_radial_mean_profile.tif", ra.data)

                for lv in range(self.num_comp):
                    if also_dp:
                        fig_lv, ax_lv = plt.subplots(3, 2, figsize=(14, 21), dpi=100)
                    else:
                        fig_lv, ax_lv = plt.subplots(2, 2, figsize=(14, 14), dpi=100)
                    ax_lv[0, 0].plot(self.x_axis, self.run_SI.DR_comp_vectors[lv], self.color_rep[lv+1])
                    ax_lv[0, 0].set_title("Loading vector %d"%(lv+1))
                    peaks = find_peaks(self.run_SI.DR_comp_vectors[lv], prominence=prominence_lv)[0]
                    
                    peaks = peaks * self.pixel_size_inv_Ang
                    peaks = peaks + self.from_
                    print("Peak positions of loading vector %d"%(lv+1))
                    for ip, peak in enumerate(peaks):
                        if peak >= self.from_ and peak <= self.to_:
                            print(peak)
                            ax_lv[0, 0].axvline(peak, ls=':', lw=1.5, c='r')
                            ax_lv[0, 0].fill_between([peak-fill_width, peak+fill_width], y1=np.max(self.run_SI.DR_comp_vectors[lv]), y2=np.min(self.run_SI.DR_comp_vectors[lv]), alpha=0.5, color='orange')
                            ax_lv[0, 0].text(peak, np.max(self.run_SI.DR_comp_vectors[lv]), "%.3f"%(peak))
                    
                    ax_lv[0, 1].plot(self.x_axis, self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]], 'k-')
                    ax_lv[0, 1].set_title("Mean profile for the lv %d coeff threshold map"%(lv+1))
                    peaks = find_peaks(self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]], prominence=prominence_profile)[0]
                    
                    peaks = peaks * self.pixel_size_inv_Ang
                    peaks = peaks + self.from_
                    print("Peak positions of the mean profile")
                    for ip, peak in enumerate(peaks):
                        if peak >= self.from_ and peak <= self.to_:
                            print(peak)
                            ax_lv[0, 1].axvline(peak, ls=':', lw=1.5, c='r')
                            ax_lv[0, 1].fill_between([peak-fill_width, peak+fill_width], 
                                                  y1=np.max(self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]]), 
                                                  y2=np.min(self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]]), 
                                                  alpha=0.5, color='orange')
                            ax_lv[0, 1].text(peak, np.max(self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]]), "%.3f"%(peak))

                    ax_lv[1, 0].imshow(self.coeff_split[lv][i][j], cmap='gray')
                    ax_lv[1, 0].set_title("lv %d coefficient map"%(lv+1))
                    ax_lv[1, 0].axis("off")
                    ax_lv[1, 1].imshow(self.thresh_coeff_split[lv][i][j], cmap='gray')
                    ax_lv[1, 1].set_title("lv %d threshold map"%(lv+1))
                    ax_lv[1, 1].axis("off")

                    if save:
                        coeff_map = hs.signals.Signal2D(self.coeff_split[lv][i][j])
                        coeff_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                        coeff_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                        coeff_map.save(save_path+'/'+data_name+"_%d_lv_coeff_map.hspy"%(lv+1), overwrite=True)
                        th_map = hs.signals.Signal2D(self.thresh_coeff_split[lv][i][j])
                        th_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                        th_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                        th_map.save(save_path+'/'+data_name+"_%d_lv_coeff_threshold_map.hspy"%(lv+1), overwrite=True)
                        if also_tiff:
                            tifffile.imwrite(save_path+'/'+data_name+"_%d_lv_coeff_map.tif"%(lv+1), coeff_map.data)
                            tifffile.imwrite(save_path+'/'+data_name+"_%d_lv_coeff_threshold_map.tif"%(lv+1), th_map.data)
                    
                    if also_dp and len(np.nonzero(self.thresh_coeff_split[lv][i][j])[0]) != 0:
                        mean_dp = np.sum(dataset.data[np.where(self.thresh_coeff_split[lv][i][j]==1)], axis=0)
                        if save:
                            mean_dp_save = hs.signals.Signal2D(mean_dp)
                            mean_dp_save.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                            mean_dp_save.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                            mean_dp_save.save(save_path+'/'+data_name+"_mean_diffraction_pattern_%d_lv_coeff_threshold_map.hspy"%(lv+1), overwrite=True)
                            if also_tiff:
                                tifffile.imwrite(save_path+'/'+data_name+"_mean_diffraction_pattern_%d_lv_coeff_threshold_map.tif"%(lv+1), mean_dp_save.data)
                                
                        if log_scale_dp:
                            mean_dp[mean_dp <= 0] = 1.0
                            ax_lv[2, 0].imshow(np.log(mean_dp).clip(min=0.0), cmap='gray')
                            ax_lv[2, 0].set_title('(log-scale) Mean diffraction pattern\nfor the high-variance map')
                            ax_lv[2, 0].axis("off")
                        else:
                            ax_lv[2, 0].imshow(mean_dp.clip(min=0.0), cmap='gray')
                            ax_lv[2, 0].set_title('Mean diffraction pattern\nfor the high-variance map')
                            ax_lv[2, 0].axis("off")
                            
                        max_dp = np.max(dataset.data[np.where(self.thresh_coeff_split[lv][i][j]==1)], axis=0)
                        if save:
                            max_dp_save = hs.signals.Signal2D(max_dp)
                            max_dp_save.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                            max_dp_save.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                            max_dp_save.save(save_path+'/'+data_name+"_max_diffraction_pattern_%d_lv_coeff_threshold_map.hspy"%(lv+1), overwrite=True)
                            if also_tiff:
                                tifffile.imwrite(save_path+'/'+data_name+"_max_diffraction_pattern_%d_lv_coeff_threshold_map.tif"%(lv+1), max_dp_save.data)
                                
                        if log_scale_dp:
                            max_dp[max_dp <= 0] = 1.0
                            ax_lv[2, 1].imshow(np.log(max_dp).clip(min=0.0), cmap='gray')
                            ax_lv[2, 1].set_title('(log-scale) Maximum diffraction pattern\nfor the thresholding map')
                            ax_lv[2, 1].axis("off")
                        else:
                            ax_lv[2, 1].imshow(max_dp.clip(min=0.0), cmap='gray')
                            ax_lv[2, 1].set_title('Maximum diffraction pattern\nfor the high-variance map')
                            ax_lv[2, 1].axis("off")

                    fig_lv.tight_layout()
                    fig_lv.savefig(save_path+'/'+data_name+"_NMF_%d_lv_summary.png"%(lv+1))
                    if _eff_save is not None:
                        os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                        base, ext = os.path.splitext(_eff_save)
                        if not ext:
                            ext = '.png'
                        fig_lv.savefig(f"{base}_lv_{lv+1}_{i}_{j}{ext}", bbox_inches='tight')
                        plt.close(fig_lv)
                    else:
                        plt.show()
                    
                if also_dp:
                    del dataset # release the occupied memory

          
    def NMF_summary_save_specific(self, save=False, also_dp=False, log_scale_dp=True, also_tiff=False, fill_width=0.01, prominence_lv=0.001, prominence_profile=0.001, specific_data=[], figure_save_path=None):
        _eff_save = figure_save_path if figure_save_path is not None else (os.path.join(self._default_figure_save_path, 'NMF_summary_save_specific') if self._default_figure_save_path is not None else None)
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_split[i])
            for j in range(num_img):
                for key in specific_data:
                    if key in self.loaded_data_path[i][j]:
                        if also_dp:
                            dataset = hs.load(self.loaded_data_path[i][j][:-18]+'corrected_scaled.hspy')
                            if self.rebin_256:
                                if dataset.data.shape[1] > 250:
                                    dataset = dataset.rebin(scale = (2,2,1,1))               

                        save_path = os.path.dirname(self.loaded_data_path[i][j]) # able to change the base directory for saving
                        print("save directory: ", save_path)
                        data_name = os.path.basename(self.loaded_data_path[i][j]).split("_")
                        data_name = data_name[0]+'_'+data_name[1]
                        print("save prefix: ", data_name)
                        top, bottom, left, right = self.crop
                        fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
                        ax[0].imshow(self.BF_disc_align[i][j][top:bottom, left:right], cmap='inferno')
                        ax[0].set_title("Aligned BF disc")
                        ax[0].axis("off")

                        sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                        ax[1].imshow(sum_map, cmap='inferno')
                        ax[1].set_title("Intensity map")
                        ax[1].axis("off")                 

                        rv = self.radial_var_sum_split[i][j]
                        ax[2].plot(self.x_axis, rv[self.range_ind[0]:self.range_ind[1]], 'k-', label="var_sum")
                        ax[2].set_title("Sum of radial variance/mean profiles")
                        ax[2].legend(loc='upper right')

                        ra = self.radial_avg_sum_split[i][j]
                        ax_twin = ax[2].twinx()
                        ax_twin.plot(self.x_axis, ra[self.range_ind[0]:self.range_ind[1]], 'r:', label="mean_sum")
                        ax_twin.legend(loc='right')

                        fig.suptitle(self.subfolders[i]+" - "+os.path.basename(self.loaded_data_path[i][j])[:15])
                        fig.tight_layout()
                        fig.savefig(save_path+'/'+data_name+"_summary.png")
                        if _eff_save is not None:
                            os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                            base, ext = os.path.splitext(_eff_save)
                            if not ext:
                                ext = '.png'
                            fig.savefig(f"{base}_summary_{i}_{j}{ext}", bbox_inches='tight')
                            plt.close(fig)
                        else:
                            plt.show()

                        if save:
                            sum_map = hs.signals.Signal2D(sum_map)
                            sum_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                            sum_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                            sum_map.save(save_path+'/'+data_name+"_intensity_map.hspy", overwrite=True)
                            rv = hs.signals.Signal1D(rv)
                            rv.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                            rv.save(save_path+'/'+data_name+"_mean_radial_variance_profile.hspy", overwrite=True)
                            ra = hs.signals.Signal1D(ra)
                            ra.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                            ra.save(save_path+'/'+data_name+"_mean_radial_mean_profile.hspy", overwrite=True)
                            if also_tiff:
                                tifffile.imwrite(save_path+'/'+data_name+"_intensity_map.tif", sum_map.data)
                                tifffile.imwrite(save_path+'/'+data_name+"_mean_radial_variance_profile.tif", rv.data)
                                tifffile.imwrite(save_path+'/'+data_name+"_mean_radial_mean_profile.tif", ra.data)

                        for lv in range(self.num_comp):
                            if also_dp:
                                fig_lv, ax_lv = plt.subplots(3, 2, figsize=(14, 21), dpi=100)
                            else:
                                fig_lv, ax_lv = plt.subplots(2, 2, figsize=(14, 14), dpi=100)
                            ax_lv[0, 0].plot(self.x_axis, self.run_SI.DR_comp_vectors[lv], self.color_rep[lv+1])
                            ax_lv[0, 0].set_title("Loading vector %d"%(lv+1))
                            peaks = find_peaks(self.run_SI.DR_comp_vectors[lv], prominence=prominence_lv)[0]

                            peaks = peaks * self.pixel_size_inv_Ang
                            peaks = peaks + self.from_
                            print("Peak positions of loading vector %d"%(lv+1))
                            for ip, peak in enumerate(peaks):
                                if peak >= self.from_ and peak <= self.to_:
                                    print(peak)
                                    ax_lv[0, 0].axvline(peak, ls=':', lw=1.5, c='r')
                                    ax_lv[0, 0].fill_between([peak-fill_width, peak+fill_width], y1=np.max(self.run_SI.DR_comp_vectors[lv]), y2=np.min(self.run_SI.DR_comp_vectors[lv]), alpha=0.5, color='orange')
                                    ax_lv[0, 0].text(peak, np.max(self.run_SI.DR_comp_vectors[lv]), "%.3f"%(peak))

                            ax_lv[0, 1].plot(self.x_axis, self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]], 'k-')
                            ax_lv[0, 1].set_title("Mean profile for the lv %d coeff threshold map"%(lv+1))
                            peaks = find_peaks(self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]], prominence=prominence_profile)[0]

                            peaks = peaks * self.pixel_size_inv_Ang
                            peaks = peaks + self.from_
                            print("Peak positions of the mean profile")
                            for ip, peak in enumerate(peaks):
                                if peak >= self.from_ and peak <= self.to_:
                                    print(peak)
                                    ax_lv[0, 1].axvline(peak, ls=':', lw=1.5, c='r')
                                    ax_lv[0, 1].fill_between([peak-fill_width, peak+fill_width], 
                                                          y1=np.max(self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]]), 
                                                          y2=np.min(self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]]), 
                                                          alpha=0.5, color='orange')
                                    ax_lv[0, 1].text(peak, np.max(self.lv_mean_split[lv][i][j][self.range_ind[0]:self.range_ind[1]]), "%.3f"%(peak))

                            ax_lv[1, 0].imshow(self.coeff_split[lv][i][j], cmap='gray')
                            ax_lv[1, 0].set_title("lv %d coefficient map"%(lv+1))
                            ax_lv[1, 0].axis("off")
                            ax_lv[1, 1].imshow(self.thresh_coeff_split[lv][i][j], cmap='gray')
                            ax_lv[1, 1].set_title("lv %d threshold map"%(lv+1))
                            ax_lv[1, 1].axis("off")

                            if save:
                                coeff_map = hs.signals.Signal2D(self.coeff_split[lv][i][j])
                                coeff_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                                coeff_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                                coeff_map.save(save_path+'/'+data_name+"_%d_lv_coeff_map.hspy"%(lv+1), overwrite=True)
                                th_map = hs.signals.Signal2D(self.thresh_coeff_split[lv][i][j])
                                th_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                                th_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                                th_map.save(save_path+'/'+data_name+"_%d_lv_coeff_threshold_map.hspy"%(lv+1), overwrite=True)
                                if also_tiff:
                                    tifffile.imwrite(save_path+'/'+data_name+"_%d_lv_coeff_map.tif"%(lv+1), coeff_map.data)
                                    tifffile.imwrite(save_path+'/'+data_name+"_%d_lv_coeff_threshold_map.tif"%(lv+1), th_map.data)

                            if also_dp and len(np.nonzero(self.thresh_coeff_split[lv][i][j])[0]) != 0:
                                mean_dp = np.sum(dataset.data[np.where(self.thresh_coeff_split[lv][i][j]==1)], axis=0)
                                if save:
                                    mean_dp_save = hs.signals.Signal2D(mean_dp)
                                    mean_dp_save.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                                    mean_dp_save.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                                    mean_dp_save.save(save_path+'/'+data_name+"_mean_diffraction_pattern_%d_lv_coeff_threshold_map.hspy"%(lv+1), overwrite=True)
                                    if also_tiff:
                                        tifffile.imwrite(save_path+'/'+data_name+"_mean_diffraction_pattern_%d_lv_coeff_threshold_map.tif"%(lv+1), mean_dp_save.data)

                                if log_scale_dp:
                                    mean_dp[mean_dp <= 0] = 1.0
                                    ax_lv[2, 0].imshow(np.log(mean_dp).clip(min=0.0), cmap='gray')
                                    ax_lv[2, 0].set_title('(log-scale) Mean diffraction pattern\nfor the high-variance map')
                                    ax_lv[2, 0].axis("off")
                                else:
                                    ax_lv[2, 0].imshow(mean_dp.clip(min=0.0), cmap='gray')
                                    ax_lv[2, 0].set_title('Mean diffraction pattern\nfor the high-variance map')
                                    ax_lv[2, 0].axis("off")

                                max_dp = np.max(dataset.data[np.where(self.thresh_coeff_split[lv][i][j]==1)], axis=0)
                                if save:
                                    max_dp_save = hs.signals.Signal2D(max_dp)
                                    max_dp_save.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                                    max_dp_save.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                                    max_dp_save.save(save_path+'/'+data_name+"_max_diffraction_pattern_%d_lv_coeff_threshold_map.hspy"%(lv+1), overwrite=True)
                                    if also_tiff:
                                        tifffile.imwrite(save_path+'/'+data_name+"_max_diffraction_pattern_%d_lv_coeff_threshold_map.tif"%(lv+1), max_dp_save.data)

                                if log_scale_dp:
                                    max_dp[max_dp <= 0] = 1.0
                                    ax_lv[2, 1].imshow(np.log(max_dp).clip(min=0.0), cmap='gray')
                                    ax_lv[2, 1].set_title('(log-scale) Maximum diffraction pattern\nfor the thresholding map')
                                    ax_lv[2, 1].axis("off")
                                else:
                                    ax_lv[2, 1].imshow(max_dp.clip(min=0.0), cmap='gray')
                                    ax_lv[2, 1].set_title('Maximum diffraction pattern\nfor the high-variance map')
                                    ax_lv[2, 1].axis("off")

                            fig_lv.tight_layout()
                            fig_lv.savefig(save_path+'/'+data_name+"_NMF_%d_lv_summary.png"%(lv+1))
                            if _eff_save is not None:
                                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                                base, ext = os.path.splitext(_eff_save)
                                if not ext:
                                    ext = '.png'
                                fig_lv.savefig(f"{base}_lv_{lv+1}_{i}_{j}{ext}", bbox_inches='tight')
                                plt.close(fig_lv)
                            else:
                                plt.show()

                        if also_dp:
                            del dataset # release the occupied memory        

                                                
    def effective_small_area(self, data_key, threshold_map="NMF", algorithm="DBSCAN", eps=1.5, min_sample=16, visual_result=True, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'effective_small_area') if self._default_figure_save_path is not None else None)
        self.threshold_map_small = threshold_map
        self.clustering_params = {
            "data_key": data_key,
            "threshold_map": threshold_map,
            "algorithm": algorithm,
            "eps": eps,
            "min_sample": min_sample
        }
        if self.threshold_map_small == "NMF":
            for i in range(len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                for j in range(num_img):
                    if data_key in self.loaded_data_path[i][j]:
                        self.selected_data_path = self.loaded_data_path[i][j]
                        self.sub_ind = i
                        self.img_ind = j
            clustered_lv = []                
            for lv in range(self.num_comp):
                binary_map = self.thresh_coeff_split[lv][self.sub_ind][self.img_ind]
                sel_coor = np.nonzero(binary_map)
                X = np.stack((sel_coor[0], sel_coor[1]), axis=1)
                if algorithm == "DBSCAN":
                    db = DBSCAN(eps=eps, min_samples=min_sample)
                elif algorithm == "HDBSCAN":
                    db = HDBSCAN(min_samples=min_sample)
                if len(X) == 0:
                    X = np.array([[0,0]])
                    
                db.fit(X)
                label = db.labels_
                clustered = np.zeros_like(binary_map)
                clustered[X[:, 0], X[:, 1]] = label+1
                clustered_lv.append(clustered)

                if visual_result:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
                    ax[0].imshow(binary_map, cmap='gray')
                    ax[1].imshow(clustered, cmap='tab20')
                    fig.suptitle(self.subfolders[self.sub_ind]+'\nLoading vector %d\n'%(lv+1)+os.path.basename(self.loaded_data_path[self.sub_ind][self.img_ind])[:15]+"\nBefore and After clustering")
                    fig.tight_layout()
                    if _eff_save is not None:
                        os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                        base, ext = os.path.splitext(_eff_save)
                        if not ext:
                            ext = '.png'
                        fig.savefig(f"{base}_lv_{lv+1}{ext}", bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()
            
            self.clustered_lv = clustered_lv
            
        if self.threshold_map_small == "variance":
            for i in range(len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                for j in range(num_img):
                    if data_key in self.loaded_data_path[i][j]:
                        self.selected_data_path = self.loaded_data_path[i][j]
                        self.sub_ind = i
                        self.img_ind = j            

            db = DBSCAN(eps=eps, min_samples=min_sample)
            binary_map = self.thresh_var_split[self.sub_ind][self.img_ind]
            sel_coor = np.nonzero(binary_map)
            X = np.stack((sel_coor[0], sel_coor[1]), axis=1)
            if len(X) == 0:
                X = np.array([[0,0]])
            db.fit(X)
            label = db.labels_
            clustered = np.zeros_like(binary_map)
            clustered[X[:, 0], X[:, 1]] = label+1
            
            if visual_result:
                fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
                ax[0].imshow(binary_map, cmap='gray')
                ax[1].imshow(clustered, cmap='tab20')
                fig.suptitle(self.subfolders[self.sub_ind]+' - '+os.path.basename(self.loaded_data_path[self.sub_ind][self.img_ind])[:15]+"\nBefore and After clustering")
                fig.tight_layout()
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig.savefig(f"{base}_variance{ext}", bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
            
            self.clustered = clustered
        

    def small_area_investigation(self, visual_cluster=True, visual_dp=False, log_dp=True, save=False, also_tiff=False, virtual_4D=False, save_path=None, boundary_method=None, concave_ratio=None, figures_save_path=None): 
        _eff_save = figures_save_path if figures_save_path is not None else (os.path.join(self._default_figure_save_path, 'small_area_investigation') if self._default_figure_save_path is not None else None)
        if boundary_method is None:
            boundary_method = self.boundary_method
        if concave_ratio is None:
            concave_ratio = self.concave_ratio

        if self.threshold_map_small == 'NMF':
            if virtual_4D:
                dataset = hs.load(self.selected_data_path[:-18]+'corrected_scaled.hspy')
                if self.rebin_256:
                    if dataset.data.shape[1] > 250:
                        dataset = dataset.rebin(scale = (2,2,1,1))

            if save_path is None:
                save_path = os.path.dirname(self.selected_data_path)
            print("save directory: ", save_path)
            data_name = os.path.basename(self.selected_data_path).split("_")
            data_name = data_name[0]+'_'+data_name[1]
            print("save prefix: ", data_name)

            virtual_lv = []
            centroid_lv = []
            boundary_lv = []
            for lv in range(self.num_comp):
                centroid_label = []
                boundary_label = []
                virtual_label = []
                label_cluster = self.clustered_lv[lv]
                label_list = np.unique(label_cluster)
                
                if visual_cluster:
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
                    ax.imshow(label_cluster, cmap='tab20')

                for l in label_list[1:]:
                    sel_coor = np.where(label_cluster == l)
                    xy = np.stack((sel_coor[0], sel_coor[1]), axis=1)

                    if boundary_method in ["shapely", "native"]:
                        from shapely.geometry import MultiPoint
                        hull_poly = shapely.concave_hull(MultiPoint(xy), ratio=concave_ratio)
                        if isinstance(hull_poly, Polygon):
                            hull = np.array(hull_poly.exterior.coords)
                        else:
                            hull = xy
                    else:
                        obj = ConcaveHull(xy, 2, use_gpu=self.use_gpu)
                        hull = obj.calculate() # boundary pixel positions

                    com_x, com_y = np.mean(sel_coor[1]), np.mean(sel_coor[0])
                    if visual_cluster:
                        ax.scatter(com_x, com_y, s=15, c='k', marker='*')
                        try:
                            ax.plot(hull[:, 1], hull[:, 0], 'b-')
                            ax.text(com_x, com_y, "%d"%(l))
                            ax.axis("off")
                        except:
                            ax.text(com_x, com_y, "%d"%(l))
                            ax.axis("off")

                    centroid_label.append([com_y, com_x])
                    boundary_label.append(hull)
                
                    if virtual_4D:
                        mean_dp = np.sum(dataset.data[sel_coor], axis=0)
                        max_dp = np.max(dataset.data[sel_coor], axis=0)
                        virtual_label.append(mean_dp)    
                        
                        if save:
                            mean_dp_sig = hs.signals.Signal2D(mean_dp)
                            mean_dp_sig.axes_manager[0].scale = self.radial_var_split[self.sub_ind][self.img_ind].axes_manager[-1].scale
                            mean_dp_sig.axes_manager[1].scale = self.radial_var_split[self.sub_ind][self.img_ind].axes_manager[-1].scale
                            mean_dp_sig.save(save_path+'/'+data_name+"_mean_diffraction_pattern_%d_lv_%02d_cluster.hspy"%(lv+1, l), overwrite=True)
                            if also_tiff:
                                tifffile.imwrite(save_path+'/'+data_name+"_mean_diffraction_pattern_%02d_lv_%02d_cluster.tif"%(lv+1, l), mean_dp_sig.data)

                            max_dp_sig = hs.signals.Signal2D(max_dp)
                            max_dp_sig.axes_manager[0].scale = self.radial_var_split[self.sub_ind][self.img_ind].axes_manager[-1].scale
                            max_dp_sig.axes_manager[1].scale = self.radial_var_split[self.sub_ind][self.img_ind].axes_manager[-1].scale
                            max_dp_sig.save(save_path+'/'+data_name+"_max_diffraction_pattern_%d_lv_%02d_cluster.hspy"%(lv+1, l), overwrite=True)
                            if also_tiff:
                                tifffile.imwrite(save_path+'/'+data_name+"_max_diffraction_pattern_%02d_lv_%02d_cluster.tif"%(lv+1, l), max_dp_sig.data)

                    if visual_dp:
                        fig_dp, ax_dp = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
                        if log_dp:
                            mean_dp_log = mean_dp.copy()
                            max_dp_log = max_dp.copy()
                            mean_dp_log[np.where(mean_dp_log<=0)] = 1.0
                            max_dp_log[np.where(max_dp_log<=0)] = 1.0
                            ax_dp[0].imshow(np.log(mean_dp_log).clip(min=0), cmap='gray')
                            ax_dp[1].imshow(np.log(max_dp_log).clip(min=0), cmap='gray')
                        else:
                            ax_dp[0].imshow(mean_dp.clip(min=0), cmap='gray')
                            ax_dp[1].imshow(max_dp.clip(min=0), cmap='gray')    
                        fig_dp.suptitle(self.subfolders[self.sub_ind]+" - "+os.path.basename(self.selected_data_path)[:15]+'\n%02d lv %02d cluster'%(lv+1, l))

                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    if visual_cluster:
                        fig.savefig(f"{base}_lv_{lv+1}_cluster{ext}", bbox_inches='tight')
                        plt.close(fig)
                    if visual_dp:
                        fig_dp.savefig(f"{base}_lv_{lv+1}_dp{ext}", bbox_inches='tight')
                        plt.close(fig_dp)
                else:
                    plt.show()
                centroid_lv.append(centroid_label)
                boundary_lv.append(boundary_label)
                virtual_lv.append(virtual_label)

            self.centroid_lv = centroid_lv
            self.boundary_lv = boundary_lv
            
            if virtual_4D:
                self.virtual_lv = virtual_lv
            if 'dataset' in locals():
                del dataset
            import gc
            gc.collect()
            

        if self.threshold_map_small == 'variance':
            if virtual_4D:
                dataset = hs.load(self.selected_data_path[:-18]+'corrected_scaled.hspy')
                if self.rebin_256:
                    if dataset.data.shape[1] > 250:
                        dataset = dataset.rebin(scale = (2,2,1,1))

            if save_path is None:
                save_path = os.path.dirname(self.selected_data_path)
            print("save directory: ", save_path)
            data_name = os.path.basename(self.selected_data_path).split("_")
            data_name = data_name[0]+'_'+data_name[1]
            print("save prefix: ", data_name)

            label_cluster = self.clustered
            label_list = np.unique(label_cluster)
            print(label_list)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
            ax.imshow(label_cluster, cmap='tab20')

            centroid_label = []
            boundary_label = []
            virtual_label = []
            for l in label_list[1:]:
                sel_coor = np.where(label_cluster == l)
                xy = np.stack((sel_coor[0], sel_coor[1]), axis=1)

                if boundary_method in ["shapely", "native"]:
                    from shapely.geometry import MultiPoint
                    hull_poly = shapely.concave_hull(MultiPoint(xy), ratio=concave_ratio)
                    if isinstance(hull_poly, Polygon):
                        hull = np.array(hull_poly.exterior.coords)
                    else:
                        hull = xy
                else:
                    obj = ConcaveHull(xy, 2, use_gpu=self.use_gpu)
                    hull = obj.calculate() # boundary pixel positions

                com_x, com_y = np.mean(sel_coor[1]), np.mean(sel_coor[0])
                ax.scatter(com_x, com_y, s=15, c='k', marker='*')
                try:
                    ax.plot(hull[:, 1], hull[:, 0], 'b-')
                    ax.text(com_x, com_y, "%d"%(l))
                    ax.axis("off")
                except:
                    ax.text(com_x, com_y, "%d"%(l))
                    ax.axis("off")

                centroid_label.append([com_y, com_x])
                boundary_label.append(hull)

                if virtual_4D:
                    mean_dp = np.sum(dataset.data[sel_coor], axis=0)
                    max_dp = np.max(dataset.data[sel_coor], axis=0)     
                    virtual_label.append(mean_dp)
                    
                    if save:
                        mean_dp_sig = hs.signals.Signal2D(mean_dp)
                        mean_dp_sig.axes_manager[0].scale = self.radial_var_split[self.sub_ind][self.img_ind].axes_manager[-1].scale
                        mean_dp_sig.axes_manager[1].scale = self.radial_var_split[self.sub_ind][self.img_ind].axes_manager[-1].scale
                        mean_dp_sig.save(save_path+'/'+data_name+"_mean_diffraction_pattern_%d_lv_%02d_cluster.hspy"%(lv+1, l), overwrite=True)
                        if also_tiff:
                            tifffile.imwrite(save_path+'/'+data_name+"_mean_diffraction_pattern_%02d_lv_%02d_cluster.tif"%(lv+1, l), mean_dp_sig.data)

                        max_dp_sig = hs.signals.Signal2D(max_dp)
                        max_dp_sig.axes_manager[0].scale = self.radial_var_split[self.sub_ind][self.img_ind].axes_manager[-1].scale
                        max_dp_sig.axes_manager[1].scale = self.radial_var_split[self.sub_ind][self.img_ind].axes_manager[-1].scale
                        max_dp_sig.save(save_path+'/'+data_name+"_max_diffraction_pattern_%d_lv_%02d_cluster.hspy"%(lv+1, l), overwrite=True)
                        if also_tiff:
                            tifffile.imwrite(save_path+'/'+data_name+"_max_diffraction_pattern_%02d_lv_%02d_cluster.tif"%(lv+1, l), max_dp_sig.data)

                if visual_dp:
                    fig_dp, ax_dp = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
                    if log_dp:
                        mean_dp_log = mean_dp.copy()
                        max_dp_log = max_dp.copy()
                        mean_dp_log[np.where(mean_dp_log<=0)] = 1.0
                        max_dp_log[np.where(max_dp_log<=0)] = 1.0
                        ax_dp[0].imshow(np.log(mean_dp_log).clip(min=0), cmap='gray')
                        ax_dp[1].imshow(np.log(max_dp_log).clip(min=0), cmap='gray')
                    else:
                        ax_dp[0].imshow(mean_dp.clip(min=0), cmap='gray')
                        ax_dp[1].imshow(max_dp.clip(min=0), cmap='gray')                    
                    fig_dp.suptitle(self.subfolders[self.sub_ind]+" - "+os.path.basename(self.selected_data_path)[:15]+'\n%02d lv %02d cluster'%(lv+1, l))

            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig.savefig(f"{base}_variance_cluster{ext}", bbox_inches='tight')
                plt.close(fig)
                if visual_dp:
                    fig_dp.savefig(f"{base}_variance_dp{ext}", bbox_inches='tight')
                    plt.close(fig_dp)
            else:
                plt.show()
            self.centroid_label = centroid_label
            self.boundary_label = boundary_label
            if virtual_4D:
                self.virtual_label = virtual_label
            if 'dataset' in locals():
                del dataset
            import gc
            gc.collect()


    def overlap_check(self, visual_lv=False, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'overlap_check') if self._default_figure_save_path is not None else None)
        if self.threshold_map_small == 'NMF':
            fig_tot, ax_tot = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
            for lv in range(self.num_comp):
                label_cluster = self.clustered_lv[lv]
                label_list = np.unique(label_cluster).astype(int)
                if visual_lv:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
                    ax[0].imshow(label_cluster, cmap='tab20')
                for l in range(0, len(label_list)-1, 1):
                    try:
                        ax_tot.fill(self.boundary_lv[lv][l][:, 1], self.boundary_lv[lv][l][:, 0], 
                                facecolor=self.color_rep[lv+1], 
                                edgecolor=self.color_rep[lv+1], linewidth=3, alpha=0.5)
                        ax_tot.axis("off")
                    
                        if visual_lv:
                            ax[0].scatter(self.centroid_lv[lv][l][1], self.centroid_lv[lv][l][0], s=15, c='k', marker='*')
                            ax[0].text(self.centroid_lv[lv][l][1], self.centroid_lv[lv][l][0], "%d"%(l+1))
                            ax[0].plot(self.boundary_lv[lv][l][:, 1], self.boundary_lv[lv][l][:, 0], 'b-')
                            ax[0].axis("off")

                            ax[1].imshow(np.zeros_like(label_cluster), alpha=0.0)
                            ax[1].fill(self.boundary_lv[lv][l][:, 1], self.boundary_lv[lv][l][:, 0], 
                                    facecolor=self.color_rep[lv+1], 
                                    edgecolor=self.color_rep[lv+1], linewidth=3, alpha=0.5)
                            ax[1].set_xticks([])
                            ax[1].set_yticks([])
                            ax[1].set_xticklabels([])
                            ax[1].set_yticklabels([])
                            ax[1].set_facecolor('lightgray')
                            ax[1].axis("off")
                    
                    except:
                        ax_tot.axis("off")

                if visual_lv:
                    fig.tight_layout()
            
            ax_tot.set_xticks([])
            ax_tot.set_yticks([])
            ax_tot.set_xticklabels([])
            ax_tot.set_yticklabels([])
            ax_tot.set_facecolor("lightgray")
            fig_tot.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig_tot.savefig(f"{base}_overlap{ext}", bbox_inches='tight')
                plt.close(fig_tot)
            else:
                plt.show()
            
        else:
            return


    def single_phase_investigation(self, visual=True, fig_save=False, dp_shape=[515, 515], crop_ind=[0, 515, 0, 515],
                                   eps=4.5, min_sample=30, virtual_4D=True, diff_size=False, size_list=None, cut_too_large=None,
                                   boundary_method=None, fill_method=None, concave_ratio=None, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'single_phase_investigation') if self._default_figure_save_path is not None else None)
        boundary_method_val = boundary_method if boundary_method is not None else self.boundary_method
        fill_method_val = fill_method if fill_method is not None else self.fill_method
        concave_ratio_val = concave_ratio if concave_ratio is not None else self.concave_ratio

        self.mean_edx = {}
        self.mean_zernike = {}
        self.mean_rvp = {}
        self.mean_rmp = {}
        self.num_pixel = {}
        self.dp_storage = {}
        for s in range(len(self.subfolders)):
            if self.simult_edx and self.edx_range_flag:    
                for i in range(self.num_comp):
                    self.mean_edx['sub_index_%d_LV%d'%(s+1, i+1)] = np.zeros(self.edx_dim)
                    
            if self.zernike:
                for i in range(self.num_comp):
                    self.mean_zernike['sub_index_%d_LV%d'%(s+1, i+1)] = np.zeros(self.zernike_length)
                    
            for i in range(self.num_comp):
                self.mean_rvp['sub_index_%d_LV%d'%(s+1, i+1)] = np.zeros(self.profile_length)

            for i in range(self.num_comp):
                self.mean_rmp['sub_index_%d_LV%d'%(s+1, i+1)] = np.zeros(self.profile_length)

            for i in range(self.num_comp):
                self.num_pixel['sub_index_%d_LV%d'%(s+1, i+1)] = 0

            for i in range(self.num_comp):
                self.dp_storage['sub_index_%d_LV%d'%(s+1, i+1)] = []


        self.num_lv_pixel_split = []
        self.pos_lv_pixel_split = []
        self.clustered_lv_split = []
        self.centroid_lv_split = []
        self.boundary_lv_split = []
        for i in range(len(self.subfolders)):
            self.sub_num_pixel = []
            self.sub_pos_pixel = []
            self.sub_clustered_lv = []
            self.sub_centroid_lv = []
            self.sub_boundary_lv = []

            for j, adr in enumerate(self.loaded_data_path[i]):
                print(adr)
                self.data_num_pixel = {}
                for lv in range(self.num_comp):
                    self.data_num_pixel['sub_index_%d_LV%d'%(i+1, lv+1)] = 0

                self.data_pos_pixel = {}
                for lv in range(self.num_comp):
                    self.data_pos_pixel['sub_index_%d_LV%d'%(i+1, lv+1)] = []

                data_key = os.path.basename(adr)[:15]
                
                size = self.radial_avg_split[i][j].data.shape[1]
                
                if diff_size:
                    min_size = np.min(size_list)
                    self.effective_small_area(data_key=data_key, threshold_map="NMF", eps=eps, min_sample=int(min_sample*size/min_size), visual_result=False)
                else:
                    self.effective_small_area(data_key=data_key, threshold_map="NMF", eps=eps, min_sample=min_sample, visual_result=False)
                    
                self.small_area_investigation(visual_cluster=False, visual_dp=False, save=False, also_tiff=False, virtual_4D=virtual_4D,
                                              boundary_method=boundary_method_val, concave_ratio=concave_ratio_val)
                
                self.sub_clustered_lv.append(self.clustered_lv)
                self.sub_centroid_lv.append(self.centroid_lv)
                self.sub_boundary_lv.append(self.boundary_lv)
                
                datacube = []
                lv_label = []

                if virtual_4D:
                    for lv in range(self.num_comp):
                        label = [lv+1] * len(self.virtual_lv[lv])
                        lv_label.extend(label)
                        if virtual_4D:
                            datacube.extend(self.virtual_lv[lv])

                    lv_label = np.asarray(lv_label)
                    datacube = np.asarray(datacube).reshape(-1, dp_shape[0], dp_shape[1])
                    for lv in range(self.num_comp):
                        ind = np.where(lv_label == lv+1)[0]
                        for k in ind:
                            self.dp_storage['sub_index_%d_LV%d'%(i+1, lv+1)].append(datacube[k][crop_ind[0]:crop_ind[1], crop_ind[2]:crop_ind[3]].copy())
                
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(np.sum(self.radial_avg_split[self.sub_ind][self.img_ind].data, axis=2), cmap="gray")
                for lv in range(self.num_comp):
                    label_cluster = self.clustered_lv[lv]
                    label_list = np.unique(label_cluster).astype(int)
                    for l in range(0, len(label_list)-1, 1):
                        try:
                            hull = self.boundary_lv[lv][l]
                            if fill_method_val == "scipy":
                                cluster_mask = (label_cluster == l)
                                filled_mask = binary_fill_holes(cluster_mask)
                                inside_points = np.argwhere(filled_mask)
                                if hull is not None and len(hull) > 0:
                                    inside_points = np.vstack((inside_points, hull))
                                inside_points = np.unique(inside_points, axis=0).astype(int)
                            elif fill_method_val in ["path", "matplotlib"]:
                                polygon = Polygon(hull)
                                x_min, y_min, x_max, y_max = polygon.bounds
                                
                                grid_y, grid_x = np.mgrid[int(x_min):int(x_max), int(y_min):int(y_max)]
                                # Pre-cast to float32: mpath.Path.contains_points internally converts
                                # int64 -> float64, creating a hidden transient copy that doubles
                                # peak memory. float32 avoids this and halves the allocation.
                                grid_pts = np.stack((grid_y.ravel(), grid_x.ravel()), axis=1).astype(np.float32)
                                del grid_y, grid_x  # free immediately; grid_pts holds the data
                                path = mpath.Path(hull.astype(np.float32))
                                mask = path.contains_points(grid_pts, radius=-1e-9)
                                inside_points = grid_pts[mask].astype(int)
                                del grid_pts, mask  # free large intermediates before vstack
                                
                                inside_points = np.vstack((inside_points, hull))
                                inside_points = np.unique(inside_points, axis=0).astype(int)
                            else:
                                polygon = Polygon(hull)
                                x_min, y_min, x_max, y_max = polygon.bounds
                                
                                grid_y, grid_x = np.mgrid[int(x_min):int(x_max), int(y_min):int(y_max)]
                                grid_pts = np.stack((grid_y.ravel(), grid_x.ravel()), axis=1)
                                pts_geom = shapely.points(grid_pts)
                                mask = shapely.contains(polygon, pts_geom)
                                inside_points = grid_pts[mask]
                                
                                inside_points = np.vstack((inside_points, hull))
                                inside_points = np.unique(inside_points, axis=0).astype(int)
                            
                            if cut_too_large != None and len(inside_points) > int(cut_too_large*size*(size-1)):
                                self.data_pos_pixel['sub_index_%d_LV%d'%(i+1, lv+1)].append([])
                            else:                           
                                ax.scatter(inside_points[:, 1], inside_points[:, 0], s=0.5, color=self.color_rep[lv+1], alpha=0.5)
                                
                                if self.simult_edx and self.edx_range_flag: 
                                    self.mean_edx['sub_index_%d_LV%d'%(i+1, lv+1)] += np.sum(self.edx_split[self.sub_ind][self.img_ind].data[inside_points[:, 0], inside_points[:, 1]], axis=0)
                                    
                                if self.zernike:
                                    self.mean_zernike['sub_index_%d_LV%d'%(i+1, lv+1)] += np.sum(self.zernike_split[self.sub_ind][self.img_ind].data[inside_points[:, 0], inside_points[:, 1]], axis=0)
                                    
                                self.num_pixel['sub_index_%d_LV%d'%(i+1, lv+1)] += len(inside_points)
                                self.mean_rvp['sub_index_%d_LV%d'%(i+1, lv+1)] += np.sum(self.radial_var_split[self.sub_ind][self.img_ind].data[inside_points[:, 0], inside_points[:, 1]], axis=0)
                                self.mean_rmp['sub_index_%d_LV%d'%(i+1, lv+1)] += np.sum(self.radial_avg_split[self.sub_ind][self.img_ind].data[inside_points[:, 0], inside_points[:, 1]], axis=0)
                                self.data_num_pixel['sub_index_%d_LV%d'%(i+1, lv+1)] += len(inside_points)
                                # Store as numpy array (not .tolist()) to avoid ~5x Python object overhead
                                self.data_pos_pixel['sub_index_%d_LV%d'%(i+1, lv+1)].append(inside_points)
                                del inside_points  # free immediately; stored reference above keeps it alive
                        except:
                            self.data_pos_pixel['sub_index_%d_LV%d'%(i+1, lv+1)].append([])
                
                if fig_save:
                    fig.savefig("%s_%s_single_phase_area.png"%(self.subfolders[i], data_key), dpi=100)
                if visual:            
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])  
                    fig.tight_layout()
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig.savefig(f"{base}_sub_{i}_data_{j}{ext}", bbox_inches='tight')
                    plt.close(fig)
                elif visual:
                    plt.show()
                plt.close(fig)
                        
                self.sub_num_pixel.append(self.data_num_pixel)
                self.sub_pos_pixel.append(self.data_pos_pixel)
                if 'datacube' in locals():
                    del datacube
                import gc
                gc.collect()
            self.num_lv_pixel_split.append(self.sub_num_pixel)
            self.pos_lv_pixel_split.append(self.sub_pos_pixel)
            self.clustered_lv_split.append(self.sub_clustered_lv)
            self.centroid_lv_split.append(self.sub_centroid_lv)
            self.boundary_lv_split.append(self.sub_boundary_lv)                       


    def scattering_range_of_interest(self, profile_type="variance", str_name=None, fill_width=0.1, height=None, width=None, threshold=None, distance=None, prominence=0.001, save_path=None):

        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'scattering_range_of_interest') if self._default_figure_save_path is not None else None)
        if width != None:
            width = width/self.pixel_size_inv_Ang

        if distance != None:
            distance = distance/self.pixel_size_inv_Ang
        
        # sum of radial variance profile by subfolder
        total_sum_split = []
        if profile_type == "variance":
            for split in self.radial_var_sum_split:
                total_sum_split.append(np.mean(split, axis=0))
        elif profile_type == "mean":
            for split in self.radial_avg_sum_split:
                total_sum_split.append(np.mean(split, axis=0))  
        else:
            print("Warning! wrong profile type!")
            return

        peak_sub = {}
        for i, sp in enumerate(total_sum_split):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
            tmp_sp = sp[self.range_ind[0]:self.range_ind[1]]
            if np.max(tmp_sp) != 0:
                tmp_sp /= np.max(tmp_sp)

            peaks = find_peaks(tmp_sp, height=height, 
                               width=width, 
                               threshold=threshold, 
                               distance=distance, 
                               prominence=prominence)[0]
            
            peaks = peaks * self.pixel_size_inv_Ang
            peaks = peaks + self.from_
            peak_sub[self.subfolders[i]] = peaks
            
            tmp_ax, = ax.plot(self.x_axis, tmp_sp, c=self.color_rep[i+1], label=self.subfolders[i])
            shadow_effect = path_effects.withStroke(linewidth=3, foreground='gray')
            tmp_ax.set_path_effects([shadow_effect])

            if str_name != None and str_name != []:
                ax_twin = ax.twinx()
                for key in str_name:
                    ax_twin.plot(self.x_axis, self.int_sf[key], label=key, linestyle=":")
                ax_twin.legend(loc="right")

            for j, peak in enumerate(peaks):
                if peak >= self.from_ and peak <= self.to_:
                    print("%d peak position (1/Å):\t"%(j+1), peak)
                    ax.axvline(peak, ls=':', lw=1.5, c=self.color_rep[i+1])
                    ax.fill_between([peak-fill_width, peak+fill_width], y1=np.max(tmp_sp), y2=np.min(tmp_sp), alpha=0.5, color='orange')
                    ax.text(peak, 1.0, "%d"%(j+1))

            ax.set_xlabel('scattering vector (1/Å)')
            # ax.set_facecolor("lightgray")
            ax.legend(loc="right")
            fig.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig.savefig(f"{base}_sub_{i}{ext}", bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
            
        self.peak_sub = peak_sub

    
    def variance_map(self, sv_range=None, peaks=None, fill_width=0.1, visual_title=True, save_path=None):

        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'variance_map') if self._default_figure_save_path is not None else None)
        if peaks != None:
            for i, peak in enumerate(peaks):
                sv_range = [peak-fill_width, peak+fill_width]
                mean_var_map = []
                std_var_map = []
                for i in range(len(self.subfolders)):
                    num_img = len(self.radial_var_split[i])
                    grid_size = int(np.around(np.sqrt(num_img)))
                    if num_img == 1:
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                        ax = np.array([ax])
                    elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                        fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
                    else:
                        fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
                    for j in range(num_img):
                        var_map = np.sum(self.radial_var_split[i][j].isig[sv_range[0]:sv_range[1]].data, axis=2)
                        mean_var_map.append(np.mean(var_map))
                        std_var_map.append(np.std(var_map))
                        ax.flat[j].imshow(var_map, cmap='inferno')
                        if visual_title:
                            ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    for a in ax.flat:
                        a.axis('off')
                    fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(sv_range[0], sv_range[1]))
                    plt.subplots_adjust(hspace=0.1, wspace=0.1)
                    if visual_title:
                        fig.tight_layout()
                    if _eff_save is not None:
                        os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                        base, ext = os.path.splitext(_eff_save)
                        if not ext:
                            ext = '.png'
                        fig.savefig(f"{base}_peak_{p}_sub_{i}{ext}", bbox_inches='tight')
                        plt.close(fig)
                    else:
                        plt.show()
                
        
                # to specify the absolute threshold value to make the binary variance map
                total_num = 0
                fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
                for i in range(0, len(self.subfolders)):
                    num_img = len(self.radial_var_split[i])
                    if i == 0:
                        num_range = np.arange(0, num_img)
                        ax.plot(num_range, mean_var_map[:num_img], c=self.color_rep[i+1], marker='s', label=self.subfolders[i])
                        ax.errorbar(num_range, mean_var_map[:num_img], yerr=std_var_map[:num_img], capsize=5, c=self.color_rep[i+1])
                
                    else:
                        num_range = np.arange(total_num, total_num+num_img)
                        ax.plot(num_range, mean_var_map[total_num:total_num+num_img], c=self.color_rep[i+1], marker='s', label=self.subfolders[i])
                        ax.errorbar(num_range, mean_var_map[total_num:total_num+num_img], 
                                    yerr=std_var_map[total_num:total_num+num_img], capsize=5, c=self.color_rep[i+1])
                    total_num += num_img
                ax.grid()
                ax.legend()
                fig.suptitle("mean and standard deviation of the variance maps above")
                fig.tight_layout()
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig.savefig(f"{base}_peak_{p}_mean_std{ext}", bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
            
        elif sv_range != None and sv_range != []:
            self.sv_range = sv_range
            mean_var_map = []
            std_var_map = []
            for i in range(len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if num_img == 1:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                    ax = np.array([ax])
                elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
                for j in range(num_img):
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                    mean_var_map.append(np.mean(var_map))
                    std_var_map.append(np.std(var_map))
                    ax.flat[j].imshow(var_map, cmap='inferno')
                    if visual_title:
                        ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                for a in ax.flat:
                    a.axis('off')
                fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(self.sv_range[0], self.sv_range[1]))
                plt.subplots_adjust(hspace=0.1, wspace=0.1)
                if visual_title:
                    fig.tight_layout()
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig.savefig(f"{base}_svrange_sub_{i}{ext}", bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
            
    
            # to specify the absolute threshold value to make the binary variance map
            total_num = 0
            fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
            for i in range(0, len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                if i == 0:
                    num_range = np.arange(0, num_img)
                    ax.plot(num_range, mean_var_map[:num_img], c=self.color_rep[i+1], marker='s', label=self.subfolders[i])
                    ax.errorbar(num_range, mean_var_map[:num_img], yerr=std_var_map[:num_img], capsize=5, c=self.color_rep[i+1])
            
                else:
                    num_range = np.arange(total_num, total_num+num_img)
                    ax.plot(num_range, mean_var_map[total_num:total_num+num_img], c=self.color_rep[i+1], marker='s', label=self.subfolders[i])
                    ax.errorbar(num_range, mean_var_map[total_num:total_num+num_img], 
                                yerr=std_var_map[total_num:total_num+num_img], capsize=5, c=self.color_rep[i+1])
                total_num += num_img
            ax.grid()
            ax.legend()
            fig.suptitle("mean and standard deviation of the variance maps above")
            fig.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig.savefig(f"{base}_svrange_mean_std{ext}", bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
        
        else:
            print("The scattering vector range must be specified!")
            return
 

    def high_variance_map(self, abs_threshold=None, peaks=None, fill_width=0.1, percentile_threshold=90, visual_title=True, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'high_variance_map') if self._default_figure_save_path is not None else None)
        # binary variance map (leave only large variances for the range specified above)
        # abosulte variance map threshold (pixel value > abs_threshold will be 1, otherwise it will be 0)
        if peaks != None:
            for p, peak in enumerate(peaks):
                fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
                sp_tot = np.zeros(self.profile_length)
                total_num = 0
                sv_range = [peak-fill_width, peak+fill_width]
                
                for i in range(len(self.subfolders)):
                    sp_sub = np.zeros(self.profile_length)
                    sub_num = 0
                    num_img = len(self.radial_var_split[i])
                    grid_size = int(np.around(np.sqrt(num_img)))
                if num_img == 1:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                    ax = np.array([ax])
                elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
                    for j in range(num_img):
                        var_map = np.sum(self.radial_var_split[i][j].isig[sv_range[0]:sv_range[1]].data, axis=2)
                        abs_threshold = np.percentile(var_map, percentile_threshold)
                        var_map[var_map<=abs_threshold] = 0
                        var_map[var_map>abs_threshold] = 1
                        ax.flat[j].imshow(var_map, cmap='gray')
                        if visual_title:
                            ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15]+"_threshold value=%f"%abs_threshold)

                        tmp_num = len(np.where(var_map==1)[0])
                        total_num += tmp_num
                        sub_num += tmp_num
                        sp_tot += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                        sp_sub += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)

                    if sub_num != 0:
                        sp_sub /= sub_num
                    ax_tot[1].plot(self.x_axis, sp_sub[self.range_ind[0]:self.range_ind[1]], c=self.color_rep[i], label=self.subfolders[i])
                    
                    for a in ax.flat:
                        a.axis('off')
                    fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(sv_range[0], sv_range[1]))
                    plt.subplots_adjust(hspace=0.1, wspace=0.1)
                    if visual_title:
                        fig.tight_layout()

                ax_tot[1].legend()
                if total_num != 0:
                    sp_tot /= total_num
                ax_tot[0].plot(self.x_axis, sp_tot[self.range_ind[0]:self.range_ind[1]], 'k-')
                fig_tot.tight_layout()
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig_tot.savefig(f"{base}_peaks_{p}{ext}", bbox_inches='tight')
                    plt.close(fig_tot)
                else:
                    plt.show()

        elif percentile_threshold != None:     
            thresh_var_split = []
            fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
            sp_tot = np.zeros(self.profile_length)
            total_num = 0

            # to obtain the threshold value
            temp_stored_values = []
            for i in range(len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                for j in range(num_img):
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                    temp_stored_values.extend(var_map.flatten().tolist())
            self.abs_threshold = np.percentile(temp_stored_values, percentile_threshold)

            # leave the pixels of the high variances
            for i in range(len(self.subfolders)):
                sp_sub = np.zeros(self.profile_length)
                sub_num = 0
                thresh_var = []
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if num_img == 1:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                    ax = np.array([ax])
                elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
                for j in range(num_img):
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                    var_map[var_map<=self.abs_threshold] = 0
                    var_map[var_map>self.abs_threshold] = 1
                    thresh_var.append(var_map)
                    ax.flat[j].imshow(var_map, cmap='gray')
                    if visual_title:
                        ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])

                    tmp_num = len(np.where(var_map==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num
                    sp_tot += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                    sp_sub += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                if sub_num != 0:
                    sp_sub /= sub_num
                ax_tot[1].plot(self.x_axis, sp_sub[self.range_ind[0]:self.range_ind[1]], c=self.color_rep[i], label=self.subfolders[i])
                
                for a in ax.flat:
                    a.axis('off')
                fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(self.sv_range[0], self.sv_range[1]))
                plt.subplots_adjust(hspace=0.1, wspace=0.1)
                if visual_title:
                    fig.tight_layout()
                thresh_var_split.append(thresh_var)
                
            ax_tot[1].legend()
            if total_num != 0:
                sp_tot /= total_num
            ax_tot[0].plot(self.x_axis, sp_tot[self.range_ind[0]:self.range_ind[1]], 'k-')
            fig_tot.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig_tot.savefig(f"{base}_percentile{ext}", bbox_inches='tight')
                plt.close(fig_tot)
            else:
                plt.show()
                
            self.thresh_var_split = thresh_var_split
            return sp_tot

        elif abs_threshold != None:     
            thresh_var_split = []
            self.abs_threshold = abs_threshold
            fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
            sp_tot = np.zeros(self.profile_length)
            total_num = 0
            
            for i in range(len(self.subfolders)):
                sp_sub = np.zeros(self.profile_length)
                sub_num = 0
                thresh_var = []
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if num_img == 1:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
                    ax = np.array([ax])
                elif (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=100)
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10), dpi=100)
                for j in range(num_img):
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                    var_map[var_map<=self.abs_threshold] = 0
                    var_map[var_map>self.abs_threshold] = 1
                    thresh_var.append(var_map)
                    ax.flat[j].imshow(var_map, cmap='gray')
                    if visual_title:
                        ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    
                    tmp_num = len(np.where(var_map==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num                    
                    sp_tot += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                    sp_sub += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                    
                if sub_num != 0:
                    sp_sub /= sub_num
                ax_tot[1].plot(self.x_axis, sp_sub[self.range_ind[0]:self.range_ind[1]], c=self.color_rep[i], label=self.subfolders[i])
                
                for a in ax.flat:
                    a.axis('off')
                fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(self.sv_range[0], self.sv_range[1]))
                fig.tight_layout()
                thresh_var_split.append(thresh_var)
                
            ax_tot[1].legend()
            if total_num != 0:
                sp_tot /= total_num
            ax_tot[0].plot(self.x_axis, sp_tot[self.range_ind[0]:self.range_ind[1]], 'k-')
            fig_tot.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig_tot.savefig(f"{base}_abs_threshold{ext}", bbox_inches='tight')
                plt.close(fig_tot)
            else:
                plt.show()
                
            self.thresh_var_split = thresh_var_split
            return sp_tot
            
        else:
            print("The absolute threshold value or the percentile threshold must be specified")
            return

    
    def Xcorrel(self, str_name=None, profile_type="mean", visual_title=True, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'Xcorrel') if self._default_figure_save_path is not None else None)
        xcor_val_split = []
        xcor_sh_split = []

        self.xcor_profile = profile_type
        self.xcor_str = str_name
        
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_split[i])
            xcor_val_list = []
            xcor_sh_list = []
            for j in range(num_img):
                if profile_type=="variance":
                    var_data = self.radial_var_split[i][j].data
                elif profile_type=="mean":
                    var_data = self.radial_avg_split[i][j].data
                else:
                    print("Warning! wrong profile type!")
                    return
                    
                xcor_val = []
                xcor_sh = []
                for sy in range(var_data.shape[0]):
                    for sx in range(var_data.shape[1]):
                        sp = var_data[sy, sx][self.range_ind[0]:self.range_ind[1]]
                        xcor = np.correlate(self.int_sf[str_name]/np.max(self.int_sf[str_name]), sp/np.max(sp), mode='full')
                        xcor_val.append(np.max(xcor))
                        xcor_sh.append(np.argmax(xcor))
                xcor_val = np.asarray(xcor_val).reshape(var_data.shape[:2])
                xcor_sh = np.asarray(xcor_sh).reshape(var_data.shape[:2])*self.pixel_size_inv_Ang - np.median(self.x_axis)
                xcor_val_list.append(xcor_val)
                xcor_sh_list.append(xcor_sh)
            xcor_val_split.append(xcor_val_list)
            xcor_sh_split.append(xcor_sh_list)

        self.xcor_val_split = xcor_val_split
        self.xcor_sh_split = xcor_sh_split

        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_split[i])
            grid_size = int(np.around(np.sqrt(num_img)))
            if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12), dpi=100)
            else:
                fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10), dpi=100)
            for j in range(num_img):     
                ax.flat[j*2].imshow(xcor_val_split[i][j], cmap='inferno')
                if visual_title:
                    ax.flat[j*2].set_title(self.loaded_data_path[i][j][-29:-14])
                ax.flat[j*2].axis("off")
        
                ax.flat[j*2+1].hist(xcor_val_split[i][j].flatten(), bins=100)
                ax.flat[j*2+1].set_title("cross-correlation values")
                ax.flat[j*2+1].set_facecolor("lightgray")
        
            fig.suptitle(self.subfolders[i]+' cross-correlation - value')
            fig.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig.savefig(f"{base}_sub_{i}{ext}", bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

    
    def high_Xcorr(self, value_threshold=5.0, shift_threshold=0.3, visual_title=True, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'high_Xcorr') if self._default_figure_save_path is not None else None)
        thresh_xcor_split = []
        fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
        sp_tot = np.zeros(self.profile_length)
        total_num = 0
        for i in range(len(self.subfolders)):
            sp_sub = np.zeros(self.profile_length)
            thresh_xcor = []
            num_img = len(self.radial_var_split[i])
            grid_size = int(np.around(np.sqrt(num_img)))
            sub_num = 0
            
            if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12), dpi=100)
            else:
                fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10), dpi=100)
            for j in range(num_img):
                xcor_val_map = self.xcor_val_split[i][j].copy()
                xcor_sh_map = self.xcor_sh_split[i][j].copy()
                val_map = self.xcor_val_split[i][j].copy()
                sh_map = self.xcor_sh_split[i][j].copy()        
                xcor_val_map[val_map<=value_threshold] = 0
                xcor_val_map[val_map>value_threshold] = 1
                xcor_sh_map[np.abs(sh_map)>shift_threshold] = 0
                xcor_sh_map[np.abs(sh_map)<=shift_threshold] = 1
        
                bool_mask = xcor_val_map * xcor_sh_map
                thresh_xcor.append(bool_mask)
                ax.flat[j*2].imshow(bool_mask, cmap='gray')
                if visual_title:
                    ax.flat[j*2].set_title(self.loaded_data_path[i][j][-29:-14])
                ax.flat[j*2].axis("off")
                
                if len(np.nonzero(bool_mask)[0]) != 0:
                    if self.xcor_profile == "variance":
                        avg_rv = np.sum(self.radial_var_split[i][j].data[np.where(bool_mask==1)], axis=0)
                    elif self.xcor_profile == "mean":
                        avg_rv = np.sum(self.radial_avg_split[i][j].data[np.where(bool_mask==1)], axis=0)
                    else:
                        print("Warning! wrong profile type!")
                        return

                    tmp_num = len(np.where(bool_mask==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num
                    sp_tot += avg_rv
                    sp_sub += avg_rv
                    if np.max(avg_rv) != 0:
                        avg_rv /= np.max(avg_rv)
                    ax.flat[j*2+1].plot(self.x_axis, avg_rv[self.range_ind[0]:self.range_ind[1]], 'k-')
                    ax_twin = ax.flat[j*2+1].twinx()
                    ax_twin.plot(self.x_axis, self.int_sf[self.xcor_str], 'k:')
                    ax.flat[j*2+1].set_facecolor("lightgray")

            if sub_num != 0:
                sp_sub /= sub_num
            ax_tot[1].plot(self.x_axis, sp_sub[self.range_ind[0]:self.range_ind[1]], c=self.color_rep[i], label=self.subfolders[i])

            thresh_xcor_split.append(thresh_xcor)
            fig.suptitle(self.subfolders[i]+' large cross-correlation - value')
            fig.tight_layout()
        
        ax_tot[1].legend()
        if total_num != 0:
            sp_tot /= total_num
        ax_tot[0].plot(self.x_axis, sp_tot[self.range_ind[0]:self.range_ind[1]], 'k-')
        fig_tot.tight_layout()
        if _eff_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
            base, ext = os.path.splitext(_eff_save)
            if not ext:
                ext = '.png'
            fig_tot.savefig(f"{base}_xcorr_total{ext}", bbox_inches='tight')
            plt.close(fig_tot)
        else:
            plt.show()
        self.thresh_xcor_split = thresh_xcor_split
        return sp_tot


    def sum_edx(self, edx_from, edx_to, offset=0.0, edx_scale=0.01, total_edx=False, visual=True, visual_title=True, title_font_size=10, axis_off=True, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'sum_edx') if self._default_figure_save_path is not None else None)
        if self.simult_edx == False:
            self.edx_range_flag = False
            print("Warning! EDX data not loaded!")
            return


        self.edx_range_flag = True
        self.edx_dim = self.edx_split[0][0].data.shape[2]
        self.edx_range = np.linspace(0.0, self.edx_dim*edx_scale, self.edx_dim)
        self.edx_offset = offset

        self.edx_from = edx_from
        self.edx_to = edx_to
        self.edx_scale = edx_scale

        self.edx_range_ind = [int(np.around(self.edx_from/self.edx_scale)), int(np.around(self.edx_to/self.edx_scale))]
        self.edx_offset_ind = [int(np.around((self.edx_from+self.edx_offset)/self.edx_scale)), int(np.around((self.edx_to+self.edx_offset)/self.edx_scale))]

        if total_edx:
            tot_edx = np.zeros(self.edx_dim)
            for i in range(len(self.subfolders)):
                num_img = len(self.edx_split[i])
                for j in range(num_img):
                    tot_edx += np.sum(self.edx_split[i][j].data, axis=(0, 1))
                    
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=100)
            ax.plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], tot_edx[self.edx_offset_ind[0]:self.edx_offset_ind[1]], 'k-')
            ax.tick_params(axis="both", labelsize=15)
            fig.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig.savefig(f"{base}_total_edx{ext}", bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

            self.tot_edx = tot_edx     

        if visual:
            for i in range(len(self.subfolders)):
                num_img = len(self.edx_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12), dpi=100)
                else:
                    fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10), dpi=100)
                    
                for j in range(num_img):
                    edx_sum_map = np.sum(self.edx_split[i][j].data[:, :, self.edx_range_ind[0]:self.edx_range_ind[1]], axis=2)
                    ax.flat[j*2].imshow(edx_sum_map, cmap='inferno')
                    if visual_title:
                        ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                    ax.flat[j*2].axis("off")
                    
                    edx_sum = np.mean(self.edx_split[i][j].data[:, :, self.edx_offset_ind[0]:self.edx_offset_ind[1]], axis=(0, 1))
                    ax.flat[j*2+1].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], edx_sum, 'k-')
                    if axis_off:
                        ax.flat[j*2+1].tick_params(axis="y", labelsize=0, color='white')
                fig.suptitle(self.subfolders[i]+' EDX intensity map and mean EDX spectrum')
                fig.tight_layout()
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig.savefig(f"{base}_sub_{i}{ext}", bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()        


    def edx_count(self, save_path=None):
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'edx_count') if self._default_figure_save_path is not None else None)
        if self.simult_edx == False:
            print("Warning! EDX data not loaded!")
            return
        
        count_list = []
        for i in range(len(self.subfolders)):
            for e in self.edx_split[i]:
                edx_data = e.data
                count = np.sum(edx_data, axis=2)
                count = count.flatten()
                count = count.tolist()
                count_list.extend(count)

        uq_count = np.unique(count_list)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi = 300)
        ax.hist(count_list, color='black', log=True, bins=len(uq_count))
        ax.tick_params(axis="both", labelsize=15)
        fig.tight_layout()
        if _eff_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
            base, ext = os.path.splitext(_eff_save)
            if not ext:
                ext = '.png'
            fig.savefig(f"{base}_edx_count{ext}", bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        self.uq_count = uq_count
        self.count_list = count_list


    def edx_classification(self, threshold_map="NMF", visual_title=True, 
                           title_font_size=10, axis_off=True, visual_individual=True, save_path=None):
        
        _eff_save = save_path if save_path is not None else (os.path.join(self._default_figure_save_path, 'edx_classification') if self._default_figure_save_path is not None else None)
        if self.simult_edx == False:
            print("Warning! EDX data not loaded!")
            return

        if threshold_map == "variance":
            fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
            sp_tot = np.zeros_like(self.edx_range)
            total_num = 0
            for i in range(len(self.subfolders)):
                num_img = len(self.edx_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                sub_num = 0
                
                if visual_individual:
                    if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                        fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12), dpi=100)
                    else:
                        fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10), dpi=100)

                sp_sub = np.zeros_like(self.edx_range)
                for j in range(num_img):
                    thresh_map = self.thresh_var_split[i][j]
                    if visual_individual:
                        ax.flat[j*2].imshow(thresh_map, cmap='gray')
                        if visual_title:
                            ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                        ax.flat[j*2].axis("off")
                    
                    tmp_num = len(np.where(thresh_map==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num
                    edx_sum = np.sum(self.edx_split[i][j].data[np.where(thresh_map==1)], axis=0)
                    sp_tot += edx_sum
                    sp_sub += edx_sum
                    if visual_individual:
                        ax.flat[j*2+1].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], sp_sub[self.edx_offset_ind[0]:self.edx_offset_ind[1]], 'k-')
                        if axis_off:
                            ax.flat[j*2+1].tick_params(axis="y", labelsize=0, color='white')

                if visual_individual:
                    fig.suptitle(self.subfolders[i]+' mean EDX spectrum for each high-variance map')
                    fig.tight_layout()
                
                if sub_num != 0:
                    sp_sub /= sub_num
                ax_tot[1].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], sp_sub[self.edx_offset_ind[0]:self.edx_offset_ind[1]], c=self.color_rep[i], label=self.subfolders[i])
                
            if total_num != 0:
                sp_tot /= total_num
            ax_tot[0].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], sp_tot[self.edx_offset_ind[0]:self.edx_offset_ind[1]], 'k-')
            ax_tot[1].legend()
            fig_tot.suptitle(self.subfolders[i]+' mean EDX spectrum for all high-variance maps')
            fig_tot.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig_tot.savefig(f"{base}_variance_edx_total{ext}", bbox_inches='tight')
                plt.close(fig_tot)
            else:
                plt.show()

            return sp_tot

        elif threshold_map == "cross-correlation":
            fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
            sp_tot = np.zeros_like(self.edx_range)
            total_num = 0
            for i in range(len(self.subfolders)):
                num_img = len(self.edx_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                sub_num = 0
                
                if visual_individual:
                    if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                        fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12), dpi=100)
                    else:
                        fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10), dpi=100)

                sp_sub = np.zeros_like(self.edx_range)
                for j in range(num_img):
                    thresh_map = self.thresh_xcor_split[i][j]
                    if visual_individual:
                        ax.flat[j*2].imshow(thresh_map, cmap='gray')
                        if visual_title:
                            ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                        ax.flat[j*2].axis("off")
                    
                    edx_sum = np.sum(self.edx_split[i][j].data[np.where(thresh_map==1)], axis=0)
                    tmp_num = len(np.where(thresh_map==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num
                    sp_tot += edx_sum
                    sp_sub += edx_sum
                    
                    if visual_individual:
                        ax.flat[j*2+1].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], sp_sub[self.edx_offset_ind[0]:self.edx_offset_ind[1]], 'k-')
                        if axis_off:
                            ax.flat[j*2+1].tick_params(axis="y", labelsize=0, color='white')

                if visual_individual:        
                    fig.suptitle(self.subfolders[i]+' mean EDX spectrum for each high-cross-correlation map')
                    fig.tight_layout()
                    
                if sub_num != 0:
                    sp_sub /= sub_num
                ax_tot[1].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], sp_sub[self.edx_offset_ind[0]:self.edx_offset_ind[1]], c=self.color_rep[i+1], label=self.subfolders[i])

            if total_num != 0:
                sp_tot /= total_num
            ax_tot[0].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], sp_tot[self.edx_offset_ind[0]:self.edx_offset_ind[1]], 'k-')
            ax_tot[1].legend()
            fig_tot.suptitle(self.subfolders[i]+' mean EDX spectrum for all high-cross-correlation maps')
            fig_tot.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig_tot.savefig(f"{base}_xcorr_edx_total{ext}", bbox_inches='tight')
                plt.close(fig_tot)
            else:
                plt.show()

            return sp_tot
        
        elif threshold_map == "NMF":
            lv_tot = []
            for lv in range(self.num_comp):
                fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
                sp_tot = np.zeros_like(self.edx_range)
                total_num = 0
                for i in range(len(self.subfolders)):
                    num_img = len(self.edx_split[i])
                    grid_size = int(np.around(np.sqrt(num_img)))
                    sub_num = 0
                    
                    if visual_individual:
                        if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                            fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12), dpi=100)
                        else:
                            fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10), dpi=100)
    
                    sp_sub = np.zeros_like(self.edx_range)
                    for j in range(num_img):
                        thresh_map = self.thresh_coeff_split[lv][i][j]
                    
                        if visual_individual:
                            ax.flat[j*2].imshow(thresh_map, cmap='gray')
                            if visual_title:
                                ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15], fontsize=title_font_size)
                            ax.flat[j*2].axis("off")
                        
                        edx_sum = np.sum(self.edx_split[i][j].data[np.where(thresh_map==1)], axis=0)
                        tmp_num = len(np.where(thresh_map==1)[0])
                        total_num += tmp_num
                        sub_num += tmp_num
                        sp_tot += edx_sum
                        sp_sub += edx_sum
                        
                        if visual_individual:
                            ax.flat[j*2+1].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], sp_sub[self.edx_offset_ind[0]:self.edx_offset_ind[1]], 'k-')
                            if axis_off:
                                ax.flat[j*2+1].tick_params(axis="y", labelsize=0, color='white')

                    if visual_individual:
                        fig.suptitle(self.subfolders[i]+' mean EDX spectrum for each high-coefficient map for loading vector %d'%(lv+1))
                        fig.tight_layout()                        
                            
                    if sub_num != 0:
                        sp_sub /= sub_num
                    ax_tot[1].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], sp_sub[self.edx_offset_ind[0]:self.edx_offset_ind[1]], c=self.color_rep[i+1], label=self.subfolders[i])

                if total_num != 0:
                    sp_tot /= total_num
                lv_tot.append(sp_tot)
                ax_tot[0].plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], sp_tot[self.edx_offset_ind[0]:self.edx_offset_ind[1]], 'k-')
                ax_tot[1].legend()
                fig_tot.suptitle(self.subfolders[i]+' mean EDX spectrum for all high-coefficient maps for loading vector %d'%(lv+1))
                fig_tot.tight_layout()

            fig_lv, ax_lv = plt.subplots(1, 1, figsize=(12, 4), dpi=100)
            for l, line in enumerate(lv_tot):
                ax_lv.plot(self.edx_range[self.edx_range_ind[0]:self.edx_range_ind[1]], 
                           line[self.edx_offset_ind[0]:self.edx_offset_ind[1]], c=self.color_rep[l+1], label='lv %d'%(l+1))
            ax_lv.legend()
            fig_lv.suptitle("Compare the mean of EDX spectra between loading vectors")
            fig_lv.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig_lv.savefig(f"{base}_nmf_edx_lv_compare{ext}", bbox_inches='tight')
                plt.close(fig_lv)
            else:
                plt.show()

            return lv_tot
        
        else:
            print("Warning! unavailable type!")

    
    def summary_save(self, sv_range=None, percentile_threshold=None, save=False, also_dp=False, log_scale_dp=False, also_tiff=False, specific_data=[], figure_save_path=None):
        
        _eff_save = figure_save_path if figure_save_path is not None else (os.path.join(self._default_figure_save_path, 'summary_save') if self._default_figure_save_path is not None else None)
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_split[i])
            max_dps = []
            mean_dps = []
            for j in range(num_img):
                for key in specific_data:
                    if key in self.loaded_data_path[i][j]:
                        save_path = os.path.dirname(self.loaded_data_path[i][j]) # able to change the base directory for saving
                        print("save directory: ", save_path)
                        data_name = os.path.basename(self.loaded_data_path[i][j]).split("_")
                        data_name = data_name[0]+'_'+data_name[1]
                        print("save prefix: ", data_name)
                        top, bottom, left, right = self.crop
                        fig, ax = plt.subplots(3, 3, figsize=(15, 15), dpi=100)
                        ax[0, 0].imshow(self.BF_disc_align[i][j][top:bottom, left:right], cmap='inferno')
                        ax[0, 0].set_title("Aligned BF disc")
                        ax[0, 0].axis("off")

                        sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                        ax[0, 1].imshow(sum_map, cmap='inferno')
                        ax[0, 1].set_title("Intensity map")
                        ax[0, 1].axis("off")                 

                        rv = self.radial_var_sum_split[i][j]
                        ax[0, 2].plot(self.x_axis, rv[self.range_ind[0]:self.range_ind[1]], 'k-', label="var_sum")
                        ax[0, 2].set_title("Sum of radial variance/mean profiles")
                        ax[0, 2].legend(loc='upper right')

                        ra = self.radial_avg_sum_split[i][j]
                        ax_twin = ax[0, 2].twinx()
                        ax_twin.plot(self.x_axis, ra[self.range_ind[0]:self.range_ind[1]], 'r:', label="mean_sum")
                        ax_twin.legend(loc='right')

                        if sv_range != None and sv_range != []:
                            var_map = np.sum(self.radial_var_split[i][j].isig[sv_range[0]:sv_range[1]].data, axis=2)
                        else:
                            var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                        ax[1, 0].imshow(var_map, cmap='inferno')
                        ax[1, 0].set_title('Variance map\nscattering vector range %.3f-%.3f (1/Å)'%(self.sv_range[0], self.sv_range[1]))

                        th_map = var_map.copy()
                        if percentile_threshold != None:
                            abs_threshold = np.percentile(var_map, percentile_threshold)
                            th_map[var_map<=abs_threshold] = 0
                            th_map[var_map>abs_threshold] = 1
                        else:
                            th_map[var_map<=self.abs_threshold] = 0
                            th_map[var_map>self.abs_threshold] = 1                    
                        ax[1, 1].imshow(th_map, cmap='gray')
                        ax[1, 1].set_title('High-variance map\nabsolute threshold %.3f'%self.abs_threshold)

                        if also_dp and len(np.nonzero(th_map)[0]) != 0:
                            dataset = hs.load(self.loaded_data_path[i][j][:-18]+'corrected_scaled.hspy')
                            if self.rebin_256:
                                if dataset.data.shape[1] > 250:
                                    dataset = dataset.rebin(scale = (2,2,1,1))


                            mean_dp = np.mean(dataset.data[np.where(th_map==1)], axis=0)
                            mean_dps.append(np.sum(dataset.data[np.where(th_map==1)], axis=0))
                            if save:
                                mean_dp_save = hs.signals.Signal2D(mean_dp)
                                mean_dp_save.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                                mean_dp_save.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                                mean_dp_save.save(save_path+'/'+data_name+"_mean_diffraction_pattern_for_threshold_map.hspy", overwrite=True)
                                if also_tiff:
                                    tifffile.imwrite(save_path+'/'+data_name+"_mean_diffraction_pattern_for_threshold_map.tiff", mean_dp_save.data)
                            if log_scale_dp:
                                mean_dp[mean_dp <= 0] = 1.0
                                ax[2, 1].imshow(np.log(mean_dp), cmap='inferno')
                                ax[2, 1].set_title('(log-scale) Mean diffraction pattern\nfor the high-variance map')
                            else:
                                ax[2, 1].imshow(mean_dp, cmap='inferno')
                                ax[2, 1].set_title('Mean diffraction pattern\nfor the high-variance map')


                            max_dp = np.max(dataset.data[np.where(th_map==1)], axis=0)
                            max_dps.append(max_dp)
                            if save:
                                max_dp_save = hs.signals.Signal2D(max_dp)
                                max_dp_save.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                                max_dp_save.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                                max_dp_save.save(save_path+'/'+data_name+"_max_diffraction_pattern_for_threshold_map.hspy", overwrite=True)
                                if also_tiff:
                                    tifffile.imwrite(save_path+'/'+data_name+"_max_diffraction_pattern_for_threshold_map.tif", max_dp_save.data)
                            if log_scale_dp:
                                max_dp[max_dp <= 0] = 1.0
                                ax[1, 2].imshow(np.log(max_dp), cmap='inferno')
                                ax[1, 2].set_title('(log-scale) Maximum diffraction pattern\nfor the thresholding map')
                            else:
                                ax[1, 2].imshow(max_dp, cmap='inferno')
                                ax[1, 2].set_title('Maximum diffraction pattern\nfor the high-variance map')

                            del dataset # release the occupied memory

                        if len(np.nonzero(th_map)[0]) != 0:
                            avg_rv = np.mean(self.radial_var_split[i][j].data[np.where(th_map==1)], axis=0)
                            ax[2, 2].plot(self.x_axis, avg_rv[self.range_ind[0]:self.range_ind[1]], 'k-')
                            ax[2, 2].set_title('Averaged radial variance profile\nfor the high-variance map')

                            avg_ra = np.mean(self.radial_avg_split[i][j].data[np.where(th_map==1)], axis=0)
                            ax22_twin = ax[2, 2].twinx()
                            ax22_twin.plot(self.x_axis, avg_ra[self.range_ind[0]:self.range_ind[1]], 'k-')
                            ax22_twin.set_title('Averaged radial mean profile\nfor the high-variance map')                    
                            if save:
                                avg_rv = hs.signals.Signal1D(avg_rv)
                                avg_rv.axes_manager[0].scale = self.pixel_size_inv_Ang
                                avg_rv.save(save_path+'/'+data_name+"_mean_radial_variance_profile_for_threshold_map.hspy", overwrite=True)
                                avg_ra = hs.signals.Signal1D(avg_ra)
                                avg_ra.axes_manager[0].scale = self.pixel_size_inv_Ang
                                avg_ra.save(save_path+'/'+data_name+"_mean_radial_mean_profile_for_threshold_map.hspy", overwrite=True) 

                        ax[1, 0].axis("off")
                        ax[1, 1].axis("off")
                        ax[1, 2].axis("off")
                        ax[2, 0].axis("off")
                        ax[2, 1].axis("off")

                        fig.suptitle(self.subfolders[i]+" - "+os.path.basename(self.loaded_data_path[i][j])[:15])
                        fig.tight_layout()
                        if _eff_save is not None:
                            os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                            base, ext = os.path.splitext(_eff_save)
                            if not ext:
                                ext = '.png'
                            fig.savefig(f"{base}_summary_{i}_{j}{ext}", bbox_inches='tight')
                            plt.close(fig)
                        else:
                            plt.show()

                        if save:
                            sum_map = hs.signals.Signal2D(sum_map)
                            sum_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                            sum_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                            sum_map.save(save_path+'/'+data_name+"_intensity_map.hspy", overwrite=True)
                            rv = hs.signals.Signal1D(rv)
                            rv.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                            rv.save(save_path+'/'+data_name+"_mean_radial_variance_profile.hspy", overwrite=True)
                            ra = hs.signals.Signal1D(ra)
                            ra.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                            ra.save(save_path+'/'+data_name+"_mean_radial_mean_profile.hspy", overwrite=True)            
                            var_map = hs.signals.Signal2D(var_map)
                            var_map = hs.signals.Signal2D(var_map)
                            var_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                            var_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                            var_map.save(save_path+'/'+data_name+"_variance_map.hspy", overwrite=True)
                            th_map = hs.signals.Signal2D(th_map)
                            th_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                            th_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                            th_map.save(save_path+'/'+data_name+"_threshold_map.hspy", overwrite=True)
                            fig.savefig(save_path+'/'+data_name+"_summary.png")
                            if also_tiff:
                                tifffile.imwrite(save_path+'/'+data_name+"_intensity_map.tif", sum_map.data)
                                tifffile.imwrite(save_path+'/'+data_name+"_mean_radial_variance_profile.tif", rv.data)
                                tifffile.imwrite(save_path+'/'+data_name+"_mean_radial_mean_profile.tif", ra.data)
                                tifffile.imwrite(save_path+'/'+data_name+"_variance_map.tif", var_map.data)
                                tifffile.imwrite(save_path+'/'+data_name+"_threshold_map.tif", th_map.data)

                    max_dps = np.asarray(max_dps)
                    mean_dps = np.asarray(mean_dps)
                    fig_dp, ax_dp = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
                    if log_scale_dp:
                        ax_dp[0].imshow(np.log(np.mean(max_dps, axis=0)), cmap="inferno")
                    else:
                        ax_dp[0].imshow(np.sum(max_dps, axis=0), cmap="inferno")
                    ax_dp[0].axis("off")
                    ax_dp[0].set_title("sum of all max diffraction patterns from each data")
                    if log_scale_dp:
                        ax_dp[1].imshow(np.log(np.mean(mean_dps, axis=0)), cmap="inferno")
                    else:
                        ax_dp[1].imshow(np.sum(mean_dps, axis=0), cmap="inferno")
                    ax_dp[1].axis("off")
                    ax_dp[1].set_title("sum of all diffraction patterns")
                    fig_dp.tight_layout()
                    if _eff_save is not None:
                        os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                        base, ext = os.path.splitext(_eff_save)
                        if not ext:
                            ext = '.png'
                        fig_dp.savefig(f"{base}_dp{ext}", bbox_inches='tight')
                        plt.close(fig_dp)
                    else:
                        plt.show()


    def summary_save(self, sv_range=None, percentile_threshold=None, save=False, also_dp=False, log_scale_dp=False, also_tiff=False, figure_save_path=None):

        _eff_save = figure_save_path if figure_save_path is not None else (os.path.join(self._default_figure_save_path, 'summary_save') if self._default_figure_save_path is not None else None)
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_split[i])
            max_dps = []
            mean_dps = []
            for j in range(num_img):
                save_path = os.path.dirname(self.loaded_data_path[i][j]) # able to change the base directory for saving
                print("save directory: ", save_path)
                data_name = os.path.basename(self.loaded_data_path[i][j]).split("_")
                data_name = data_name[0]+'_'+data_name[1]
                print("save prefix: ", data_name)
                top, bottom, left, right = self.crop
                fig, ax = plt.subplots(3, 3, figsize=(15, 15), dpi=100)
                ax[0, 0].imshow(self.BF_disc_align[i][j][top:bottom, left:right], cmap='inferno')
                ax[0, 0].set_title("Aligned BF disc")
                ax[0, 0].axis("off")

                sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                ax[0, 1].imshow(sum_map, cmap='inferno')
                ax[0, 1].set_title("Intensity map")
                ax[0, 1].axis("off")                 
        
                rv = self.radial_var_sum_split[i][j]
                ax[0, 2].plot(self.x_axis, rv[self.range_ind[0]:self.range_ind[1]], 'k-', label="var_sum")
                ax[0, 2].set_title("Sum of radial variance/mean profiles")
                ax[0, 2].legend(loc='upper right')
                
                ra = self.radial_avg_sum_split[i][j]
                ax_twin = ax[0, 2].twinx()
                ax_twin.plot(self.x_axis, ra[self.range_ind[0]:self.range_ind[1]], 'r:', label="mean_sum")
                ax_twin.legend(loc='right')

                if sv_range != None and sv_range != []:
                    var_map = np.sum(self.radial_var_split[i][j].isig[sv_range[0]:sv_range[1]].data, axis=2)
                else:
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                ax[1, 0].imshow(var_map, cmap='inferno')
                ax[1, 0].set_title('Variance map\nscattering vector range %.3f-%.3f (1/Å)'%(self.sv_range[0], self.sv_range[1]))
        
                th_map = var_map.copy()
                if percentile_threshold != None:
                    abs_threshold = np.percentile(var_map, percentile_threshold)
                    th_map[var_map<=abs_threshold] = 0
                    th_map[var_map>abs_threshold] = 1
                else:
                    th_map[var_map<=self.abs_threshold] = 0
                    th_map[var_map>self.abs_threshold] = 1                    
                ax[1, 1].imshow(th_map, cmap='gray')
                ax[1, 1].set_title('High-variance map\nabsolute threshold %.3f'%self.abs_threshold)

                if also_dp and len(np.nonzero(th_map)[0]) != 0:
                    dataset = hs.load(self.loaded_data_path[i][j][:-18]+'corrected_scaled.hspy')
                    if self.rebin_256:
                        if dataset.data.shape[1] > 250:
                            dataset = dataset.rebin(scale = (2,2,1,1))
                        
                    mean_dp = np.mean(dataset.data[np.where(th_map==1)], axis=0)
                    mean_dps.append(np.sum(dataset.data[np.where(th_map==1)], axis=0))
                    if save:
                        mean_dp_save = hs.signals.Signal2D(mean_dp)
                        mean_dp_save.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                        mean_dp_save.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                        mean_dp_save.save(save_path+'/'+data_name+"_mean_diffraction_pattern_for_threshold_map.hspy", overwrite=True)
                        if also_tiff:
                            tifffile.imwrite(save_path+'/'+data_name+"_mean_diffraction_pattern_for_threshold_map.tiff", mean_dp_save.data)
                    if log_scale_dp:
                        mean_dp[mean_dp <= 0] = 1.0
                        ax[2, 1].imshow(np.log(mean_dp), cmap='inferno')
                        ax[2, 1].set_title('(log-scale) Mean diffraction pattern\nfor the high-variance map')
                    else:
                        ax[2, 1].imshow(mean_dp, cmap='inferno')
                        ax[2, 1].set_title('Mean diffraction pattern\nfor the high-variance map')
                        
                        
                    max_dp = np.max(dataset.data[np.where(th_map==1)], axis=0)
                    max_dps.append(max_dp)
                    if save:
                        max_dp_save = hs.signals.Signal2D(max_dp)
                        max_dp_save.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                        max_dp_save.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                        max_dp_save.save(save_path+'/'+data_name+"_max_diffraction_pattern_for_threshold_map.hspy", overwrite=True)
                        if also_tiff:
                            tifffile.imwrite(save_path+'/'+data_name+"_max_diffraction_pattern_for_threshold_map.tif", max_dp_save.data)
                    if log_scale_dp:
                        max_dp[max_dp <= 0] = 1.0
                        ax[1, 2].imshow(np.log(max_dp), cmap='inferno')
                        ax[1, 2].set_title('(log-scale) Maximum diffraction pattern\nfor the thresholding map')
                    else:
                        ax[1, 2].imshow(max_dp, cmap='inferno')
                        ax[1, 2].set_title('Maximum diffraction pattern\nfor the high-variance map')
                        
                    del dataset # release the occupied memory
                    
                if len(np.nonzero(th_map)[0]) != 0:
                    avg_rv = np.mean(self.radial_var_split[i][j].data[np.where(th_map==1)], axis=0)
                    ax[2, 2].plot(self.x_axis, avg_rv[self.range_ind[0]:self.range_ind[1]], 'k-')
                    ax[2, 2].set_title('Averaged radial variance profile\nfor the high-variance map')

                    avg_ra = np.mean(self.radial_avg_split[i][j].data[np.where(th_map==1)], axis=0)
                    ax22_twin = ax[2, 2].twinx()
                    ax22_twin.plot(self.x_axis, avg_ra[self.range_ind[0]:self.range_ind[1]], 'k-')
                    ax22_twin.set_title('Averaged radial mean profile\nfor the high-variance map')                    
                    if save:
                        avg_rv = hs.signals.Signal1D(avg_rv)
                        avg_rv.axes_manager[0].scale = self.pixel_size_inv_Ang
                        avg_rv.save(save_path+'/'+data_name+"_mean_radial_variance_profile_for_threshold_map.hspy", overwrite=True)
                        avg_ra = hs.signals.Signal1D(avg_ra)
                        avg_ra.axes_manager[0].scale = self.pixel_size_inv_Ang
                        avg_ra.save(save_path+'/'+data_name+"_mean_radial_mean_profile_for_threshold_map.hspy", overwrite=True) 
        
                ax[1, 0].axis("off")
                ax[1, 1].axis("off")
                ax[1, 2].axis("off")
                ax[2, 0].axis("off")
                ax[2, 1].axis("off")
         
                fig.suptitle(self.subfolders[i]+" - "+os.path.basename(self.loaded_data_path[i][j])[:15])
                fig.tight_layout()
                if _eff_save is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                    base, ext = os.path.splitext(_eff_save)
                    if not ext:
                        ext = '.png'
                    fig.savefig(f"{base}_summary_{i}_{j}{ext}", bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
        
                if save:
                    sum_map = hs.signals.Signal2D(sum_map)
                    sum_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                    sum_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                    sum_map.save(save_path+'/'+data_name+"_intensity_map.hspy", overwrite=True)
                    rv = hs.signals.Signal1D(rv)
                    rv.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                    rv.save(save_path+'/'+data_name+"_mean_radial_variance_profile.hspy", overwrite=True)
                    ra = hs.signals.Signal1D(ra)
                    ra.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                    ra.save(save_path+'/'+data_name+"_mean_radial_mean_profile.hspy", overwrite=True)            
                    var_map = hs.signals.Signal2D(var_map)
                    var_map = hs.signals.Signal2D(var_map)
                    var_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                    var_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                    var_map.save(save_path+'/'+data_name+"_variance_map.hspy", overwrite=True)
                    th_map = hs.signals.Signal2D(th_map)
                    th_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                    th_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                    th_map.save(save_path+'/'+data_name+"_threshold_map.hspy", overwrite=True)
                    fig.savefig(save_path+'/'+data_name+"_summary.png")
                    if also_tiff:
                        tifffile.imwrite(save_path+'/'+data_name+"_intensity_map.tif", sum_map.data)
                        tifffile.imwrite(save_path+'/'+data_name+"_mean_radial_variance_profile.tif", rv.data)
                        tifffile.imwrite(save_path+'/'+data_name+"_mean_radial_mean_profile.tif", ra.data)
                        tifffile.imwrite(save_path+'/'+data_name+"_variance_map.tif", var_map.data)
                        tifffile.imwrite(save_path+'/'+data_name+"_threshold_map.tif", th_map.data)

            max_dps = np.asarray(max_dps)
            mean_dps = np.asarray(mean_dps)
            fig_dp, ax_dp = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
            if log_scale_dp:
                ax_dp[0].imshow(np.log(np.mean(max_dps, axis=0)), cmap="inferno")
            else:
                ax_dp[0].imshow(np.sum(max_dps, axis=0), cmap="inferno")
            ax_dp[0].axis("off")
            ax_dp[0].set_title("sum of all max diffraction patterns from each data")
            if log_scale_dp:
                ax_dp[1].imshow(np.log(np.mean(mean_dps, axis=0)), cmap="inferno")
            else:
                ax_dp[1].imshow(np.sum(mean_dps, axis=0), cmap="inferno")
            ax_dp[1].axis("off")
            ax_dp[1].set_title("sum of all diffraction patterns")
            fig_dp.tight_layout()
            if _eff_save is not None:
                os.makedirs(os.path.dirname(os.path.abspath(_eff_save)), exist_ok=True)
                base, ext = os.path.splitext(_eff_save)
                if not ext:
                    ext = '.png'
                fig_dp.savefig(f"{base}_dp_summary_{i}{ext}", bbox_inches='tight')
                plt.close(fig_dp)
            else:
                plt.show()

    def summary_report(self):
        """Generates a comprehensive Markdown report summarizing the data processing and analysis results."""
        report = []
        report.append("# Radial Profile Analysis Summary Report\n")
        
        report.append("## 1. General Configuration")
        report.append(f"- **Base Directory**: `{self.base_dir}`")
        report.append(f"- **Subfolders**: {self.subfolders}")
        report.append(f"- **Profile Length**: {self.profile_length}")
        report.append(f"- **Number of Files to Load (per Subfolder)**: {self.num_load}")
        report.append(f"- **Rebin 256**: {self.rebin_256}")
        report.append(f"- **Simultaneous EDX**: {self.simult_edx}")
        report.append(f"- **Zernike Moments**: {self.zernike}")
        if self.include_key:
            report.append(f"- **Include Keys**: {self.include_key}")
        if self.exclude_key:
            report.append(f"- **Exclude Keys**: {self.exclude_key}")
        report.append("")
        
        report.append("## 2. Loaded Datasets")
        for i, sub in enumerate(self.subfolders):
            report.append(f"### Subfolder: `{sub}`")
            for j, fpath in enumerate(self.loaded_data_path[i]):
                edx_info = ""
                if self.simult_edx:
                    edx_info = f" (EDX: `{os.path.basename(self.loaded_edx_path[i][j])}`)"
                report.append(f"- Sample {j+1}: `{os.path.basename(fpath)}`{edx_info}")
        report.append("")
        
        if hasattr(self, 'crop') and self.crop:
            report.append("## 3. Preprocessing")
            report.append(f"- **Center Beam Alignment Crop**: `top:bottom, left:right = {self.crop}`")
            report.append("")
            
        if hasattr(self, 'from_unit'):
            report.append("## 4. Reciprocal Space Settings")
            report.append(f"- **Scattering Vector Range**: `[{self.from_unit:.4f}, {self.to_unit:.4f}] 1/Å` (Indices: `[{self.from_ind}, {self.to_ind}]`)")
            report.append(f"- **Reciprocal Pixel Size**: `{self.pixel_size_inv_Ang:.6f} 1/Å`")
            if hasattr(self, 'str_path') and self.str_path:
                report.append("- **CIF Files Loaded**:")
                for adr in self.str_path:
                    report.append(f"  - `{os.path.basename(adr)}`")
                if hasattr(self, 'peak_sf') and self.peak_sf:
                    report.append("- **Simulated Phase Peak Positions (1/Å)**:")
                    for key, peaks in self.peak_sf.items():
                        peaks_str = ", ".join([f"{p:.3f}" for p in peaks])
                        report.append(f"  - `{key}`: [{peaks_str}]")
            report.append("")
            
        if hasattr(self, 'num_comp'):
            report.append("## 5. NMF Decomposition & Area Ratios")
            report.append(f"- **Number of Components**: {self.num_comp}")
            report.append(f"- **Profile Type Decomposed**: `{self.NMF_profile_type}`")
            if hasattr(self, 'nmf_reconstruction_err') and self.nmf_reconstruction_err is not None:
                report.append(f"- **Reconstruction Error**: {self.nmf_reconstruction_err:.6f}")
            if hasattr(self, 'nmf_params'):
                params_str = ", ".join([f"{k}={v}" for k, v in self.nmf_params.items()])
                report.append(f"- **Decomposition Parameters**: {params_str}")
            if hasattr(self, 'percentile_threshold'):
                report.append(f"- **Coefficient Percentile Threshold**: {self.percentile_threshold}%")
            report.append("")
            
            # Compute and show the area occupancy table
            if hasattr(self, 'thresh_coeff_split') and self.thresh_coeff_split:
                report.append("### Phase Area Occupancy Ratios")
                headers = ["Subfolder", "Sample Filename"]
                for lv in range(self.num_comp):
                    headers.append(f"LV{lv+1} Area % (px)")
                
                rows = []
                for i, sub in enumerate(self.subfolders):
                    num_img = len(self.radial_var_split[i])
                    for j in range(num_img):
                        file_name = os.path.basename(self.loaded_data_path[i][j])
                        row = [sub, file_name]
                        for lv in range(self.num_comp):
                            binary_map = self.thresh_coeff_split[lv][i][j]
                            total_pixels = binary_map.size
                            active_pixels = np.sum(binary_map)
                            pct = (active_pixels / total_pixels) * 100
                            row.append(f"{pct:.2f}% ({active_pixels}/{total_pixels})")
                        rows.append(row)
                
                table_lines = []
                table_lines.append("| " + " | ".join(headers) + " |")
                table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    table_lines.append("| " + " | ".join(map(str, row)) + " |")
                report.append("\n".join(table_lines))
                report.append("")

        if hasattr(self, 'clustering_params') and self.clustering_params:
            report.append("## 6. Spatial Clustering of Grain Areas")
            report.append(f"- **Algorithm**: `{self.clustering_params.get('algorithm')}`")
            report.append(f"- **Parameters**: `eps={self.clustering_params.get('eps')}`, `min_sample={self.clustering_params.get('min_sample')}`")
            report.append(f"- **Thresholding Map**: `{self.clustering_params.get('threshold_map')}`")
            report.append(f"- **Selected Dataset**: `{os.path.basename(self.selected_data_path)}`")
            if hasattr(self, 'clustered_lv') and self.clustered_lv:
                report.append("- **Clusters Found by Loading Vector**:")
                for lv, clustered_map in enumerate(self.clustered_lv):
                    num_clusters = int(np.max(clustered_map))
                    report.append(f"  - **LV{lv+1}**: {num_clusters} cluster(s) detected")
            report.append("")

        return "\n".join(report)

    def __getstate__(self):
        state = self.__dict__.copy()
        blacklist = [
            'data_3d', 'radial_var_sum_split',
            'radial_avg_split', 'radial_avg_sum_split', 'edx_split',
            'edx_raw', 'BF_disc_align', 'run_SI', 'zernike_split',
            'zernike_sum_split', 'zernike', 'virtual_lv'
        ]
        for key in blacklist:
            if key in state:
                state[key] = None
        
        if 'radial_var_split' in state and state['radial_var_split'] is not None:
            try:
                state['radial_var_split'] = [[None for _ in sublist] for sublist in state['radial_var_split']]
            except Exception:
                state['radial_var_split'] = None
                
        return state

    def save_state(self, filepath):
        """Pickles the object state to a file for downstream cross-analysis."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Object state successfully saved to {filepath}")
            
