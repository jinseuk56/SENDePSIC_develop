import sys
import os
import time
import numpy as np
import hyperspy.api as hs
import tempfile
import shutil

# Make sure we import from the installed package
from sendepsic.radial_profile import radial_profile_analysis
from sendepsic.feature_extract import feature_extract

class MockRadialProfileAnalysis(radial_profile_analysis):
    def __init__(self, mock_signals, mock_paths):
        # Bypass file loading __init__ and set attributes manually
        self.formatted = "mock_test_run"
        self.color_rep = ["black", "red", "green", "blue", "orange", "purple", "yellow", "lime", 
                    "cyan", "magenta", "lightgray", "peru", "springgreen", "deepskyblue", 
                    "hotpink", "darkgray"]
        self.pixel_size_inv_Ang = 0.05
        self.subfolders = ["mock_sub"]
        self.profile_length = 360
        self.num_load = 2
        
        # Populate mock splits
        self.radial_var_split = [mock_signals]
        self.radial_avg_split = [mock_signals] # simplified for test
        self.loaded_data_path = [mock_paths]
        
        # Slicing indexes/scales
        self.from_ = 0
        self.to_ = 100
        self.from_ind = 0
        self.to_ind = 100
        self.rebin_256 = False

def run_test():
    print("=== SENDePSIC Refactoring Verification Test ===")
    
    # 1. Create mock 3D spectrum data (e.g. 2 images, each 10x10 scan with 360 energy/radial channels)
    print("Generating mock data...")
    img1_data = np.random.rand(10, 10, 360).astype(np.float32)
    img2_data = np.random.rand(10, 10, 360).astype(np.float32)
    
    sig1 = hs.signals.Signal1D(img1_data)
    sig2 = hs.signals.Signal1D(img2_data)
    
    # 2. Write them to temporary files so we can test the "disk-load" feature_extract (old behavior)
    temp_dir = tempfile.mkdtemp()
    img1_path = os.path.join(temp_dir, "img1_azimuthal_var.hspy")
    img2_path = os.path.join(temp_dir, "img2_azimuthal_var.hspy")
    
    try:
        print(f"Writing mock files to temporary directory: {temp_dir}")
        sig1.save(img1_path, overwrite=True)
        sig2.save(img2_path, overwrite=True)
        
        mock_signals = [sig1, sig2]
        mock_paths = [img1_path, img2_path]
        
        # 3. Instantiate Mock RPA
        rpa_obj = MockRadialProfileAnalysis(mock_signals, mock_paths)
        
        # 4. Measure old behavior (loading from disk)
        print("\n--- Running NMF with disk-loading (Simulating original behavior) ---")
        start_time = time.time()
        fe_disk = feature_extract(
            mock_paths, dat_dim=3, dat_unit='1/Å', 
            cr_range=[0, 100, 1], dat_scale=1, rescale=False, DM_file=True, verbose=False, rebin_256=False
        )
        fe_disk.make_input(min_val=0.0, max_normalize=False, rescale_0to1=True)
        fe_disk.ini_DR(method="nmf", num_comp=3, result_visual=False)
        disk_time = time.time() - start_time
        print(f"Disk NMF components shape: {fe_disk.DR_comp_vectors.shape}")
        print(f"Disk NMF coefficients shape: {fe_disk.DR_coeffs.shape}")
        print(f"Disk NMF time: {disk_time:.4f} seconds")
        
        # 5. Measure new behavior (in-memory)
        print("\n--- Running NMF in-memory (Optimized behavior) ---")
        start_time = time.time()
        fe_mem = feature_extract(
            mock_paths, dat_dim=3, dat_unit='1/Å', 
            cr_range=[0, 100, 1], dat_scale=1, rescale=False, DM_file=True, verbose=False, rebin_256=False,
            data_storage=mock_signals
        )
        fe_mem.make_input(min_val=0.0, max_normalize=False, rescale_0to1=True)
        fe_mem.ini_DR(method="nmf", num_comp=3, result_visual=False)
        mem_time = time.time() - start_time
        print(f"In-memory NMF components shape: {fe_mem.DR_comp_vectors.shape}")
        print(f"In-memory NMF coefficients shape: {fe_mem.DR_coeffs.shape}")
        print(f"In-memory NMF time: {mem_time:.4f} seconds")
        
        # Check correctness/equivalence of shapes
        assert fe_disk.DR_comp_vectors.shape == fe_mem.DR_comp_vectors.shape
        assert fe_disk.DR_coeffs.shape == fe_mem.DR_coeffs.shape
        assert len(fe_disk.coeffs_reshape) == len(fe_mem.coeffs_reshape)
        assert fe_disk.coeffs_reshape[0].shape == fe_mem.coeffs_reshape[0].shape
        
        print("\n--> Shape assertions passed!")
        print(f"--> Speedup ratio: {disk_time / mem_time:.2f}x faster in-memory!")
        
        # 6. Test RPA class wrapper NMF_decompose integration
        print("\n--- Running RPA NMF_decompose wrapper ---")
        rpa_obj.NMF_decompose(num_comp=3, scale_crop=True, rescale_SI=False, max_normalize=False, rescale_0to1=True, profile_type="variance", verbose=False)
        print("RPA wrapper executed successfully!")
        print(f"run_SI type: {type(rpa_obj.run_SI)}")
        print(f"run_SI coefficients shape: {rpa_obj.run_SI.DR_coeffs.shape}")
        print(f"run_SI components shape: {rpa_obj.run_SI.DR_comp_vectors.shape}")
        print(f"run_SI reshaped coefficients shape: {[c.shape for c in rpa_obj.run_SI.coeffs_reshape]}")
        
        assert rpa_obj.run_SI.DR_comp_vectors.shape == (3, 100)
        assert rpa_obj.run_SI.DR_coeffs.shape == (200, 3)
        assert rpa_obj.run_SI.coeffs_reshape[0].shape == (10, 10, 3)
        
        print("\n=== All Tests Passed Successfully! ===")
        
    finally:
        # Clean up temp files
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    run_test()
