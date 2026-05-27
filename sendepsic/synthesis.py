import os
import pickle
import numpy as np
import pandas as pd

# Import from peer module
from .acom import map_3d_orientations

def comprehensive_scientific_synthesis(rpa, pa, acom_pkl_path, crystal_system='cubic'):
    """
    Performs a cross-analysis by merging results from:
      1. NMF components and area ratios (from radial_profile_analysis `rpa`)
      2. Crystallographic phase matching (from phase_analysis `pa`)
      3. Dominant orientation clustering (from ACOM result pickle file `acom_pkl_path`)
    Generates a unified Markdown synthesis report containing data tables and scientific insights.
    """
    # Load RPA from pickle if path is provided
    if isinstance(rpa, str) and os.path.exists(rpa):
        try:
            with open(rpa, 'rb') as f:
                rpa = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load RPA pickle from {rpa}: {e}")
            
    # Load PA from pickle if path is provided
    if isinstance(pa, str) and os.path.exists(pa):
        try:
            with open(pa, 'rb') as f:
                pa = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load PA pickle from {pa}: {e}")
            
    # 1. Load ACOM results if file exists
    acom_results = None
    if acom_pkl_path:
        actual_acom_path = acom_pkl_path
        if not os.path.exists(actual_acom_path):
            alt_path = '/data/ryuserve/LHP/analysis/FA_BA/ctrl/ACOM_result.pkl'
            if os.path.exists(alt_path):
                actual_acom_path = alt_path
        if os.path.exists(actual_acom_path):
            try:
                with open(actual_acom_path, "rb") as f:
                    acom_results = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load ACOM pickle from {actual_acom_path}: {e}")
            
    report = []
    report.append("# Comprehensive Data Synthesis & Scientific Insights Report\n")
    
    report.append("## 1. Executive Summary")
    report.append("This report synthesizes results across three different steps of the characterization pipeline: "
                  "phase segmentation (NMF), phase identification (MMAD pattern matching), and crystal orientation mapping (ACOM). "
                  "The objective is to map out the spatial distribution and texture (orientation) of various crystallographic phases "
                  "across the analyzed samples to uncover phase evolution and structure-property relationships.\n")
    
    report.append("## 2. Integrated Phase and Orientation Synthesis")
    
    # Check what RPA and PA attributes are available
    has_rpa = hasattr(rpa, 'loaded_data_path') and rpa.loaded_data_path
    has_pa = hasattr(pa, 'match_results') and pa.match_results
    
    if not has_rpa:
        report.append("*Error: Radial Profile Analysis data not loaded in the provided rpa object.*")
        return "\n".join(report)
        
    headers = ["Sample (Sub Index)", "Component", "Identified Phase Structure", "Area Ratio % (px)", "Dominant Out-of-Plane", "Dominant In-Plane"]
    rows = []
    
    for i, sub in enumerate(rpa.subfolders):
        num_img = len(rpa.radial_var_split[i])
        for j in range(num_img):
            fpath = rpa.loaded_data_path[i][j]
            filename = os.path.basename(fpath)
            # extract sample prefix e.g., '20240418_155246'
            parts = filename.split("_")
            sample_prefix = parts[0] + "_" + parts[1] if len(parts) >= 2 else filename
            
            sub_name = f"SI{i+1}"
            
            for lv in range(rpa.num_comp):
                lv_name = f"LV{lv+1}"
                
                # Phase identification lookup
                phase_name = "Unknown Phase"
                if has_pa:
                    # Try Median Winner
                    if hasattr(pa, 'median_winners') and sub_name in pa.median_winners:
                        phase_name = pa.median_winners[sub_name].get(lv_name, "Unknown")
                    # Try Frequency Winner
                    elif hasattr(pa, 'frequency_winners') and sub_name in pa.frequency_winners:
                        phase_name = pa.frequency_winners[sub_name].get(lv_name, "Unknown")
                    # Try Diff Winner
                    elif hasattr(pa, 'matchings') and pa.matchings and hasattr(pa, 'diffs') and pa.diffs:
                        diffs_sub = pa.diffs[i]
                        matchings_sub = pa.matchings[i]
                        sort_diff_ind = np.argsort(diffs_sub[:, 0])
                        best_diff_match = matchings_sub[sort_diff_ind[-1]]
                        for row in best_diff_match:
                            if row[1] == lv_name and row[2] == 1:
                                phase_name = row[3]
                                break
                    # Try raw match_results
                    elif hasattr(pa, 'match_results') and len(pa.match_results[i]) > 0:
                        run_data = pa.match_results[i][0] # first run
                        good_matches = run_data[1]
                        for match in good_matches:
                            if match[1] == lv_name:
                                phase_name = match[2]
                                break
                
                # Area ratio calculation
                area_str = "N/A"
                if hasattr(rpa, 'thresh_coeff_split') and rpa.thresh_coeff_split:
                    binary_map = rpa.thresh_coeff_split[lv][i][j]
                    pct = (np.sum(binary_map) / binary_map.size) * 100
                    area_str = f"{pct:.1f}%"
                    
                # ACOM dominant orientations lookup
                oop_orient = "N/A"
                ip_orient = "N/A"
                
                if acom_results and phase_name != "Unknown Phase" and phase_name != "Unknown":
                    # Look for matching ACOM entries
                    matching_acom_key = None
                    for key in acom_results.keys():
                        if "image_orientation" in key and sample_prefix in key:
                            matching_acom_key = key
                            break
                            
                    if matching_acom_key and phase_name in acom_results[matching_acom_key]:
                        dominant_orientation = acom_results[matching_acom_key][phase_name]
                        # Run orientation clustering
                        out_of_plane = dominant_orientation[:, :, :, 0]
                        in_plane = dominant_orientation[:, :, :, 1]
                        
                        oop_res = map_3d_orientations(out_of_plane, num_clusters=2, crystal_system=crystal_system, threshold=0.1)
                        ip_res = map_3d_orientations(in_plane, num_clusters=2, crystal_system=crystal_system, threshold=0.1)
                        
                        if len(oop_res["centers"]) > 0:
                            top_oop_idx = np.argmax([np.sum(oop_res["labels_grid"] == c) for c in range(len(oop_res["centers"]))])
                            top_oop_pct = (np.sum(oop_res["labels_grid"] == top_oop_idx) / oop_res["labels_grid"].size) * 100
                            oop_orient = f"{oop_res['cluster_names'][top_oop_idx]} ({top_oop_pct:.1f}%)"
                            
                        if len(ip_res["centers"]) > 0:
                            top_ip_idx = np.argmax([np.sum(ip_res["labels_grid"] == c) for c in range(len(ip_res["centers"]))])
                            top_ip_pct = (np.sum(ip_res["labels_grid"] == top_ip_idx) / ip_res["labels_grid"].size) * 100
                            ip_orient = f"{ip_res['cluster_names'][top_ip_idx]} ({top_ip_pct:.1f}%)"
                
                rows.append([f"{sample_prefix} ({sub_name})", lv_name, phase_name, area_str, oop_orient, ip_orient])
                
    table_lines = []
    table_lines.append("| " + " | ".join(headers) + " |")
    table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    report.append("\n".join(table_lines))
    report.append("")
    
    # 3. Scientific Insights & Discussion
    report.append("## 3. Scientific Discussion & Trend Discovery")
    
    # Aggregate stats to report scientific findings
    df_rows = pd.DataFrame(rows, columns=headers)
    valid_phases = df_rows[~df_rows["Identified Phase Structure"].isin(["Unknown Phase", "Unknown"])]
    
    if not valid_phases.empty:
        phase_counts = valid_phases["Identified Phase Structure"].value_counts()
        most_common_phase = phase_counts.index[0]
        report.append(f"### 3.1 Phase Distribution & Abundance Trends")
        report.append(f"- **Dominant Phase Observed**: The most frequently identified phase across samples is **{most_common_phase}** (found in {phase_counts.iloc[0]} components).")
        
        # Check area ratios
        valid_phases_with_area = valid_phases[valid_phases["Area Ratio % (px)"] != "N/A"]
        if not valid_phases_with_area.empty:
            valid_phases_with_area["Area_Val"] = valid_phases_with_area["Area Ratio % (px)"].str.replace("%", "").astype(float)
            grouped_area = valid_phases_with_area.groupby("Identified Phase Structure")["Area_Val"].mean().sort_values(ascending=False)
            highest_area_phase = grouped_area.index[0]
            report.append(f"- **Highest Volume Fraction**: **{highest_area_phase}** shows the highest average area coverage of **{grouped_area.iloc[0]:.1f}%** in regions where it is present.")
            report.append(f"  - Full phase average area coverage: " + ", ".join([f"{p}: {v:.1f}%" for p, v in grouped_area.items()]))
            
        # Check preferred orientation (texture)
        textured_phases = valid_phases[(valid_phases["Dominant Out-of-Plane"] != "N/A") | (valid_phases["Dominant In-Plane"] != "N/A")]
        if not textured_phases.empty:
            report.append(f"\n### 3.2 Crystallographic Texture & Orientation Alignment")
            for phase, group in textured_phases.groupby("Identified Phase Structure"):
                oop_list = group[group["Dominant Out-of-Plane"] != "N/A"]["Dominant Out-of-Plane"].tolist()
                ip_list = group[group["Dominant In-Plane"] != "N/A"]["Dominant In-Plane"].tolist()
                
                report.append(f"#### Phase: **{phase}**")
                if oop_list:
                    report.append(f"- **Out-of-Plane Orientations**: {', '.join(set(oop_list))}")
                if ip_list:
                    report.append(f"- **In-Plane Orientations**: {', '.join(set(ip_list))}")
                    
            report.append("\n#### Epitaxy and Texture Insights")
            report.append("By comparing the dominant out-of-plane and in-plane orientations, we can determine whether there is "
                          "preferred grain growth (texture) or epitaxy. For example, if a specific phase consistently displays "
                          "the same dominant out-of-plane pole (e.g., `~[100]`), this points to a strong growth direction preference "
                          "normal to the substrate.")
    else:
        report.append("No identified phases available for trend analysis. Please run `phase_analysis` matching first.")
        
    return "\n".join(report)
