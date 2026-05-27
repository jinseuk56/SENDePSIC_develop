import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import py4DSTEM

# Import shared helpers
from .utils import (
    profile_peak, simul_xrd_peak, mmad_score, ConcaveHull
)

class phase_analysis:
    def phase_matching_initialization(self, str_path, target_structures, profile, num_comp, num_sub=1, crop=None, color_rep=None):
        self.str_path = str_path
        self.target_structures = target_structures
        self.profile = profile
        self.num_comp = num_comp
        self.num_sub = num_sub
        
        if self.target_structures == None:
            self.target_structures = []
            for adr in self.str_path:
                self.target_structures.append(os.path.basename(adr).split('.')[0])

        if crop != None:
            self.crop = crop
        else:
            self.crop = [0, len(profile[0])]

        if color_rep != None:
            self.color_rep = color_rep
        else:
            self.color_rep = ["black", "red", "green", "blue", "orange", "purple", "yellow", "lime", 
                    "cyan", "magenta", "lightgray", "peru", "springgreen", "deepskyblue", 
                    "hotpink", "darkgray"]

        if len(self.color_rep) < self.num_comp:
            print("The number of colors for plotting is less than that of profiles.")
            print("Plotting may not work.")


    def profile_comparison(self, profile_k_step, max_num_peaks=3, ylim=None, line_color=None, show_legend=True):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        self.peak_list = []
        for lv in range(self.num_comp):
            line = self.profile[lv]
            if line_color:
                ax.plot(np.arange(len(line))*profile_k_step, line, c=line_color[lv], label="loading vector %d"%(lv+1))
            else:
                ax.plot(np.arange(len(line))*profile_k_step, line, c=self.color_rep[lv+1], label="loading vector %d"%(lv+1))
            
            p = profile_peak(line, profile_k_step, self.crop, max_num_peaks=max_num_peaks)
            self.peak_list.append(p)
            for peak in p:
                ax.axvline(peak, c="gray", linestyle="--")
                ax.text(peak, np.max(line), "%.3f"%(peak))
                
        if show_legend:
            ax.legend(loc='upper right')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel("1/Å")
        fig.tight_layout()
        plt.show()


    def simple_comparsion(self, cif_adr, profile_k_step, max_num_peaks, xrd_k_step, broadening, xrd_peak_prominence, ylim=None):
        fig, ax = plt.subplots(self.num_comp, 1, figsize=(6, 4*self.num_comp), dpi=100)
        self.xrd_peak_list = []
        k_range = np.arange(self.crop[0], self.crop[1], 1) * profile_k_step
        basename, from_cif, peaks_cif = simul_xrd_peak(cif_adr, k_range, broadening, xrd_peak_prominence)
        
        for lv in range(self.num_comp):
            line = self.profile[lv][self.crop[0]:self.crop[1]].copy()
            line /= np.max(line)
            
            if self.num_comp == 1:
                ax_ = ax
            else:
                ax_ = ax[lv]
                
            ax_.plot(k_range, line, c=self.color_rep[lv+1], label="LV %d"%(lv+1))
            ax_.plot(k_range, from_cif, c="black", label=basename)
            ax_.legend(loc='upper right')
            
            p = self.peak_list[lv]
            for peak in p:
                ax_.axvline(peak, c=self.color_rep[lv+1], linestyle="--")
                
            for peak in peaks_cif:
                ax_.axvline(peak, c="black", linestyle="--")
                
            if ylim:
                ax_.set_ylim(ylim)
                
        fig.tight_layout()
        plt.show()


    def MMAD_matching(self, profile_step_list, 
                      xrd_step_list, 
                      broadening_list, 
                      xrd_peak_prominence_list,
                      max_num_peaks=3):
        
        self.match_results = []
        self.params = []
        self.diffs = []
        
        k_range = np.arange(self.crop[0], self.crop[1], 1) * profile_step_list[0]
        
        for i in range(self.num_sub):
            print("=========================================")
            print("Sub Index: SI%d"%(i+1))
            
            sub_results = []
            
            # 1. Simulate profiles for target structures using grid parameters
            run_idx = 0
            for p_step in profile_step_list:
                for x_step in xrd_step_list:
                    for broad in broadening_list:
                        for prom in xrd_peak_prominence_list:
                            
                            cif_profiles = {}
                            cif_peaks = {}
                            
                            for adr in self.str_path:
                                name, prof, peaks = simul_xrd_peak(adr, k_range, broad, prom)
                                cif_profiles[name] = prof
                                cif_peaks[name] = peaks
                            
                            # 2. Extract experimental peaks for this sub-index
                            exp_peaks = {}
                            experimental_cols = []
                            for lv in range(self.num_comp):
                                lv_name = "LV%d"%(lv+1)
                                experimental_cols.append(lv_name)
                                exp_peaks[lv_name] = self.peak_list[lv]
                            
                            # 3. Create dataframe with experimental and simulation peaks
                            df_peaks = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in {**exp_peaks, **cif_peaks}.items() ]))
                            
                            # 4. Perform Reciprocal Distance Matching (Mean of Minimum Absolute Differences, MMAD)
                            scores = mmad_score(df_peaks, experimental_cols)
                            
                            # Compile summary table for this run
                            run_summary = []
                            for lv_col_name in experimental_cols:
                                for rank, item in enumerate(scores[lv_col_name], 1):
                                    run_summary.append([
                                        "SI%d"%(i+1),
                                        lv_col_name,
                                        rank,
                                        item['simulation'],
                                        item['mmad_score']
                                    ])
                            
                            df_summary = pd.DataFrame(run_summary, columns=["Sub Index", "LV", "Rank (reciprocal distance)", "Structure", "SAD Score"])
                            
                            # Compile best matches (rank 1)
                            good_match = []
                            for lv_col_name in experimental_cols:
                                item = scores[lv_col_name][0]
                                good_match.append([
                                    "SI%d"%(i+1),
                                    lv_col_name,
                                    item['simulation'],
                                    item['mmad_score']
                                ])
                                
                            sub_results.append([
                                [p_step, x_step, broad, prom],
                                good_match,
                                df_summary
                            ])
                            
                            run_idx += 1
            
            # Select the best parameter set for this sub-index
            # Criterion: largest mean of differences between matching scores (best vs second best)
            diff_mean_std = []
            param_combinations = []
            for run in sub_results:
                params = run[0]
                df_sum = run[2]
                
                # Calculate difference between 1st and 2nd rank for each LV
                diffs = []
                for lv in range(self.num_comp):
                    lv_name = "LV%d"%(lv+1)
                    subset = df_sum[df_sum["LV"] == lv_name].sort_values("Rank (reciprocal distance)")
                    if len(subset) >= 2:
                        first_score = subset.iloc[0]["SAD Score"]
                        second_score = subset.iloc[1]["SAD Score"]
                        if first_score == float('inf') or second_score == float('inf'):
                            diffs.append(0.0)
                        else:
                            diffs.append(abs(second_score - first_score))
                    else:
                        diffs.append(0.0)
                
                diff_mean_std.append([np.mean(diffs), np.std(diffs)])
                param_combinations.append(params)
                
            diff_mean_std = np.asarray(diff_mean_std)
            sort_diff_ind = np.argsort(diff_mean_std[:, 0])
            best_idx = sort_diff_ind[-1]
            
            print(f"Total number of grid parameter sets: {len(sub_results)}")
            print(f"Selected best parameter index: {best_idx+1} (0-indexed: {best_idx})")
            print(f"Selected parameters: Profile step={param_combinations[best_idx][0]:.6f}, XRD step={param_combinations[best_idx][1]:.6f}, Broadening={param_combinations[best_idx][2]:.6f}, Prominence={param_combinations[best_idx][3]:.6f}")
            print(f"Mean Difference of MMAD scores: {diff_mean_std[best_idx][0]:.6f} (std={diff_mean_std[best_idx][1]:.6f})")
            
            # Save all match results
            self.match_results.append(sub_results)
            self.params.append(param_combinations)
            self.diffs.append(diff_mean_std)
            
            # Run the best match and print results
            best_run = sub_results[best_idx]
            good_match = best_run[1]
            for m in good_match:
                print(f"Best match for Sub Index: {m[0]}, {m[1]} is {m[2]} with MMAD score {m[3]:.6f}")
                

    def best_match_diff(self):
        # Difference method (Diff): Selection of parameters based on the parameter set 
        # that yields the maximum average difference between the 1st and 2nd matched structures
        # self.diffs has shape (self.num_sub, len(sub_results), 2)
        # self.match_results has shape (self.num_sub, len(sub_results), 3)
        self.matchings = []
        for i in range(self.num_sub):
            diffs_sub = self.diffs[i]
            matchings_sub = self.match_results[i]
            sort_diff_ind = np.argsort(diffs_sub[:, 0])
            best_diff_match = matchings_sub[sort_diff_ind[-1]][1]
            self.matchings.append(best_diff_match)
            print("=========================================")
            print("Sub Index: SI%d (Difference method)"%(i+1))
            for m in best_diff_match:
                print(f"Best match for Sub Index: {m[0]}, {m[1]} is {m[2]} with MMAD score {m[3]:.6f}")


    def best_match_median(self):
        # Median method: Compiles MMAD scores across all parameter sets and selects the structure
        # with the lowest median MMAD score for each Sub-index/LV combination
        self.median_winners = {}
        self.median_scores = {}
        
        for i in range(self.num_sub):
            sub_name = f"SI{i+1}"
            sub_results = self.match_results[i]
            
            # Dictionary to accumulate SAD scores
            # Format: {LV: {structure: [scores]}}
            lv_struct_scores = defaultdict(lambda: defaultdict(list))
            
            for run in sub_results:
                df_sum = run[2]
                for _, row in df_sum.iterrows():
                    lv_struct_scores[row['LV']][row['Structure']].append(row['SAD Score'])
            
            self.median_winners[sub_name] = {}
            self.median_scores[sub_name] = {}
            print("=========================================")
            print(f"Sub Index: {sub_name} (Median method)")
            
            for lv in range(self.num_comp):
                lv_name = f"LV{lv+1}"
                self.median_scores[sub_name][lv_name] = {}
                best_structure = None
                lowest_median = float('inf')
                
                for struct, scores in lv_struct_scores[lv_name].items():
                    median_val = np.median(scores)
                    self.median_scores[sub_name][lv_name][struct] = median_val
                    if median_val < lowest_median:
                        lowest_median = median_val
                        best_structure = struct
                        
                self.median_winners[sub_name][lv_name] = best_structure
                print(f"Best match for {sub_name}, {lv_name} is {best_structure} with median MMAD score {lowest_median:.6f}")


    def best_match_frequency(self):
        # Frequency method: Selects the structure that is most frequently ranked 1st across all parameter sets
        self.frequency_winners = {}
        self.frequency_counts = {}
        
        for i in range(self.num_sub):
            sub_name = f"SI{i+1}"
            sub_results = self.match_results[i]
            
            # Dictionary to count rank 1 appearances
            # Format: {LV: {structure: count}}
            lv_struct_counts = defaultdict(lambda: defaultdict(int))
            
            for run in sub_results:
                df_sum = run[2]
                rank1_df = df_sum[df_sum['Rank (reciprocal distance)'] == 1]
                for _, row in rank1_df.iterrows():
                    lv_struct_counts[row['LV']][row['Structure']] += 1
            
            self.frequency_winners[sub_name] = {}
            self.frequency_counts[sub_name] = {}
            print("=========================================")
            print(f"Sub Index: {sub_name} (Frequency method)")
            
            for lv in range(self.num_comp):
                lv_name = f"LV{lv+1}"
                self.frequency_counts[sub_name][lv_name] = dict(lv_struct_counts[lv_name])
                
                if lv_struct_counts[lv_name]:
                    best_structure = max(lv_struct_counts[lv_name], key=lv_struct_counts[lv_name].get)
                    count_val = lv_struct_counts[lv_name][best_structure]
                else:
                    best_structure = "None"
                    count_val = 0
                    
                self.frequency_winners[sub_name][lv_name] = best_structure
                total_runs = len(sub_results)
                print(f"Best match for {sub_name}, {lv_name} is {best_structure} (selected {count_val}/{total_runs} times)")


    def compare_all_methods(self):
        # Compares matching results of the three methods: Diff, Frequency, and Median
        for i in range(self.num_sub):
            sub_name = f"SI{i+1}"
            print("=========================================")
            print(f"Sub Index: {sub_name} Comparison of Methods")
            print("LV\t\tDiff\t\t\tFrequency\t\t\tMedian")
            
            diff_match = self.matchings[i]
            for lv in range(self.num_comp):
                lv_name = f"LV{lv+1}"
                
                d_winner = "N/A"
                for row in diff_match:
                    if row[1] == lv_name and row[2] == 1:
                        d_winner = row[3]
                        break
                        
                f_winner = self.frequency_winners.get(sub_name, {}).get(lv_name, "N/A")
                m_winner = self.median_winners.get(sub_name, {}).get(lv_name, "N/A")
                
                print(f"{lv_name}\t\t{d_winner[:15]:<15}\t\t{f_winner[:15]:<15}\t\t{m_winner[:15]:<15}")


    def neighbor_phase_initialization(self, pkl_adr):
        # Loads a summary file from RPA object (which contains the locations of NMF components)
        import pickle
        with open(pkl_adr, 'rb') as f:
            self.df_summary = pickle.load(f)
        
        self.df_summary = self.df_summary.drop(self.df_summary[self.df_summary['LV'] == 'LV0'].index)
        self.df_summary['LV'] = self.df_summary['LV'].str.replace('LV', '').astype(int)
        self.df_summary['Sub Index'] = self.df_summary['Sub Index'].str.replace('sub_index_', '').astype(int)
        self.df_summary['Data Index'] = self.df_summary['Data Index'].astype(int)
        print(self.df_summary)


    def area_analysis(self):
        grouped = self.df_summary.groupby(['Sub Index', 'LV'])
        df_area_sum = grouped['Area Ratio (%)'].mean().unstack(fill_value=0.0)
        
        # Calculate standard deviation
        df_area_std = grouped['Area Ratio (%)'].std().unstack(fill_value=0.0)
        
        print("Average Area Ratio (%)")
        print(df_area_sum)
        print("Standard Deviation of Area Ratio (%)")
        print(df_area_std)
        
        # Heatmap visualization
        self._plot_heatmap(df_area_sum, "Area Ratio (%)", 'LV', 'Sub Index', 'viridis', title="Average Area Ratio (%)", is_percentage=True)
        self._plot_heatmap(df_area_std, "Std Dev (%)", 'LV', 'Sub Index', 'plasma', title="Standard Deviation of Area Ratio (%)", is_percentage=True)


    def closest_neighbor_analysis(self, prox_neighbors=3, plot_result=False):
        self.df_summary['ID'] = range(len(self.df_summary))
        global_lv_neighborhoods = defaultdict(list)
        sub_lv_neighborhoods = defaultdict(lambda: defaultdict(list))
        neighbor_data = []
        grouped = self.df_summary.groupby(['Sub Index', 'Data Index'])

        for name, group in grouped:
            if len(group) <= 1:
                continue

            sub_index = name[0]
            coordinates = group[['Centroid X', 'Centroid Y']].values
            distance_matrix = squareform(pdist(coordinates, 'euclidean'))
            
            for i in range(len(group)):
                distances = distance_matrix[i]
                sorted_indices = np.argsort(distances)
                
                num_neighbors = min(5, len(group) - 1)
                closest_indices = sorted_indices[1:1+num_neighbors]
                
                source_centroid = group.iloc[i]
                source_lv = source_centroid['LV']

                for rank, neighbor_index in enumerate(closest_indices, 1):
                    neighbor = group.iloc[neighbor_index]
                    distance = distances[neighbor_index]
                    neighbor_data.append({
                        'Source_ID': source_centroid['ID'], 'Source_LV': source_centroid['LV'],
                        'Source_X': source_centroid['Centroid X'], 'Source_Y': source_centroid['Centroid Y'],
                        'Neighbor_Rank': rank, 'Neighbor_ID': neighbor['ID'], 'Neighbor_LV': neighbor['LV'],
                        'Neighbor_X': neighbor['Centroid X'], 'Neighbor_Y': neighbor['Centroid Y'],
                        'Distance': distance
                    })
                
                for neighbor_index in closest_indices[:prox_neighbors]:
                    neighbor_lv = group.iloc[neighbor_index]['LV']
                    global_lv_neighborhoods[source_lv].append(neighbor_lv)
                    sub_lv_neighborhoods[sub_index][source_lv].append(neighbor_lv)

        all_lvs = sorted(list(set(self.df_summary['LV'])))
        
        def create_matrices(neighborhood_dict):
            h_df = pd.DataFrame(0.0, index=all_lvs, columns=all_lvs)
            p_df = pd.DataFrame(0.0, index=all_lvs, columns=all_lvs)
            for s_lv, neighbors in neighborhood_dict.items():
                total_neighbors = len(neighbors)
                if total_neighbors > 0:
                    counts = pd.Series(neighbors).value_counts()
                    for n_lv, count in counts.items():
                        h_df.loc[s_lv, n_lv] = int(count)
                        p_df.loc[s_lv, n_lv] = (count / total_neighbors * 100)
            return h_df, p_df

        global_hist_df, global_prox_df = create_matrices(global_lv_neighborhoods)
        
        sub_index_histograms = {}
        sub_index_proximities = {}
        for sub_idx, neighborhoods in sub_lv_neighborhoods.items():
            h_df, p_df = create_matrices(neighborhoods)
            sub_index_histograms[sub_idx] = h_df
            sub_index_proximities[sub_idx] = p_df

        if plot_result:
            print("--- Global Nearest Neighboring Phase ---")
            print("Source LV\tNeighbor LV")
            for i, row in global_prox_df.iterrows():
                values = row.tolist()
                if sum(values) > 0:
                    print("%d\t\t%d" % (i, all_lvs[np.argmax(values)]))

            self._plot_heatmap(global_hist_df, "Count", 'Closest Neighbor LV', 'Source LV', 'seismic', title="Global Neighborhood: Count")
            self._plot_heatmap(global_prox_df, "Percentage (%)", 'Neighbor LV', 'Source LV', 'seismic', title="Global Neighborhood: Percentage", is_percentage=True)

            if len(sub_index_histograms) > 1:
                for sub_idx in sub_index_histograms:
                    print(f"--- Nearest Neighboring Phase for Sub Index [{sub_idx}] ---")
                    print("Source LV\tNeighbor LV")
                    for i, row in sub_index_histograms[sub_idx].iterrows():
                        values = row.tolist()
                        if sum(values) > 0:
                            print("%d\t\t%d" % (i, all_lvs[np.argmax(values)]))
                    
                    self._plot_heatmap(
                        sub_index_histograms[sub_idx], "Count", 'Closest Neighbor LV', 'Source LV', 'seismic', 
                        title=f"Sub Index [{sub_idx}]: Count"
                    )
                    self._plot_heatmap(
                        sub_index_proximities[sub_idx], "Percentage (%)", 'Neighbor LV', 'Source LV', 'seismic', 
                        title=f"Sub Index [{sub_idx}]: Percentage", is_percentage=True
                    )

        return global_hist_df, global_prox_df, sub_index_histograms, sub_index_proximities


    def _plot_heatmap(self, df, cbar_label, xlabel, ylabel, cmap, title="", is_percentage=False):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        im = ax.imshow(df, cmap=cmap)

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, pad=15)

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                color = "black" if 0.2 < im.norm(val) < 0.8 else "white"
                text_format = f"{val:.1f}" if is_percentage else f"{int(val)}"
                ax.text(j, i, text_format, ha="center", va="center", color=color)
        
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        fig.tight_layout()
        plt.show()


    def summary_report(self):
        report = []
        report.append("# Phase Analysis Summary Report\n")
        report.append("## 1. Configuration")
        report.append(f"- **Target Structures (CIF Library)**: {self.target_structures}")
        report.append(f"- **Profile Shape**: {self.profile.shape}")
        report.append(f"- **Crop Window**: `{self.crop}` (indices: `{self.crop[0]}~{self.crop[1]}`)")
        report.append(f"- **Number of Loading Vectors**: {self.num_comp}")
        report.append(f"- **Number of Sub-indices (Samples)**: {self.num_sub}")
        report.append("")
        
        if hasattr(self, 'match_results') and self.match_results:
            report.append("## 2. Match Scores & Best Allocations")
            
            has_diff = hasattr(self, 'matchings') and self.matchings
            has_freq = hasattr(self, 'frequency_winners') and self.frequency_winners
            has_med = hasattr(self, 'median_winners') and self.median_winners
            
            if has_diff and has_freq and has_med:
                report.append("### Comprehensive Match Summary (Diff vs. Frequency vs. Median)")
                headers = ["Sub Index", "Loading Vector", "Diff Winner", "Frequency Winner", "Median Winner"]
                rows = []
                for i in range(self.num_sub):
                    sub_name = f"SI{i+1}"
                    diffs_sub = self.diffs[i]
                    matchings_sub = self.matchings[i]
                    sort_diff_ind = np.argsort(diffs_sub[:, 0])
                    best_diff_match_result = matchings_sub[sort_diff_ind[-1]] 
                    
                    for lv in range(self.num_comp):
                        lv_name = f"LV{lv+1}"
                        diff_winner = "N/A"
                        for row in best_diff_match_result:
                            if row[1] == lv_name and row[2] == 1:
                                diff_winner = row[3]
                                break
                        freq_winner = self.frequency_winners.get(sub_name, {}).get(lv_name, "N/A")
                        median_winner = self.median_winners.get(sub_name, {}).get(lv_name, "N/A")
                        rows.append([sub_name, lv_name, diff_winner, freq_winner, median_winner])
                
                table_lines = []
                table_lines.append("| " + " | ".join(headers) + " |")
                table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    table_lines.append("| " + " | ".join(row) + " |")
                report.append("\n".join(table_lines))
                report.append("")
                
                report.append("## 3. Raw Calculation Data by Matching Criterion")
                report.append("### 3.1 Difference (Diff) Criterion Raw MMAD Scores")
                report.append("Below are the full MMAD ranking scores for the best matching parameter set (based on the largest mean of differences between matching scores):")
                diff_headers = ["Sub Index", "LV", "Rank", "Structure", "MMAD Score"]
                diff_rows = []
                for i in range(self.num_sub):
                    diffs_sub = self.diffs[i]
                    matchings_sub = self.matchings[i]
                    sort_diff_ind = np.argsort(diffs_sub[:, 0])
                    best_diff_match_result = matchings_sub[sort_diff_ind[-1]]
                    for row in best_diff_match_result:
                        diff_rows.append([str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])])
                
                table_lines = []
                table_lines.append("| " + " | ".join(diff_headers) + " |")
                table_lines.append("| " + " | ".join(["---"] * len(diff_headers)) + " |")
                for row in diff_rows:
                    table_lines.append("| " + " | ".join(row) + " |")
                report.append("\n".join(table_lines))
                report.append("")

                report.append("### 3.1.1 Parameter Set Selection (Mean Differences)")
                report.append("Below are the evaluated parameter sets and their corresponding mean and standard deviation of differences in MMAD scores:")
                sel_headers = ["Run Index", "Profile Step", "XRD Step", "Broadening", "XRD Prominence", "Mean Diff", "Std Diff", "Selected?"]
                for i in range(self.num_sub):
                    report.append(f"#### Sub Index SI{i+1}:")
                    sel_rows = []
                    diffs_sub = self.diffs[i]
                    params_sub = self.params[i]
                    sort_diff_ind = np.argsort(diffs_sub[:, 0])
                    best_idx = sort_diff_ind[-1]
                    
                    for r_idx in range(len(diffs_sub)):
                        p = params_sub[r_idx]
                        mean_d = diffs_sub[r_idx][0]
                        std_d = diffs_sub[r_idx][1]
                        is_best = "Yes" if r_idx == best_idx else "No"
                        sel_rows.append([
                            str(r_idx + 1),
                            f"{p[0]:.6f}",
                            f"{p[1]:.6f}",
                            f"{p[2]:.6f}",
                            f"{p[3]:.6f}",
                            f"{mean_d:.6f}" if mean_d != float('inf') else "inf",
                            f"{std_d:.6f}" if std_d != float('inf') else "inf",
                            is_best
                        ])
                    
                    table_lines = []
                    table_lines.append("| " + " | ".join(sel_headers) + " |")
                    table_lines.append("| " + " | ".join(["---"] * len(sel_headers)) + " |")
                    for row in sel_rows:
                        table_lines.append("| " + " | ".join(row) + " |")
                    report.append("\n".join(table_lines))
                    report.append("")

                report.append("### 3.1.2 Complete Raw MMAD Scores (All Parameter Sets)")
                report.append("<details>")
                report.append("<summary>Click to expand/collapse the complete raw MMAD scores for all evaluated parameter combinations</summary>\n")
                raw_headers = ["Sub Index", "LV", "Parameters (p_step / x_step / broad / prom)", "Structure", "MMAD Score", "Rank"]
                raw_rows = []
                for i in range(self.num_sub):
                    for r_idx, run_data in enumerate(self.match_results[i]):
                        p = run_data[0]
                        p_str = f"{p[0]:.5f} / {p[1]:.5f} / {p[2]:.4f} / {p[3]:.4f}"
                        df_sum = run_data[2]
                        for idx, row in df_sum.iterrows():
                            raw_rows.append([
                                str(row["Sub Index"]),
                                str(row["LV"]),
                                p_str,
                                str(row["Structure"]),
                                str(row["SAD Score"]),
                                str(row["Rank (reciprocal distance)"])
                            ])
                table_lines = []
                table_lines.append("| " + " | ".join(raw_headers) + " |")
                table_lines.append("| " + " | ".join(["---"] * len(raw_headers)) + " |")
                for row in raw_rows:
                    table_lines.append("| " + " | ".join(row) + " |")
                report.append("\n".join(table_lines))
                report.append("\n</details>\n")
                
                if hasattr(self, 'frequency_counts'):
                    report.append("### 3.2 Frequency Criterion Raw Counts")
                    report.append("Below are the number of times each structure was ranked 1st across all evaluated parameter sets:")
                    freq_headers = ["Sub Index", "LV", "Structure", "First-Rank Count"]
                    freq_rows = []
                    for sub_name, lvs in self.frequency_counts.items():
                        for lv_name, counts in lvs.items():
                            for struct_name, count_val in counts.items():
                                freq_rows.append([sub_name, lv_name, struct_name, str(count_val)])
                                
                    table_lines = []
                    table_lines.append("| " + " | ".join(freq_headers) + " |")
                    table_lines.append("| " + " | ".join(["---"] * len(freq_headers)) + " |")
                    for row in freq_rows:
                        table_lines.append("| " + " | ".join(row) + " |")
                    report.append("\n".join(table_lines))
                    report.append("")
                    
                if hasattr(self, 'median_scores'):
                    report.append("### 3.3 Median Criterion Raw Scores")
                    report.append("Below are the median MMAD scores for each target structure computed across all parameter sets:")
                    med_headers = ["Sub Index", "LV", "Structure", "Median MMAD Score"]
                    med_rows = []
                    for sub_name, lvs in self.median_scores.items():
                        for lv_name, scores in lvs.items():
                            for struct_name, score_val in scores.items():
                                med_rows.append([sub_name, lv_name, struct_name, f"{score_val:.6f}"])
                                
                    table_lines = []
                    table_lines.append("| " + " | ".join(med_headers) + " |")
                    table_lines.append("| " + " | ".join(["---"] * len(med_headers)) + " |")
                    for row in med_rows:
                        table_lines.append("| " + " | ".join(row) + " |")
                    report.append("\n".join(table_lines))
                    report.append("")
            else:
                report.append("### MMAD Best Matches (Rank 1)")
                headers = ["Sub Index", "LV", "Best Matched Structure", "MMAD Score"]
                rows = []
                for i in range(self.num_sub):
                    if len(self.match_results[i]) > 0:
                        for run_data in self.match_results[i]:
                            good_matches = run_data[1]
                            for match in good_matches:
                                rows.append([match[0], match[1], match[2], match[3]])
                            break
                
                table_lines = []
                table_lines.append("| " + " | ".join(headers) + " |")
                table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    table_lines.append("| " + " | ".join(row) + " |")
                report.append("\n".join(table_lines))
                report.append("")
        else:
            report.append("## 2. Match Scores & Best Allocations")
            report.append("*MMAD phase matching has not been executed yet.*")
            report.append("")
        return "\n".join(report)

    def save_state(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Object state successfully saved to {filepath}")
