import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from collections import defaultdict, Counter
from scipy.spatial.distance import pdist, squareform
import py4DSTEM

# Import shared helpers
from .utils import (
    profile_peak, simul_xrd_peak, mmad_score
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

        self.experimental_cols = [f'LV{i}' for i in range(1, self.num_comp+1)]

    def profile_comparison(self, profile_k_step, max_num_peaks=3, ylim=None, line_color=None, show_legend=True):
        k_range = []
        for i in range(self.crop[0], self.crop[1], 1):
            k_range.append(profile_k_step*i)

        for lv in range(self.num_comp):
            fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=100)
            for i in range(self.num_sub):
                line = self.profile[lv+i*self.num_comp]
                profile_peaks = profile_peak(line=line, 
                                        k_step=profile_k_step, 
                                        crop=self.crop, 
                                        peak_prominence=1E-9, 
                                        max_num_peaks=max_num_peaks)
                if line_color != None:
                    l_color = line_color
                else:
                    l_color = self.color_rep[i+1]
                tmp_ax, = ax.plot(k_range, line[self.crop[0]:self.crop[1]], c=l_color, label="Sub Index %d"%(i+1), linewidth=2)
                shadow_effect = path_effects.withStroke(linewidth=3, foreground='gray')
                tmp_ax.set_path_effects([shadow_effect])
                
                if max_num_peaks != 0:
                    for peak in profile_peaks:
                        ax.axvline(peak, ls=':', lw=1.5, c=l_color, alpha=0.5)
                        if ylim != None: 
                            ax_txt = ax.text(peak, ylim[1]*0.7, "%.3f"%(peak), c=l_color, fontsize=7)
                        else:
                            ax_txt = ax.text(peak, np.max(line[self.crop[0]:self.crop[1]])*0.7, "%.3f"%(peak), c=l_color, fontsize=7)
                        shadow_effect_txt = path_effects.withStroke(linewidth=1, foreground='gray')
                        ax_txt.set_path_effects([shadow_effect_txt])

                if ylim != None:
                    ax.set_ylim(ylim[0], ylim[1])
            
            if show_legend:
                ax.legend()
            fig.suptitle("LV%d"%(lv+1))
            fig.tight_layout()
            plt.show()


    def simple_comparsion(self, cif_adr, profile_k_step, max_num_peaks, xrd_k_step, broadening, xrd_peak_prominence, ylim=None):
        k_range = []
        for i in range(self.crop[0], self.crop[1], 1):
            k_range.append(profile_k_step*i)

        k_xrd = np.arange(k_range[0], k_range[-1]+xrd_k_step, xrd_k_step) 

        basename, simul_xrd, xrd_peaks = simul_xrd_peak(cif_adr, k_xrd, broadening, xrd_peak_prominence)

        k = 0
        for i in range(self.num_sub):
            for lv in range(self.num_comp):
                line = self.profile[lv+k]       
                profile_peaks = profile_peak(line=line, 
                                        k_step=profile_k_step, 
                                        crop=self.crop, 
                                        peak_prominence=1E-9, 
                                        max_num_peaks=max_num_peaks)
        
                line = line[self.crop[0]:self.crop[1]].copy()
                line = line / np.max(line)
                fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=100)
                ax.plot(k_xrd, simul_xrd, 'k:', label=basename, linewidth=2)
                tmp_ax, = ax.plot(k_range, line, c=self.color_rep[lv+1], label="sub index %d LV%d"%(i+1, lv+1), linewidth=2)
                shadow_effect = path_effects.withStroke(linewidth=3, foreground='gray')
                tmp_ax.set_path_effects([shadow_effect])

                for peak in profile_peaks:
                    ax.axvline(peak, ls=':', lw=1.5, c=self.color_rep[lv+1], alpha=0.5)
                    if ylim != None: 
                        ax_txt = ax.text(peak, ylim[1]*0.7, "%.3f"%(peak), c=self.color_rep[lv+1], fontsize=7)
                    else:
                        ax_txt = ax.text(peak, np.max(line[self.crop[0]:self.crop[1]])*0.7, "%.3f"%(peak), c=self.color_rep[lv+1], fontsize=7)
                    shadow_effect_txt = path_effects.withStroke(linewidth=1, foreground='gray')
                    ax_txt.set_path_effects([shadow_effect_txt])

                for peak in xrd_peaks:
                    ax.axvline(peak, ls=':', lw=1.5, c='black', alpha=0.5)
                    if ylim != None: 
                        ax_txt = ax.text(peak, ylim[1]*0.9, "%.3f"%(peak), c="black", fontsize=7)
                    else:
                        ax_txt = ax.text(peak, 0.9, "%.3f"%(peak), c="black", fontsize=7)
                    shadow_effect_txt = path_effects.withStroke(linewidth=1, foreground='gray')
                    ax_txt.set_path_effects([shadow_effect_txt])                   

                if ylim != None:
                    ax.set_ylim(ylim[0], ylim[1])
                
                ax.legend()
                fig.tight_layout()
                plt.show()

            k+=self.num_comp        

    def MMAD_matching(self, profile_step_list, 
                      profile_prominence,
                      max_num_peaks, 
                      xrd_step_list, 
                      xrd_broadening_list, 
                      xrd_prominence_list,
                      num_good_match=1):

        if len(profile_step_list) != len(xrd_step_list):
            print("The step sizes of the profile and xrd data must be aligned")
            return

        
        results = []
        k = 0
        for i in range(self.num_sub):
            result_sub = []
            for scale, scale_xrd in zip(profile_step_list, xrd_step_list):
                k_range = []
                for ci in range(self.crop[0], self.crop[1], 1):
                    k_range.append(scale*ci)

                k_xrd = np.arange(k_range[0], k_range[-1]+scale_xrd, scale_xrd)
                for broadening in xrd_broadening_list:
                    for prominence_xrd in xrd_prominence_list:
                        # Find peak positions in the profile data
                        peak_positions = []
                        for lv in range(self.num_comp):
                            line = self.profile[lv+k]
                            peak_positions.append(profile_peak(line=line, 
                                                                k_step=scale, 
                                                                crop=self.crop, 
                                                                peak_prominence=profile_prominence, 
                                                                max_num_peaks=max_num_peaks))

                        # Simulate XRD data and find peak positions
                        basenames = []
                        xrd_peak_list = []
                        simul_xrds = []
                        for adr in self.str_path:
                            basename, simul_xrd, xrd_peaks = simul_xrd_peak(adr=adr, 
                                                                            k_range=k_xrd, 
                                                                            broadening=broadening, 
                                                                            peak_prominence=prominence_xrd)
                            simul_xrds.append(simul_xrd)
                            basenames.append(basename)
                            xrd_peak_list.append(xrd_peaks)

                        # Sort out the peak position data
                        table_data = {}
                        for lv in range(self.num_comp):
                            table_data["LV%d"%(lv+1)] = pd.Series(peak_positions[lv])
                        for bi, name in enumerate(basenames):
                            table_data[name] = pd.Series(xrd_peak_list[bi])
                        df_data = pd.DataFrame(table_data)

                        # Calculate the mean of minimum absolute differences (MMAD)
                        results_mmad_subset = mmad_score(df_data=df_data, 
                                                        experimental_cols=self.experimental_cols, 
                                                        target_structures=self.target_structures)

                        # Leave only good match results
                        table_data = []
                        good_match = []
                        # print(scale, scale_xrd, broadening)
                        for lv_name, ranked_matches in results_mmad_subset.items():
                            prev_mmad_score = 0
                            prev_rank = 0
                            three_kind = 1
                            for rm, match in enumerate(ranked_matches):
                                same_rank_flag = False
                                mmad_score_str = f"{match['mmad_score']:.6f}" if match['mmad_score'] != float('inf') else "inf"

                                if rm != 0:
                                    if mmad_score_str != "inf" and prev_mmad_score != "inf" and float(mmad_score_str) == float(prev_mmad_score):
                                        same_rank_flag = True

                                table_data.append({
                                    "Sub Index": "SI%d"%(i+1),
                                    "LV": lv_name,
                                    "Rank (reciprocal distance)": prev_rank if same_rank_flag else rm+1,
                                    "Structure": match['simulation'],
                                    "SAD Score": mmad_score_str,
                                })

                                prev_mmad_score = mmad_score_str

                                if same_rank_flag:
                                    prev_rank = prev_rank
                                else:
                                    prev_rank = rm+1

                                if rm < num_good_match:
                                    # print(lv_name, match['simulation'], mmad_score_str)
                                    good_match.append(["SI%d"%(i+1), lv_name, match['simulation'], mmad_score_str])

                        df_summary = pd.DataFrame(table_data)
                        parameters = [scale, scale_xrd, broadening, prominence_xrd]
                        result_sub.append([parameters, good_match, df_summary])
            
            results.append(result_sub)
            k += self.num_comp
        
        self.match_results = results

    def best_match_diff(self):

        self.diffs = []
        self.params = []
        self.matchings = []

        for i in range(self.num_sub):
            print("*"*50)
            print("Matching Result for Sub Index %d"%(i+1))
            print("*"*50)
            diffs_sub = []
            params_sub = []
            matchings_sub = []
            for r, result in enumerate(self.match_results[i]):
                diff = []
                for j in range(1, len(result)):
                    if result[1][j][1] == result[1][j-1][1]:
                        diff.append(np.abs(eval(result[1][j][3]) - eval(result[1][j-1][3])))

                diffs_sub.append([np.mean(diff), np.std(diff)])
                params_sub.append(result[0])
                matchings_sub.append(result[2].values)

            diffs_sub = np.asarray(diffs_sub)
            params_sub = np.asarray(params_sub)
            matchings_sub = np.asarray(matchings_sub)
            sort_diff_ind = np.argsort(diffs_sub[:, 0])
            
            print("Three largest means of differences")
            print(diffs_sub[sort_diff_ind[-1]], diffs_sub[sort_diff_ind[-2]], diffs_sub[sort_diff_ind[-3]])

            best_diff, best_params, best_match_result = diffs_sub[sort_diff_ind[-1]], params_sub[sort_diff_ind[-1]], matchings_sub[sort_diff_ind[-1]]

            print("Best result in terms of MMAD")
            print("Mean of MMAD differences between phases")
            print(best_diff)
            print("\n")
            print("Parameters (profile step size, xrd step size, broadening, xrd peak prominence)")
            print(best_params)
            print("\n")
            print("Full MMAD scores")
            print(best_match_result)
            print("\n")
            print("Best Match")
            for i in range(0, len(best_match_result), len(self.target_structures)):
                print(best_match_result[i])

            print("*"*50)
            print("*"*50)

            self.diffs.append(diffs_sub)
            self.params.append(params_sub)
            self.matchings.append(matchings_sub)

    def best_match_median(self):
            print("*"*50)
            print("Median MMAD Score Analysis")
            print("*"*50)

            # Dictionary to store the #1 ranked structures for future use
            self.median_winners = {} 

            for i in range(self.num_sub):
                sub_name = f"SI{i+1}"
                self.median_winners[sub_name] = {}
                print(f"\nEvaluating Sub Index {i+1}...")

                all_summaries = [run[2] for run in self.match_results[i]]
                if not all_summaries:
                    continue

                combined_df = pd.concat(all_summaries, ignore_index=True)
                combined_df['SAD Score'] = pd.to_numeric(combined_df['SAD Score'].replace('inf', np.nan))
                clean_df = combined_df.dropna(subset=['SAD Score'])

                rank_dict = {}

                for lv in range(self.num_comp):
                    lv_name = f"LV{lv+1}"
                    lv_data = clean_df[clean_df['LV'] == lv_name]

                    if lv_data.empty:
                        continue

                    median_scores = lv_data.groupby('Structure')['SAD Score'].median()
                    ranked_medians = median_scores.sort_values(ascending=True)

                    # Store the absolute best (Rank 1) match for later comparison
                    self.median_winners[sub_name][lv_name] = ranked_medians.index[0]
                    if not hasattr(self, 'median_scores'):
                        self.median_scores = {}
                    if sub_name not in self.median_scores:
                        self.median_scores[sub_name] = {}
                    self.median_scores[sub_name][lv_name] = ranked_medians.to_dict()

                    current_rank = 1
                    for structure, score in ranked_medians.items():
                        if current_rank not in rank_dict:
                            rank_dict[current_rank] = []
                        rank_dict[current_rank].append((lv_name, structure, score))
                        current_rank += 1

                max_rank = max(rank_dict.keys()) if rank_dict else 0
                for r in range(1, max_rank + 1):
                    if r in rank_dict:
                        print(f"\n--- Rank {r} ---")
                        for lv_name, structure, score in rank_dict[r]:
                            print(f"{lv_name}: {structure} (Median Score: {score:.6f})")

    def best_match_frequency(self):
            
            # Dictionary to store the #1 ranked structures for future use
            self.frequency_winners = {} 

            for i in range(self.num_sub):
                sub_name = f"SI{i+1}"
                self.frequency_winners[sub_name] = {}
                
                ind = np.where(self.matchings[i] == 1)
                print("*"*50)
                print("Matching Result for Sub Index %d"%(i+1))
                print("Best result in terms of frequency")
                print("*"*50)
                
                for lv in range(self.num_comp):
                    lv_name = f"LV{lv+1}"
                    temp_sort = self.matchings[i][ind[0], ind[1]]
                    ind_lv = np.where(temp_sort == lv_name)
                    
                    count = Counter(temp_sort[ind_lv[0]][:, 3])
                    
                    if count:
                        # Store the absolute best match 
                        best_structure = count.most_common(1)[0][0]
                        self.frequency_winners[sub_name][lv_name] = best_structure
                        if not hasattr(self, 'frequency_counts'):
                            self.frequency_counts = {}
                        if sub_name not in self.frequency_counts:
                            self.frequency_counts[sub_name] = {}
                        self.frequency_counts[sub_name][lv_name] = dict(count)
                    
                    print(f"Sub Index {i+1} {lv_name}", count.most_common(3))
                print("*"*50)
                print("*"*50)


    def compare_all_methods(self):
        print("\n" + "*"*70)
        print("Comprehensive Match Summary (Diff vs. Frequency vs. Median)")
        print("*"*70)

        if not hasattr(self, 'matchings') or not self.matchings:
            print("Error: Please run best_match_diff() first.")
            return
            
        if not hasattr(self, 'frequency_winners') or not hasattr(self, 'median_winners'):
            print("Error: Please run both best_match_frequency() and best_match_median() first.")
            return

        summary_data = []

        for i in range(self.num_sub):
            sub_name = f"SI{i+1}"
            
            # Reconstruct the best match result array from the diff method's stored variables
            diffs_sub = self.diffs[i]
            matchings_sub = self.matchings[i]
            sort_diff_ind = np.argsort(diffs_sub[:, 0])
            best_diff_match_result = matchings_sub[sort_diff_ind[-1]] 

            for lv in range(self.num_comp):
                lv_name = f"LV{lv+1}"
                
                # 1. Extract Diff Winner
                diff_winner = "N/A"
                for row in best_diff_match_result:
                    if row[1] == lv_name and row[2] == 1:
                        diff_winner = row[3]
                        break

                # 2. Extract Frequency Winner from the newly stored dictionary
                freq_winner = self.frequency_winners.get(sub_name, {}).get(lv_name, "N/A")

                # 3. Extract Median Winner from the newly stored dictionary
                median_winner = self.median_winners.get(sub_name, {}).get(lv_name, "N/A")

                summary_data.append({
                    "Sub Index": sub_name,
                    "LV": lv_name,
                    "Diff Winner": diff_winner,
                    "Frequency Winner": freq_winner,
                    "Median Winner": median_winner
                })

        # Display the final comparative table
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.set_index(['Sub Index', 'LV'], inplace=True)
            print("\n")
            print(summary_df.to_string())
            print("\n" + "*"*70)
        else:
            print("No valid data found to summarize.")


    def neighbor_phase_initialization(self, pkl_adr):
        print(pkl_adr)

        with open(pkl_adr, "rb") as f:
            run_object = pickle.load(f)
            
        print(run_object.keys())

        self.pos_lv_pixel_split = run_object["pos_lv_pixel_split"]

        small_area_split = []
        for i in range(len(self.pos_lv_pixel_split)):
            sub_data = self.pos_lv_pixel_split[i]
            small_area_sub = []
            for j in range(len(sub_data)):
                indv_data = sub_data[j]
                small_area_indv = []
                for lv in range(len(indv_data)):
                    lv_data = indv_data['sub_index_%d_LV%d'%(i+1, lv+1)]
                    small_area_lv = []
                    # print(len(lv_data))
                    for cluster in lv_data:
                        small_area_lv.append(len(cluster))
                    small_area_indv.append(small_area_lv)
                small_area_sub.append(small_area_indv)
            small_area_split.append(small_area_sub)

        self.small_area_split = small_area_split

        self.centroid_lv_split = run_object["centroid_lv_split"]

        table_data = []
        for i in range(len(self.centroid_lv_split)):
            sub_centroid = self.centroid_lv_split[i]
            for j in range(len(sub_centroid)):
                indv_centroid = sub_centroid[j]
                for lv in range(len(indv_centroid)):
                    lv_centroid = indv_centroid[lv]
                    for k, ct in enumerate(lv_centroid):
                        try:
                            table_data.append({
                            "Sub Index": i+1,
                            "Data Index": j+1,
                            "LV": lv+1,
                            "Centroid Y": ct[0],
                            "Centroid X": ct[1],
                            "Area": small_area_split[i][j][lv][k]    
                            })
                        except:
                            pass

        self.df_summary = pd.DataFrame(table_data)

    def area_analysis(self):
        label_list = np.unique(self.df_summary['LV'].values)
        label_list = np.sort(label_list)

        mean_areas = []
        std_areas = []
        total_areas = []
        sub_index_list = np.unique(self.df_summary['Sub Index'].values)
        sub_index_list = np.sort(sub_index_list)
        
        for sub_index in sub_index_list:
            mean_area_sub = []
            std_area_sub = []
            total_area_sub = []
            for label in label_list:            
                area = self.df_summary[self.df_summary['LV'] == label]
                area = area[area["Sub Index"] == sub_index]["Area"].values
                mean_area_sub.append(np.mean(area))
                std_area_sub.append(np.std(area))
                total_area_sub.append(np.sum(area))

            mean_areas.append(mean_area_sub)
            total_areas.append(total_area_sub)
            std_areas.append(std_area_sub)

        self.mean_areas = np.asarray(mean_areas)
        # print(self.mean_areas.shape)
        self.std_areas = np.asarray(std_areas)
        # print(self.std_areas.shape)
        self.total_areas = np.asarray(total_areas)
        # print(self.total_areas.shape)

        print("Cluster Size and Area Information")
        for s, sub_index in enumerate(sub_index_list):
            print("Subfolder Index: %d"%(sub_index))
            print("LV\tMean\tSTD\tTotal\tPercentage")
            for i, label in enumerate(label_list):
                print(f"{label}\t{int(mean_areas[s][i])}\t{std_areas[s][i]:.2f}\t{int(total_areas[s][i])}\t{total_areas[s][i]*100/np.sum(total_areas[s]):.1f}")

    def closest_neighbor_analysis(self, prox_neighbors=3, plot_result=False):
        self.df_summary['ID'] = range(len(self.df_summary))
        
        # Trackers for global and sub-index levels
        global_lv_neighborhoods = defaultdict(list)
        sub_lv_neighborhoods = defaultdict(lambda: defaultdict(list))
        
        neighbor_data = []

        grouped = self.df_summary.groupby(['Sub Index', 'Data Index'])

        for name, group in grouped:
            if len(group) <= 1:
                continue

            # Extract the Sub Index from the groupby tuple
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

                # --- Data collection for CSV output ---
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
                
                # --- Data collection for matrices ---
                for neighbor_index in closest_indices[:prox_neighbors]:
                    neighbor_lv = group.iloc[neighbor_index]['LV']
                    
                    # 1. Append to Global tracker
                    global_lv_neighborhoods[source_lv].append(neighbor_lv)
                    
                    # 2. Append to Sub Index tracker
                    sub_lv_neighborhoods[sub_index][source_lv].append(neighbor_lv)

        # --- Generate DataFrames for summaries ---
        all_lvs = sorted(list(set(self.df_summary['LV'])))
        
        # Helper function to build matrices
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

        # Build Global Matrices
        global_hist_df, global_prox_df = create_matrices(global_lv_neighborhoods)
        
        # Build Sub Index Matrices
        sub_index_histograms = {}
        sub_index_proximities = {}
        for sub_idx, neighborhoods in sub_lv_neighborhoods.items():
            h_df, p_df = create_matrices(neighborhoods)
            sub_index_histograms[sub_idx] = h_df
            sub_index_proximities[sub_idx] = p_df

        # --- Generate and display plots ---
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

    # Updated to accept a 'title' parameter
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
        
        # Add the title if one was provided
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
        """Generates a comprehensive Markdown report of the phase matching and structure allocation results."""
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
            
            # Check if winners dictionaries are available to build the comprehensive comparative table
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
                
                # --- ADDING RAW CALCULATION DATA ---
                report.append("## 3. Raw Calculation Data by Matching Criterion")
                
                # A. Raw Data for Diff matching (full ranks and scores for the best parameter set)
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

                # A.1. Parameter Set Selection (Mean Differences)
                report.append("### 3.1.1 Parameter Set Selection (Mean Differences)")
                report.append("Below are the evaluated parameter sets and their corresponding mean and standard deviation of differences in MMAD scores (used to determine the best parameter set for the Diff matching criterion):")
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

                # A.2. All Raw MMAD Scores (All Parameter Sets)
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
                
                # B. Raw Data for Frequency matching (counts of top-rank appearances across parameter sets)
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
                    
                # C. Raw Data for Median matching (median MMAD score across parameter sets)
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
                # If comparative winners aren't compiled yet, show best match from match_results
                report.append("### MMAD Best Matches (Rank 1)")
                headers = ["Sub Index", "LV", "Best Matched Structure", "MMAD Score"]
                rows = []
                for i in range(self.num_sub):
                    if len(self.match_results[i]) > 0:
                        # good_match is a list of ["SI%d"%(i+1), lv_name, structure, score]
                        for run_data in self.match_results[i]:
                            good_matches = run_data[1]
                            for match in good_matches:
                                rows.append([match[0], match[1], match[2], match[3]])
                            break # just show the first run's top matches
                
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
        """Pickles the object state to a file for downstream cross-analysis."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Object state successfully saved to {filepath}")
