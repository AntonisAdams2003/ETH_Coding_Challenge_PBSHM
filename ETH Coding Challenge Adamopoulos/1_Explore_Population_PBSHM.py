from data_loader import get_data
import numpy as np
import matplotlib.pyplot as plt
import os


## IMPORTING THE DATA  ##

# Call the data_loader.py to get the data
data = get_data()
measurements_data = data['measurements_data']
labels_array = data['labels_array']
population_weights_array = data['population_weights_array']
storeys = data['storeys'] 
heights = data['heights'] 
frequencies = data['frequencies']
Adj_weighted_matrix = data['Adj_weighted_matrix']
features = data['features']



## EXTRACTING IMPORTANT DATA ##

# Note: Use np.nan... in all vectrorized calculations containing the "heights" and "frequencies", since they containt NaN

# Total number of buildings 
number_of_buildings = len(measurements_data)  # 50 total
# Average storeys of a building
avg_storeys = np.mean(storeys)
#
# Total Height of each building
height_of_buildings = np.nansum(heights, axis=1)
# Average height of a building
avg_height = np.mean(height_of_buildings)
# Average height per storey 
avg_height_per_storey = np.mean(height_of_buildings / storeys)
#
# Average frequency of each building
avg_frequency_of_building = np.nanmean(frequencies, axis=1)



## VISUALIZE THE SIZE AND GEOMETRIC DATA ##

# The 1st figure contains 2 plots that show the distribution of storeys in the population,
# and the distribution of the total height of each building

fig1,axes1 = plt.subplots(2,3, figsize=(12,6), gridspec_kw={'height_ratios': [0.15, 0.85]}, sharex='col')
fig1.suptitle('Characteristics of the Population', fontsize=16, fontweight='bold')


# Get the unique storey heights and the frequency that they appear
unique_storeys, unique_storeys_count = np.unique(storeys, return_counts=True)
#
axes1[1,0].bar(unique_storeys, unique_storeys_count, color='darkred', edgecolor='black')
axes1[1,0].set_xlabel('Number of Storeys')
axes1[1,0].set_ylabel('Number of Buildings')
axes1[1,0].axvline(avg_storeys, color='k', linestyle='dashed', label=f'Average Storeys: {avg_storeys:.1f}')  # add line for the mean
axes1[1,0].legend()

# Adding box plot for clarity
axes1[0,0].boxplot(storeys, vert=False, widths=0.5, patch_artist=True, boxprops=dict(facecolor='steelblue'), medianprops=dict(color='red', linewidth=1.5),
                  showmeans=True, meanline=True, meanprops=dict(color='black', linestyle='dashed'))
axes1[0,0].set_title('Distribution of Structure Storeys')
axes1[0,0].axis('off')

# Total height of each building"
axes1[1,1].hist(height_of_buildings, color='darkred', edgecolor='black')
axes1[1,1].set_xlabel('Height [m]')
axes1[1,1].set_ylabel('Number of Buildings')
axes1[1,1].axvline(avg_height, color='k', linestyle='dashed', label=f'Average Height: {avg_height:.1f}[m]')  # add line for the mean
axes1[1,1].plot([], [], ' ', label=f'Average Height/Storey: {avg_height_per_storey:.1f}[m]')  # Add only the legend
axes1[1,1].legend()

# Adding box plot for clarity
axes1[0,1].boxplot(height_of_buildings, vert=False, widths=0.5, patch_artist=True, boxprops=dict(facecolor='steelblue'), medianprops=dict(color='red', linewidth=1.5),
                  showmeans=True, meanline=True, meanprops=dict(color='black', linestyle='dashed'))
axes1[0,1].set_title('Distribution of Total Heights')
axes1[0,1].axis('off')

# Number of healthy vs damaged buildings
y = labels_array[:,1].astype(int)   # binary health state of the buildings
health_state_names = ['Healthy','Damaged']
# Calculate the number fo healthy and damaged buildings
numb_of_healthy = np.sum(y==0)
numb_of_damaged = np.sum(y==1)

axes1[1,2].bar(health_state_names, [numb_of_healthy, numb_of_damaged] , color='darkred', edgecolor='black')
axes1[1,2].set_title("Healthy State Distribution")
axes1[1,2].set_xticks(range(len(health_state_names)))
axes1[1,2].set_xticklabels(health_state_names, rotation=55)

axes1[0,2].axis('off') # delete the 0,2 box

fig1.tight_layout()
fig1.subplots_adjust(hspace=0.05) # remove the gap



## VISUALIZE THE STARTER POPULATION RELATIONS ##

# The 2nd figure contains 2 plots that show how many buildings share a non-zero cosine similarity number with respect to their absolute storey difference,
# and the box plot of each category.
# Result: Buildings with similar geometric characteristics (number of storeys) tend to have higher cosine similaritiy weights 

# Search for the non zero values of the adjacency matrix
upper_Adj_matrix = np.triu(Adj_weighted_matrix, k=1) # Search only in the upper, since the Adj is symmetric
row_idx, col_idx = np.where(upper_Adj_matrix > 0)
# Storey differences and the coresponding weight
storey_differences = np.abs(storeys[row_idx] - storeys[col_idx])
weights = Adj_weighted_matrix[row_idx, col_idx]
# Compute the possible storey differences between the population buildings
unique_diffs, counts = np.unique(storey_differences, return_counts=True)


fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
fig2.suptitle('Population Similarities', fontsize=16, fontweight='bold')
# Subplot 1: Bar Chart of Counts 
axes2[0].bar(unique_diffs, counts, color='darkred', edgecolor='black')
axes2[0].set_xlabel("Absolute Difference in Storeys")
axes2[0].set_ylabel("Number of Connections (Edges)")
axes2[0].set_title("Similarity vs Geometric Difference")
axes2[0].set_xticks(unique_diffs)

# Subplot 2: Boxplot of Weights 
weights_per_category = [weights[storey_differences == i] for i in unique_diffs]
axes2[1].boxplot(weights_per_category, positions=unique_diffs, patch_artist=True, medianprops=dict(color='red', linewidth=1.5))
axes2[1].set_xlabel("Absolute Difference in Storeys")
axes2[1].set_ylabel("Cosine Similarity Weight")
axes2[1].set_title("Cosine Similarity Distribution per Category")
axes2[1].set_xticks(unique_diffs)
fig2.tight_layout()




## VISUALIZE MEASUREMENT-LIKE FEATURES ##

# The 3rd figure contains a plot that examines the average frequency of buildings divided in categories based on their number of storeys
# and their health condition. 
# Result: There average frequency of a building is not an indicator of the damage

# Initialize the vectors for the boxplots
plot_data = []
plot_labels = []

for i, storey_group in enumerate(unique_storeys):
    # Sort in groups based on total floors
    idx = (storeys == storey_group)
    group_frequencies = avg_frequency_of_building[idx]
    group_structure_labels = labels_array[:,1][idx]  # grap only the column 1 -> damaged or not

    # Separate healthy (=0) and damaged (=1) buildings into two groups
    healthy_frequencies = group_frequencies[group_structure_labels == 0]
    damaged_frequencies = group_frequencies[group_structure_labels == 1]

    # Append the separated data to the master plotting lists
    plot_data.append(healthy_frequencies)
    plot_labels.append(f"{int(storey_group)}-Storey \n(Healthy)")
    
    plot_data.append(damaged_frequencies)
    plot_labels.append(f"{int(storey_group)}-Storey \n(Damaged)")
    

# Generate a standard boxplot using the grouped lists
fig3, ax3 = plt.subplots(figsize=(10, 6))
box_plot = ax3.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True, medianprops=dict(color='red', linewidth=1.5))

for i, box in enumerate(box_plot['boxes']):
    if i % 2 != 0:  
        box.set_facecolor('darkred') 
    else:        
        box.set_facecolor('steelblue')

ax3.set_title('Average Building Frequency by Health Stage', fontsize=16, fontweight='bold')
ax3.set_ylabel('Average Frequency [Hz]')
fig3.tight_layout()



## OTHER ESTIMATORS ##

# We may use the Maximum Absolute Inter-Storey Frequency Difference, since the damage was
# modeled as a "localized stiffness reduction". 
max_frequency_drops = features['max_frequency_drops']


fig4, axes4 = plt.subplots(1,2,figsize=(12, 6))
fig4.suptitle('Comparison of derived damage-sensitive features', fontsize=16, fontweight='bold')

# Get true labels directly from your labels_array (column 1 is 'damaged')
damaged_status = labels_array[:, 1].astype(int)
healthy_idx = np.where(damaged_status == 0)[0]
damaged_idx = np.where(damaged_status == 1)[0]

# Plot Healthy Buildings
axes4[0].scatter(storeys[healthy_idx], max_frequency_drops[healthy_idx], c='steelblue', label='Healthy', s=100, alpha=0.8, edgecolors='black')
# Plot Damaged Buildings
axes4[0].scatter(storeys[damaged_idx], max_frequency_drops[damaged_idx], c='darkred', label='Damaged', marker='X', s=150, linewidths=2)
#
axes4[0].set_title('Feature: Max Inter-Storey Frequency Drop')
axes4[0].set_xlabel('Number of Storeys', fontsize=12)
axes4[0].set_ylabel('Max Frequency Drop Between Adjacent Floors [Hz]', fontsize=12)
axes4[0].set_xticks(np.unique(storeys))
axes4[0].grid(True, linestyle='--', alpha=0.6)
axes4[0].legend(fontsize=12)


# Another estimator could be the quantity [(f_i)^2-(f_i-1)^2]*storeys to account for both frequancy drop and mass of building
delta_k_estimator = features['delta_k_estimator']

# Plot Damaged Buildings
axes4[1].scatter(storeys[healthy_idx], delta_k_estimator[healthy_idx], c='steelblue', label='Healthy', s=100, alpha=0.8, edgecolors='black')
# Plot Damaged Buildings
axes4[1].scatter(storeys[damaged_idx], delta_k_estimator[damaged_idx], c='darkred', label='Damaged', marker='X', s=150, linewidths=2)
#
axes4[1].set_title('Feature: Estimator for Localized Stiffness Reduction (Δk)')
axes4[1].set_xlabel('Number of Storeys', fontsize=12)
axes4[1].set_ylabel('Estimator of Δk', fontsize=12)
axes4[1].set_xticks(np.unique(storeys))
axes4[1].grid(True, linestyle='--', alpha=0.6)
axes4[1].legend(fontsize=12)
fig4.tight_layout()


plt.show()


## SAVE THE FIGURES ##
os.makedirs('figures', exist_ok=True)  # create a directory fo figures
# Fig 1
fig1.savefig(os.path.join('figures','task1_population_characteristics.svg'), format='svg', bbox_inches='tight', transparent=True)
fig1.savefig(os.path.join('figures','task1_population_characteristics.png'), dpi=300, bbox_inches='tight', transparent=False)
# Fig 2
fig2.savefig(os.path.join('figures','task1_population_similarities.svg'), format='svg', bbox_inches='tight', transparent=True)
fig2.savefig(os.path.join('figures','task1_population_similarities.png'), dpi=300, bbox_inches='tight', transparent=False)
# Fig 3
fig3.savefig(os.path.join('figures','task1_avg_frequency_vs_health_stage.svg'), format='svg', bbox_inches='tight', transparent=True)
fig3.savefig(os.path.join('figures','task1_avg_frequency_vs_health_stage.png'), dpi=300, bbox_inches='tight', transparent=False)
# Fig 4
fig4.savefig(os.path.join('figures','task1_estimator_comparison.svg'), format='svg', bbox_inches='tight', transparent=True)
fig4.savefig(os.path.join('figures','task1_estimator_comparison.png'), dpi=300, bbox_inches='tight', transparent=False)


