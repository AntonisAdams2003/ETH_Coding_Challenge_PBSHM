import json
import pandas as pd
import numpy as np
import os


## This code will load all the provided data and excecute the process to convert them to usefull arrays ##

# Define the data function
def get_data():

    # Give the data directory in the folders
    data_dir = 'data'

    # Import data from the .json file
    with open(os.path.join(data_dir, 'structures_measurements.json'), 'r') as file:
        measurements_data = json.load(file)

    # Import data from the .csv files as data frames
    labels_df = pd.read_csv(os.path.join(data_dir, 'structure_labels.csv'))
    population_edges_df = pd.read_csv(os.path.join(data_dir, 'population_edges_geometry.csv'))
    population_weights_df = pd.read_csv(os.path.join(data_dir, 'population_edge_weights_geometry.csv'))

    # Convert them to numpy arrays
    labels_array = labels_df.to_numpy()
    population_edges_array = population_edges_df.to_numpy()
    population_weights_array = population_weights_df.to_numpy()


    ## EXCTRACT INFO ##

    # Total number of buildings
    number_of_buildings = len(measurements_data)  # 50 total

    # Number of Storeys of each building
    storeys = np.array([building['n_storeys'] for building in measurements_data])

    # Extract the height and frequency of each storey of each building (building,storey)
    heights = np.full((number_of_buildings, np.max(storeys)), np.nan)  # Initialize the matrix with NaNs
    frequencies = np.full((number_of_buildings, np.max(storeys)), np.nan)
    for i in range(number_of_buildings):
        for j in range(storeys[i]):
            heights[i,j] = measurements_data[i]['node_features'][j]['height_m']                  
            frequencies[i,j] = measurements_data[i]['node_features'][j]['dominant_modal_frequency_Hz']
            
    # Note: Buildings with <8 floors are going to have NaN heights and frequencies. Use np.nan... is all functions for safery!


    # Extract the graph by forming the adjacency weighted matrix
    # The connected buildings are characterised by their weight number (cosine similarity) and NOT 1
    Adj_weighted_matrix = np.zeros((number_of_buildings,number_of_buildings))
    #
    adj_rows = population_edges_array[:,0] # Extract all rows for building A
    adj_cols = population_edges_array[:,1] # Extract all rows for building B
    adj_weights = population_weights_array[:,2]  # Extract the weights
    # Form the matrix and force symmetry (since the graph is undirected)
    Adj_weighted_matrix[adj_rows, adj_cols] = adj_weights
    Adj_weighted_matrix[adj_cols, adj_rows] = adj_weights   # Symmetry 


    # Calculate damage-sensitive features
    mean_frequency = np.nanmean(frequencies,axis=1)
    max_frequency = np.nanmax(frequencies,axis=1)
    min_frequency = np.nanmin(frequencies,axis=1)
    std_frequency = np.nanstd(frequencies,axis=1)
    squared_frequencies = frequencies**2
    delta_k_estimator = np.nanmax(np.abs(np.diff(squared_frequencies,axis=1)),axis=1) * storeys
    max_frequency_drops = np.nanmax(np.abs(np.diff(frequencies,axis=1)),axis=1)


    # Return the important quantities
    return {
        'measurements_data': measurements_data,
        'labels_array': labels_array,
        'population_weights_array': population_weights_array,
        'storeys': storeys,
        'heights': heights,
        'frequencies': frequencies,
        'Adj_weighted_matrix': Adj_weighted_matrix,
        'features': {
            'mean_frequency': mean_frequency,
            'max_frequency': max_frequency,
            'min_frequency': min_frequency,
            'std_frequency': std_frequency,
            'delta_k_estimator': delta_k_estimator,
            'max_frequency_drops': max_frequency_drops
        }
    }
