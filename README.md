# ETH_Coding_Challenge_PBSHM: Population-Based Structural Health Monitoring

## OVERVIEW
This repository contains my solution for the coding challenge. It explores the starting population characteristics and relationships of the 50-building dataset, tests baseline ML models (supervised and unsupervised),
and lastly applies a simple Graph Convolutional Network (GCN) to account for the spatial connections of the storeys.


## REPOSITORY STRUCTURE
The project is organized as follows:
- 'data': Contains the candidate-provided data.
- 'figures': Contains the figures produced by the Python scripts.
- 'data_loader.py': Script that imports and processes raw data, outputting information in a structured format.
- '1_Explore_Population_PBSHM.py': Script that plots the main characteristics of the population (Task 1).
- '2_3_Baselines_PBSHM.py': Script that applies the Random Forest Algorithm (supervised) and subsequently PCA and K-Means (unsupervised) to investigate if damage-sensitive parameters separate the health state (Tasks 2 & 3).
- '4_Graph_Based_Extension.PBSHM.py': Script that uses a simple GCN to account for the inter-storey connections of each building (Task 4).


## KEY DESIGN DECISIONS
Data Handling('data_loader.py'): 
- The buildings have variable storeys, so the data matrices were initialized using the maximum matrix size with NaN to allow vectorized operations.
- Potential damage-sensitive features were designed and tested to try to account for localized stiffness reduction (such as max inter-storey frequency drops and a $\Delta k$ estimator)
Baseline Modeling('2_3_Baselines_PBSHM.py'):
- Random Forest was selected as the supervised baseline because it efficiently handles non-linearities in data and has built-in feature importance tracking.
- For the unsupervised approach, PCA was applied prior to K-Means to reduce dimensionality. The case of keeping 95% of the data was tested, and then keeping only 2 components for visual separation.
Graph Approach('4_Graph_Based_Extension.PBSHM.py'):
- A Graph Convolutional Network (GCN) was implemented via `torch_geometric` to model the inter-storey relationships of buildings (storeys as nodes, physical connections as edges).
- An extra layer was added to the network, and 16 paths were simulated to allow easier data flow in the network.
- Non-linearities were modeled using GELU (Gaussian Error Linear Unit) for a smoother result.
- A 50% dropout was added to mitigate overfitting.


## LIBRARIES REQUIRED
To run the scripts, you will need to install the following Python libraries:
- 'pandas'
- 'numpy'
- 'matplotlib'
- 'scikit-learn'
- 'torch'
- 'torch_geometric'


## HOW TO RUN THE CODE
To reproduce the analysis, please follow these steps:
- Ensure the raw data is located in the 'data' folder.
- Run '1_Explore_Population_PBSHM.py' for Task 1 results.
- Run '2_3_Baselines_PBSHM.py' for Tasks 2 and 3 results.
- Run '4_Graph_Based_Extension.PBSHM.py' for Task 4 results.


## NOTES
- There is no OS restriction for the code since the os module was used for cross-platform file pathing.
- The 'data_loader.py' is called dynamically by all scripts.


## AUTHOR
Antonis Adamopoulos, Email:anadamopoulos@uth.gr
