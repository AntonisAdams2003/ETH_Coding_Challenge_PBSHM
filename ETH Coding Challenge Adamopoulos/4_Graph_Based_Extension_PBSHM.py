from data_loader import get_data
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import random
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool



## IMPORTING THE DATA  ##

# Call the data_loader.py to get the data
data = get_data()
measurements_data = data['measurements_data']
labels_array = data['labels_array']
population_weights_array = data['population_weights_array']
storeys = data['storeys']
frequencies = data['frequencies']

# Total number of buildings 
number_of_buildings = len(measurements_data)  # 50 total



## CREATE THE DATA LIST ##

# Create the list that will contain the graph data for all the buildings
graph_dataset = []

for i in range(number_of_buildings):

    # Create the temporary feature list of the building
    feature_list = []

    for j in range(storeys[i]):

        features = [
            measurements_data[i]['node_features'][j]["storey"],
            measurements_data[i]['node_features'][j]["height_m"],
            measurements_data[i]['node_features'][j]["dominant_modal_frequency_Hz"]
        ]
        # Add the features to le building list
        feature_list.append(features)
    # Node features as tensor
    x = torch.tensor(feature_list, dtype=torch.float)


    # Create the connectivity tensor
    edges = measurements_data[i]["edges"] 
    edge_index = torch.tensor(edges, dtype=torch.long).T   
    # The graph is udirected
    edge_index = to_undirected(edge_index)


    # The binary label for this building
    label = int(labels_array[i, 1])
    #  Create the answer keys tensor
    y = torch.tensor([label], dtype=torch.long)


    # Pack the building i into a PyTorch Geometric Data object
    graph = Data(x=x, edge_index=edge_index, y=y)
    # Add building i to the graph list
    graph_dataset.append(graph)


# Shuffle to ensure a random distribution
random.seed(42) 
random.shuffle(graph_dataset)

# Pick the first 40 for train and the rest for test
train_dataset = graph_dataset[:40]
test_dataset = graph_dataset[40:]


# Use dataloader to fix the problem of diiferent input sizes (different storyes)
# Group the 40 training specimens into batches of 10.
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Group the 10 holdout specimens into a single batch.
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


## FORM THE GRAPHIC CONVOLUTIONAL NETWORK (GCN) ##

# Inherit 
class BuildingGCN(torch.nn.Module):

    # Define the Neutral Network Layers
    def __init__(self):
        super(BuildingGCN, self).__init__()
        # Layer 1
        # We have 3 input features (storey index, height, frequency). 
        # We set an arbitary number of paths for data flow
        numb_hidden_paths = 16
        self.convolution1 = GCNConv(3,numb_hidden_paths)    # store convolution1 to the GCN object

        # Layer 2
        # We add an extra layes for data
        self.convolution2 = GCNConv(numb_hidden_paths,numb_hidden_paths)

        # Layer 3
        # Two outputs Healthy or Damaged
        self.convolution3 = torch.nn.Linear(numb_hidden_paths,2)

    # The forward pass
    def forward(self, x, edge_index, batch):

        # Caclulate the output x after the weights from the 1st layer neighbohrs
        x = self.convolution1(x, edge_index)
        # Add technical non-linearity
        x = F.gelu(x) # Gaussian Error Linear Unit

        # Repeat for the second "artificial" layer
        x = self.convolution2(x,edge_index)
        x = F.gelu(x)

        # Averages the features of all storeys into a single vector for each individual building
        x = global_mean_pool(x, batch)

        # We additionally add a dropout of 50%, to prevent oferfitting bu random;y zeroing neutron (in train)
        x = F.dropout(x, p=0.5, training=self.training)  # self.training to dropout only during training
        
        # return the final layer (the classifier)
        return self.convolution3(x)

# Create the model
model = BuildingGCN()


# Set the learning rate
learning_rate = 0.01 
# Define the optimizer (Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()


# We need to run the code for some epoch numbers (iterations). 
# Inside the loop we train the model using the core lines:
# # Zero the gradients
# optimizer.zero_grad()      
# # Forwad Pass: Pass the building information throung the network
# y_output = model(data.x, data.edge_index, data.batch)      
# # Loss estimation
# loss = loss_fn(y_output, data.y)     
# # Backward pass: Compute the gradients
# loss.backward()     
# # Update the paraneters
# optimizer.step()     
# # Find the total loss
# total_loss += loss.item()


# In the test phase we disable the dropout and apply torch.no_grad(), so the model doesnt learn while being tested
# Finally, we compare the predictions with the label key answers to get its accuracy

