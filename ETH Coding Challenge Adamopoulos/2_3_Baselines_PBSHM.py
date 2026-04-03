from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict 
from data_loader import get_data
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score , confusion_matrix,ConfusionMatrixDisplay , roc_curve,auc
from sklearn.cluster import KMeans


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

# Total number of buildings 
number_of_buildings = len(measurements_data)  # 50 total


## FORM THE RANDOM FOREST SOLVER ##

# Form the Feature Matrix
X = np.column_stack((
    storeys,
    features['std_frequency'],
    features['min_frequency'],
    features['max_frequency_drops'],
    features['delta_k_estimator']
    ))

# Form the Target Vector
y = labels_array[:,1].astype(int)  # Make the binary health state integer


# Call the Random Forest (initiate the algorithm)
rf_baseline_model = RandomForestClassifier(random_state=2)


# Run the cross-validation to get the probability of RF predictions 
y_prob_rf = cross_val_predict(rf_baseline_model,X,y, cv=5,method='predict_proba')  # returns [P(0:healthy) , P(1:damaged)]
# We mostly care about the probability of a building being damaged:
y_prob_damaged_rf = y_prob_rf[:,1]

# If the model is >=50% sure that the building is damaged then =1, elseif <50% then =0. The predictions are:
y_pred_rf = (y_prob_damaged_rf >= 0.5).astype(int)  # 0.5 threshold

# Find the metrics for Random Forest
# Accuracy score
accuracy_score_rf = accuracy_score(y,y_pred_rf)
# F1 score
f1_score_rf = f1_score(y,y_pred_rf)
# ROC AUC score
roc_auc_score_rf = roc_auc_score(y,y_prob_damaged_rf)  # ROC AUC evaluates the confidence of the model, so it need the probabilities y

# Print the metrics
print("\n Random Forest Metrics:")
print(f"Accuracy score: {accuracy_score_rf:.3f}")
print(f"F1 score: {f1_score_rf:.3f}")
print(f"ROC AUC score: {roc_auc_score_rf:.3f}")
print('\n')


# Confusion Matrix
conf_matrix_rf = confusion_matrix(y,y_pred_rf)
conf_matrix_disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=['Healthy','Damaged'])

# ROC Curve
roc_curve_rf = roc_curve(y,y_prob_damaged_rf)   # returns [fpr, tpr, thresholds]
roc_auc_score_curve_rf = auc(roc_curve_rf[0], roc_curve_rf[1])

# Plot the Confusion Matrix and the ROC Curve
fig1, axes1 = plt.subplots(1,2, figsize=(12, 6))
fig1.suptitle('Random Forest: Results evaluation', fontsize=16, fontweight='bold')

conf_matrix_disp_rf.plot(ax=axes1[0], cmap=plt.cm.Blues, text_kw={'fontsize': 16})
axes1[0].set_title('Confusion Matrix of Random Forest Predictions')

axes1[1].plot(roc_curve_rf[0], roc_curve_rf[1], color='darkred', lw=2, label=f'ROC curve (area = {roc_auc_score_curve_rf:.3f})')
axes1[1].plot([0, 1], [0, 1], color='steelblue', lw=2, linestyle='--', label='Random guess (AUC = 0.50)')  # line of ROC AUC score = 0.5
axes1[1].set_xlim([0.0, 1.0])
axes1[1].set_ylim([0.0, 1.05])
axes1[1].set_xlabel('False Positive Rate')
axes1[1].set_ylabel('True Positive Rate')
axes1[1].set_title('ROC Curve (Random Forest)')
axes1[1].legend(loc="lower right")
fig1.tight_layout()


# Extract the feature importances
rf_baseline_model.fit(X, y)
# See the importance of each feature in training the model
# Note: The array follows the X matrix: storeys,mean_frequency,max_frequency,min_frequency,std_frequency,max_frequency_drops
importances_rf = rf_baseline_model.feature_importances_   # percentage that each feature was used

# Names of X columns
feature_names = [
 'Number of Storeys', 
 'Frequency Std', 
 'Min Frequency', 
 'Max Frequency Drop', 
 'Δk estimator'
]

# Sort the features by importance
importance_idx = np.argsort(importances_rf)[::-1]  # get from highest to lower importance
# Apply the index
sorted_feature_names = [feature_names[i] for i in importance_idx]
sorted_importances = importances_rf[importance_idx]
# Count the number of features
num_features_rf = len(sorted_feature_names)

fig2, axes2 = plt.subplots(figsize=(10, 6))
axes2.bar(range(num_features_rf), sorted_importances, color='steelblue', edgecolor='black', align="center")
axes2.set_title("Feature Importances for Random Forest",fontsize=16, fontweight='bold')
axes2.set_xticks(range(num_features_rf))
axes2.set_xticklabels(sorted_feature_names, rotation=55)
fig2.tight_layout()






## FORM THE PCA and K-MEANS SOLVER ##


# Feature Matrix has columns with different units and different scales.
# We have to scale all the columns to have a mean of 0 and variance 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compress the data in order for the K-means to handle them better
pca_compressed = PCA(n_components=0.95)     # Keep the 95% of the information
X_pca_compressed = pca_compressed.fit_transform(X_scaled)   # the compressed feature matrix
# See the dimentionality reduction
print(f"Feature Matrix after size compresion: {X_pca_compressed.shape}")

# Feed the reduced-order matrix into the K-means clustering algorithm
kmeans_model = KMeans(n_clusters=2, random_state=2, n_init=10)  # Select 2 clusters: Healthy (=0), Damaged (=1)
y_pred_kmeans = kmeans_model.fit_predict(X_pca_compressed)


# K-Means separates the data into two categories withought knowing the physical meaning of each group (damaged and healthy)
y_pred_kmeans_case1 = y_pred_kmeans    # the indexes of kmeans match the healthy and damaged ones
y_pred_kmeans_case2 = 1 - y_pred_kmeans    # they dont, so we swap

# Accuracy score
accuracy_score_kmeans_case1 = accuracy_score(y, y_pred_kmeans_case1)
accuracy_score_kmeans_case2 = accuracy_score(y, y_pred_kmeans_case2)
# F1 score
f1_score_kmeans_case1 = f1_score(y, y_pred_kmeans_case1)
f1_score_kmeans_case2 = f1_score(y, y_pred_kmeans_case2)
# Note: ROC AUC score is omitted since the K-means algorithms just sorts and doesnt produce a "confidence probability"

# Print the metrics
print("\n PCA and K-Means Metrics:")
print(f"Accuracy score case 1: {accuracy_score_kmeans_case1:.3f}")
print(f"F1 score case 1: {f1_score_kmeans_case1:.3f}")
print(f"Accuracy score case 2: {accuracy_score_kmeans_case2:.3f}")
print(f"F1 score case 2: {f1_score_kmeans_case2:.3f}")



# Confusion Matrix for the best prediction
conf_matrix_kmeans = confusion_matrix(y,y_pred_kmeans_case2)
conf_matrix_disp_kmeans = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_kmeans, display_labels=['Healthy','Damaged'])

# Plot the Confusion Matrix 
fig3, axes3 = plt.subplots(figsize=(8, 6))
conf_matrix_disp_kmeans.plot(ax=axes3, cmap=plt.cm.Blues, text_kw={'fontsize': 16})
axes3.set_title('Confusion Matrix of K-Means Predictions',fontsize=16, fontweight='bold')


# For visualization purposes ONLY, we are forcing the PCA to keep only 2 components (2D)
pca_visualization = PCA(n_components=2)
X_pca_visualization = pca_visualization.fit_transform(X_scaled)   # or use X_pca_compressed



fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle('2D compression PCA scatter plots', fontsize=16, fontweight='bold')

# Scatter plot of the true labels 
axes4[0].scatter(X_pca_visualization[y == 0,0], X_pca_visualization[y == 0,1], c='steelblue', label='Healthy', edgecolor='black', s=60, alpha=0.8)
axes4[0].scatter(X_pca_visualization[y == 1,0], X_pca_visualization[y == 1,1], c='darkred', label='Damaged', edgecolor='black', s=60, alpha=0.8)
axes4[0].set_title('True Health State')
axes4[0].set_xlabel('Principal Component 1')
axes4[0].set_ylabel('Principal Component 2')
axes4[0].legend()
axes4[0].grid(True, linestyle='--', alpha=0.6)


# Scatter plot of K-Means clusters 
axes4[1].scatter(X_pca_visualization[y_pred_kmeans == 0,0], X_pca_visualization[y_pred_kmeans == 0,1], c='darkred', label=f'Cluster 0', edgecolor='black', s=60, alpha=0.8)
axes4[1].scatter(X_pca_visualization[y_pred_kmeans == 1,0], X_pca_visualization[y_pred_kmeans == 1,1], c='orange', label=f'Cluster 1', edgecolor='black', s=60, alpha=0.8)
axes4[1].set_title('K-Means Clusters')
axes4[1].set_xlabel('Principal Component 1')
axes4[1].legend()
axes4[1].grid(True, linestyle='--', alpha=0.6)
fig4.tight_layout()


plt.show()



## SAVE THE FIGURES ##

os.makedirs('figures', exist_ok=True)  # create a directory fo figures
# Fig 1
fig1.savefig(os.path.join('figures','task23_rf_confusion_ROC.svg'), format='svg', bbox_inches='tight', transparent=True)
fig1.savefig(os.path.join('figures','task23_rf_confusion_ROC.png'), dpi=300, bbox_inches='tight', transparent=False)
# Fig 2
fig2.savefig(os.path.join('figures','task23_rf_feature_importances.svg'), format='svg', bbox_inches='tight', transparent=True)
fig2.savefig(os.path.join('figures','task23_rf_feature_importances.png'), dpi=300, bbox_inches='tight', transparent=False)
# Fig 3
fig3.savefig(os.path.join('figures','task23_kmeans_confusion.svg'), format='svg', bbox_inches='tight', transparent=True)
fig3.savefig(os.path.join('figures','task23_kmeans_confusion.png'), dpi=300, bbox_inches='tight', transparent=False)
# Fig 4
fig4.savefig(os.path.join('figures','task23_2d_pca.svg'), format='svg', bbox_inches='tight', transparent=True)
fig4.savefig(os.path.join('figures','task23_2d_pca.png'), dpi=300, bbox_inches='tight', transparent=False)
