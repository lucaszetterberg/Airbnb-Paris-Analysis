import numpy as np
import statistics
import seaborn as sns
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.cluster import MiniBatchKMeans

features = [
    "price",
    "accommodates",
    "bedrooms",
    "minimum_nights",
    "maximum_nights",
    "host_total_listings_count",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "latitude",
    "longitude"
]

def plot_correlation_circle(initial_variables, new_variables, feature_names,c1=0,c2=1):
    plt.figure(figsize=(8, 8))
    pc_labels=[f'PC{c1}', f'PC{c2}']
    
    # Tracer un cercle unité
    circle = plt.Circle((0, 0), 1, color='gray', fill=False)
    plt.gca().add_artist(circle)
    
    # Tracer les flèches (correlations des variables)
    for i in range(len(feature_names)):
        x=np.corrcoef(new_variables[:,c1],initial_variables[:,i])[0,1]
        y=np.corrcoef(new_variables[:,c2],initial_variables[:,i])[0,1]
        plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.001, fc='red', ec='red')
        plt.text(x * 1.1, y * 1.1, feature_names[i], color='black', ha='center', va='center')
    
    # Ajustements esthétiques
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel(pc_labels[0])
    plt.ylabel(pc_labels[1])
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.title('Cercle de corrélation')
    plt.show()
    
def elbow_analysis(X_scaled, K=20):
    inertia = []
    K_range = range(1, K)

    for k in K_range:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow Curve
    plt.plot(K_range, inertia, 'bx-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Intra-cluster Inertia')
    plt.title('Elbow Method for Paris Airbnb Peer Groups')
    plt.show()
    
def inertia_barplot(pca):
    i=np.arange(1,pca.n_components_+1)
    plt.bar(i,pca.explained_variance_ratio_)
    plt.title('Barplot of principal component decomposition')
    plt.grid()
    plt.show()