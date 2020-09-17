# ML_Supervised-and-unsupervised
### Introduction

Having one or two drinks now are proved to be good to human body. A glass of wine keeps a docuter away! The goal of this assignment is to perform supervised and unsupervised machine learning to demonstrate classification and clustering using the built-in toy dataset in sklearn.datasets module.

#### Dataset: dataset load_wine in sklearn.datasets module
- from sklearn.datasets import load_wine
- wine = load_wine()
- print(wine.DESCR)

Number of Instances: 178 (50 in each of three classes) Number of Attributes: 13 numeric, predictive attributes and the class Attribute Information:
- Alcohol
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

class:
-class_0
-class_1
-class_2

### Perform k-means clustering - Unsupervised machine learning
##### 1. create a pair plot to demonstrate the relationship and clusters between each attribute
- %matplotlib inline
- import seaborn as sns
- sns.set(font_scale=1.1)
- sns.set_style('whitegrid')
- grid = sns.pairplot(data=wine_df, vars=wine_df.columns[0:5], hue="wine classification")

##### 2.1 Perform dimensionality reduction down to two dimensions using TSNE
- from sklearn.manifold import TSNE
- tsne = TSNE(n_components=2, random_state=22)
- reduced_wine1 = tsne.fit_transform(wine.data)
- reduced_wine1.shape

##### 2.2 Perform dimensionality reduction down to two dimensions using PCA¶
- from sklearn.decomposition import PCA
- pca = PCA(n_components=2, random_state=22)  # reduce to two components
- reduced_wine2 = pca.fit_transform(wine.data)
- reduced_wine2.shape

##### 3.1 Perform k-means clustering with TSNE dimensionality reduction method
- from sklearn.cluster import KMeans
- kmeans = KMeans(n_clusters=3, random_state=22) 
- kmeans.fit(reduced_wine1.data)

##### 3.2 Perform k-means clustering with PCA dimensionality reduction method
- kmeans.fit(reduced_wine2.data)

#### 4. create scatter plots to determine which reducion methods is better
##### 4.0 show the cluster centers for each dimensionalirty reduction method
if kmeans have the input data without dimentionality reducing:
-print("cluster centers:\n", kmeans.cluster_centers_) 
- the cluster centers before reducing dimensionality => 3 x 13 arrays
- wine_centers = pca.transform(kmeans.cluster_centers_) 
- apply the pca to reduce the dimentionality of cluster centers

##### 4.1 graph the scatter plot with old labels using TSNE dimensionality reduction method
##### 4.2 graph the scatter plot with new labels using TSNE dimensionality reduction method
##### 4.3 graph the scatter plot with old labels using PCA dimensionality reduction method
##### 4.4 graph the scatter plot with new labels using PCA dimensionality reduction method

### Supervised Machine Learning with new labels using TSNE to reduce dimensionality
##### 1. convert the wine class from string to integer so it can perform k-neighborsclassifier
##### 2. split the dataset to train and test sets
##### 3. perform k-neighborsClassigier with train dataset to build the predictive model
##### 4. make prediction with the predicitive model and x test set, and compare with the expected test dataset
##### 5. create a confusion matrix and the heatmap of the matrix
##### 6. Take a look at the classification report

### Further experiments
Assuming that the wine dataset is not a cheating data that has already been labeled. I try to apply different number of cluster to kmeans clustering method and create scatter plot to see the clustering
