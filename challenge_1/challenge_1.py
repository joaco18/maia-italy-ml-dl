
"""
Import all libraries you need.

In this example there are only libraries needed for the example code.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier     #KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
Read the data.

In this challenge you will use the toy dataset Breast Cancer Wisconsin (Diagnostic) Database.

In the following example, how to load and read the infos of this dataset.
"""

breast_dataset=load_breast_cancer()
#let's print the dataset description
print(breast_dataset.DESCR+"\n\n")
#let's print the size of the dataset
print("Size:" +str(breast_dataset.data.shape)+"\n")
#let's print the features names
print("Features:\n"+str(breast_dataset.feature_names)+"\n")
#let's print the class names
print("Classes:\n"+str(breast_dataset.target_names))

"""
Study the features.

As an example, in the following code we use pandas (after a conversion from a  scikit-learn dataset to a pandas dataset) and seaborn libraries to:
- build a scatter matrix to study the covariance between features (in the example, all features are used, but you can plot only the features you need)
- build a heatmap of correlation between features (again of all features, but you can choose what you need) using the Pearson correlation coefficient
"""

#convert the scikit-learn dataset into pandas dataset
breast_data_pd = pd.DataFrame(breast_dataset.data,columns=breast_dataset.feature_names)
breast_data_pd['target'] = breast_dataset.target
#plot the scatter matrix
pd.plotting.scatter_matrix(breast_data_pd[breast_dataset.feature_names],figsize=(25,25))
#correlation plot
corr = breast_data_pd[breast_dataset.feature_names].corr()
f, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(corr,annot=True)

"""
Apply all the feature transformations and/or feature selection and/or dimensionality reduction that you think are necessary to improve the performance.

Follow the Scikit-learn user guide, and in particular: 
- for feature transformations see https://scikit-learn.org/stable/modules/preprocessing.html 
- for dimensionaity reduction see this example on LDA and PCA https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html
- for feature selection see https://scikit-learn.org/stable/modules/feature_selection.html  

Don't modify this part where we split the data into training and test, then train a 3-NN classifier and evaluate the accuracy on the test set.
"""

#DO NOT MODIFY THE FOLLOWING SPLIT OF THE DATASET
X_train, X_test, y_train, y_test = train_test_split(breast_dataset.data, breast_dataset.target, stratify=breast_dataset.target,random_state=42)

#BE CAREFUL: To find the right transformation only use the training set (variable X_train) 
#Then apply the same transformation to the data in the test set (variable X_test)

#Write here your code to transform, select or reduce features (don't forget to import libraries)



#DO NOT MODIFY THE CLASSIFIER AND THE PERFORMANCE EVALUATION
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
scores = knn.predict(X_test)

# Show prediction accuracy
print('\nPrediction accuracy:')
print('{:.2%}\n'.format(accuracy_score(y_test, scores)))