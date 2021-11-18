# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:48:37 2020

@author: ucanr
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from sklearn.cluster import KMeans
%matplotlib inline

#Import the data
trainFile = "C:/Users/ucanr/iCloudDrive/Statistics/Statistics for Me/Codes/drug200.csv"
pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
myData=pd.read_csv(os.path.basename(trainFile))


#Subset the data to just the numerical variables
feature_cols=['Age','Na_to_K']
myDataSubset=myData[feature_cols]


#Lets look at our data
print(myDataSubset.head(3))
plt.scatter(myDataSubset.Age,myDataSubset.Na_to_K)

#Assign the number of clusters
myGroups=KMeans(n_clusters=3)
myGroups.fit(myDataSubset)
#Assign centroids
centroids=myGroups.cluster_centers_
labels=myGroups.labels_

#Set up the color palette
colors=["b.","g.","r.","c.","m."]

#Plot each point
for i in range(len(myDataSubset)):
    plt.plot(myDataSubset.iloc[[i],[0]],myDataSubset.iloc[[i],[1]],colors[labels[i]],markersize=10)

#Generate the view
plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=150,linewidth=5)
plt.show()
