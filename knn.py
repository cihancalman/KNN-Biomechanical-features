# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:18:12 2023

@author: cihan
"""

# %% import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% load and plot data


data = pd.read_csv("column_2C_weka.csv")

N = data[data["class"] == "Normal"]
A = data[data["class"] == "Abnormal"]

plt.scatter(N.pelvic_incidence, N.pelvic_radius, color="blue", label = "Normal",alpha=0.3)
plt.scatter(A.pelvic_incidence, A.pelvic_radius, color="red", label = "Abnormal",alpha=0.3)
plt.xlabel("Pelvic Incidence")
plt.ylabel("Pelvic Radius")
plt.legend()

#%% edit data
data["class"] = [1 if each =="Normal" else 0   for each in data["class"]]



y = data["class"]
x = data.drop(["class"],axis=1)

#%% Data Normalization

x = (x - np.min(x))/ (np.max(x) - np.min(x))

#%% data train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


#%% KNN Model and find k

from sklearn.neighbors import KNeighborsClassifier

score_list = []
max = 0 
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    score_list.append(knn.score(x_test, y_test))
    if(score_list[i-1] > max) :
        max , k = score_list[i-1],i
    
    
plt.plot(range(1,20), score_list)
plt.show()

print("{} nn score: {}".format(k, max))
























