# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:56:01 2015

@author: nandannayak
"""

import pandas as pd
import StringIO
import pydot
from sklearn import tree
from time import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split

#Macros

TARGET_COLMN = "Category"

if __name__=='__main__':
    totalTime = time()
    """Please change the below path to where you have stored the database in your machine"""
    df=pd.read_csv('/media/nandan/Store/Python Excersises/FinalProjectML/train.csv')
    train_index, test_index=train_test_split(df.index, train_size = 0.7, test_size = 0.3)
    train_df = df.iloc[train_index]
    
    df_mod=train_df.copy()
    targets=df[TARGET_COLMN].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[TARGET_COLMN]=df_mod[TARGET_COLMN].replace(map_to_int)
    features=list(df_mod.columns[7:9])
    y=df_mod[TARGET_COLMN]
    x=df_mod[features]
    
    dt=DecisionTreeClassifier(min_samples_split=10000,random_state=99)
    dt.fit(x,y)
    dotData = StringIO.StringIO()
    tree.export_graphviz(dt, out_file= dotData,
                         feature_names = features)
    
    graphVisual = pydot.graph_from_dot_data(dotData.getvalue())
    graphVisual.write_pdf("DeciTrainDataTree.pdf")
    
    #Test DataFrame
    test_df = df.iloc[test_index]
    x_test1 = test_df[features]
    predictedValues = dt.predict(x_test1)
    np.savetxt("PredictedDtreeValues.txt", predictedValues)
    dotData = StringIO.StringIO()
    tree.export_graphviz(dt, out_file= dotData,
                         feature_names = features)
    
    graphVisual = pydot.graph_from_dot_data(dotData.getvalue())
    graphVisual.write_pdf("DeciTestDataTree.pdf")
    
    
    #Use test dataset as Trainig dataset    
    test_df_mod=test_df.copy()
    test_targets=test_df_mod[TARGET_COLMN].unique()
    test_map_to_int = {name: n for n, name in enumerate(test_targets)}
    test_df_mod[TARGET_COLMN]=test_df_mod[TARGET_COLMN].replace(test_map_to_int)
    features=list(test_df_mod.columns[7:9])
    test_y=test_df_mod[TARGET_COLMN]
    test_x=test_df_mod[features]
    
    test_dt=DecisionTreeClassifier(min_samples_split=1000,random_state=99)
    test_dt.fit(test_x,test_y)
    print test_dt.score(test_x,test_y)
    totalTime = time()-totalTime
    print("Total Time taken for the algorithm:"+str(totalTime)+" seconds.")
    
