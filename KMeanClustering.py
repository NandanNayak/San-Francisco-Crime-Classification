# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:11:50 2015

@author: nandannayak
"""

from pandas import pandas as ps
import numpy as np
from sklearn import cluster
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import  metrics
from time import time

def MeasurementOfAlgorithmMetrics(modelAlgo,name, data, labelData):
    initialTime = time()
    labels = labelData
    modelAlgo.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f'
          % (name, (time() - initialTime), modelAlgo.inertia_,
             metrics.homogeneity_score(labels, modelAlgo.labels_),
             metrics.completeness_score(labels, modelAlgo.labels_),
             metrics.v_measure_score(labels, modelAlgo.labels_),
             metrics.adjusted_rand_score(labels, modelAlgo.labels_)
             ))

def EncodedColumnValues(documentDataFrame, columnToBeEncoded):
    """This function enumerates the string values of the columns to numbers"""
    modifiedDataFrame = documentDataFrame.copy()
    encodingTargetList = modifiedDataFrame[columnToBeEncoded].unique()
    convert_to_int = {strName: float(n) for n, strName in enumerate(encodingTargetList)}
    modifiedDataFrame["Target"+columnToBeEncoded] = modifiedDataFrame[columnToBeEncoded].replace(convert_to_int).astype(np.float64)
    return (modifiedDataFrame)

if __name__ == '__main__':
    totalTime = time()
    """Please change the below path to where you have stored the database in your machine"""
    documentDataFrame= ps.read_csv('/media/nandan/Store/Python Excersises/FinalProjectML/train.csv')
    trainDataFrameIndex, testDataFrameIndex= train_test_split(documentDataFrame.index, train_size = 0.7)
    trainDataFrame = documentDataFrame.iloc[trainDataFrameIndex]
    testDataFrame = documentDataFrame.iloc[testDataFrameIndex]
    modifiedtrainDataFrame = EncodedColumnValues(trainDataFrame, "Category")
    modifiedtestDataFrame = EncodedColumnValues(testDataFrame, "Category")
    
    featuresList = []
    featuresTestList = []
    modifiedtrainDataFrame = EncodedColumnValues(modifiedtrainDataFrame, "DayOfWeek")
    modifiedtestDataFrame = EncodedColumnValues(modifiedtestDataFrame, "DayOfWeek")
    modifiedtrainDataFrame = EncodedColumnValues(modifiedtrainDataFrame, "PdDistrict")
    modifiedtestDataFrame = EncodedColumnValues(modifiedtestDataFrame, "PdDistrict")
    
    featuresList.append(modifiedtrainDataFrame.columns[11])
    featuresList.append(modifiedtrainDataFrame.columns[7])
    featuresList.append(modifiedtrainDataFrame.columns[8])
    featuresList.append(modifiedtrainDataFrame.columns[9])
    featuresTestList.append(modifiedtestDataFrame.columns[11])
    featuresTestList.append(modifiedtestDataFrame.columns[7])
    featuresTestList.append(modifiedtestDataFrame.columns[8])
    featuresTestList.append(modifiedtestDataFrame.columns[9])
    k = len(documentDataFrame["Category"].unique())
    y= modifiedtrainDataFrame["TargetCategory"]
    yT = modifiedtestDataFrame["TargetCategory"]
    X= modifiedtrainDataFrame[featuresList]
    XT = modifiedtestDataFrame[featuresTestList]
    X= np.array(X)
    XT = np.array(XT)
    y=np.array(y)
    print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI')
    MeasurementOfAlgorithmMetrics(cluster.KMeans(init='k-means++', n_clusters=k, n_init=10), name='kmeans++', data = X, labelData=y)
    reducedData = PCA(n_components=2).fit_transform(X)
    reduced_test_Data = PCA(n_components=2).fit_transform(XT)
    kmeansModel = cluster.KMeans(init='k-means++',n_clusters=k, n_init=10)
    kmeansModel.fit(reducedData)
    labelsOfData = kmeansModel.labels_
    
#############################################################################
#      Plot the Graph
#############################################################################
#    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reducedData[:, 0].min() - 1, reducedData[:, 0].max() + 1
    x_minT, x_maxT = reduced_test_Data[:,0].min() - 1, reduced_test_Data[:,0].max() + 1
    y_min, y_max = reducedData[:, 1].min() - 1, reducedData[:, 1].max() + 1
    y_minT, y_maxT = reduced_test_Data[:,1].min() -1, reduced_test_Data[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xxT, yyT = np.meshgrid(np.arange(x_minT, x_maxT, h), np.arange(y_minT, y_maxT, h))
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeansModel.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

    plt.plot(reducedData[:, 0], reducedData[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centeroidsOfCluster = kmeansModel.cluster_centers_
    plt.scatter(centeroidsOfCluster[:, 0], centeroidsOfCluster[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title('K-means clustering on the training part of dataset(PCA-reduced data)\n'
          'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.savefig("TrainDataCluster.png")
    
    ZT = kmeansModel.predict(np.c_[xxT.ravel(), yyT.ravel()])

    # Put the result into a color plot
    ZT = ZT.reshape(xxT.shape)
    plt.figure(2)
    plt.clf()
    plt.imshow(ZT, interpolation='nearest',
           extent=(xxT.min(), xxT.max(), yyT.min(), yyT.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

    plt.plot(reduced_test_Data[:, 0], reduced_test_Data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centeroidsOfClusterTest = kmeansModel.cluster_centers_
    plt.scatter(centeroidsOfClusterTest[:, 0], centeroidsOfClusterTest[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title('K-means clustering on the testing part of dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
    plt.xlim(x_minT, x_maxT)
    plt.ylim(y_minT, y_maxT)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.savefig("TestDataCluster.png")
    totalTime = time()-totalTime
    print("Total Time taken for the algorithm:"+str(totalTime)+" seconds.")
    
