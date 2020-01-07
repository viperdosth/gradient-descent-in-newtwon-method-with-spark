# -*- coding: utf-8 -*-
import numpy as np
import argparse
from time import time
from SparseVector import SparseVector
from operator import add
from pyspark import SparkContext
import os
import math

def readDataRDD(input_file,spark_context):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)  

         x is a dictionary of the form:                 

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The result is stored in an RDD containing tuples of the form
                 (SparseVector(x),y)             
    """ 
    return spark_context.textFile(input_file)\
                        .map(eval)\
                        .map(lambda datapoint:(SparseVector(datapoint[0]),datapoint[1]))

##
# sigmod函数
# #
def sigmoid(x):
    return 1.0 / (1+math.exp(-x))


    ##
# 计算hessian矩阵
# #
def computeHessianMatrix(data, hypothesis):
    hessianMatrix = []
    n = len(data)

    for i in range(n):
        row = []
        for j in range(n):
            row.append(-data[i]*data[j]*(1-hypothesis)*hypothesis)
        hessianMatrix.append(row)
    return hessianMatrix

##
# 计算两个向量的点积
# #

def computeDotProduct(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    dotProduct = 0
    for i in range(n):
        dotProduct += a[i] * b[i]
    return dotProduct

##
# 计算两个向量的和
# #
def computeVectPlus(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    sum = []
    for i in range(n):
        sum.append(a[i]+b[i])
    return sum

##
# 计算某个向量的n倍
# #
def computeTimesVect(vect, n):
    nTimesVect = []
    for i in range(len(vect)):
        nTimesVect.append(n * vect[i])
    return nTimesVect


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


def test(testRDD,theta):
    testRDD = testRDD.map(lambda pair: (np.array(list(pair[0].values())),pair[1]))

    Prdd = testRDD.filter(lambda pair: classifyVector(pair[0],theta) > 0).cache()
    Nrdd = testRDD.filter(lambda pair: classifyVector(pair[0],theta) <= 0).cache()

    TP = float(Prdd.filter(lambda pair: pair[1] == 1).count())
    FP = float(Prdd.filter(lambda pair: pair[1] == -1).count())
    TN = float(Nrdd.filter(lambda pair: pair[1] == -1).count())
    FN = float(Nrdd.filter(lambda pair: pair[1] == 1).count())

    ACC = (TP + TN) / (Prdd.count() + Nrdd.count())
    PRE = TP / (TP + FP)
    REC = TP / (TP + FN)

    return ACC, PRE, REC

def train(data,iterNum,testRDD=None):
    
    m = data.count()
    n = len(data.take(1)[0][0])
    z = 0
    theta = [0.0] * n

    start = time()
    while z < iterNum:
        print(z)
        gradientSum = [0.0] * n
        hessianMatSum = [[0.0] * n] * n


        hypothesis_label = data.map(lambda pair: (list(pair[0].values()), pair[0],pair[1]))\
                               .map(lambda pair: (sigmoid(pair[1].dot(theta)),sigmoid(computeDotProduct(pair[0],theta)),pair[1], pair[2]))


        data_predictionRDD = hypothesis_label.map(lambda pair: (pair[2], pair[3] - pair[1]))

        gradient = data_predictionRDD.map(lambda pair: (pair[0],pair[1]/m))\
                                .map(lambda pair: pair[0] * pair[1])
        

        gradientSum = gradient.reduce(lambda x,y: x + y)

        hessianRDD = hypothesis_label.map(lambda pair: (list(pair[2].values()),pair[1]/m))\
                                  .map(lambda pair: computeHessianMatrix(pair[0],pair[1]))

        hessian = hessianRDD.collect()

        for i in range(m):
            for j in range(n):
                hessianMatSum[j] = computeVectPlus(hessianMatSum[j], hessian[i][j])

        try:

            hessianMatInv = np.mat(hessianMatSum).I.tolist()
            print(hessianMatInv)
        except:
            hessianMatInv = hessianMatSum
            print("出异常了")

        gradientSumList = list(gradientSum.values()) 
        for k in range(n):
            theta[k] -= computeDotProduct(hessianMatInv[k], gradientSumList)

        if testRDD != None:
            acc,pre,rec = test(testRDD,theta)
            print('z = ',z,'\tt = ',time()-start,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec)

        z = z + 1

    return theta














