import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from random import seed

MAX_ITERATIONS = 50

def dist(x,y):
    return np.sqrt(sum((x - y) ** 2))

def generateCentroid(qtd,k):
     return [[2,2,2,2],
             [3,3,3,3],
             [4,4,4,4]]
    #return np.random.random_sample((k,qtd))

def somaVet(ori,dest,qtd):
    i = 0
    for o in ori:
        dest[i] += o
        i += 1
    qtd += 1
    return dest,qtd

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def ajusteCluster(x, l, k):
    listCentroids = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    sumQtd = [0,0,0]
    qtd = len(l)    
    
    i = 0
    while(i < qtd):
        listCentroids[l[i]],sumQtd[l[i]] = somaVet(x[i],listCentroids[l[i]],sumQtd[l[i]])
        i += 1   

    i = 0
    for cent in listCentroids:
        if(sumQtd[i] == 0):
            cent = np.random.random_sample((1,4))
        else :
            j = 0
            for c in cent:
                cent[j] = truncate(c/sumQtd[i],2)
                j += 1
        
        i += 1

    return listCentroids

def agruparDados(X, C):
    labels = []    
    for x in X:
        minValue = sys.maxsize
        interation = 0
        for c in C:
            d = dist(x,c)
            if(d < minValue):
                minValue = d
                label = interation
            interation += 1
        labels.append(label)
    return labels

def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS: 
        return True
    return oldCentroids == centroids



def kmeans(X,k):
    centroids = generateCentroid(len(X[0]),k)

    iterations = 0
    oldCentroids = None

    while not shouldStop(oldCentroids, centroids, iterations):
        
        oldCentroids = centroids
        iterations += 1        
        labels = agruparDados(X, centroids)        
        centroids = ajusteCluster(X, labels, k)

        # print(labels)
        # print('\n')
        # print(centroids)
        # print('\n')
        # print(iterations)
        # print('\n')

    return centroids

k = 3


df = pd.read_csv('irisCsv.csv')

print("--- MENU ---")
op =  input("1 - Mostrar cluster final\n2 - Plotar grafico com agrupamento final\n3 - Plotar grafico com função alvo original\n")
op = int(op)

if(op == 1):
    X = np.array(df.drop('target', axis =1))
    print(kmeans(X,k))
else :
    sns.set()
    compar = 'target'
    if(op == 2):
        compar = 'k-means'       
        X = np.array(df.drop('target', axis =1))
        retorno = kmeans(X,k)
        agrupados = agruparDados(X,retorno)
        df.insert(5, "k-means", agrupados , True)
    
    sns.pairplot(df, hue=compar)
    plt.show()        
    
