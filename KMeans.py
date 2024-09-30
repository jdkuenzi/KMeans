'''
 Projet: KMeans
 Cours : Machine Learning
 Auteurs : Ottavio Buonomo & Jean-Daniel Kuenzi
 Version : 1.0
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import itertools
import sys
import copy

def distanceEuclidienne(v1, v2):
    dist = 0
    for i, j in zip(v1, v2):
        dist += (i - j) ** 2
    return round(np.sqrt(dist), 3)

def centroid(clusters, p):
    minCluster = None
    minDist = None
    for c, pos in clusters.items():
        tmp = distanceEuclidienne(p, pos[0])
        if  minDist is None or tmp < minDist:
            minCluster = c
            minDist = tmp
    clusters[minCluster][1].append(p)

def moyCluster(clusters):
    for c, lst in clusters.items():
        if lst[1] != []:
            clusters[c][0] = np.mean(lst[1], axis=0)

def resetGroupsPoints(clusters):
    for c, lst in clusters.items():
        clusters[c][1] = []

def initialClusters(clusters, nbClusters):
    for i in range(nbClusters):
        clusters[i] = [[], []]
        clusters[i][0] = np.random.rand(1, 2)[0]

def isDifferent(d1, d2):
    if len(d1.keys()) != len(d2.keys()):
        return True 
    for lst1, lst2 in itertools.zip_longest(d1.values(), d2.values()):
        lst1 = np.array(lst1[1])
        lst2 = np.array(lst2[1])
        if lst1.shape != lst2.shape:
            return True
        if not (lst1==lst2).all():
            return True
    return False

def afficheClustering(clusters, nbClusters, title = ''):
    plt.figure()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    x = np.arange(nbClusters)
    ys = [i+x+(i*x)**2 for i in range(nbClusters)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    for group, c in zip(clusters.values(), colors):
        xs = [p[0] for p in group[1]]
        ys = [p[1] for p in group[1]]
        x = group[0][0]
        y = group[0][1]
        plt.scatter(xs, ys, marker='o', s=20, color=c)
        plt.scatter(x, y, marker='x', s=50, color=c)
    plt.show()

def kmeans(nbPoints, nbClusters):
    clusters = {}
    points = np.random.rand(nbPoints, 2)
    initialClusters(clusters, nbClusters)
    old_clusters = {}
    nbEpoch = 1
    run = True
    while run:
        resetGroupsPoints(clusters)
        for n in points:
            centroid(clusters, n)
        afficheClustering(clusters, nbClusters, 'Epoch n°{}'.format(nbEpoch))
        run = isDifferent(clusters, old_clusters)
        old_clusters = copy.deepcopy(clusters)
        moyCluster(clusters)
        nbEpoch += 1
    afficheClustering(clusters, nbClusters, 'Final epoch')
    print("La variance : " + str(evaluation(clusters, nbPoints)))

def evaluation(clusters, nbPoints):
    l = 0
    for c, pts in clusters.values():
        for i in pts:
            l += distanceEuclidienne(i, c)
    l = l / nbPoints
    return l
        

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print('La commande est : python3 KMeans.py <nbPoints> <nbGroup>')
        exit(0)
    nbPoints = int(sys.argv[1])
    nbGroups = int(sys.argv[2])
    kmeans(nbPoints, nbGroups)
