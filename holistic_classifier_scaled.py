
import csv
from matplotlib.pyplot import *
from pyhull.convex_hull import ConvexHull
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import statistics
import operator
from mpl_toolkits import mplot3d
import sklearn
import scipy
#read training dataset from imput csv file
cluster1 = np.array([])
cluster2 = np.array([])
def read_data(cluster_num, n_dimensions):
    #load data into list of lists, then convert to numpy array.
    global cluster1
    global cluster2
    input_data = []
    if cluster_num == 1:
        csvfile = open('cluster1.csv', newline = '')
        reader = csv.reader(csvfile, delimiter=' ')
    elif cluster_num == 2:
        csvfile = open('cluster2.csv', newline = '')
        reader = csv.reader(csvfile, delimiter=' ')

    for row in reader:
        #Try/Except in case data file has a row containing information/titles
        try:
            row = row[0].split(',')
            dimension_number = 0
            point = []
            for i in row:
                point.append(float(i))
            #get rid of [:-1] if the data file does not have a cloumn of indexes.
            if n_dimensions == '2D':
                input_data.append(point[0:2])
            else:
                input_data.append(point)

        except:
            pass

    if cluster_num == 1:
        cluster1 = np.array(input_data)
        return cluster1
    elif cluster_num == 2:
        cluster2 = np.array(input_data)
        return cluster2


#execute convex hull algorithm using pyhull
def convex_hull(cluster):
    list_cluster = cluster.tolist()
    hull = ConvexHull(list_cluster, True)
    #STORE N DIMENSIONAL CONVEX HULL
    cluster_hull = []
    for vertex in hull.vertices:
        #"point" is the index of the coordinate at a particular dimension.
        for point in vertex:
            cluster_hull.append(list_cluster[point])

    #return a numpy array
    cluster_hull = np.array(cluster_hull)
    return cluster_hull


#create function that isolates inner faces of each cluster
def inner_face(cluster_num, cluster_hull):
#find the cluster centers (average of all data points in a cluster)
    #TODO:Consider using median to minimize skewing from outlier data points
    #calculate distance between each point and opposing cluster center
    if cluster_num == 1:
        opposing_center = np.mean(cluster2, axis=0).tolist()
    else:
        opposing_center = np.mean(cluster1, axis=0).tolist()

    distance_matrix = []
    for point in cluster_hull:
        dist = np.linalg.norm(point-opposing_center)
        distance_matrix.append(dist)

    #if a points distance is below median, include in the interior face.
    median_distance = statistics.mean(distance_matrix)
    inner_face = []
    for distance in distance_matrix:
        if distance < median_distance:
            #add a check to prevent duplicates
            if cluster_hull.tolist()[distance_matrix.index(distance)] not in inner_face:
                inner_face.append(cluster_hull.tolist()[distance_matrix.index(distance)])

    if cluster_num == 1:
        cluster1_face = inner_face
        return cluster1_face
    elif cluster_num == 2:
        cluster2_face = inner_face
        return cluster2_face

def hull_orientation(cluster_hull1, cluster_hull2):
    #take convex hull of the 2 convex hulls
    cluster_hull1 = cluster_hull1.tolist()
    cluster_hull2 = cluster_hull2.tolist()
    combined_hulls = np.vstack((cluster_hull1,cluster_hull2)).tolist()
    convex_megahull = ConvexHull(combined_hulls)


    c1_connecting_points = []
    c1_deltas = []
    c2_connecting_points = []
    c2_deltas = []
    for vertex in convex_megahull.vertices:
        #find and isolate the verticies than joins both of the hulls
        #seperate vertex points depending on its cluster or orgin
        if (combined_hulls[vertex[0]] in cluster_hull1 and combined_hulls[vertex[1]] in cluster_hull2):
            c1_connecting_points.append(combined_hulls[vertex[0]])
            c2_connecting_points.append(combined_hulls[vertex[1]])
        elif (combined_hulls[vertex[1]] in cluster_hull1 and combined_hulls[vertex[0]] in cluster_hull2):
            c1_connecting_points.append(combined_hulls[vertex[1]])
            c2_connecting_points.append(combined_hulls[vertex[0]])

    #find deltas for each cluster
    for i in range(0,len(c1_connecting_points[0])):
        c1_deltas.append(np.linalg.norm(c1_connecting_points[0][i]-c1_connecting_points[1][i]))
        c2_deltas.append(np.linalg.norm(c2_connecting_points[0][i]-c2_connecting_points[1][i]))

    combined_deltas = np.vstack((c1_deltas,c2_deltas)).tolist()
    #find average delta for each dimension across both clusters
    average_deltas = np.mean(combined_deltas, axis=0).tolist()
    #find dimension with the largest delta
    max_delta = max(average_deltas)

    #find bounds
    c1_bounds = []
    c2_bounds = []

    for i in range(0, len(c1_connecting_points[0])):
        c1_bounds.append([c1_connecting_points[0][i], c1_connecting_points[1][i]])
        c2_bounds.append([c2_connecting_points[0][i], c2_connecting_points[1][i]])

    return [average_deltas.index(max_delta), c1_bounds, c2_bounds]


    #find x and y bounds for each cluster
    #if delta y bounds > than delta x bounds, sort the hulls vertically
    #else, sort them horizontally.
        #Increases accuracy of algo

def boundary_points(cluster1_face, cluster2_face, orientation):
    if len(cluster1_face) <= len(cluster2_face):
        cluster2_face = cluster2_face[0:len(cluster1_face)]
    else:
        cluster1_face = cluster1_face[0:len(cluster2_face)]

    boundary_points = []

    for i in range(0, len(cluster1_face)):
        pair = [cluster1_face[i], cluster2_face[i]]
        point = np.mean(pair, axis=0).tolist()
        boundary_points.append(point)

    return boundary_points

def func_to_fit(ind_v, *params):
    # out is the function -> ax^4 + by^3+ cz^2 + d
    #this function needs to handle a variable amount of parameters depending on training dataset
    #used loops to handle this. Ugly, but works
    try:
        ind_v = ind_v.tolist()
    except:
        pass
    #there is a case where 2nd arg is a tuple
    if type(params[0]) == list:
        params = (params[0])

    out = 0
    for i in params:
        try:
            out = out + (i*(ind_v[params.index(i)]))**(params.index(i) + 1)
        except:
            out = out + i


    return out
def fit_hyperplane(points):
    first_point = points[0]
    p0 = []
    for i in range(0, len(first_point) + 1):
        p0.append(0.00000000005)

    ydata = []
    for point in points:
        ydata.append(point[-1])


    optimized_params = scipy.optimize.curve_fit(func_to_fit, np.array(points)[:,:-1].T, np.array(ydata), np.array(p0))
    print(optimized_params[0])

    if len(points[0]) > 2:
        x = []
        y = []
        for point in points:
            x.append(point[0])
            y.append(point[1])
        z = []
        for point in points:
            z.append(func_to_fit(point, list(optimized_params[0])))
        ax = plt.axes(projection='3d')
        triang = mtri.Triangulation(x, y)
        ax.plot_trisurf(triang, z, cmap='jet')
        ax.scatter(x,y,z, marker='.', s=10, c="black", alpha=0.5)
        ax.view_init(elev=60, azim=-45)
        plt.show()
    else:
        x = []
        y = []
        for point in points:
            x.append(point[0])
            #y.append(func_to_fit(point[0], list(optimized_params[0])))
            y.append(point[1])
        plt.plot(x,y, 'c--')
        plt.show()

    return optimized_params[0]

def classify_point(point, hyperplane):
    #enter independent variables into hyperplane function.
    boundary = func_to_fit(point[:-1],list(hyperplane))
    test_point = point[-1]
        #compare point_coordinate[-1] to output of hyperplane
        #assign to side 1 or side 2
    if boundary > test_point:
        return 'class1'
    elif boundary < test_point:
        return 'class2'
'------------------------------------------------------------------------------'
n_dimensions = '3D'
read_data(1, n_dimensions)
read_data(2, n_dimensions)

cluster1_hull = convex_hull(cluster1)
cluster2_hull = convex_hull(cluster2)
orientation = hull_orientation(np.array(cluster1_hull), np.array(cluster2_hull))

cluster1_face = inner_face(1, cluster1_hull)
cluster2_face = inner_face(2, cluster2_hull)

cluster1_face = sorted(cluster1_face, key=operator.itemgetter(orientation[0]), reverse=False)
cluster2_face = sorted(cluster2_face, key=operator.itemgetter(orientation[0]), reverse=False)

boundary_points = boundary_points(cluster1_face, cluster2_face, orientation[0])
hyperplane = fit_hyperplane(boundary_points)

for p2 in cluster2.tolist():
    if n_dimensions == '3D':
        print(classify_point(p2[0:2], hyperplane))
    else:
        print(classify_point( p2, hyperplane))
#TODO: FIX THE CURVE FITTING PART, VERY INNACURATE
