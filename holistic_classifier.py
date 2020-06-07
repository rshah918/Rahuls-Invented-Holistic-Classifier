
from matplotlib.pyplot import *
from dionysus_test import alpha_shape
import numpy as np
from scipy.spatial import ConvexHull
import csv
from operator import itemgetter
def concave_hull(points, cluster):
    # Constructing the input point data
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    points = np.vstack([x, y]).T
    # Computing the alpha shape
    edges = alpha_shape(points,alpha=9000, only_outer=True)
    hull_points = []
    x = []
    y = []
    #reformat into list of coordinates
    for i, j in edges:
        hull_points.append([points[[i, j][1], 0].tolist(), points[[i, j][1], 1].tolist()])
    return hull_points

def pw_function_generator(points,cluster_num):
    pw_function_cluster1 = []
    pw_function_cluster2 = []
    concave_hull = points
    #describe each cluster outline by a peicewise function.
        #will be used later to isolate interior faces of each cluster
    #TODO: replace this with a library for optimization and scalability
    #format for peicewise function:
        #{((X1, X2), (Y1, Y2)): (SLOPE, Y_INT)})

    for point in concave_hull:
        first_point = point
        if first_point == concave_hull[-1]:
            second_point = concave_hull[0]
        else:
            second_point = concave_hull[concave_hull.index(point) + 1]
        pair = [first_point, second_point]
        try:
            slope = (second_point[1]-first_point[1]) / (second_point[0]-first_point[0])
        except:
            slope = 'V'
        if slope != 'V':
            y_int = first_point[1] - (slope * first_point[0])
        else:
            y_int = 'V'
        x_bound = (first_point[0], second_point[0])
        y_bound = (first_point[1], second_point[1])
        peice = {(x_bound, y_bound):(slope, y_int)}
        if cluster_num == 1:
            pw_function_cluster1.append(peice)
        elif cluster_num == 2:
            pw_function_cluster2.append(peice)
        else:
            print('Cluster Number must be either "1" or "2"')
            #returns concave hull represented as a giant peicewise function
    if cluster_num == 1:
        return pw_function_cluster1
    elif cluster_num == 2:
        return pw_function_cluster2

def pw_function_executer(cluster,x_input, y_input):
    #Runs input through peicewise function, returns output

    outputs = []
    if y_input == None:
        for peice in cluster:
            first_bound = list(peice.keys())[0][0][0]
            second_bound = list(peice.keys())[0][0][1]
            if (x_input > first_bound and x_input < second_bound and (first_bound-second_bound != 0)) or (x_input < first_bound and x_input > second_bound and (first_bound-second_bound != 0)):
                slope = list(peice.values())[0][0]
                y_int = list(peice.values())[0][1]
                if slope != 'V':
                    output = (slope * x_input) + (y_int)
                else:
                    output = 'V'
                outputs.append(output)
        if len(outputs) > 1:
            float_outputs = []
            for element in outputs:
                if element != 'V':
                    float_outputs.append(element)
            outputs = [max(float_outputs), min(float_outputs)]
            return outputs
    if x_input == None:
        for peice in cluster:
            first_bound = list(peice.keys())[0][1][0]
            second_bound = list(peice.keys())[0][1][1]
            if (y_input > first_bound and y_input < second_bound) and (first_bound-second_bound != 0)or (y_input < first_bound and y_input > second_bound and (first_bound-second_bound != 0)):
                slope = list(peice.values())[0][0]
                y_int = list(peice.values())[0][1]
                if slope != 'V':
                    try:
                        output = (y_input - y_int)/slope
                    except:
                        output = (y_int)
                else:
                    output = 'V'
                outputs.append(output)
        if len(outputs) > 1:
            float_outputs = []
            for element in outputs:
                if element != 'V':
                    float_outputs.append(element)
            outputs = [max(float_outputs), min(float_outputs)]
            return outputs


def inter_cluster_boundary(cluster1x, cluster1y, cluster2x, cluster2y):
    mega_clusterx = cluster1x + cluster2x
    mega_clustery = cluster1y + cluster2y
    points = np.vstack([mega_clusterx, mega_clustery]).T
    hull = ConvexHull(points)
    convex_hull = []
    for index in hull.vertices:
        convex_hull.append([float(mega_clusterx[index]),float(mega_clustery[index])])
    cluster1points = np.vstack([cluster1x, cluster1y]).T
    boundary_targets = []
    for points in convex_hull:
        if points in cluster1points.tolist():
            try:
                if convex_hull[convex_hull.index(points) + 1] not in cluster1points.tolist():
                    boundary_targets.append([points, convex_hull[convex_hull.index(points) + 1]])
            except:
                if convex_hull[0] not in cluster1points.tolist():
                    boundary_targets.append([points, convex_hull[0]])
        elif points not in cluster1points.tolist():
            try:
                if convex_hull[convex_hull.index(points) + 1] in cluster1points.tolist():
                    boundary_targets.append([convex_hull[convex_hull.index(points) + 1], points])
            except:
                if convex_hull[0] in cluster1points.tolist():
                    boundary_targets.append([convex_hull[0], points])
    cluster1_Xbounds = []
    cluster1_Ybounds = []
    cluster2_Xbounds = []
    cluster2_Ybounds = []
    cluster1_Xbounds.append(boundary_targets[0][0][0])
    cluster1_Xbounds.append(boundary_targets[1][0][0])
    cluster1_Ybounds.append(boundary_targets[0][0][1])
    cluster1_Ybounds.append(boundary_targets[1][0][1])
    cluster2_Xbounds.append(boundary_targets[0][1][0])
    cluster2_Xbounds.append(boundary_targets[1][1][0])
    cluster2_Ybounds.append(boundary_targets[0][1][1])
    cluster2_Ybounds.append(boundary_targets[1][1][1])

    xdiff = abs((cluster1_Xbounds[1] - cluster1_Xbounds[0])) + abs((cluster2_Xbounds[0] - cluster2_Xbounds[1]))
    ydiff = abs((cluster1_Ybounds[1] - cluster1_Ybounds[0])) + abs((cluster2_Ybounds[0] - cluster2_Ybounds[1]))
    if xdiff > ydiff:
        idealbounds = [cluster1_Xbounds, cluster2_Xbounds,'x']
        return idealbounds
    elif xdiff < ydiff:
        idealbounds = [cluster1_Ybounds, cluster2_Ybounds,'y']
        return idealbounds
    else:
        idealbounds = [cluster1_Xbounds, cluster2_Xbounds,'x']
        return idealbounds

def interior_points(points1, points2, x, y, x2, y2):
    cluster1set1 = []
    cluster1set2 = []
    cluster2set1 = []
    cluster2set2 = []
    idealbounds = inter_cluster_boundary(x,y,x2,y2)
    points = [points1, points2]
    index_counter = -1
    for bound_set in idealbounds:
        index_counter = index_counter + 1
        if index_counter != 2:
            cluster_number = index_counter + 1
            #run concave hull algorithm, then feed output to function generator
            hull = concave_hull(points[index_counter], cluster_number)
            func = pw_function_generator(hull, cluster_number)
            test_inputs = np.linspace(bound_set[0], bound_set[1], 45).tolist()
            for input_ in test_inputs:
                if idealbounds[2] == 'x':
                    output = pw_function_executer(func, input_, None)
                    if index_counter == 0 and output != None:
                        if output[0] != 'V':
                            cluster1set1.append([input_, output[0]])
                        if output[1] != 'V':
                            cluster1set2.append([input_, output[1]])
                    if index_counter == 1 and output != None:
                        if output[0] != 'V':
                            cluster2set1.append([input_, output[0]])
                        if output[1] != 'V':
                            cluster2set2.append([input_, output[1]])
                if idealbounds[2] == 'y':
                    output = pw_function_executer(func, None,input_)
                    if index_counter == 0 and output != None:
                        if output[0] != 'V':
                            cluster1set1.append([output[0], input_])
                        if output[1] != 'V':
                            cluster1set2.append([output[1], input_])
                    if index_counter == 1 and output != None:
                        if output[0] != 'V':
                            cluster2set1.append([output[0], input_])
                        if output[1] != 'V':
                            cluster2set2.append([output[1], input_])

    cluster1_mean = [sum(x)/len(x), sum(y)/len(y)]
    cluster2_mean = [sum(x2)/len(x2), sum(y2)/len(y2)]
    c1s1d = 0
    c1s2d = 0
    c2s1d = 0
    c2s2d = 0
    try:
        for element in cluster1set1:
            c1s1d += (((element[0] - cluster2_mean[0])**2) + ((element[1]- cluster2_mean[1])**2))**(1/2)
    except:
        pass
    try:
        for element in cluster1set2:
            c1s2d += (((element[0] - cluster2_mean[0])**2) + ((element[1]- cluster2_mean[1])**2))**(1/2)
    except:
        pass
    try:
        for element in cluster2set1:
            c2s1d += (((element[0] - cluster1_mean[0])**2) + ((element[1]- cluster1_mean[1])**2))**(1/2)
    except:
        pass
    try:
        for element in cluster2set2:
            c2s2d += (((element[0] - cluster1_mean[0])**2) + ((element[1]- cluster1_mean[1])**2))**(1/2)
    except:
        pass
    cluster1_interior_set = []
    cluster2_interior_set = []
    if c1s1d < c1s2d:
        cluster1_interior_set = cluster1set1
    else:
        cluster1_interior_set = cluster1set2
    if c2s1d < c2s2d:
        cluster2_interior_set = cluster2set1
    else:
        cluster2_interior_set = cluster2set2
    interiorsets = [cluster1_interior_set, cluster2_interior_set, idealbounds[2]]
    return interiorsets

def boundary_line(interiorsets, x,y,x2,y2, bound_axis):
    #Boundary midpoints depend on how the interior points of each cluster are arranged.
    #Below I find the optimum midpoint arrangement
    #organize midpoints from left to right or up to down depending on the bounds
    cluster1_interior_setX = sorted(interiorsets[0], key=itemgetter(0))
    cluster2_interior_setX = sorted(interiorsets[1], key=itemgetter(0))
    cluster1_interior_setY = sorted(interiorsets[0], key=itemgetter(1))
    cluster2_interior_setY = sorted(interiorsets[1], key=itemgetter(1))
    midpoint_set1 = []
    if bound_axis == 'x':
        if len(cluster1_interior_setX) < len(cluster2_interior_setX):
            for point in cluster1_interior_setX:
                index = cluster1_interior_setX.index(point)
                midpointx = (point[0] + cluster2_interior_setX[index][0])/2
                midpointy = (point[1] + cluster2_interior_setX[index][1])/2
                midpoint = (midpointx, midpointy)
                midpoint_set1.append(midpoint)
        else:
            for point in cluster2_interior_setX:
                index = cluster2_interior_setX.index(point)
                midpointx = (point[0] + cluster1_interior_setX[index][0])/2
                midpointy = (point[1] + cluster1_interior_setX[index][1])/2
                midpoint = (midpointx, midpointy)
                midpoint_set1.append(midpoint)
    elif bound_axis == 'y':
        if len(cluster1_interior_setY) < len(cluster2_interior_setY):
            for point in cluster1_interior_setY:
                index = cluster1_interior_setY.index(point)
                midpointx = (point[0] + cluster2_interior_setY[index][0])/2
                midpointy = (point[1] + cluster2_interior_setY[index][1])/2
                midpoint = (midpointx, midpointy)
                midpoint_set1.append(midpoint)
        else:
            for point in cluster2_interior_setY:
                index = cluster2_interior_setY.index(point)
                midpointx = (point[0] + cluster1_interior_setY[index][0])/2
                midpointy = (point[1] + cluster1_interior_setY[index][1])/2
                midpoint = (midpointx, midpointy)
                midpoint_set1.append(midpoint)
    midpoint_set1 = sorted(midpoint_set1, key=itemgetter(1))
    points = np.array(midpoint_set1)
    y = points[:,0]
    x = points[:,1]
    # calculate polynomial
    z = np.polyfit(x, y,2)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(-2, 2, 70)
    y_new = f(x_new)

    plot(y_new, x_new)
    xlim([x[0]-1, x[-1] + 1 ])

    return midpoint_set1

def execute(points1, points2, x, y, x2, y2):
    #find interior faces of each cluster
    interior_set = interior_points(points1, points2, x, y, x2, y2)
    #separate the 2 faces for the sake of plotting
    cluster1_interior_set = interior_set[0]
    cluster2_interior_set = interior_set[1]
    #pass in the interior sets to find the mid-boundary points
    midpoints = boundary_line(interior_set, x,y,x2,y2, interior_set[2])
    plot(x,y,'go')
    plot(x2,y2,'bo')
    c1x = []
    c1y = []
    for element in cluster1_interior_set:
        c1x.append(element[0])
    for element in cluster1_interior_set:
        c1y.append(element[1])
    plot(c1x, c1y, 'g--')
    c2x = []
    c2y = []
    for element in cluster2_interior_set:
        c2x.append(element[0])
    for element in cluster2_interior_set:
        c2y.append(element[1])
    plot(c2x, c2y, 'b--')
    mx = []
    my = []
    for element in midpoints:
        mx.append(element[0])
    for element in midpoints:
        my.append(element[1])
    plot(mx, my, 'ro')
    show()

'----------------------------------------------------------------------------'
csvfile = open('cluster1.csv', newline = '')
reader = csv.reader(csvfile, delimiter=' ')
x = []
y = []
for row in reader:
    try:
        row = row[0].split(',')
        x.append(float(row[0]))
        y.append(float(row[1]))
    except:
        pass
csvfile = open('cluster2.csv', newline = '')
reader = csv.reader(csvfile, delimiter=' ')
x2 = []
y2 = []
for row in reader:
    try:
        row = row[0].split(',')
        x2.append(float(row[0]))
        y2.append(float(row[1]))
    except:
        pass
points1 = np.vstack([x,y]).T.tolist()
points2 = np.vstack([x2,y2]).T.tolist()
execute(points1, points2, x, y, x2, y2)
