############################################

# imports
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pulp import *
import cheapest_insertion_algorithm
from IntegerProblemFormulation import *

#############################################

# take in data:
demands = pd.read_csv('demandDataUpdated.csv')
distances = pd.read_csv('WarehouseDistances.csv')
times = pd.read_csv('WarehouseDurations.csv')
coords = pd.read_csv('WarehouseLocations.csv')

# for col in times.columns:
#         times.loc[:, col] += mon_fri_demands.loc[col, 'Mean']*600    # 10 minutes = 600 seconds


#############################################

# FUNCTIONS:
def plot_nodes(dictionary, name, route = [],save=True):
    """ Plots a given set of nodes in Auckland.

    Parameters
    ----------
    dictionary : dictionary
        Dictionary object containing node names, longitude, latitudes.
    save: str or None
        If a string is provided, then saves the figure to the path given by the string
        If None, then displays the figure to the screen
    """
    # print(dictionary)
    lats = [dictionary[p][1] for p in dictionary.keys()]
    # print(len(lats))
    lngs = [dictionary[p][0] for p in dictionary.keys()]
    # print(len(lngs))
    sector = [dictionary[p][2] for p in dictionary.keys()]
    # print(len(sector))

    fig, ax = plt.subplots(figsize = (18,15))
    ext = [174.48866, 175.001869, -37.09336, -36.69258]
    plt.imshow(plt.imread("akl_zoom.png"), extent=ext)

    
    for r in route:
        path=[]
        for i in r:
            path.append([lngs[i],lats[i]]) 

        if sector[r[1]] == 'A':
            colour = 'y'
        elif sector[r[1]] == 'B':
            colour = 'r'
        elif sector[r[1]] == 'C':
            colour = 'b'
        else:
            colour = 'm'
        plt.plot([p[0] for p in path[:]],[p[1] for p in path[:]],color = colour)

    for i in range(len(lats)):

        if sector[i] == 'A':
            colour = 'y'
        elif sector[i] == 'B':
            colour = 'r'
        elif sector[i] == 'C':
            colour = 'b'
        else:
            colour = 'm'

        c1 = plt.Circle((lngs[i], lats[i]), 0.003, color=colour)
        ax.add_artist(c1)

        for p in dictionary.keys():
            if p[0] == 'D':
                c2 = plt.Circle((dictionary[p][0], dictionary[p][1]), 0.004, color='y',fill=True)
                ax.add_artist(c2)
                c3 = plt.Circle((dictionary[p][0], dictionary[p][1]), 0.004, color='k',fill=False)
                ax.add_artist(c3)

        c4 = plt.Circle((lngs[i], lats[i]), 0.003, color='k',fill=False)
        ax.add_artist(c4)

    if save:
        plt.savefig(name, dpi=300)
    else:
        plt.show()

def cut(dictionary, lngsplit = 174.78, latsplitA=-36.81, latsplitB=-36.98, band=False):
    ''' Seperate a subset of a network by latitude.
	
		Parameters
		----------
		dictionary : dictionary
            Dictionary object containing node names, longitude, latitudes.
        lngsplit : float
			longitude to seperate the nodes into left and right groups.
		latsplitA : float
			latitude to seperate the left nodes from lngsplit.
        latsplitB : float
			latitude to seperate the right nodes from lngsplit.
			
		Returns
		-------
		dictionary : dictionary
			Dictionary object containing node names, longitude, latitudes.
            Additional attributes for the 'quadrant' are added.
		Notes
		-----
		If no input for latsplit, the average of all home 
        latitudes in the homes subset is used. 
    '''
    # If we want another way to determine the splits, put here (rather than ad-hoc).

    # --->

    # based on splits, assign a quadrant label to the dictionary key.
    if band:
        for store in dictionary.keys():
            if dictionary[store][1] > latsplitA:
                dictionary[store].append('A')
            else:
                dictionary[store].append('B')

            if dictionary[store][2] == 'B' and dictionary[store][1] < latsplitB:
                dictionary[store][2] = 'C'
    else:
        for store in dictionary.keys():
            # if dictionary[store][0] > lngsplit:
            dictionary[store].append('B')
            # else:
            #     dictionary[store].append('A')
            
            # if dictionary[store][2] == 'A' and dictionary[store][1] < latsplitA:
            #     dictionary[store][2] = 'C'

            # if dictionary[store][2] == 'B' and dictionary[store][1] < latsplitB:
            #     dictionary[store][2] = 'D'
        
    return dictionary


def subset_sum(weights1, names, target, partialnames=[], partialdemands=[],paths=[]):
    '''
    Finds all paths in a subset which add to 20 or slightly less pallets.
    '''
    # This is recursive...

    # sum of new set.
    s = sum(partialdemands)

    # want set sum to be less than or equal 20 pallets and keep ones which are still large.
    if s <= 20 and s>= 1: 
        paths.append(partialnames)

    if s >= target:
        return  # if we reach the number why bother to continue

    # recursively loop through nodes. Keep moving along remaining nodes to check for each loop...
    for i in range(len(names)):
        n = weights1[names[i]][0]
        nm = names[i]
        remaining = names[i+1:]
        subset_sum(weights,remaining, target, partialnames + [nm], partialdemands + [n],paths) 
    
    return paths

def writematrix(namem,matrix):
    '''
    '''
    namef = namem + '.txt'
    file1 = open(namef,'w')
    for L in [matrix[i][:] for i in range(len(matrix))]:
        LL = ','.join(str(i) for i in L) + '\n'
        file1.write(LL)
    file1.close()

def writeroutes(routenumbers,orderedpaths,nodes,storelist,filenamer,filenameno):
    '''
    '''
    r1 = []
    for r in routenumbers:
        index = int(r[5:])-1
        # print(index)
        # print(orderedpaths)
        path = orderedpaths[index]
        rn=[]
        for t in path:
            for p in range(len(list(nodes))):
                if list(nodes)[p] in [t]:
                    rn.append(p)
        r1.append(rn)
    
    nameno = filenameno + '.txt'
    fileno1 = open(nameno,'w')
    for j in r1:
        L = [i for i in j]
        LL = ','.join(str(i) for i in L) + '\n'
        fileno1.write(LL)
    fileno1.close()

    namer = filenamer + '.txt'
    filer1 = open(namer,'w')
    for j in r1:
        L = [storelist[i] for i in j]
        LL = ','.join(str(i) for i in L) + '\n'
        filer1.write(LL)
    filer1.close()

    return r1

##################################################################################

if __name__ == '__main__':

    nodes = {}
    Warehousenodes = {}
    weights = {}
    
    # make coordinate dictionary
    for i in range(len(coords)):
        row = coords.iloc[i]
        nodes[row['Store']] = [row['Long'],row['Lat']]

    # make average demand dictionary for weekend and week.
    for i in range(len(demands)):
        weekdemands = []
        enddemands = []
        demanddates = demands.iloc[i]

        for j in range(len(demanddates)):
            if j%7 in [1,2,3,4,5]:
                weekdemands.append(demanddates[j])

            if j%7 == 6:
                enddemands.append(demanddates[j])

        weights[demanddates[0]] = [int(np.ceil(np.mean(weekdemands))),int(np.ceil(np.mean(enddemands)))]
 
    # create a subset of the names for testing (I think it works now).
    names = list(weights)
    
    # names = list(weights)
    A,B,C,D = [],[],[],[]
    Aw,Bw,Cw,Dw = [],[],[],[]

    # cut up nodes.
    nodes = cut(nodes)

    for i in range(2,len(coords)):
        row = coords.iloc[i]
        quad = nodes[row['Store']][2]
        if quad == 'A':
            A.append(row['Store'])

        elif quad == 'B':
            B.append(row['Store'])

        elif quad == 'C':
            C.append(row['Store'])

        elif quad == 'D':
            D.append(row['Store'])
    
    for i in list(nodes):
        if i[0] in ['T','D']:
            Warehousenodes[i] = [nodes[i][0],nodes[i][1],nodes[i][2]]

    # find all paths in the subset which sum to or almost to 20.
    # (can adjust to include all paths if needed for linear program).

    # WEEKDAY:
    pathsA = subset_sum(weights, A, 20,paths = [])
    pathsB = subset_sum(weights, B, 20,paths = [])
    pathsC = subset_sum(weights, C, 20, paths = [])
    pathsD = subset_sum(weights, D, 20,paths = [])
    # WEEKEND:
    weekendnodes = {}

    for j in list(weights):
        if weights[j][1] != 0:
            weekendnodes[j] = [weights[j][1]]
    
    for n in list(weekendnodes):
        if n in A:
            Aw.append(n)
        elif n in B:
            Bw.append(n)
        elif n in C:
            Cw.append(n)
        elif n in D:
            Dw.append(n)
    
    pathsAw = subset_sum(weekendnodes, Aw, 20,paths = [])
    pathsBw = subset_sum(weekendnodes, Bw, 20,paths = [])
    pathsCw = subset_sum(weekendnodes, Cw, 20,paths = [])
    pathsDw = subset_sum(weekendnodes, Dw, 20,paths = [])

    paths = pathsA+pathsC+pathsB+pathsD
    pathsw = pathsAw+pathsCw+pathsBw+pathsDw

    stores = ['Name']+[i for i in list(weights)]+['Cost']
    storesw = ['Name']+[i for i in list(weekendnodes)]+['Cost']
    storeslist = ['Distribution South','Distribution North']+[i for i in list(weights)]
    storeslistw = ['Distribution South','Distribution North']+[i for i in list(weekendnodes)]

    ##############################################
    # USE PATH FINDING ALGORITHM HERE:::
    orderedpaths_week1 = []
    orderedpaths_sat1 = []
    orderedpaths_week2 = []
    orderedpaths_sat2 = []

    times = pd.read_csv('WarehouseDurations.csv', header=0, index_col=0, parse_dates=True)
    mon_fri_demands = pd.read_csv('MonFriDemand.csv', header=0, index_col=0, parse_dates=True)
    sat_demands = pd.read_csv('SatDemand.csv', header=0, index_col=0, parse_dates=True)
 
    orderedpaths1,orderedtimes1 = cheapest_insertion_algorithm.Cheapest_insertion(paths,'Distribution North',times*1.4,mon_fri_demands)
    orderedpaths2,orderedtimes2 = cheapest_insertion_algorithm.Cheapest_insertion(paths,'Distribution South',times*1.4,mon_fri_demands)

    order1 = []
    path1 = []
    order2 = []
    path2 = []

    for i in range(len(orderedpaths1)):
        if orderedtimes1[i] != np.inf and orderedtimes2[i] != np.inf:
            order2.append(orderedtimes2[i])
            path2.append(orderedpaths2[i])

            if orderedtimes1[i]>orderedtimes2[i]:
                order1.append(orderedtimes2[i])
                path1.append(orderedpaths2[i])
            else:
                order1.append(orderedtimes1[i])
                path1.append(orderedpaths1[i])
            
        elif orderedtimes1[i] == np.inf and orderedtimes2[i] == np.inf:
            continue
        
        elif orderedtimes1[i] != np.inf and orderedtimes2[i] == np.inf:
            order1.append(orderedtimes1[i])
            path1.append(orderedpaths1[i])

        elif orderedtimes1[i] == np.inf and orderedtimes2[i] != np.inf:
            order1.append(orderedtimes2[i])
            path1.append(orderedpaths2[i])
            order2.append(orderedtimes2[i])
            path2.append(orderedpaths2[i])
    
    orderedtimes1 = order1
    orderedtimes2 = order2
    orderedpaths1 = path1
    orderedpaths2 = path2

    # MATRIX 1
    matrix1 = np.zeros((len(stores),len(orderedpaths1)+1))
    matrix1 = matrix1.tolist()
    
    for k in range(1,len(orderedpaths1)+1):
        matrix1[0][k] = str(k)

    for j in range(len(stores)):
        matrix1[j][0] = stores[j]

    for p in range(len(orderedpaths1)):
        currentp = orderedpaths1[p]
        for j in range(len(stores)-2):
            for i in currentp:
                if i==stores[j+1]:
                    matrix1[j+1][p+1]=1
    
    # MATRIX 3
    matrix3 = np.zeros((len(stores),len(orderedpaths2)+1))
    matrix3 = matrix3.tolist()
    
    for k in range(1,len(orderedpaths2)+1):
        matrix3[0][k] = str(k)

    for j in range(len(stores)):
        matrix3[j][0] = stores[j]

    for p in range(len(orderedpaths2)):
        currentp = orderedpaths2[p]
        for j in range(len(stores)-2):
            for i in currentp:
                if i==stores[j+1]:
                    matrix3[j+1][p+1]=1
    
    for k in range(len(orderedpaths1)):
        
        matrix1[-1][k+1] = orderedtimes1[k]
        orderedpaths_week1.append(orderedpaths1[k])

    for k in range(len(orderedpaths2)):
       
        matrix3[-1][k+1] = orderedtimes2[k]
        orderedpaths_week2.append(orderedpaths2[k])
        
    # ############
         
    orderedpaths1w,orderedtimes1w = cheapest_insertion_algorithm.Cheapest_insertion(pathsw,'Distribution North',times*1.19,sat_demands)
    orderedpaths2w,orderedtimes2w = cheapest_insertion_algorithm.Cheapest_insertion(pathsw,'Distribution South',times*1.19,sat_demands)

    order1w = []
    path1w = []
    order2w= []
    path2w = []
    # print(orderedpaths1w)

    for i in range(len(orderedpaths1w)):
        if orderedtimes1w[i] != np.inf and orderedtimes2w[i] != np.inf:
            order2w.append(orderedtimes2w[i])
            path2w.append(orderedpaths2w[i])

            if orderedtimes1w[i]>orderedtimes2w[i]:
                order1w.append(orderedtimes2w[i])
                path1w.append(orderedpaths2w[i])
            else:
                order1w.append(orderedtimes1w[i])
                path1w.append(orderedpaths1w[i])
            
        elif orderedtimes1w[i] == np.inf and orderedtimes2w[i] == np.inf:
            continue
        
        elif orderedtimes1w[i] != np.inf and orderedtimes2w[i] == np.inf:
            order1w.append(orderedtimes1w[i])
            path1w.append(orderedpaths1w[i])

        elif orderedtimes1w[i] == np.inf and orderedtimes2w[i] != np.inf:
            order1w.append(orderedtimes2w[i])
            path1w.append(orderedpaths2w[i])
            order2w.append(orderedtimes2w[i])
            path2w.append(orderedpaths2w[i])
    
    orderedtimes1w = order1w
    orderedtimes2w = order2w
    orderedpaths1w = path1w
    orderedpaths2w = path2w

    # print(orderedtimes1w)

    # MATRIX 2
    matrix2 = np.zeros((len(storesw),len(orderedpaths1w)+1))
    matrix2 = matrix2.tolist()
    
    for k in range(1,len(orderedpaths1w)+1):
        matrix2[0][k] = str(k)

    for j in range(len(storesw)):
        matrix2[j][0] = storesw[j]

    for p in range(len(orderedpaths1w)):
        currentp = orderedpaths1w[p]
        for j in range(len(storesw)-2):
            for i in currentp:
                if i==storesw[j+1]:
                    matrix2[j+1][p+1]=1

    # MATRIX 4
    matrix4 = np.zeros((len(storesw),len(orderedpaths2w)+1))
    matrix4 = matrix4.tolist()
    
    for k in range(1,len(orderedpaths2w)+1):
        matrix4[0][k] = str(k)

    for j in range(len(storesw)):
        matrix4[j][0] = storesw[j]

    for p in range(len(orderedpaths2w)):
        currentp = orderedpaths2w[p]
        for j in range(len(storesw)-2):
            for i in currentp:
                if i==storesw[j+1]:
                    matrix4[j+1][p+1]=1

    for l in range(len(orderedpaths1w)):
        
        matrix2[-1][l+1] = orderedtimes1w[l]
        orderedpaths_sat1.append(orderedpaths1w[l])
    
    for l in range(len(orderedpaths2w)):
       
        matrix4[-1][l+1] = orderedtimes2w[l]
        orderedpaths_sat2.append(orderedpaths2w[l])

    # # INPUT ALL THE PATHS
    # # COULD SPLIT AND USE GENERAL ALGORITHM AND THEN COMBINE LATER...
    # # NOW, OUTPUTS SHOULD BE COSTS OF ALL AND LIST OF ORDERED ROUTES SO WE CAN PLOT...
    # ##############################################
 
    writematrix('matrix1',matrix1)
    writematrix('matrix2',matrix2)
    writematrix('matrix3',matrix3)
    writematrix('matrix4',matrix4)

    # ####################################################

    totaltime1,routeno1 = IntegerProgramw('matrix1.txt')
    totaltime2,routeno2 = IntegerProgramw('matrix2.txt')
    totaltime3,routeno3 = IntegerProgramw('matrix3.txt')
    totaltime4,routeno4 = IntegerProgramw('matrix4.txt')

    print(routeno1)
    print(routeno2)
    print(routeno3)
    print(routeno4)

    # for i in routeno3:
    #     print(orderedtimes2[int(i[5:])-1], orderedpaths2[int(i[5:])-1])

    print('\n',totaltime1)
    print(totaltime2)
    print(totaltime3)
    print(totaltime4)

    r1 = writeroutes(routeno1,orderedpaths1,nodes,storeslist,'routes1','rnumbers1')
    r2 = writeroutes(routeno2,orderedpaths1w,Warehousenodes,storeslistw,'routes2','rnumbers2')
    r3 = writeroutes(routeno3,orderedpaths2,nodes,storeslist,'routes3','rnumbers3')
    r4 = writeroutes(routeno4,orderedpaths2w,Warehousenodes,storeslistw,'routes4','rnumbers4')

    print(r1)
    print(r2)
    print(r3)
    print(r4)

    # # plot graphs.
    plot_nodes(nodes,name = 'Week-both centres',route= r1)
    plot_nodes(Warehousenodes,name = 'Weekend-both centres',route= r2)
    plot_nodes(nodes,name = 'Week-South centre',route= r3)
    plot_nodes(Warehousenodes,name = 'Weekend-South centre',route= r4)

    plot_nodes(nodes,name = 'Nodes picture')

    # #####################################################

    # # orderedpath1,orderedtime1 = cheapest_insertion_algorithm.Cheapest_insertion(paths[5],'Distribution North',times,mon_fri_demands)
    # # print(orderedtime1/3600*12*175)