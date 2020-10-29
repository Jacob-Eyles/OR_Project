import numpy as np
import pandas as pd
import copy

times = pd.read_csv('WarehouseDurations.csv', header=0, index_col=0, parse_dates=True)
mon_fri_demands = pd.read_csv('MonFriDemand.csv', header=0, index_col=0, parse_dates=True)
sat_demands = pd.read_csv('SatDemand.csv', header=0, index_col=0, parse_dates=True)
 
def Cheapest_insertion(nodes,start,traveltimes,demand):
    '''
    '''
    allpaths = []
    alltime = []
    ######################################################
    addtimes = copy.deepcopy(traveltimes) 
    
    # Go through every column and add on the corresponding extra time onto original time depending on pallet demands
    for col in addtimes.columns:
        if col == 'Distribution South' or col == 'Distribution North':
            continue
        else:
            
            addtimes.loc[:, col] += mon_fri_demands.loc[col, 'Mean']*600    # 10 minutes = 600 seconds
            addtimes.loc[col, col] = 0

    traveltimes = addtimes
    
    for path in nodes:
        # print(path)
    # for i in range(1,len(addtimes)):
    #     for j in range(1,len(addtimes)):
    #         if i==j:
    #             addtimes.loc[i,j] = 0
    # Maybe we can make the weekday or saturday demands a input for a function so we don't have to hard code the time dataframe everytime.
    # But yeah for now I coded the rest using the variable name "times" so I just switched the name back
    
    # print(traveltimes)
    # An example of the final output from your route code.
    # all_paths=[['Noel Leeming Albany', 'Noel Leeming Botany', 'Noel Leeming Glenfield Clearance', 'Noel Leeming Henderson'], ['Noel Leeming Albany', 'Noel Leeming Botany', 'Noel Leeming Glenfield Clearance', 'Noel Leeming Lunn Avenue'], ['Noel Leeming Albany', 'Noel Leeming Botany', 'Noel Leeming Glenfield Clearance', 'Noel Leeming Manukau Supa Centre'], ['Noel Leeming Albany', 'Noel Leeming Botany', 'Noel Leeming Glenfield Clearance', 'Noel Leeming Manukau Westfield']]

    # print(nodes1==nodes)
    # We could probably also have to make the distribution center an input asw. For now I chose it as Distribution North and hard coded it in.
        nodes1 = list(np.copy(path))
        unsorted_path = nodes1
        unsorted_path.insert(0,start)
        unsorted_path.append(start)
        # I need this later to transform the edges into a one-way path.
        unsorted_path1 = copy.deepcopy(unsorted_path)

        # Cheapest Insertion Algorithm. Just going through all the distances and finding the shortest path everytime
        # a new location is added. I've done a few tests and it seems to be correct.

        initial_location = start
        unsorted_path.remove(initial_location)

        i = initial_location
        j = unsorted_path[0]

        distance_ij = traveltimes.loc[i, j]

        for point in unsorted_path:
            if traveltimes.loc[i, point] < distance_ij:
                distance_ij = traveltimes.loc[i, point]
                j = point

        unsorted_path.remove(j)

        edges = [(i,j)]

        visited = []
        visited.append(i)
        visited.append(j)

        while len(unsorted_path) > 0:
            i = visited[0]
            k = unsorted_path[0]
            distance_ki = traveltimes.loc[k, i]

            for point in unsorted_path:
                for c in visited:
                    distance = traveltimes.loc[point, c]
                    if distance < distance_ki:
                        k = point
                
            i = edges[0][0]
            j = edges[0][1]
            c_min = traveltimes.loc[i,k] + traveltimes.loc[k, j] - traveltimes.loc[i,j]
            
            for e in edges:
                current_i = e[0]
                current_j = e[1]
                distance = traveltimes.loc[current_i, k] + traveltimes.loc[k, current_j] - traveltimes.loc[current_i, current_j]
                if distance < c_min:
                    c_min = distance
                    i = current_i
                    j = current_j

            edges.remove((i,j))
            edges.append((i,k))
            edges.append((k,j))

            visited.append(k)
            unsorted_path.remove(k)
        # print(edges)
        total_time = 0
        for e in edges:
            i = e[0]
            j = e[1]
            total_time += traveltimes.loc[i,j]
        # print(total_time)
        if total_time>14400:
            allpaths.append([None])
            alltime.append(np.inf)
            continue

        # The code gives me tuples of starting node and destination node. So I needed to format it into
        # a list of path
        ordered_path = []

        for e in edges:
            # print(e)
            if e[0] == start:
                initial = e[0]
                next_point = e[1]
                unsorted_path1.remove(initial)
                ordered_path.append(initial)
                
                break
        # print(next_point)
        # print(ordered_path)
        # print(edges)
        while len(ordered_path) != (len(path)+1):
        # for j in range(len(nodes)):
            # print(next_pot)
            for e in edges:   
                # print(e[0]==next_point,e) 
                    # if len(ordered_path) == (len(nodes)+1):
                    #     break
                    # else:
                # print(e)
                # print(next_point)
                if e[0] == next_point:
                    # unsorted_path1.remove(e[0])
                    ordered_path.append(e[0])
                    next_point = e[1]
                    # print(next_point)
                # print('unsorted',unsorted_path1)
                # print('ordered',ordered_path)
            
                if len(ordered_path) == len(path)+1:
                    break
                # print('here')
                
 
        ordered_path.append(initial)
        allpaths.append(ordered_path)
        alltime.append(total_time)
    # print("Total Time Taken:", total_time)
    # print("Quickest Path:", ordered_path)

    return allpaths, alltime