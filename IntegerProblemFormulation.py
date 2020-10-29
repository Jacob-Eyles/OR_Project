import numpy as np
import pandas as pd 
from pulp import *
 
def IntegerProgramw(file):
    #All below here remains constant
    #Finds the number of columns in the file
    with open(file, 'r') as f:
        num_cols = len(f.readline().split(","))

    #Finds the number of rows in the file
    num_row = sum(1 for line in open(file))

    #Import the list of locations
    NodeList = np.genfromtxt(file, dtype=str, delimiter= ',',  skip_header = 1, usecols= 0, skip_footer=1)

    #Import the list of route names

    Routes = np.genfromtxt(file, dtype=str, delimiter= ',', max_rows=1, usecols=range(2, num_cols))[1:]
    # print(Routes)
    # print(len(Routes))



    #import total times from each route
    cols = np.arange(1, num_cols)
    TimeRead = np.genfromtxt(file, dtype=float, delimiter= ',', skip_header = num_row-1, usecols= range(1, num_cols))
    # print(TimeRead)

    Routes = np.genfromtxt(file, dtype=str, delimiter= ',', max_rows=1, usecols=range(1, num_cols))


    #import total times from each route
    TimeRead = np.genfromtxt(file, dtype=float, delimiter= ',', skip_header = num_row-1, usecols=range(1, num_cols))

    Time = pd.Series(TimeRead, index = Routes)

    #Importing a matrix with the rows consisting of the locations and columns
    # of the potential routes. If the location is in each route its value is a 1, and 0 otherwise.
    RouteLocations = np.genfromtxt(file, dtype=int, delimiter= ',', skip_header=1, usecols=range(1, num_cols), skip_footer = 1)

    #Making the route data into a dictionary
    RouteData = makeDict([NodeList, Routes],RouteLocations,0)

    # The 'prob' variable to contain the problem data, wanting to minimise the objective function
    prob = LpProblem("WarehouseProblem", LpMinimize)

    #A dictionary to contain the referenced variables
    route_vars = LpVariable.dicts(name="Rout",indexs=Routes,lowBound=0,upBound=1, cat = 'Integer')
    # print(route_vars['1'])

    #Adding the objective function to prob
    prob += lpSum([Time[i]*route_vars[i] for i in Routes])
    #Making sure every location (excluding D/Cs) gets visited only once
    for i in NodeList:
        prob += lpSum([RouteData[i][j]*route_vars[j] for j in Routes]) >= 1
        prob += lpSum([RouteData[i][j]*route_vars[j] for j in Routes]) <= 1

    #Constraining to only include routes less than or equal to the number of truck available
    prob += lpSum([route_vars[i] for i in Routes]) <= 50

    #Writes the problem data to an .lp file
    prob.writeLP("RouteSelection.lp")

    # The problem is solved using PuLP's choice of solver
    prob.solve()

    #Initalising Values
 
    UsedRoutes = []

    #Prints optimal objective function and names of the variables/routes used

    TotalCost = value(prob.objective)*175/3600
    for v in prob.variables():
        # print(v,v.varValue)
        if v.varValue == 1:
            # UsedRoute = v.name[6:len(v.name)]
            UsedRoutes.append(v.name)
    # print(route_vars)
    return(TotalCost, UsedRoutes)

    # C, R = IntegerProgram('matrix1.txt')
    # print(C, R)

    # TotalCost = value(prob.objective)*2.91666
    # for v in prob.variables():
    #     print(v.name, '=', v.varValue)
    # for v in prob.variables():
    #     if v.varValue > 0:
    #         UsedRoute = v.name
    #         UsedRoutes.append(UsedRoute)
    # #print(prob)
    # y = ['1412','1452','154','1580','1647','1695','2','266','33','657','715','985']
    # checked=[]
    # for j in y:
    #     for i in NodeList:
    #         if RouteData[i][j] == 1:
    #             # print(i)
    #             checked.append(i)
    # # print(NodeList)
    # return(TotalCost, UsedRoutes)
    
# C, R = IntegerProgramw('matrix4.txt')
# print(C, R)

