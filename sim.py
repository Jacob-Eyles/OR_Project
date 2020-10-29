import numpy as np
from fitter import Fitter
import pandas as pd 
from scipy import stats
from random import * 
import seaborn as sns
import matplotlib.pyplot as plt

def DemandSim (Weekday, num):
	'''
	Parameters:
	-----------
	Weekday: binary 
		1 if Monday to Friday demands, 0 if saturday
	Num : integer
		Number of samples
	Returns:
	--------
	StoreDemands: Dictionary
		Demand varitaions of each store
	
	DemandS: array-like
		Demand varitaions of each store in array
	'''
	if Weekday == 1:
		file = "DemandDataMonFri.csv"
		NoofDays = 20
	else:
		file = "DemandDataSat.csv"
		NoofDays = 4
	rangeStores = np.arange(1,NoofDays+1)
	Demand = np.genfromtxt(file, delimiter= ',', usecols=rangeStores, skip_header=1)
	Stores = np.genfromtxt(file, dtype=str, delimiter= ',', usecols=0, skip_header=True)
	DemandS = np.zeros((len(Stores),num))
	
	if Weekday == 1:
		for i in range(len(Stores)):
			selectStore = Demand[i,:]
			DemandFit = stats.norm.fit(selectStore)
			DemandRange = stats.norm.rvs(loc=DemandFit[0], scale=DemandFit[1], size=num)
			DemandS[i,:] = np.around(DemandRange)

	else:
		for i in range(len(Stores)):
			selectStore = Demand[i,:]
			DemandFit = stats.uniform.fit(selectStore)
			DemandRange = stats.uniform.rvs(loc=DemandFit[0], scale=DemandFit[1], size=num)
			DemandS[i,:] = np.around(DemandRange)

	StoresDemands = {Stores[i]: DemandS[i] for i in range(len(Stores))}
	
	return StoresDemands, DemandS

def TrafficFactor(num, run =0):
	'''
	Fits a disturbtion for TomTom traffic congestion on Mon-Fri and Sat.

	Parameters:
	-----------
	num : float
		Number of trafiic congestion variables to return for Weekdays and Saturday.
	run : binary
		1 if you want to rerun Fitter function

	Returns:
	--------
	MonFriRV : array-like
		Returns congestion factor array 1 x num array.
	SatRV : array-like
		Congestion factor array 1 x num size 
	'''

	MonFri = np.genfromtxt('TrafficCongestion.csv', delimiter =',',usecols=(1,2,3,4,5), skip_header = 1)
	MonFriCong = MonFri.flatten()
	if (run ==1):
		f = Fitter(MonFriCong)
		f.fit()
		print(f.summary()) # halfcauchy is best sums = 531.25
	MonFrifitted = stats.norm.fit(MonFriCong)
	MonFriRV = stats.norm.rvs(loc=MonFrifitted[0], scale=MonFrifitted[1], size=num)
	
	Sat = np.genfromtxt('TrafficCongestion.csv', delimiter =',',usecols=(6), skip_header = 1)
	SatCong = Sat.flatten()
	SatRV = stats.uniform.rvs(loc= min(SatCong), scale = (max(SatCong)-min(SatCong)), size = num)

	return MonFriRV, SatRV 

def duration(Routes, num, Weekday):
	'''
	Runs the simulation of the traffic and demand variation
	Returns the cost of each route, for each variation 

	Parameters:
	-----------
	Routes : array-like
		List of routes to iterate over. 
	num : float
		number of simulations to run.
	Weekday : binary 
		1 if weekday routes, 0 for saturday
	'''
	WeekDayT, SatT = TrafficFactor(num) # creates congestion percentage
	times = pd.read_csv('WarehouseDurations.csv',header=0, index_col=0) # reads in non-traffic times
	if (Weekday == 1):
		factor = WeekDayT
	else:
		factor = SatT

	demands, x = DemandSim (Weekday, num)
	TotalVarRouteTimes = np.zeros((num, len(Routes)))
	TotalVarRouteDemand = np.zeros((num, len(Routes)))

	for n in range(num):
		OneVarRouteTimes = np.zeros(len(Routes))
		# print(factor[n])
		TrafficTimes = times*factor[n] # Creates the times with traffic facotred in
		OneVarRouteDemand = np.zeros(len(Routes))
		for i in range(len(Routes)):
			singleRoute_time =0
			singleDemand = 0
			for j in range( len (Routes[i]) - 1 ):
				fromNode = Routes[i][j]
				ToNode = Routes[i][j+1]
				if ToNode != 'Distribution South':
					if ToNode != 'Distribution North':
						singleRoute_time += 600*demands[ToNode][n]
						singleDemand += demands[ToNode][n]
				singleRoute_time += TrafficTimes.loc[fromNode][ToNode] #Calculates travel time from 
			OneVarRouteTimes[i] = singleRoute_time 
			OneVarRouteDemand[i] = singleDemand
		TotalVarRouteTimes[n] = OneVarRouteTimes
		TotalVarRouteDemand[n] = OneVarRouteDemand

	return TotalVarRouteTimes, TotalVarRouteDemand

def Simulation(num, Routes, Weekday=1):
	'''
	Parameters:
	-----------
	num : integer
		Number of times to run the simulation
	Routes : Array-like
		Ordered routes that are choosen
	Weekday : binary 
		1 if simulation is for Monday to Friday, 0 if Saturday 
	Returns:
	--------
	None 
	'''
	RouteTimes, RouteDemands = duration(Routes, num, Weekday = 1)
	RouteCosts = np.zeros((num, len(Routes)))

	ka = []
	over4a = []
	over20a = []
	mainfa = []
	

	for i in range(num):
		k = 50 - len(Routes)
		over4 = 0
		over20 = 0
		mainf = 0
		for j in range(len(Routes)):
			# print(k)
			if RouteDemands[i][j] > 20: # IF route demand is more then 20, mainfreight does it
				
				if k>0:
					newcost = (RouteTimes[i][j]*2 - RouteDemands[i][j]*600)*175/3600
					f = (RouteTimes[i][j]-600*20)*175/3600+1500
					if newcost<f:
						RouteCosts[i][j] = newcost
						over20 +=1
						k = k-1
					else:
						RouteCosts[i][j] = f
						mainf+=1
				else:
					RouteCosts[i][j] = f
					mainf+=1

			
				# elif k>0:
				# 	newcost = (RouteTimes[i][j] - RouteDemands[i][j]*600)*175/3600
				# 	RouteCosts[i][j] += np.min([newcost,1500])
				# 	over20 +=1
				

			elif RouteTimes[i][j] > 14400: # If routes takes more then four hours, overtime pay 
				over4 +=1
				RouteCosts[i][j] = 175/60/60*14400 + (RouteTimes[i][j] - 14400)*250/60/60 

			else:  
				RouteCosts[i][j] = RouteTimes[i][j]*175/60/60	
		ka.append(k)
		over4a.append(over4)
		over20a.append(over20)
		mainfa.append(mainf)

	for i in range(len(Routes)):
		Route = RouteCosts[:,i]
		print('Mean travel time of Route', i, '=', np.mean(Route))
		print('95% confidence interval for Route', i, '=', mean_confidence_interval(Route))
		

	totalCost = np.zeros(num)
	for i in range(num):
		totalCost[i] = np.sum(RouteCosts[i,:]) 

	print(np.mean(over4a)/len(Routes))
	print(np.mean(over4a))
	print(np.mean(over20a)/len(Routes))
	print(np.mean(over20a))
	print(np.mean(ka))
	print(np.mean(mainfa)/len(Routes))

	return RouteCosts, totalCost

def mean_confidence_interval(data):
	''' Returns a 95% confidence interval
	Parameters:
	----------
	Data : Array-Like 
		Data to calculate CI interval over
	
	Returns:
	-------- 
	[m-h, m+h] : Array - Like
		The bounds of the CI 
	'''
	n = len(data)
	m, sd = np.mean(data), np.std(data)
	h = 1.96 * (sd/np.sqrt(n))
	return [m-h, m+h] 

def ReadOddSizedTxtFile(File):

	# open the file
	fp = open(File,'r')

	# enter a while loop to read the data
	ln = fp.readline()         # read the first line of the data
	lines=[]                
	while ln != '':            # check: is it the end of the file?
		# "strip" off the "newline" character at the end
		ln = ln.strip()

		# "split" the string into two substrings, using the comma as a delimiter
		line = ln.split(',')
		lines.append(line)
		# read the next line before going back to the start of the loop
		ln = fp.readline()         

	# close the file
	fp.close()

	return lines
	
def RouteDemSim (Routes, Scenario):
    x, demands = DemandSim(1, 1000)
    RouteNew = []
    RoutesNew = []
    for i in range(len(Routes)):
        for j in range(len(Routes[i])):
            if Routes[i][j] > 0:
                RouteNew.append(Routes[i][j] - 2)
        RoutesNew.append(RouteNew)
        RouteNew = []
    RouteDemands = np.zeros((len(RoutesNew), 1000))
    for i in range (0, 1000):
        for j in range(len(RoutesNew)):
            RouteDemand = 0
            for k in range(len(RoutesNew[j])):
                RouteDemand += demands[k][i]
            RouteDemands[j][i] = RouteDemand

    mergearray = RouteDemands.flatten()
    sns.distplot(mergearray)
    plt.savefig(Scenario + '.png')
    confint = mean_confidence_interval(mergearray)
    print(confint)


Routes = ReadOddSizedTxtFile('routes1.txt')
# print(Routes)
RouteCosts, totalCosts = Simulation(1000, Routes, 1)
totalCostsA = totalCosts
totalCosts = np.sort(totalCosts)
lower = totalCosts[25]
upper = totalCosts[975]
mean = np.mean(totalCosts)
print('\n',20*upper,20*lower,20*mean,'\n')
print('\n',1*upper,1*lower,1*mean,'\n')
plt.hist(totalCosts/1000, density=True)
plt.title('Histogram of the One-Day Costs for the Monday to Friday Trucking Plan With Both Distribution Centers', fontsize= 18)
plt.xlabel('Cost [Thousands of Dollars]', fontsize= 18)
plt.ylabel('Probability', fontsize= 18)
plt.show()

Routes = ReadOddSizedTxtFile('routes2.txt')
RouteCosts1, totalCosts1 = Simulation(1000, Routes, 0)
totalCostsB = totalCosts1
totalCosts1 = np.sort(totalCosts1)
lower = totalCosts1[25]
upper = totalCosts1[975]
mean = np.mean(totalCosts1)
print('\n',4*upper,4*lower,4*mean,'\n')
print('\n',1*upper,1*lower,1*mean,'\n')
plt.hist(totalCosts1/1000, density=True)
plt.title('Histogram of the One-Day Costs for the Saturday Trucking Plan With Both Distribution Centers', fontsize= 18)
plt.xlabel('Cost [Thousands of Dollars]', fontsize= 18)
plt.ylabel('Probability', fontsize= 18)
plt.show()

print('\n')
totalweek = 20*totalCostsA+4*totalCostsB
# print((totalweek))
# print((totalCostsA))
totalweek = np.sort(totalweek)
lower = totalweek[25]
upper = totalweek[975]
mean = np.mean(totalweek)
print(upper,lower,mean)
print('\n')

Routes = ReadOddSizedTxtFile('routes3.txt')
# print(Routes)
RouteCosts, totalCosts = Simulation(1000, Routes, 1)
totalCostsA = totalCosts
totalCosts = np.sort(totalCosts)
lower = totalCosts[25]
upper = totalCosts[975]
mean = np.mean(totalCosts)
print('\n',20*upper,20*lower,20*mean,'\n')
print('\n',1*upper,1*lower,1*mean,'\n')
plt.hist(totalCosts/1000, density=True)
plt.title('Histogram of the One-Day Costs for the Monday to Friday Trucking Plan With One Distribution Centers', fontsize= 18)
plt.xlabel('Cost [Thousands of Dollars]', fontsize= 18)
plt.ylabel('Probability', fontsize= 18)
plt.show()

Routes = ReadOddSizedTxtFile('routes4.txt')
RouteCosts1, totalCosts1 = Simulation(1000, Routes, 0)
totalCostsB = totalCosts1
totalCosts1 = np.sort(totalCosts1)
lower = totalCosts1[25]
upper = totalCosts1[975]
mean = np.mean(totalCosts1)
print('\n',4*upper,4*lower,4*mean,'\n')
print('\n',1*upper,1*lower,1*mean,'\n')
plt.hist(totalCosts1/1000, density=True)
plt.title('Histogram of the One-Day Costs for the Saturday Trucking Plan With One Distribution Centers', fontsize= 18)
plt.xlabel('Cost [Thousands of Dollars]', fontsize= 18)
plt.ylabel('Probability', fontsize= 18)
plt.show()

print('\n')
totalweek = 20*totalCostsA+4*totalCostsB
# print((totalweek))
# print((totalCostsA))
totalweek = np.sort(totalweek)
lower = totalweek[25]
upper = totalweek[975]
mean = np.mean(totalweek)
print(upper,lower,mean)
print('\n')