#Anthony Maylath 3/18/2019 - functions to compute valuation surfaces in python
import proj_helpers.FinObj as fin
import numpy as np
import scipy.stats as sp

def getTheta(stock,auto,paths,time_s,t_del,theta_times):
	'''
	Computes valuations surface accross time and the corresponding theta
	Assumes only one timestep!
	stock : object fo type stock
	auto : object of type autocall
	paths : number of paths for each simulation
	time_s : number of time steps
	t_del : size of each timestep
	theta_times : array of tuples; dates to value the product
		(time units, number of days)

	returns dictionary of shocked valuations
	'''
	#Save down original time units
	begin_th = (stock.t_unit, auto.t_unit)

	t_prices = {}
	for th, days in theta_times:
		#Rescale to new time horizon
		auto.rescale(th)
		stock.rescale(th)
		t_prices[days] = fin.getPrice(stock, auto, paths, time_s, t_del,scheme=fin.mc_Mil)

	#Restore original time units
	stock.rescale(begin_th[0]); auto.rescale(begin_th[1])

	return t_prices

def getThetaIter(stock,auto,paths,time_s,t_del):
	'''
	Computes valuations surface accross time and the corresponding theta
	by incrementing down days
	stock : object fo type stock
	auto : object of type autocall
	paths : number of paths for each simulation
	time_s : number of time steps
	t_del : size of each timestep
	theta_times : array of tuples; dates to value the product
		(time units, number of days)

	returns dictionary of shocked valuations
	'''

	t_prices = {}
	for th in range(0,time_s):
		t_prices[time_s-th] = fin.getPrice(stock, auto, paths, time_s,\
		 t_del, scheme=fin.mc_Mil, anti=True, ini_t=th)

	return t_prices

def numTheta(t_dict):
	'''
	t_dict : dictionary with number of days as keys and valuations as values
	returns a dictionary with number of days as keys and theta as values 
	where theta is numerically computed
	'''
	#plot theta (assume day increment) numerical
	xvals = [x for x in t_dict.keys()]
	theta = {xvals[x]:t_dict[xvals[x+1]]-t_dict[xvals[x]] for x in range(0,len(xvals)-1)}
	return theta

def getSpot(stock,auto,paths,time_s,t_del,spots,ini_t=0):
	'''
	Computes valuation for various spot prices
		stock : object fo type stock
	auto : object of type autocall
	paths : number of paths for each simulation
	time_s : number of time steps
	t_del : size of each timestep
	spots : spot prices used in valuation

	Returns a dictionary with spots as keys and valuations as values
	'''
	#Save down initial spots
	begin_sp = stock.spot

	prices = {}
	for s in spots:
	    stock.setSpot(s)
	    prices[s] = fin.getPrice(stock, auto, paths, time_s, t_del, ini_t=ini_t)

	#Reset to initial spot
	stock.setSpot(begin_sp)

	return prices

def getVol(stock,auto,paths,time_s,t_del,vols,ini_t=0):
	'''
	Computes valuation for various spot prices
		stock : object fo type stock
	auto : object of type autocall
	paths : number of paths for each simulation
	time_s : number of time steps
	t_del : size of each timestep
	vols : spot prices used in valuation

	Returns a dictionary with vols as keys and valuations as values
	'''
	#Save down initial vol
	begin_sp = stock.vol

	prices = {}
	for v in vols:
	    stock.setVol(v)
	    prices[v] = fin.getPrice(stock, auto, paths, time_s, t_del, ini_t=ini_t)

	#Reset inital vol
	stock.setVol(begin_sp)

	return prices

def getSpotVol(stock,auto,paths,time_s,t_del,spots,vols,ini_t=0):
	'''
	Computes valuation for various spot and vol
	Repeatedly calls getSpot

	returns numpy array with dim len(spots) x len(vols)
	'''
	prices = np.zeros((len(vols),len(spots)))
	#Save down initial vol
	begin_sp = stock.vol
	for v in range(0,len(vols)):
		stock.setVol(vols[v])
		prices[v] = list(getSpot(stock,auto,paths,time_s,t_del,spots,ini_t).values())

	#Reset inital vol
	stock.setVol(begin_sp)

	return prices

def getd1(spot = 100, strike = 100, div = 0, rate = 0, vol = 0.15, time = 1):
	'''
	Compute d1 in Black Scholes formula
	'''
	d1 = np.log(spot/strike) + (rate - div + vol*vol/2)*time
	return d1/(np.sqrt(time)*vol)

def BSCall(spot = 100, strike = 100, div = 0, rate = 0, vol = 0.15, time = 1):
	'''
	Compute the price of vanilla option
	'''
	d1 = getd1(spot, strike, div, rate, vol, time)
	d2 = d1 - vol*np.sqrt(time)

	result = spot*np.exp(-div*time)*sp.norm.cdf(d1)
	return result - strike*np.exp(-rate*time)*sp.norm.cdf(d2)

def BSPut(spot = 100, strike = 100, div = 0, rate = 0, vol = 0.15, time = 1):
	'''
	Compute the price of vanilla option
	'''
	d1 = getd1(spot, strike, div, rate, vol, time)
	d2 = d1 - vol*np.sqrt(time)

	result = sp.norm.cdf(-d2)*strike*np.exp(-rate*time)
	return result - spot*np.exp(-time*div)*sp.norm.cdf(-d1)

def BSPutDig(spot = 100, strike = 100, div = 0, rate = 0, vol = 0.15, time = 1):
	'''
	Compute the price of digital vanilla put
	'''
	d1 = getd1(spot, strike, div, rate, vol, time)
	d2 = d1 - vol*np.sqrt(time)
	return np.exp(-rate*time)*sp.norm.cdf(-d2)

def BSCallDig(spot = 100, strike = 100, div = 0, rate = 0, vol = 0.15, time = 1):
	'''
	Compute the price of digital vanilla call
	'''
	d1 = getd1(spot, strike, div, rate, vol, time)
	d2 = d1 - vol*np.sqrt(time)
	return np.exp(-rate*time)*sp.norm.cdf(d2)
