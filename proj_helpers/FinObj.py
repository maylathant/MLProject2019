#Define Financial Objects and Helper Functions
#Anthony Maylath 2/15/2019
#Create class to hold stock infornmation

import numpy as np
import time
import sys
from sklearn.ensemble import RandomForestRegressor
import pickle

class Stock:
    def __init__(self, spot, vol, t_unit=1, repo=0.0):
        self.spot = spot
        self.vol = vol/np.sqrt(t_unit)
        self.t_unit = t_unit
        self.repo = repo/t_unit

    def rescale(self, t_unit):
        '''
        Change scaling of vol and rate to match new units
        1 = years, 360 = days, 12 = months, etc...
        '''
        self.vol = self.vol*np.sqrt(self.t_unit)/np.sqrt(t_unit)
        self.repo = self.repo*self.t_unit/t_unit
        self.t_unit = t_unit

    def setSpot(self, spot):
    	self.spot = spot

    def setVol(self, vol):
    	self.vol = vol

#Option class to represent American options
class Option:
    '''
    Payoff is a function that describes the payoff
    of the option
    birthday : dictionary with rng and pay elements
        rng represents the range from the strike price in ascending order
        pay represents the payoff for the corresponding range
    '''
    def __init__(self, stock, payoff, exp, strike=100, t_unit=1,\
     birthday={"rng":[5,10,15],"pay":[1.0,0.5,0.25]},ctlV=0,wedge=90):
        self.stock = stock #Underlying
        self.payoff = payoff #Payoff function
        self.exp = exp #Time to expiration
        self.strike = strike #Fixed strike price
        self.t_unit = t_unit
        self.ctlV = ctlV #Expectation for control variate
        self.wedge = wedge
        self.setBirthday(birthday)

    def rescale(self, t_unit):
        '''
        Change scaling of option
        1 = years, 360 = days, 12 = months, etc...
        '''
        self.t_unit = t_unit

    def setBirthday(self, birthday):
        '''
        Set values for birthday cake payoff
        '''
        if type(birthday) != dict:
            print("Error: birthday parameter must contain range and payout amounts in a dictionary")
            sys.exit()

        if len(birthday["rng"]) != len(birthday["pay"]):
            print("Error: Range and Payoff values for Birthday cake must be equal.")
            sys.exit()

        self.birthday = birthday

class ControlVar:
    '''
    Control variate class (assume European style)
    payoff : Payoff of control variate product
    EX : expectation of control variate product
    b : CV factor
    '''
    def __init__(self, payoff=False, EX=False, b=False):
        self.payoff = payoff
        self.EX = EX
        self.b = b

#American put payoff
def amPutVec(spot,op):
    '''
    spot : vector of spot prices
    '''
    result = op.strike - spot
    return result * (result > 0.0) #Takes max between 0 and value

#American digital put payoff
def amPutDigVec(spot,op):
    '''
    spot : vector of spot prices
    op : option object
    '''
    result = op.strike - spot
    return 1.0 * (result > 0.0) #Takes max between 0 and value

#American digital call payoff
def amCallDigVec(spot,op):
    '''
    spot : vector of spot prices
    op : option object
    '''
    result = spot - op.strike
    return 1.0 * (result > 0.0) #Takes max between 0 and value

#Wedge digital option (aka double digital)
def wedgeVec(spot,op):
    return 1.0 * (spot < op.wedge) + 1.0 * (spot < op.strike)

#Birthday cake digital option
def birthdayVec(spot,op):
    dev = abs(spot-op.strike) #Deviation from strike
    result = (dev <= op.birthday["rng"][0])*1.0 #Initialize with highest payoff
    for rg in range(1,len(op.birthday["rng"])): #Set remainder of cake
        result = (dev <= op.birthday["rng"][rg])\
        *(dev >= op.birthday["rng"][rg-1])\
        *op.birthday["pay"][rg]+result #Check if value is within bound
    return result

#Birthday cake digital option with control variate
def birthdayVecCtl(spot,op):
    #Assume 0.8 for factor
    result = birthdayVec(spot,op)
    result = result - 0.8*(amPutDigVec(spot,op) - op.ctlV)
    return result

#American put payoff
def amPut(spot,op):
    return max(op.strike-spot,0.0)

#American call payoff
def amCall(spot,op):
    return max(spot-op.strike,0)

#American call payoff
def amCallVec(spot,op):
    '''
    spot : vector of spot prices
    '''
    result = spot - op.strike
    return result * (result > 0.0) #Takes max between 0 and value

#Vanilla call payoff
def vanCall(stock, spot , auto, t):
    if t == auto.exp: return max(spot-auto.strike,0)
    return 0 

def mc_Time(st,stock,wt):
    '''
    One time step for the monte carlo
    Euler scheme
    '''
    #Ensure result is not negative
    return max(st + st*stock.vol*wt,0.0)

def mc_Mil(st,stock,wt):
    '''
    One time step for the monte carlo
    Milstein
    '''
    #Ensure result is not negative
    return max(st + st*stock.vol*wt\
    	+0.5*stock.vol*stock.vol*st*(wt*wt-1),0.0)

####################################################################################################
####################################################################################################
##################      American Option Functions     ##############################################
####################################################################################################
####################################################################################################

#Monte carlo simulation for american payoffs
#One Path for the MC Simulation
def mc_PathFull(stock,rand_vec,scheme,opt):
    '''
    stock : Class stock
    rand_vec : vector of random variables for each time step
    scheme : function to iterate by one time step
    function must take spot, stock class and random variable as input

    Returns the final iteration result
    '''

    #Result hold the current spot price and product payoff for all steps
    result = np.zeros((2,len(rand_vec)+1))
    result[0][0] = stock.spot
    for step in range(1,len(rand_vec)+1):
        result[0][step] = scheme(result[0][step-1],stock,rand_vec[step-1])
        result[1][step] = opt.payoff(result[0][step], opt)
        
    return result

#Get price via monte carlo simulation, mainly for american option
def getSim(stock, opt, paths, t_steps, t_del,\
 scheme=mc_Time,anti=False):
    '''
    stock : stock object
    opt : option object
    paths : number of MC paths
    t_steps : number of time steps per path
    t_del : size of each time step
    scheme : monte carlo scheme. For instance, mc_Time = Euler
    anti : boolean to trigger antithetical variates ~ Will run
    double the amount of paths
    '''
    
    #Generate Random numbers for the simulation
    #shape = number of paths, number of time steps

    sim_ran = np.random.normal(0,t_del,(paths,t_steps))

    #Add antithetical variates
    if anti==True: sim_ran = np.concatenate((sim_ran,sim_ran*-1),axis=0)
    
    #Apply random numbers to option
    #Lambda to be evaluated for each path
    npPath = lambda x : mc_PathFull(stock, x, scheme, opt)

    #Return the average payoff over all paths (expectation)
    #result1 = np.mean(np.apply_along_axis(npPath,1,sim_ran))
    return np.array([npPath(x) for x in sim_ran])

def getCont(Ct1,St,kwargs):
    '''
    Get continuation value at time t. Assumes valid value at t+1
    Ct1 : discounted payoff from t+1
    St : stock prices at t
    kwargs holds:
        num_basis : number of basis elements to use
    '''
    if len(St) == 0: return np.array([]) #If no paths are in the money

    basis=np.eye(kwargs["num_basis"])
    #Evaluate St at laguerre basis
    X=np.array([np.polynomial.laguerre.lagval(St,basis[x]) for x in range(0,kwargs["num_basis"])])
    X=np.transpose(X)

    #Fit regression coefficients
    betas = np.linalg.lstsq(X,Ct1)[0]

    #Return fitted values
    return np.matmul(X,betas)

def getContPol(Ct1,St,kwargs):
    '''
    Get continuation value at time t with polynomial basis
    Ct1 : discounted payoff from t+1
    St : stock prices at t
    kwargs holds:
        num_basis : number of basis elements to use
    '''
    if len(St) == 0: return np.array([]) #If no paths are in the money

    basis=np.eye(kwargs["num_basis"])
    #Evaluate St at laguerre basis
    X=np.array([np.polynomial.polynomial.polyval(St,basis[x]) for x in range(0,kwargs["num_basis"])])
    X=np.transpose(X)

    #Fit regression coefficients
    betas = np.linalg.lstsq(X,Ct1)[0]

    #Return fitted values
    return np.matmul(X,betas)

def mcmcStep(St, stock, opt, paths, anti=False):
    '''
    One step of monte carlo for continuation
    St : spot price at time t (one value)
    '''
    paths = 2*paths if anti else paths #Toggle antithetical variates
    Xt = np.random.normal(0,1,(paths,)) #Generate terminal values for W(t)
    Xt = jumpEx(stock,Xt,1)
    FP = stock.spot*np.exp(Xt)  #np.array([stock.spot*np.exp(x) for x in Xt]) #Final share price
    FO = opt.payoff(FP,opt) #Final Payoff
    FO = FO*np.exp(-stock.repo) #Discount back to present
    return np.mean(FO)

def getContMC(Ct1,St,kwargs):
    '''
    Get continuation value at time t with monte carlo
    Ct1 : payoffs
    St : current stock values
    kwargs holds:
        stockC : stock object
        optC : option object
        pathsC : number of paths
        T : divisor for units
        antiC : if true than use antithetical
    '''
    t_unit = kwargs['stockC'].t_unit

    #Rescale stock and option to units that span all time steps
    kwargs['stockC'].rescale(1/kwargs['T'])
    kwargs['optC'].rescale(1/kwargs['T'])

    Cont = mcmcStep(St,kwargs['stockC'],kwargs['optC'],kwargs['pathsC'],kwargs['antiC'])

    #Rescale stock and option to normal time step
    kwargs['stockC'].rescale(t_unit)
    kwargs['optC'].rescale(t_unit)

    return Cont

def getContRF(Ct1,St,kwargs={"num_trees":5,"max_depth":5,"min_samples_leaf":0.001}):
    '''
    Get continuation value at time t with Random Forest
    Ct1 : discounted payoff from t+1
    St : stock prices at t
    kwargs holds:
        num_trees : number of trees to use
        max_depth : max depth of each tree
        min_samples_leaf : minimum %age to have in a leaf of tree
    '''
    if len(St) == 0: return np.array([]) #If no paths are in the money

    clf=RandomForestRegressor(n_estimators=kwargs["num_trees"],\
        max_depth=kwargs["max_depth"],min_samples_leaf=kwargs["min_samples_leaf"])
    tempSt=np.reshape(St,(len(St),1))
    clf.fit(tempSt,Ct1)
    return clf.predict(tempSt)

def getPV(C,g,P):
    '''
    Obtain present value of future payoffs
    C : Continuation value
    g : Payoff today
    P : payoff t+1
    '''
    return g if C<g else P

def getPayoffs(Pt1,Ct0,Ct,moneymask):
    '''
    Compute payoffs at t
    Pt1 : payoffs discounted t+1
    Ct0 : current payoffs
    Ct : continuation values
    referencing paths in the money
    moneymask : index of in the money paths
    '''
    
    # result = Pt1 #Pt1 holds default values for zero payoffs
    # for x in moneymask:
    #     result[x] = getPV(Ct[x],Ct0[x],Pt1[x])
    mask = Ct0 > Ct
    P = Pt1
    P[mask] = Ct0[mask]
    return P

def optPayoff(C,G):
    '''
    Obtain optimal payoff, given a path
    C : sequence of continuation values
    G : sequence of intrinsic values
    P : sequence of discounted payoffs
    '''
    result = G[-1]
    for c,g in zip(reversed(C),reversed(G)):
        if c < g: result = g
    return result

def priceAmerican(sim,num_basis):
    '''
    Compute the price of an American option
    sim : result of simulation. Should be path x type x time step
    where type can be either intrinsic value or stock price
    um_basis : number of basis functions to use for least squared
    bridge : flag to use brownian bridge. Does not require parameter sim
    '''

    P,C = assemblePayoffs(sim,num_basis)
    G = sim[:,1]

    #Value of optimal payoff
    end_path=[optPayoff(c,g) for c,g in zip(C,G)] #Is this already done earlier in the algo?
    return end_path

def assemblePayoffs(sim,num_basis):
    '''
    Assemble the payoffs from all timesteps
    sim : result of simulation. Should be path x type x time step
    where type can be either intrinsic value or stock price
    num_basis : number of basis functions to use for least squared
    '''

    #Assumes timestep # is in the second element and paths in 0th
    time_s = np.shape(sim)[2]
    paths = np.shape(sim)[0]

    #np.array to hold the results
    P = np.zeros((paths,time_s))
    C = np.zeros((paths,time_s))

    #Set final payoffs to the final intrinsic values
    P[:,-1] = sim[:,1][:,-1]
    C[:,-1] = sim[:,1][:,-1]



    #Walk backwards through the simulation to find all the
    #continuation values and payoffs
    for t in reversed(range(1,time_s-1)):
        Pt1 = P[:,t+1] #Get payoffs from one step ahead
        Ct0 = sim[:,1][:,t] #Current intrinsic values
        #Index of paths in the money
        moneymask = np.where(Ct0>0)[0]
        St = sim[:,0][:,t][moneymask] #Share prices in the money


        Ct = getCont(Pt1[moneymask],St,num_basis) #Obtain continuation values
        
        Ct = {x:c for x,c in zip(moneymask,Ct)} #Transform into index
        
        P[:,t] = getPayoffs(Pt1,Ct0,Ct,moneymask) #Current payoffs
        C[:,t] = [Ct[x] if x in moneymask else 0 for x in range(0,paths)]#Fill in continuation values
    
    return P,C


def mc_Bridge(xt,stock,t,wt):
    '''
    Walk one step back on brownian bridge
    xt : value at t+1
    stock : instance of stock object
    t : tuple containing t and t+1 times
    wt : random normal
    '''
    return xt*(t[0]/t[1]) + stock.vol*np.sqrt(t[0]/t[1])*wt

def mc_BridgeVec(xt,stock,t,wt,dt=1):
    '''
    Walk one step back on brownian bridge
    xt : values at t+1
    stock : instance of stock object
    t : tuple containing t and t+1 times
    wt : vector of random normals
    '''
    return xt*(t[0]/t[1]) + stock.vol*np.sqrt(dt*t[0]/t[1])*wt

def jumpEx(stock,wt,T):
    '''
    Function to get the value of a random process at time T
    stock : stock object
    wt : random standard normal variable(s)
    T : time to elapse
    '''
    return (stock.repo-stock.vol*stock.vol/2)*T + stock.vol*np.sqrt(T)*wt


def bridgePayoffs(stock, opt, paths, t_steps, t_unit,\
 anti=False,regressor=getCont,CV=ControlVar(),kwargs={"num_basis":4,"outputresults":False}):
    '''
    Compute american payoff via LSM with brownian bridge
    stock : stock object
    opt : option object
    paths : number of MC paths
    t_steps : number of time steps per path
    t_del : size of each time step
    anti : boolean to trigger antithetical variates ~ Will run
    double the amount of paths
    regressor : type of regressor to use (function)
    CV : Control variate object
    kwargs:
        num_basis : number of basis vectors to use in Laguerre expansion -> least squares
        num_trees : number of trees to use for random forests
        max_depth : max depth for each tree in random forests
    '''

    timings = [0,0,0,0,0,0,0]

    T = t_steps/t_unit #Full duration of the product

    #Rescale stock and option to units that span all time steps
    stock.rescale(1/T)
    opt.rescale(1/T)

    #Get final state of paths
    t0 = time.time()
    paths = 2*paths if anti else paths #Toggle antithetical variates
    Xt = np.random.normal(0,1,(paths,)) #Generate terminal values for W(t)
    Xt = jumpEx(stock,Xt,T)
    FP = stock.spot*np.exp(Xt)  #np.array([stock.spot*np.exp(x) for x in Xt]) #Final share price
    FO = opt.payoff(FP,opt) #Final Payoff

    #Rescale stock and option to normal time step
    stock.rescale(t_unit)
    opt.rescale(t_unit)

    dr = np.exp(-stock.repo) #Instantanious discounting (using original scaling)

    if(CV.b):
        cvTerm = CV.b*(CV.payoff(FP,opt)*dr - CV.EX) #Control variate term

    sim_ran = np.random.normal(0,1,(paths,t_steps-1)) #Get random realizations all at once

    timings[0] = t0 - time.time()


    #Start walk back
    for t in reversed(range(1,t_steps+1)):
        t0 = time.time()
        Xt = mc_BridgeVec(Xt,stock,(t-1,t),sim_ran[:,t-2])
        timings[1] = t0 - time.time()

        t0 = time.time()
        St = stock.spot*np.exp(Xt)
        timings[2] = t0 - time.time()

        t0 = time.time()
        #Ct0 = np.array([opt.payoff(x,opt) for x in St]) #myPayoff(St,opt) #Current intrinsic values
        Ct0 = opt.payoff(St,opt)
        timings[3] = t0 - time.time()

        t0 = time.time()
        moneymask = np.where(Ct0>0.0)[0] #Filter for ITM options
        if t == 2 and len(moneymask) == 0: print("Warning: No Paths in the Money. Price May Round to Zero.")
        timings[4] = t0 - time.time()

        FO = FO*dr #Get discounted payoffs for in the money paths

        t0 = time.time()
        Ct = np.zeros(paths)
        Ct[moneymask] = regressor(FO[moneymask],St[moneymask],kwargs) #Obtain continuation values
        if kwargs["outputresults"]: saveCont(FO[moneymask],St[moneymask],Ct[moneymask],t,regressor,stock,paths,opt,t_steps)
        timings[5] = t0 - time.time()


        t0 = time.time()
        FO = getPayoffs(FO,Ct0,Ct,moneymask) #Current payoffs
        
        timings[6] = t0 - time.time()

        # print("Initialization  Time : " + str(timings[0]) + " Current State Calc : " + str(timings[1])\
        #     + " Transfrom Xt to St : " + str(timings[2]) + " Compute Option Payoff : " + str(timings[3])\
        # + " Filter moneymask : " + str(timings[4]) + " Continuation Compute : " + str(timings[5]) \
        # + " Compute Payoffs : " + str(timings[6]))

    FO = FO*dr #Discount to last time step

    if CV.b: return FO - cvTerm

    return FO

def priceEur(stock, opt, paths, t_steps, t_unit,anti=False):
    '''
    To price products that are not path dependant
    Simplified version of bridgePayoffs
    '''
    
    T = t_steps/t_unit #Full duration of the product

    #Rescale stock and option to units that span all time steps
    stock.rescale(1/T)
    opt.rescale(1/T)

    paths = 2*paths if anti else paths #Toggle antithetical variates
    Xt = np.random.normal(0,1,(paths,)) #Generate terminal values for W(t)
    Xt = jumpEx(stock,Xt,T)
    FP = stock.spot*np.exp(Xt)  #np.array([stock.spot*np.exp(x) for x in Xt]) #Final share price
    FO = opt.payoff(FP,opt) #Final Payoff
    FO = FO*np.exp(-stock.repo) #Discount back to present

    #Rescale stock and option to normal time step
    stock.rescale(t_unit)
    opt.rescale(t_unit)

    return np.mean(FO)


def priceLSM(stock, opt, paths, t_steps, t_unit,\
 anti=False,regressor=getCont,CV=ControlVar(),**kwargs):
    '''
    LSM with brownian bridge
    stock : stock object
    opt : option object
    paths : number of MC paths
    t_steps : number of time steps per path
    t_del : size of each time step
    scheme : monte carlo scheme. For instance, mc_Time = Euler
    anti : boolean to trigger antithetical variates ~ Will run
    double the amount of paths
    num_basis : number of basis vectors to use in Laguerre expansion
    '''
    return np.mean(bridgePayoffs(stock, opt, paths, t_steps, t_unit,anti,regressor,CV,kwargs))

def saveCont(FO,St,C,t,regressor,stock,paths,opt,time_s):
    '''
    Save continuation values for analysis
    FO : payoffs
    St : stock prices
    C : continuation values with FO and St
    t : current time step
    regressor : type of regression used
    opt : option object
    '''
    if len(C) > 0:
        mydic = {"payoff":FO,"stockp":St,"Continuation":C}
        with open('results/step_' + str(t) + '_paths_' + str(paths) + '_'\
         + str(regressor) + '_spot_' + str(stock.spot) +\
          '_payoff_' + str(opt.payoff) + '_time_s_' + str(time_s) + '.pkl', 'wb') as f:
            pickle.dump(mydic, f)

def saveContErr(FO,St,C,t,regressor,stock,paths):
    '''
    Save continuation errors for analysis
    FO : payoffs
    St : stock prices
    C : continuation values with FO and St
    t : current time step
    regressor : type of regression used
    '''
    with open('results/regression_errors' + '.pkl', 'rb') as f:
        mydict = pickle.load(f)

    if len(C) > 0:
        L1 = sum(abs(FO - C)); L2 = sum((FO - C)*(FO - C))
        errDict = {'L1':L1, 'L2':L2}
        for l in errDict:
            mydict[(l,str(regressor),str(t),str(stock.spot),str(paths))] = errDict[l]
    
        with open('results/regression_errors' + '.pkl', 'wb') as f:
            pickle.dump(mydict,f)


def regressionErr(stock, opt, paths,anti=False,**kwargs):
    '''
    Compute regression errors for continuation
    stock : stock object
    opt : option object
    paths : number of MC paths
    anti : boolean to trigger antithetical variates ~ Will run
    double the amount of paths
    kwargs:
        num_basis : number of basis vectors to use in Laguerre expansion -> least squares
        num_trees : number of trees to use for random forests
        max_depth : max depth for each tree in random forests
    '''
    t_unit = 1; t_steps = 2;

    T = t_steps/t_unit #Full duration of the product

    #Rescale stock and option to units that span all time steps
    stock.rescale(1/T)
    opt.rescale(1/T)

    #Get final state of paths
    t0 = time.time()
    paths = 2*paths if anti else paths #Toggle antithetical variates
    Xt = np.random.normal(0,1,(paths,)) #Generate terminal values for W(t)
    Xt = jumpEx(stock,Xt,T)
    FP = stock.spot*np.exp(Xt)  #np.array([stock.spot*np.exp(x) for x in Xt]) #Final share price
    FO = opt.payoff(FP,opt) #Final Payoff

    #Rescale stock and option to normal time step
    stock.rescale(t_unit)
    opt.rescale(t_unit)

    dr = np.exp(-stock.repo) #Instantanious discounting (using original scaling)

    sim_ran = np.random.normal(0,1,(paths,t_steps-1)) #Get random realizations all at once

    regErr = {}
    mapping = {'PolBasis':getCont,'RandomF':getContRF}

    #Start walk back
    for t in reversed(range(1,t_steps+1)):
        Xt = mc_BridgeVec(Xt,stock,(t-1,t),sim_ran[:,t-2])

        St = stock.spot*np.exp(Xt)

        Ct0 = opt.payoff(St,opt)

        moneymask = np.where(Ct0>0.0)[0] #Filter for ITM options
        if t == 2 and len(moneymask) == 0: print("Warning: No Paths in the Money. Price May Round to Zero.")

        FO = FO*dr #Get discounted payoffs for in the money paths

        Ct = np.zeros(paths)
        if(len(moneymask) > 0):
            for reg in mapping:
                regErr[reg] = mapping[reg](FO[moneymask],St[moneymask],kwargs) #Obtain continuation values
                #saveCont(FO[moneymask],St[moneymask],regErr[reg],t,reg,stock,paths) #Persist chart
                regErr[reg] = regErr[reg]-FO[moneymask] #Regression error


    return regErr


