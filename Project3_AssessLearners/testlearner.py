"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import math
import sys
import util
import time
import matplotlib.pyplot as plt

import DTLearner as dt, RTLearner as rt, BagLearner as bl


def Calc_stat(trainX, trainY, testX, testY, learner):
    # evaluate in sample
    predY = learner.query(trainX)  # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
    mae  = (np.absolute(trainY - predY)).sum() / trainY.shape[0]
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    c = np.corrcoef(predY, y=trainY)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    predY = learner.query(testX)  # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    mae = (np.absolute(testY - predY)).sum() / testY.shape[0]
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    c = np.corrcoef(predY, y=testY)
    print(f"corr: {c[0, 1]}")

def analyse_overfit (trainX, trainY, testX, testY):
    ###################### Experiment 1 - DTLearner ######################
    InSamples = np.zeros((40, 1))
    OutSamples = np.zeros((40, 1))

    for leaf_size in range(1, 41):
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)     #Train DTLearner

        #In sample
        predY = learner.query (trainX)          #Predict on training sample (i.e, In Sample)
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])   #Calc prediction error
        InSamples[leaf_size-1, 0] = rmse
        c = np.corrcoef(predY, y=trainY)

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        OutSamples[leaf_size-1, 0] = rmse
        c = np.corrcoef(predY, y=testY)

    xaxis = np.arange(1, 41)
    plt.plot(xaxis, InSamples, label="In Sample")
    plt.plot(xaxis, OutSamples, label="Out Samples")
    plt.xlabel("Leaf Size")
    plt.ylabel("Root Mean Square Error")
    plt.legend()
    plt.title("Figure 1 - DT Learner overfitting across various leaf sizes")
    plt.savefig("Exp-1.png")
    plt.clf()

    ###################### Experiment 2 - Bag Learner ######################
    InSamples = np.zeros((40, 1))
    OutSamples = np.zeros((40, 1))

    for leaf_size in range(1, 41):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":leaf_size}, bags=20, boost=False, verbose=False)
        learner.addEvidence(trainX, trainY)     #Train RTLearner

        #In sample
        predY = learner.query (trainX)          #Predict on training sample (i.e, In Sample)
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])   #Calc prediction error
        InSamples[leaf_size-1, 0] = rmse
        c = np.corrcoef(predY, y=trainY)

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        OutSamples[leaf_size-1, 0] = rmse
        c = np.corrcoef(predY, y=testY)

    xaxis = np.arange(1, 41)
    plt.plot(xaxis, InSamples, label="In Sample")
    plt.plot(xaxis, OutSamples, label="Out Samples")
    plt.xlabel("Leaf Size")
    plt.ylabel("Root Mean Square Error")
    plt.legend()
    plt.title("Figure 2 - Bagging overfitting across various leaf sizes")
    plt.savefig("Exp-2.png")
    plt.clf()

def compare_RT_DT_learner (trainX, trainY, testX, testY):
    ###################### Experiment 3 - Quantative comparison DT v/s RT learner ######################

    DT_Time = np.zeros((20, 1))
    DT_MAE  = np.zeros((20, 1))
    for leaf_size in range(1, 21):
        start_time = time.time()
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)

        #in-sample
        predY = learner.query(trainX)

        #out-sample
        predY = learner.query(testX)
        end_time = time.time()
        DT_Time[leaf_size-1, 0] = end_time - start_time
        DT_MAE[leaf_size-1, 0] = (np.absolute(testY - predY)).sum() / testY.shape[0]    #out-sample MAE

    RT_Time = np.zeros((20, 1))
    RT_MAE  = np.zeros((20, 1))
    for leaf_size in range(1, 21):
        start_time = time.time()
        learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)  # constructor
        learner.addEvidence(trainX, trainY)  # training step

        #in-sample
        predY = learner.query(trainX)  #prediction

        #out-sample
        predY = learner.query(testX)  #prediction
        end_time = time.time()
        RT_Time[leaf_size-1, 0] = end_time - start_time
        RT_MAE[leaf_size-1, 0] = (np.absolute(testY - predY)).sum() / testY.shape[0]  # out-sample MAE

    ##Plot for Run Time
    xaxis = np.arange(1, 21)
    plt.plot(xaxis, DT_Time, label="DTLearner Times")
    plt.plot(xaxis, RT_Time, label="RTLearner Times")
    plt.xlabel("Leaf Sizes")
    plt.ylabel("Times")
    plt.legend()
    plt.title("Figure 3 - DT Learner v/s. RT Learner w.r.t Run Time")
    plt.savefig("Exp-3.png")
    plt.clf()

    ##Plot for Mean Absolute Error
    xaxis = np.arange(1, 21)
    plt.plot(xaxis, DT_MAE, label="DTLearner MAE")
    plt.plot(xaxis, RT_MAE, label="RTLearner MAE")
    plt.xlabel("Leaf Sizes")
    plt.ylabel("Mean Absolute Error - Out-Sample")
    plt.legend()
    plt.title("Figure 4 - DT Learner v/s. RT Learner w.r.t Mean Absolute Error (MAE)")
    plt.savefig("Exp-4.png")
    plt.clf()

if __name__=="__main__":
    if len(sys.argv) != 2:  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        sys.exit(1)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

    ''' ********Comment this code and replace with the code from grade_learners.py since we have to 
        discard the first date column and header row from the input Istanbul.csv.********
        
    inf = open(sys.argv[1])  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    data = np.array([list(map(float,s.strip().split(','))) for s in inf.readlines()])
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # compute how much of the data is training and testing  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_rows = int(0.6* data.shape[0])  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_rows = data.shape[0] - train_rows  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # separate out training and testing data  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    trainX = data[:train_rows,0:-1]  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    trainY = data[:train_rows,-1]  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    testX = data[train_rows:,0:-1]  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    testY = data[train_rows:,-1]'''

    # data_partitions=list()
    testX, testY, trainX, trainY = None, None, None, None
    permutation = None
    author = None

    datafile = sys.argv[1]
    with util.get_learner_data_file(datafile) as f:
        alldata = np.genfromtxt(f, delimiter=',')
        # Skip the date column and header row if we're working on Istanbul data
        if datafile == 'Istanbul.csv':
            alldata = alldata[1:, 1:]

        datasize = alldata.shape[0]
        cutoff = int(datasize * 0.6)
        permutation = np.random.permutation(alldata.shape[0])
        col_permutation = np.random.permutation(alldata.shape[1] - 1)
        train_data = alldata[permutation[:cutoff], :]
        # trainX = train_data[:,:-1]
        trainX = train_data[:, col_permutation]
        trainY = train_data[:, -1]
        test_data = alldata[permutation[cutoff:], :]
        # testX = test_data[:,:-1]
        testX = test_data[:, col_permutation]
        testY = test_data[:, -1]
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"{testX.shape}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"{testY.shape}")

    print(f"{trainX.shape}")
    print(f"{trainY.shape}")
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # create a learner and train it
    ###### RTLearner######
    print("*************RTLearner*************")
    import RTLearner as lrl
    learner = lrl.RTLearner(leaf_size=1, verbose=False)  # constructor
    learner.addEvidence(trainX, trainY)  # training step
    #Y = learner.query(testX)  # query
    print("Author: ", learner.author())
    Calc_stat(trainX, trainY, testX, testY, learner)

    ###### DTLearner######
    print("*************DTLearner*************")
    import DTLearner as dt
    learner = dt.DTLearner(leaf_size=1, verbose=False)  # constructor
    learner.addEvidence(trainX, trainY)  # training step
    #Y = learner.query(testX)  # query
    print("Author: ", learner.author())
    Calc_stat(trainX, trainY, testX, testY, learner)

    ###### BagLearner######
    print("*************BagLearner*************")
    import BagLearner as bl
    import DTLearner as dt
    learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)
    learner.addEvidence(trainX, trainY)
    #Y = learner.query(testX)
    print("Author: ", learner.author())
    Calc_stat(trainX, trainY, testX, testY, learner)

    ###### InsaneLearner######
    print("*************InsaneLearner*************")
    import InsaneLearner as it
    learner = it.InsaneLearner(verbose=False)  # constructor
    learner.addEvidence(trainX, trainY)  # training step
    Y = learner.query(testX)  # query
    print("Author: ", learner.author())
    Calc_stat(trainX, trainY, testX, testY, learner)

    analyse_overfit (trainX, trainY, testX, testY)

    compare_RT_DT_learner (trainX, trainY, testX, testY)

    '''
    ###### LinearRegLearner######
    learner = lrl.LinReLearner(verbose = True) # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(learner.author())  		  	'''

