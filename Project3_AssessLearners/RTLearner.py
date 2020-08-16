"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class RTLearner(object):
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(self, leaf_size = 1, verbose = False):

        self.tree = None
        self.leaf_size = leaf_size
        self.verbose = verbose
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def author(self):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        return 'sbiswas67' # replace tb34 with your Georgia Tech username

  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def addEvidence(self,dataX,dataY):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Add training data to learner  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param dataX: X values of data to add  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param dataY: the Y training values  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """

        self.tree = self.build_tree(dataX, dataY)

  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def query (self,points):
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Estimate a set of test points given the model we built.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param points: should be a numpy array with each row corresponding to a specific query.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @returns the estimated values according to the saved model.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

        estimate = []
        # Check the number of rows in array 'points'
        row_count = points.shape[0]  # [0] return number of rows, [1] return number of columns
        for row in range(0, row_count):     # Repeat for each rows in array 'points'

            #Pass the current row to queryDT() to determine corresponding value
            value = self.queryRT (points[row, :])
            estimate.append(float(value))

        return estimate

    def queryRT (self, points_record):
        row = 0
        while (self.tree [row, 0] != float('-1')):
            feature_val = self.tree[row, 0]     #column 0 of the record
            Split_Val   = self.tree[row, 1]     #column 1 of the record

            if points_record [int(float(feature_val))] <= float(Split_Val):
                row = row + int(float(self.tree[row, 2]))  # pick location of Left_Tree
            else:
                row = row + int(float(self.tree[row, 3]))  # pick location of Right_Tree

        return self.tree[row, 1]

    def build_tree(self, dataX, dataY):

        '''# Here -1 is used to mark the node as leaf in the numpy array
        if dataX.shape[0] <= self.leaf_size:
            #Number of sample (or row) of data available is less or equal to leaf_size


        if np.all(dataY == dataY[0]):
            #All the samples (or rows) have same Y value
            return np.array([-1, np.mean(dataY), np.nan, np.nan])'''
        if len(np.unique(dataY)) == 1 or dataX.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(dataY), np.nan, np.nan]])
        else:
            # Determine best feature i randomly to split on, instead of using correllation
            # Return a random column (or feature) index of the array [from 0 to (column - 1)]
            random_idx = np.random.randint(0, dataX.shape[1] - 1)

            # median of all values in column of best feature (i.e., max correlation)
            SplitVal = np.median(dataX[:,random_idx])

            # Find the index range for the subtree recursion
            lefttree_index  = dataX[:,random_idx] <= SplitVal
            righttree_index = dataX[:,random_idx] > SplitVal

            unique = np.unique(lefttree_index)
            if len(unique) == 1:
                return np.array([[-1, np.mean(dataY[lefttree_index == unique[0]]), np.nan, np.nan]])

            lefttree = self.build_tree(dataX[lefttree_index], dataY[lefttree_index])
            righttree = self.build_tree(dataX[righttree_index], dataY[righttree_index])

            root = np.array([[random_idx, SplitVal, 1, lefttree.shape[0] + 1]])
            tree = np.concatenate((root, lefttree, righttree), axis=0)    #rows concatenation

            return tree

if __name__=="__main__":
    print("the secret clue is 'zzyzx'")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
