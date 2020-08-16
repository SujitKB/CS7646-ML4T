import numpy as np, LinRegLearner as lrl, DTLearner as dt, RTLearner as rt, BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.bag_count = 20
        self.learners = []
        self.verbose = verbose
        for i in range(self.bag_count):
            self.learners.append(bl.BagLearner(lrl.LinRegLearner, kwargs={}, bags=20, verbose=self.verbose))
    def author(self):
        return 'sbiswas67' # replace tb34 with your Georgia Tech username
    def addEvidence(self,dataX,dataY):
        for i in range(self.bag_count):
            self.learners[i].addEvidence(dataX, dataY)
    def query (self,points):
        Y = []
        for i in range (self.bag_count):
            Y.append(self.learners[i].query(points))
            return np.mean(Y, axis=0)