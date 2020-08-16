"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Sujit Kanti Biswas (replace with your name)
GT User ID: sbiswas67 (replace with your User ID)
GT ID: 903549376 (replace with your GT ID)
"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import random as rand  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class QLearner(object):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        #***** This program implements Q-learner and Dyna-Q learner based on the approach explained by
        #***** Prof. Tucker Balch in his Udacity lectures.
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.verbose = verbose  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.num_actions = num_actions  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.s = 0  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha              # Learning rate
        self.gamma = gamma              # Discount Rate
        self.rar = rar                  # Random action rate
        self.radr = radr                # Random action decay rate
        self.dyna = dyna                # No. of Dyna updates

        # Q Learner table
        self.Q = np.zeros((self.num_states, self.num_actions))

        # T shall contain the probability that (s,a) -> s'. The dimensions are [s,a,s'].
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Initialize Tcount with a non-zero value to avoid 'division-by-zero' error and also very small
        # enough not to affect the true probability value.
        self.Tcount = np.zeros((self.num_states, self.num_actions, self.num_states)) + 0.0000001

        # Reward table, initialized with zero rewards
        self.R = np.zeros((self.num_states, self.num_actions))

  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def querysetstate(self, s):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Update the state without updating the Q-table  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param s: The new state  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @returns: The selected action  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        # Track the state
        self.s = s

        # Determine Next action
        if rand.uniform(0, 1) < self.rar:                  # Choose random action
            self.a = rand.randint(0, self.num_actions - 1)
        else:                                               # Fetch from Q table
            self.a = np.argmax(self.Q[self.s, :])

        if self.verbose: print(f"s = {self.s}, a = {self.a}")

        return self.a
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def query(self,s_prime,r):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Update the Q table and return an action  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param s_prime: The new state  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param r: The reward  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @returns: The selected action  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """

        # Determine Next action either random or based on new state s_prime
        if rand.uniform(0, 1) < self.rar:  # Choose random action
            a_prime = rand.randint(0, self.num_actions - 1)
        else:  # Fetch from Q table
            a_prime = np.argmax(self.Q[s_prime, :])

        # Formula from lecture to update Q table: Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmax(Q[s', a'])])
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * \
                                 (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])])

        # Decay/Reduce the random action rate (rar)
        self.rar = self.rar * self.radr

        if self.dyna > 0:

            # Update T and R matrix
            self.Tcount[self.s][self.a][s_prime] += 1    # count the occurrence of <s,a,s'>

            # T contains the probability of occurrence of <(s,a) -> s'>
            self.T[self.s][self.a][:] = self.Tcount[self.s][self.a][:] / np.sum(self.Tcount[self.s][self.a][:])

            # Calculate reward for <s,a,s'>
            # R'[s,a] = (1 - alpha) * R[s,a] + alpha * r
            self.R[self.s][self.a] = (1 - self.alpha) * self.R[self.s][self.a] + self.alpha * r

            step = 1
            while (step <= self.dyna):

                ############ Hallucinate Experience for n steps ############

                # Randomly select a state and action
                s = rand.randint(0, self.num_states - 1)
                a = rand.randint(0, self.num_actions - 1)

                # Infer the new state(s') from T based on highest probability of <s,a -> s'>
                s_prime_halcnat = np.argmax(self.T[s, a, :])
                r = self.R[s, a]

                # Update Q table
                # Formula from lecture to update Q table: Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmax(Q[s', a'])])
                self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * \
                               (r + self.gamma * self.Q[s_prime_halcnat, np.argmax(self.Q[s_prime_halcnat, :])])

                step += 1

        # Track next action and next state
        self.a = a_prime
        self.s = s_prime
        if self.verbose: print(f"s = {s_prime}, a = {self.a}, r={r}")

        return self.a

    def author(self):
        return 'sbiswas67'
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__=="__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
