"""Assess a betting strategy.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Sujit Kanti Biswas
GT User ID: sbiswas67
GT ID: 903549376
"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def author():  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return 'sbiswas67'  # replace tb34 with your Georgia Tech username.
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def gtid():  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return 903549376  # replace with your GT ID number
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def get_spin_result(win_prob):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    result = False  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if np.random.random() <= win_prob:  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        result = True  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return result  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def test_code():

    #Total 38 pockets with 18 Black and 18 Red and 2 Green
    #Hence, probability of getting Black or Red is 18/38
    win_prob = 18.0 / 38    # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

    # add your code here to implement the experiments
    ######### Experiment-1a #########
    simulation_times = 10
    simulation_results = np.zeros((simulation_times, 1001))
    win_count = 0
    for loop in range(simulation_times):
        simulation_results[loop] = roulette_Exp1(win_prob)
        if simulation_results[loop][-1] == 80:
            win_count += 1                      #Winning simulation
    #print("Probability of winnings $80 with ", simulation_times, " simulations is ", win_count/simulation_times)
    result_mean = np.mean(simulation_results, axis=0)
    #print("Expected value for Experiment 1 for 10 simulations is ", result_mean)
    plt.figure(0)
    plt.xlabel('Iteration')
    plt.ylabel('Winnings')
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.title('Figure 1')
    for loop in range(simulation_times):
        plt.plot(pd.Series(simulation_results[loop]))
    plt.savefig('Figure-1.png')

    ######### Experiment-1b #########
    simulation_times = 1000
    simulation_results = np.zeros((simulation_times, 1001))
    win_count = 0
    for loop in range(simulation_times):
        simulation_results[loop] = roulette_Exp1(win_prob)
        if simulation_results[loop][-1] == 80:
            win_count += 1                      #Winning simulation
    #print("Probability of winnings $80 with ", simulation_times, " simulations is ", win_count / simulation_times)

    result_mean=np.mean(simulation_results,axis=0)
    #print("Expected value for Experiment 1 with 1000 simulations is ", result_mean)
    result_std=np.std(simulation_results,axis=0)
    upper=result_mean+result_std
    bottom=result_mean-result_std
    plt.figure(1)
    plt.xlabel('Iteration')
    plt.ylabel('Winnings')
    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.title('Figure 2')
    middle_line,=plt.plot(pd.Series(result_mean),label = 'Mean')
    upper_line,=plt.plot(pd.Series(upper),label = 'Upper')
    bottom_line,=plt.plot(pd.Series(bottom),label = 'Bottom')
    plt.legend(handles=[middle_line, upper_line,bottom_line], loc=4)
    plt.savefig('Figure-2.png')

    ######### Experiment-1c #########
    result_median=np.median(simulation_results,axis=0)
    result_std=np.std(simulation_results,axis=0)
    upper=result_median+result_std
    bottom=result_median-result_std
    plt.figure(2)
    plt.xlabel('Iteration')
    plt.ylabel('Winnings')
    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.title('Figure 3')
    middle_line,=plt.plot(pd.Series(result_median),label = 'Median')
    upper_line,=plt.plot(pd.Series(upper),label = 'Upper')
    bottom_line,=plt.plot(pd.Series(bottom),label = 'Bottom')
    plt.legend(handles=[middle_line, upper_line,bottom_line], loc=4)
    plt.savefig('Figure-3.png')


    ######### Experiment-2 #########
    simulation_times = 1000
    simulation_results = np.zeros((simulation_times, 1001))
    win_count = 0
    for loop in range(simulation_times):
        simulation_results[loop] = roulette_Exp2(win_prob)
        if simulation_results[loop][-1] != -256 and simulation_results[loop][-1] != 80:
            win_count += 1  # Winning simulation

    #print("Exp2: Probability of winnings $80 with ", simulation_times, " simulations is ", win_count / simulation_times)

    result_mean=np.mean(simulation_results,axis=0)
    #print("Expected value for Experiment 2 is ", result_mean)
    result_std=np.std(simulation_results,axis=0)
    upper = result_mean + result_std
    bottom = result_mean - result_std
    plt.figure(3)
    plt.xlabel('Iteration')
    plt.ylabel('Winnings')
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.title('Figure 4')
    middle_line, = plt.plot(pd.Series(result_mean), label='Mean')
    upper_line, = plt.plot(pd.Series(upper), label='Upper')
    bottom_line, = plt.plot(pd.Series(bottom), label='Bottom')
    plt.legend(handles=[middle_line, upper_line, bottom_line], loc=4)
    plt.savefig('Figure-4.png')
    
    ## Experiment 2b #####
    result_median = np.median(simulation_results, axis=0)
    result_std = np.std(simulation_results, axis=0)
    upper = result_median + result_std
    bottom = result_median - result_std
    plt.figure(4)
    plt.xlabel('Iteration')
    plt.ylabel('Winnings')
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.title('Figure 5')
    middle_line, = plt.plot(pd.Series(result_median), label='Median')
    upper_line, = plt.plot(pd.Series(upper), label='Upper')
    bottom_line, = plt.plot(pd.Series(bottom), label='Bottom')
    plt.legend(handles=[middle_line, upper_line, bottom_line], loc=4)
    plt.savefig('Figure-5.png')

def roulette_Exp1 (win_prob):

    episode_winnings = 0
    winning_bet = np.zeros(1001)    #Hold 1000 bets starting with zero
    i = 0

    while episode_winnings < 80 and i < 1000:
        won = False
        bet_amount = 1  # Starting Bet

        while not won:
            #wager bet_amount on black
            won = get_spin_result(win_prob)
            i+=1

            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2
            winning_bet[i] = episode_winnings

    if episode_winnings >= 80:
        winning_bet[i+1:] = 80

    return winning_bet

def roulette_Exp2 (win_prob):

    episode_winnings = 0
    winning_bet = np.zeros(1001)    #Hold 1000 bets starting with zero
    i = 0

    while episode_winnings < 80 and episode_winnings > -256 and i < 1000:
        won = False
        bet_amount = 1  # Starting Bet

        while not won and i < 1000:
            #wager bet_amount on black
            won = get_spin_result(win_prob)
            i += 1


            if won == True:
                episode_winnings = episode_winnings + bet_amount
                winning_bet[i] = episode_winnings
            else:
                episode_winnings = episode_winnings - bet_amount
                winning_bet[i] = episode_winnings
                bet_amount = bet_amount * 2
                if bet_amount > (256 + episode_winnings):
                    bet_amount = (256 + episode_winnings)


    if episode_winnings >= 80:
        winning_bet[i+1:] = 80

    if episode_winnings <= -256:
        winning_bet[i+1:] = -256

    #print(winning_bet)
    return winning_bet

if __name__ == "__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_code()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
