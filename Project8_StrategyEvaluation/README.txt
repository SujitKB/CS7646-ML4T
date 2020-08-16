
1) testproject.py - This will generate all the plots and statistics used in report.
PYTHONPATH=../:. python testproject.py

2) experiment1.py - generates data for report for experiment 1.
PYTHONPATH=../:. python experiment1.py

3) experiment2.py - generates data for report for experiment 2.
PYTHONPATH=../:. python experiment2.py

4) ManualStrategy.py - runs manual strategy on insample/outsample data with JPM and generates all report data.
PYTHONPATH=../:. python ManualStrategy.py

5) StrategyLearner.py - ML based logic for trading strategy.
This will be run by the 1) auto grader, 2) experiment1.py, 3) experiment2.py or 4) testproject.py

6) marketsimcode.py - Calculates portfolio value and runs statistics.
This is used by experiment1.py, experiment2.py and ManualStrategy.py

7) indicators.py - Contains calculation logic for market indicators.
This is used by ManualStrategy.py and StrategyLearner.py

8) QLearner.py - Contains QLearner logic.
This is used by StrategyLearner.py

