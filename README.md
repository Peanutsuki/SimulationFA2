Karinderya Simulation

CS3110 Modelling and Simulation – Output Activity

Monte Carlo and agent-based simulations of a small karinderya (neighbourhood eatery) open 10 AM to 8 PM, with an interactive dashboard. The simulations estimate daily profit, the risk of a loss, customer waiting times, and the effect of changing tables, demand, prices and food cost.

Live dashboard: [add your Streamlit link here]

The two models
Model	File	What it does
Monte Carlo	karinderya_montecarlo.py	Simulates thousands of random days (Poisson demand, Normal prices and costs, Multinomial dish choices, Beta food waste) and reports the distribution of revenue and profit, with sensitivity analysis
Agent-Based	karinderya_abs.py	Simulates each customer arriving, queueing, eating and leaving minute by minute, and can create an animated GIF of the day

karinderya_dashboard.py is a Streamlit dashboard that runs both models with sliders for tables, demand, prices, food cost and eating time. The notebooks (Karinderya_ABS.ipynb, Karinderya_MonteCarlo.ipynb) are the same models in notebook form.

Setup

Requires Python 3.10 or newer.

pip install -r requirements.txt
Run
python karinderya_abs.py --days 100 --seed 42
python karinderya_montecarlo.py --compare-abs
python -m streamlit run karinderya_dashboard.py
Figures, CSV files and the GIF are saved in the output folder. Use --output-dir name to choose another folder.
Add --no-gif to the agent-based script to skip the animation (it takes about 30 seconds).
Use python -m streamlit run ... for the dashboard. Running the file with the normal Run button shows nothing.
All three .py files must be in the same folder.
Sample results (seed 42)
	Monte Carlo (10,000 days)	Agent-Based (100 days)
Average daily profit	about PHP 2,245	about PHP 2,808
Chance of a loss	about 0.05%	not estimated
Customers served per day	about 141	about 143

The Monte Carlo profit is lower because it also models food waste and changing ingredient prices. With waste switched off, the two models agree within about 1.3%.

Files
karinderya_abs.py           agent-based simulation
karinderya_montecarlo.py    Monte Carlo simulation
karinderya_dashboard.py     Streamlit dashboard
Karinderya_ABS.ipynb        notebook version (agent-based)
Karinderya_MonteCarlo.ipynb notebook version (Monte Carlo)
requirements.txt            Python packages
Author

- :3
