"""
Karinderya Monte Carlo Simulation
CS3110 - Modelling and Simulation

Question: given uncertain demand, food choices, ingredient prices and food
waste, what is the *distribution* of a karinderya's daily revenue and profit?
How likely is a loss, and which inputs matter most?

MATHEMATICAL MODEL (one simulated day d)
----------------------------------------
Random inputs                                        Distribution
  D   day-to-day demand level                        Normal(1, 0.10), clipped to [0.6, 1.4]
  N_p customers in period p (Morning, Lunch,        Poisson(lambda_p * D)
      Afternoon, Evening)
  T   average eating time of the day (min)           Normal(21.67, 1.5), clipped to [12, 35]
  n_j orders of dish j among the S served            Multinomial(S, p_1..p_4)
  F   ingredient price factor                        Normal(1, 0.06), clipped to [0.7, 1.3]
  W   share of cooked food that is wasted            Beta(2, 18)          (mean 10%)
  U   utilities cost (PHP)                           Normal(300, 30)

Deterministic relationships
  N         = sum_p N_p                                      customers arriving
  C         = floor( K * H * eta / T )                       seating capacity (K tables, H open minutes,
                                                              eta = table efficiency, calibrated on the ABS)
  S         = min(N, C)                                      customers served
  Revenue   = sum_j n_j * price_j
  FoodCost  = F * (1 + W) * sum_j n_j * cost_j
  OpCost    = rent + U + staff + cleaning
  Profit    = Revenue - FoodCost - OpCost

Why these distributions
  Poisson      - counts of independent arrivals in a fixed time window.
  Normal       - many small effects add up (demand level, ingredient prices, utilities).
  Multinomial  - each customer independently picks one of several dishes.
  Beta         - a proportion bounded between 0 and 1 (food waste).

Assumptions: closed daily operation 10 AM - 8 PM, customers who arrive when
the restaurant is at capacity are turned away, menu prices are fixed, days are
independent. Hourly arrival rates and the menu match karinderya_abs.py, so the
two models describe the same business (keep the settings in sync).

Usage
-----
    python karinderya_montecarlo.py
    python karinderya_montecarlo.py --sims 20000 --seed 7
    python karinderya_montecarlo.py --compare-abs      # needs karinderya_abs.py in the same folder
    python karinderya_montecarlo.py --show

Requirements: numpy, pandas, matplotlib
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ======================================================================
# 1. SETTINGS AND ASSUMPTIONS
# ======================================================================

NUM_SIMULATIONS = 10_000
SEED = 42  # set to None for a different result every run

OPENING_TIME = 10  # 10:00 AM
CLOSING_TIME = 20  # 8:00 PM
OPEN_MINUTES = (CLOSING_TIME - OPENING_TIME) * 60

NUM_TABLES = 6
SEATS_PER_TABLE = 4
TOTAL_SEATS = NUM_TABLES * SEATS_PER_TABLE

MENU = {
    "Chicken Adobo": {"price": 80, "food_cost": 45},
    "Pork Menudo": {"price": 75, "food_cost": 40},
    "Fried Fish": {"price": 70, "food_cost": 38},
    "Vegetable": {"price": 60, "food_cost": 30},
}

FOOD_PROBABILITIES = {
    "Chicken Adobo": 0.30,
    "Pork Menudo": 0.25,
    "Fried Fish": 0.25,
    "Vegetable": 0.20,
}

OPERATING_COSTS = {
    "daily_rent": 500,
    "utilities": 300,
    "staff_cost": 1000,
    "cleaning_supplies": 100,
}
DAILY_OPERATING_COST = sum(OPERATING_COSTS.values())

# Expected customers per hour: (start_hour, end_hour, rate). Same as the ABS.
ARRIVAL_RATES = [
    (10, 11, 8),
    (11, 13, 20),
    (13, 14, 25),
    (14, 16, 10),
    (16, 18, 15),
    (18, 20, 22),
]

PERIODS = {
    "Morning": (10, 11),
    "Lunch": (11, 14),
    "Afternoon": (14, 16),
    "Evening": (16, 20),
}

EATING_TIME_PARAMS = (10, 20, 35)  # triangular (min, mode, max) in the ABS
MEAN_EATING_TIME = sum(EATING_TIME_PARAMS) / 3

# Uncertainty parameters
DEMAND_FACTOR_SD = 0.10
INGREDIENT_COST_SD = 0.06
UTILITIES_SD = 30
WASTE_BETA = (2, 18)
EATING_TIME_MEAN_SD = 1.5
TABLE_EFFICIENCY = 0.86  # share of table-time actually used (calibrated on the ABS)

TARGET_PROFIT = 2000  # "good day" threshold for the probability estimate (PHP)


def expected_customers(start_hour, end_hour):
    """Expected arrivals between two hours, from the hourly rates."""
    total = 0.0
    for block_start, block_end, rate in ARRIVAL_RATES:
        overlap = max(0, min(block_end, end_hour) - max(block_start, start_hour))
        total += overlap * rate
    return total


PERIOD_LAMBDA = {name: expected_customers(*hours) for name, hours in PERIODS.items()}

# ======================================================================
# 2. THE MONTE CARLO MODEL
# ======================================================================


def simulate_days(n, rng, num_tables=NUM_TABLES, demand_scale=1.0,
                  price_scale=1.0, cost_scale=1.0, waste=True):
    """
    Simulate n independent days at once (vectorised with numpy).
    The *_scale arguments let us ask "what if?" questions (sensitivity analysis).
    """
    dishes = list(MENU)
    price = np.array([MENU[d]["price"] for d in dishes], dtype=float) * price_scale
    cost = np.array([MENU[d]["food_cost"] for d in dishes], dtype=float) * cost_scale
    probs = np.array([FOOD_PROBABILITIES[d] for d in dishes], dtype=float)
    probs /= probs.sum()

    # 1. Customer demand: Poisson counts around a random daily demand level
    demand_factor = np.clip(rng.normal(1, DEMAND_FACTOR_SD, n), 0.6, 1.4)
    lam = np.array(list(PERIOD_LAMBDA.values()))
    by_period = rng.poisson(lam[None, :] * demand_factor[:, None] * demand_scale)
    arrived = by_period.sum(axis=1)

    # 2. Capacity: how many diners the tables can seat in a day
    eating_time = np.clip(rng.normal(MEAN_EATING_TIME, EATING_TIME_MEAN_SD, n), 12, 35)
    capacity = np.floor(num_tables * OPEN_MINUTES * TABLE_EFFICIENCY / eating_time).astype(int)
    served = np.minimum(arrived, capacity)

    # 3. Food choices: Multinomial(served, probs), drawn as chained binomials
    orders = np.zeros((n, len(dishes)), dtype=int)
    remaining = served.copy()
    p_left = 1.0
    for j in range(len(dishes) - 1):
        orders[:, j] = rng.binomial(remaining, min(1.0, probs[j] / p_left))
        remaining -= orders[:, j]
        p_left -= probs[j]
    orders[:, -1] = remaining

    # 4. Money
    revenue = orders @ price
    ingredient_factor = np.clip(rng.normal(1, INGREDIENT_COST_SD, n), 0.7, 1.3)
    waste_rate = rng.beta(*WASTE_BETA, n) if waste else np.zeros(n)
    food_cost = (orders @ cost) * ingredient_factor * (1 + waste_rate)

    utilities = np.clip(rng.normal(OPERATING_COSTS["utilities"], UTILITIES_SD, n), 0, None)
    fixed = sum(v for k, v in OPERATING_COSTS.items() if k != "utilities")
    operating_cost = fixed + utilities

    profit = revenue - food_cost - operating_cost

    data = {f"{p} Customers": by_period[:, i] for i, p in enumerate(PERIODS)}
    data.update({
        "Customers Generated": arrived,
        "Capacity": capacity,
        "Customers Served": served,
        "Customers Turned Away": arrived - served,
    })
    data.update({f"{d} Orders": orders[:, j] for j, d in enumerate(dishes)})
    data.update({
        "Demand Factor": demand_factor,
        "Ingredient Cost Factor": ingredient_factor,
        "Waste Rate": waste_rate,
        "Revenue": revenue,
        "Food Cost": food_cost,
        "Operating Cost": operating_cost,
        "Estimated Profit": profit,
    })
    return pd.DataFrame(data)


def run_monte_carlo(n=NUM_SIMULATIONS, seed=SEED, **kwargs):
    rng = np.random.default_rng(seed)
    return simulate_days(n, rng, **kwargs)


def margin_per_customer():
    """Average profit per customer before operating costs (no waste)."""
    return sum(FOOD_PROBABILITIES[d] * (MENU[d]["price"] - MENU[d]["food_cost"]) for d in MENU)


def break_even_customers():
    """Customers needed per day to cover operating costs (no waste)."""
    return DAILY_OPERATING_COST / margin_per_customer()


# ======================================================================
# 3. STATISTICS
# ======================================================================


def describe(series):
    return {
        "Mean": series.mean(),
        "Standard Deviation": series.std(),
        "Variance": series.var(),
        "Minimum": series.min(),
        "5th Percentile": series.quantile(0.05),
        "Median": series.median(),
        "95th Percentile": series.quantile(0.95),
        "Maximum": series.max(),
    }


def output_statistics(df):
    columns = ["Customers Generated", "Customers Served", "Revenue",
               "Food Cost", "Estimated Profit"]
    return pd.DataFrame({c: describe(df[c]) for c in columns}).round(2)


def profit_summary(df, target=TARGET_PROFIT):
    profit = df["Estimated Profit"]
    n = len(profit)
    mean, sd = profit.mean(), profit.std()
    margin = 1.96 * sd / np.sqrt(n)
    return {
        "Simulations": n,
        "Average Profit": round(mean, 2),
        "Standard Error": round(sd / np.sqrt(n), 2),
        "95% CI for Average Profit": (round(mean - margin, 2), round(mean + margin, 2)),
        "Profit Std Deviation": round(sd, 2),
        "5th Percentile (bad day)": round(profit.quantile(0.05), 2),
        "95th Percentile (good day)": round(profit.quantile(0.95), 2),
        "Probability of Loss (%)": round((profit < 0).mean() * 100, 2),
        "Probability of Positive Profit (%)": round((profit > 0).mean() * 100, 2),
        f"Probability of Profit >= PHP {target:,} (%)": round((profit >= target).mean() * 100, 2),
        "Days Turned Customers Away (%)": round((df["Customers Turned Away"] > 0).mean() * 100, 2),
        "Average Customers Turned Away": round(df["Customers Turned Away"].mean(), 2),
    }


def print_report(df):
    print("=" * 64)
    print("   KARINDERYA MONTE CARLO SIMULATION RESULTS")
    print("=" * 64)
    for key, value in profit_summary(df).items():
        if isinstance(value, tuple):
            print(f"{key + ':':<44} PHP {value[0]:,.2f} to PHP {value[1]:,.2f}")
        else:
            print(f"{key + ':':<44} {value:,}" if isinstance(value, int) else f"{key + ':':<44} {value:,.2f}")
    print("-" * 64)
    print(f"Break-even customers per day (no waste):     {break_even_customers():.0f}")
    print(f"Average margin per customer (before costs):  PHP {margin_per_customer():.2f}")
    print("=" * 64)
    print("\nOutput statistics:")
    print(output_statistics(df).to_string())


# ======================================================================
# 4. SENSITIVITY ANALYSIS
# ======================================================================
# Every scenario reuses the same random seed ("common random numbers"), so
# differences between scenarios come from the changed input, not from luck.


def scenario_summary(df):
    profit = df["Estimated Profit"]
    return {
        "Mean Profit": profit.mean(),
        "P5": profit.quantile(0.05),
        "P95": profit.quantile(0.95),
        "Mean Served": df["Customers Served"].mean(),
        "Loss %": (profit < 0).mean() * 100,
    }


def table_sensitivity(n=3000, seed=SEED, tables=range(3, 10)):
    rows = []
    for t in tables:
        row = scenario_summary(run_monte_carlo(n, seed, num_tables=t))
        row["Tables"] = t
        rows.append(row)
    return pd.DataFrame(rows)


def demand_sensitivity(n=3000, seed=SEED, scales=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)):
    rows = []
    for s in scales:
        row = scenario_summary(run_monte_carlo(n, seed, demand_scale=s))
        row["Demand Scale"] = s
        rows.append(row)
    return pd.DataFrame(rows)


def tornado_analysis(n=3000, seed=SEED):
    """Change one input at a time (+/-10%, or +/-1 table) and record mean profit."""
    base = run_monte_carlo(n, seed)["Estimated Profit"].mean()
    scenarios = {
        "Customer demand": (dict(demand_scale=0.9), dict(demand_scale=1.1)),
        "Menu prices": (dict(price_scale=0.9), dict(price_scale=1.1)),
        "Food cost": (dict(cost_scale=0.9), dict(cost_scale=1.1)),
        "Number of tables": (dict(num_tables=NUM_TABLES - 1), dict(num_tables=NUM_TABLES + 1)),
    }
    rows = []
    for name, (low, high) in scenarios.items():
        low_profit = run_monte_carlo(n, seed, **low)["Estimated Profit"].mean()
        high_profit = run_monte_carlo(n, seed, **high)["Estimated Profit"].mean()
        rows.append({"Input": name, "Input Lower": low_profit, "Input Higher": high_profit,
                     "Swing": abs(high_profit - low_profit), "Baseline": base})
    return pd.DataFrame(rows).sort_values("Swing", ascending=True).reset_index(drop=True)


# ======================================================================
# 5. VISUALIZATION
# ======================================================================

PINK, BLUE, GREEN, ORANGE, GRAY = "#FF7FA3", "#4B86B4", "#6BBF59", "#F2A33A", "#888888"


def _finish(fig, name, output_dir, show):
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, name), dpi=150)
    if not show:
        plt.close(fig)


def plot_profit_distribution(df, output_dir, show):
    profit = df["Estimated Profit"]
    mean, sd = profit.mean(), profit.std()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(profit, bins=40, density=True, color=PINK, alpha=0.75, edgecolor="white")
    x = np.linspace(profit.min(), profit.max(), 300)
    ax.plot(x, np.exp(-((x - mean) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)),
            color=BLUE, lw=2, label="Normal curve with same mean and SD")
    ax.axvline(mean, color="black", lw=2, label=f"Mean: PHP {mean:,.0f}")
    ax.axvline(profit.quantile(0.05), color=ORANGE, ls="--", lw=2,
               label=f"5th percentile: PHP {profit.quantile(0.05):,.0f}")
    ax.axvline(profit.quantile(0.95), color=GREEN, ls="--", lw=2,
               label=f"95th percentile: PHP {profit.quantile(0.95):,.0f}")
    ax.axvline(0, color="red", lw=1.5, label=f"Break-even (loss chance {(profit < 0).mean() * 100:.1f}%)")
    ax.set_xlabel("Estimated daily profit (PHP)")
    ax.set_ylabel("Density")
    ax.set_title("Monte Carlo Distribution of Daily Profit")
    ax.legend(fontsize=8)
    _finish(fig, "mc_profit_distribution.png", output_dir, show)


def plot_output_distributions(df, output_dir, show):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("Revenue", "Daily revenue (PHP)", BLUE),
              ("Food Cost", "Daily food cost (PHP)", ORANGE),
              ("Customers Generated", "Customers arriving per day", PINK),
              ("Customers Served", "Customers served per day", GREEN)]
    for ax, (col, label, color) in zip(axes.ravel(), panels):
        ax.hist(df[col], bins=35, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(df[col].mean(), color="black", lw=1.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Number of simulations")
        ax.set_title(f"{col} (mean {df[col].mean():,.0f})", fontsize=10)
    _finish(fig, "mc_output_distributions.png", output_dir, show)


def plot_demand_by_period(df, output_dir, show):
    fig, ax = plt.subplots(figsize=(8, 5))
    columns = [f"{p} Customers" for p in PERIODS]
    ax.boxplot([df[c] for c in columns], tick_labels=list(PERIODS), patch_artist=True,
               boxprops=dict(facecolor=PINK, alpha=0.7), medianprops=dict(color="black"))
    ax.set_ylabel("Customers in the period")
    ax.set_title("Simulated Customer Demand by Period of the Day")
    _finish(fig, "mc_demand_by_period.png", output_dir, show)


def plot_profit_vs_customers(df, output_dir, show):
    sample = df.sample(min(len(df), 3000), random_state=0)
    full = sample["Customers Turned Away"] > 0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(sample.loc[~full, "Customers Generated"], sample.loc[~full, "Estimated Profit"],
               s=9, alpha=0.45, color=BLUE, label="All customers served")
    ax.scatter(sample.loc[full, "Customers Generated"], sample.loc[full, "Estimated Profit"],
               s=9, alpha=0.45, color=ORANGE, label="Restaurant full, customers turned away")
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("Customers arriving per day")
    ax.set_ylabel("Estimated daily profit (PHP)")
    ax.set_title("Profit vs. Demand (profit stops growing once the tables are full)")
    ax.legend(fontsize=8)
    _finish(fig, "mc_profit_vs_customers.png", output_dir, show)


def plot_convergence(df, output_dir, show):
    profit = df["Estimated Profit"].to_numpy()
    count = np.arange(1, len(profit) + 1)
    running_mean = np.cumsum(profit) / count
    final_se = profit.std() / np.sqrt(len(profit))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(count, running_mean, color=PINK, lw=2, label="Running average profit")
    ax.axhline(profit.mean(), color="black", ls="--", lw=1, label="Final average")
    ax.fill_between(count, profit.mean() - 1.96 * final_se, profit.mean() + 1.96 * final_se,
                    color=GRAY, alpha=0.2, label="Final 95% confidence interval")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated days (log scale)")
    ax.set_ylabel("Average profit (PHP)")
    ax.set_title("Convergence: more simulations, steadier estimate")
    ax.set_ylim(profit.mean() - 6 * final_se * 10, profit.mean() + 6 * final_se * 10)
    ax.legend(fontsize=8)
    _finish(fig, "mc_convergence.png", output_dir, show)


def plot_table_sensitivity(table_df, output_dir, show):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(table_df["Tables"], table_df["P5"], table_df["P95"], color=PINK, alpha=0.25,
                    label="5th to 95th percentile")
    ax.plot(table_df["Tables"], table_df["Mean Profit"], marker="o", color=PINK, lw=2, label="Mean profit")
    ax.axvline(NUM_TABLES, color="black", ls=":", label="Current number of tables")
    ax.set_xlabel("Number of tables")
    ax.set_ylabel("Daily profit (PHP)")
    ax.set_title("Sensitivity: Profit vs. Number of Tables")
    ax.legend(fontsize=8)
    _finish(fig, "mc_table_sensitivity.png", output_dir, show)


def plot_demand_sensitivity(demand_df, output_dir, show):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(demand_df["Demand Scale"] * 100, demand_df["P5"], demand_df["P95"],
                    color=BLUE, alpha=0.2, label="5th to 95th percentile")
    ax.plot(demand_df["Demand Scale"] * 100, demand_df["Mean Profit"], marker="o", color=BLUE,
            lw=2, label="Mean profit")
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("Customer demand (% of normal)")
    ax.set_ylabel("Daily profit (PHP)")
    ax.set_title("Sensitivity: Profit vs. Customer Demand")
    ax.legend(fontsize=8)
    _finish(fig, "mc_demand_sensitivity.png", output_dir, show)


def plot_tornado(tornado_df, output_dir, show):
    base = tornado_df["Baseline"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, row in tornado_df.iterrows():
        ax.barh(i, row["Input Lower"] - base, left=base, color=ORANGE, alpha=0.85)
        ax.barh(i, row["Input Higher"] - base, left=base, color=GREEN, alpha=0.85)
    ax.axvline(base, color="black", lw=1.5)
    ax.set_yticks(range(len(tornado_df)))
    ax.set_yticklabels(tornado_df["Input"])
    ax.set_xlabel("Average daily profit (PHP)")
    ax.set_title("Which inputs matter most? (orange: input lowered, green: input raised)")
    _finish(fig, "mc_tornado.png", output_dir, show)


def make_plots(df, table_df, demand_df, tornado_df, output_dir, show):
    plot_profit_distribution(df, output_dir, show)
    plot_output_distributions(df, output_dir, show)
    plot_demand_by_period(df, output_dir, show)
    plot_profit_vs_customers(df, output_dir, show)
    plot_convergence(df, output_dir, show)
    plot_table_sensitivity(table_df, output_dir, show)
    plot_demand_sensitivity(demand_df, output_dir, show)
    plot_tornado(tornado_df, output_dir, show)


# ======================================================================
# 6. COMPARISON WITH THE AGENT-BASED MODEL (optional)
# ======================================================================


def compare_with_abs(n_days=300, seed=SEED):
    """
    Run karinderya_abs.py (must be in the same folder) and compare its daily
    averages with this Monte Carlo model. Waste is switched off here because
    the ABS does not model it, so the comparison is like for like.
    """
    import random

    import karinderya_abs as abs_model

    random.seed(seed)
    np.random.seed(seed)
    abs_df = abs_model.run_many_days(n_days)
    mc_df = run_monte_carlo(10_000, seed, waste=False)

    rows = []
    for label, abs_col, mc_col in [
        ("Customers generated", "Customers Generated", "Customers Generated"),
        ("Customers served", "Customers Served", "Customers Served"),
        ("Revenue (PHP)", "Revenue", "Revenue"),
        ("Food cost (PHP)", "Food Cost", "Food Cost"),
        ("Estimated profit (PHP)", "Estimated Profit", "Estimated Profit"),
    ]:
        a, m = abs_df[abs_col].mean(), mc_df[mc_col].mean()
        rows.append({"Measure": label, "Agent-Based": round(a, 1), "Monte Carlo": round(m, 1),
                     "Difference %": round((m - a) / a * 100, 1)})
    return pd.DataFrame(rows)


# ======================================================================
# 7. MAIN
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Karinderya Monte Carlo simulation")
    parser.add_argument("--sims", type=int, default=NUM_SIMULATIONS, help="number of simulated days")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed (omit for the default)")
    parser.add_argument("--output-dir", default="output_montecarlo", help="folder for figures and CSV files")
    parser.add_argument("--show", action="store_true", help="display the plot windows")
    parser.add_argument("--compare-abs", action="store_true", help="compare with karinderya_abs.py")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = run_monte_carlo(args.sims, args.seed)
    df.round(2).to_csv(os.path.join(args.output_dir, "mc_results.csv"), index=False)
    print_report(df)

    table_df = table_sensitivity(seed=args.seed)
    demand_df = demand_sensitivity(seed=args.seed)
    tornado_df = tornado_analysis(seed=args.seed)
    table_df.round(2).to_csv(os.path.join(args.output_dir, "mc_table_sensitivity.csv"), index=False)
    demand_df.round(2).to_csv(os.path.join(args.output_dir, "mc_demand_sensitivity.csv"), index=False)

    print("\nSensitivity to the number of tables:")
    print(table_df[["Tables", "Mean Profit", "P5", "P95", "Mean Served", "Loss %"]].round(1).to_string(index=False))

    make_plots(df, table_df, demand_df, tornado_df, args.output_dir, args.show)
    print(f"\nFigures and CSV files saved in '{args.output_dir}/'")

    if args.compare_abs:
        try:
            print("\nComparison with the agent-based model (waste switched off):")
            print(compare_with_abs(seed=args.seed).to_string(index=False))
        except ImportError:
            print("karinderya_abs.py was not found in this folder, so the comparison was skipped.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
