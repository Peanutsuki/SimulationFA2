"""
Karinderya Simulation Dashboard
CS3110 - Modelling and Simulation

An interactive dashboard for both simulations:
  * Monte Carlo        (karinderya_montecarlo.py)
  * Agent-Based Model  (karinderya_abs.py)

Put the three .py files in the same folder, then run:

    pip install streamlit numpy pandas matplotlib pillow
    streamlit run karinderya_dashboard.py

A browser tab opens. Use the sliders in the left sidebar to change the
scenario; every chart and number updates.
"""

import copy
import os
import random
import sys
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")  # Streamlit draws the figures itself

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

st.set_page_config(page_title="Karinderya Simulation Dashboard", page_icon="🍲", layout="wide")

try:
    import karinderya_abs as abs_model
    import karinderya_montecarlo as mc
except ImportError as error:
    st.error(
        f"Could not import the simulation files ({error}). Make sure karinderya_abs.py and "
        "karinderya_montecarlo.py are in the same folder as this dashboard."
    )
    st.stop()

PINK, BLUE, GREEN, ORANGE, GRAY = mc.PINK, mc.BLUE, mc.GREEN, mc.ORANGE, mc.GRAY
DEFAULT_EAT = round(mc.MEAN_EATING_TIME, 1)

# ======================================================================
# Scenario plumbing: apply the sidebar settings to the models temporarily
# ======================================================================


@contextmanager
def abs_settings(tables, demand, price, cost, eat_scale):
    """Temporarily change the agent-based model's settings, then restore them."""
    saved = (abs_model.NUM_TABLES, abs_model.ARRIVAL_RATES,
             copy.deepcopy(abs_model.MENU), abs_model.EATING_TIME_PARAMS)
    try:
        abs_model.NUM_TABLES = tables
        abs_model.ARRIVAL_RATES = [(s, e, r * demand) for s, e, r in saved[1]]
        abs_model.MENU = {
            name: {**item, "price": item["price"] * price, "food_cost": item["food_cost"] * cost}
            for name, item in saved[2].items()
        }
        abs_model.EATING_TIME_PARAMS = tuple(x * eat_scale for x in saved[3])
        yield
    finally:
        (abs_model.NUM_TABLES, abs_model.ARRIVAL_RATES,
         abs_model.MENU, abs_model.EATING_TIME_PARAMS) = saved


@contextmanager
def mc_settings(eat_scale):
    saved = mc.MEAN_EATING_TIME
    try:
        mc.MEAN_EATING_TIME = saved * eat_scale
        yield
    finally:
        mc.MEAN_EATING_TIME = saved


@st.cache_data(show_spinner=False)
def run_mc(n, seed, tables, demand, price, cost, eat_scale, waste):
    with mc_settings(eat_scale):
        return mc.run_monte_carlo(n, seed, num_tables=tables, demand_scale=demand,
                                  price_scale=price, cost_scale=cost, waste=waste)


@st.cache_data(show_spinner=False)
def run_mc_table_sweep(seed, demand, price, cost, eat_scale, waste):
    rows = []
    with mc_settings(eat_scale):
        for t in range(3, 11):
            df = mc.run_monte_carlo(3000, seed, num_tables=t, demand_scale=demand,
                                    price_scale=price, cost_scale=cost, waste=waste)
            row = mc.scenario_summary(df)
            row["Tables"] = t
            rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_abs(days, seed, tables, demand, price, cost, eat_scale):
    with abs_settings(tables, demand, price, cost, eat_scale):
        random.seed(seed)
        np.random.seed(seed)
        multi = abs_model.run_many_days(days)

        # one detailed day, with minute-by-minute history
        customers = abs_model.generate_arrivals()
        karinderya, history = abs_model.run_simulation(customers, record_history=True)
        daily = abs_model.get_daily_results(karinderya, len(customers))
        detail = abs_model.customer_dataframe(karinderya)
    history_df = pd.DataFrame(history)[["time", "waiting", "eating", "completed", "revenue", "dirty_dishes"]]
    arrivals = np.array([c.arrival_time for c in customers]) / 60
    return {"multi": multi, "history": history_df, "detail": detail, "daily": daily, "arrivals": arrivals}


def make_animation_gif(path, seed, tables, demand, price, cost, eat_scale):
    with abs_settings(tables, demand, price, cost, eat_scale):
        random.seed(seed)
        np.random.seed(seed)
        customers = abs_model.generate_arrivals()
        _, history = abs_model.run_simulation(customers, record_history=True)
        fig, animation = abs_model.create_karinderya_animation(history, customers)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        abs_model.save_karinderya_animation(animation, path)
        plt.close(fig)


# ======================================================================
# Small plotting helpers (each returns a figure)
# ======================================================================


def show(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def hist(series, xlabel, color, bins=35, title=None):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(series, bins=bins, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(series.mean(), color="black", lw=1.5, label=f"Mean: {series.mean():,.0f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    return fig


def profit_histogram(profit, target):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(profit, bins=40, color=PINK, alpha=0.8, edgecolor="white")
    ax.axvline(profit.mean(), color="black", lw=2, label=f"Mean: PHP {profit.mean():,.0f}")
    ax.axvline(profit.quantile(0.05), color=ORANGE, ls="--", lw=2, label="5th percentile")
    ax.axvline(profit.quantile(0.95), color=GREEN, ls="--", lw=2, label="95th percentile")
    ax.axvline(0, color="red", lw=1.5, label="Break-even")
    ax.axvline(target, color=BLUE, ls=":", lw=2, label=f"Target: PHP {target:,}")
    ax.set_xlabel("Daily profit (PHP)")
    ax.set_ylabel("Number of simulations")
    ax.set_title("Distribution of daily profit", fontsize=11)
    ax.legend(fontsize=7)
    return fig


# ======================================================================
# Sidebar
# ======================================================================

st.sidebar.title("Scenario settings")
st.sidebar.caption("These settings apply to both simulations.")
tables = st.sidebar.slider("Number of tables", 3, 10, mc.NUM_TABLES)
demand = st.sidebar.slider("Customer demand (% of normal)", 60, 140, 100, 5) / 100
price = st.sidebar.slider("Menu prices (% of normal)", 80, 120, 100, 5) / 100
cost = st.sidebar.slider("Food cost (% of normal)", 80, 120, 100, 5) / 100
eat_minutes = st.sidebar.slider("Average eating time (minutes)", 12.0, 35.0, DEFAULT_EAT, 0.5)
eat_scale = 1.0 if abs(eat_minutes - DEFAULT_EAT) < 1e-9 else eat_minutes / mc.MEAN_EATING_TIME
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
st.sidebar.caption(
    f"Normal settings: {mc.NUM_TABLES} tables, 100% demand, prices and costs, "
    f"{DEFAULT_EAT} minutes eating time."
)

st.title("🍲 Karinderya Simulation Dashboard")
st.caption("CS3110 Modelling and Simulation: Monte Carlo and Agent-Based simulation of a small karinderya, 10 AM to 8 PM.")

tab_about, tab_mc, tab_abs, tab_compare = st.tabs(
    ["About the model", "Monte Carlo", "Agent-Based", "Compare both"]
)

# ======================================================================
# About
# ======================================================================

with tab_about:
    st.subheader("What this dashboard does")
    st.write(
        "A karinderya serves customers from 10 AM to 8 PM. Customer numbers, food choices, ingredient "
        "prices and food waste are uncertain, so we simulate the business in two ways: **Monte Carlo** "
        "(thousands of random days, to see the range of outcomes) and an **Agent-Based Model** "
        "(each customer is an agent who arrives, queues, eats and leaves)."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Monte Carlo: random inputs**")
        st.table(pd.DataFrame({
            "Variable": ["Daily demand level", "Customers per period", "Average eating time",
                         "Dish choices", "Ingredient price factor", "Food waste", "Utilities"],
            "Distribution": ["Normal(1, 0.10)", "Poisson(λ·D)", "Normal(21.67, 1.5)",
                             "Multinomial", "Normal(1, 0.06)", "Beta(2, 18)", "Normal(300, 30)"],
        }))
    with right:
        st.markdown("**Agent-Based Model: rules**")
        st.markdown(
            "- Customers arrive at random (Poisson), more at lunch and early evening.\n"
            "- Each customer picks a dish at random and eats for a triangular time (10, 20, 35 minutes).\n"
            "- A customer takes a free table, or waits in a first-come first-served queue.\n"
            "- When finished, the customer pays and leaves; dirty dishes pile up and are washed every 30 minutes."
        )
    st.markdown("**Monte Carlo model**")
    st.latex(r"S=\min(N,C),\quad C=\left\lfloor \frac{K\,H\,\eta}{T}\right\rfloor")
    st.latex(r"\text{Profit}=\sum_j n_j\,\text{price}_j-F(1+W)\sum_j n_j\,\text{cost}_j-\text{OperatingCost}")

# ======================================================================
# Monte Carlo
# ======================================================================

with tab_mc:
    st.subheader("Monte Carlo simulation")
    c1, c2, c3 = st.columns(3)
    n_sims = c1.slider("Number of simulated days", 1000, 50000, 10000, 1000)
    waste = c2.checkbox("Include food waste", value=True)
    target = int(c3.number_input("Target daily profit (PHP)", value=mc.TARGET_PROFIT, step=100))

    df = run_mc(n_sims, seed, tables, demand, price, cost, eat_scale, waste)
    profit = df["Estimated Profit"]
    summary = mc.profit_summary(df, target)
    low, high = summary["95% CI for Average Profit"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Average profit", f"PHP {profit.mean():,.0f}", help=f"95% CI: PHP {low:,.0f} to {high:,.0f}")
    m2.metric("Chance of a loss", f"{summary['Probability of Loss (%)']:.2f}%")
    m3.metric("Bad day (5th pct)", f"PHP {profit.quantile(0.05):,.0f}")
    m4.metric("Good day (95th pct)", f"PHP {profit.quantile(0.95):,.0f}")
    m5.metric(f"Chance profit ≥ {target:,}", f"{summary[f'Probability of Profit >= PHP {target:,} (%)']:.1f}%")

    m6, m7, m8, m9 = st.columns(4)
    m6.metric("Customers arriving", f"{df['Customers Generated'].mean():.0f}")
    m7.metric("Customers served", f"{df['Customers Served'].mean():.0f}")
    m8.metric("Turned away per day", f"{df['Customers Turned Away'].mean():.0f}")
    m9.metric("Days restaurant is full", f"{summary['Days Turned Customers Away (%)']:.0f}%")

    a, b = st.columns(2)
    with a:
        show(profit_histogram(profit, target))
    with b:
        sample = df.sample(min(len(df), 3000), random_state=0)
        full = sample["Customers Turned Away"] > 0
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.scatter(sample.loc[~full, "Customers Generated"], sample.loc[~full, "Estimated Profit"],
                   s=8, alpha=0.45, color=BLUE, label="All customers served")
        ax.scatter(sample.loc[full, "Customers Generated"], sample.loc[full, "Estimated Profit"],
                   s=8, alpha=0.45, color=ORANGE, label="Restaurant full")
        ax.axhline(0, color="red", lw=1)
        ax.set_xlabel("Customers arriving per day")
        ax.set_ylabel("Daily profit (PHP)")
        ax.set_title("Profit stops growing when the tables are full", fontsize=11)
        ax.legend(fontsize=8)
        show(fig)

    a, b = st.columns(2)
    with a:
        show(hist(df["Revenue"], "Daily revenue (PHP)", BLUE, title="Revenue"))
    with b:
        show(hist(df["Food Cost"], "Daily food cost (PHP)", ORANGE, title="Food cost"))

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        columns = [f"{p} Customers" for p in mc.PERIODS]
        ax.boxplot([df[c] for c in columns], tick_labels=list(mc.PERIODS), patch_artist=True,
                   boxprops=dict(facecolor=PINK, alpha=0.7), medianprops=dict(color="black"))
        ax.set_ylabel("Customers in the period")
        ax.set_title("Demand by period of the day", fontsize=11)
        show(fig)
    with b:
        sweep = run_mc_table_sweep(seed, demand, price, cost, eat_scale, waste)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.fill_between(sweep["Tables"], sweep["P5"], sweep["P95"], color=PINK, alpha=0.25,
                        label="5th to 95th percentile")
        ax.plot(sweep["Tables"], sweep["Mean Profit"], marker="o", color=PINK, lw=2, label="Mean profit")
        ax.axvline(tables, color="black", ls=":", label="Selected tables")
        ax.set_xlabel("Number of tables")
        ax.set_ylabel("Daily profit (PHP)")
        ax.set_title("What if we had more or fewer tables?", fontsize=11)
        ax.legend(fontsize=8)
        show(fig)

    running = profit.cumsum() / np.arange(1, len(profit) + 1)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(np.arange(1, len(profit) + 1), running, color=PINK, lw=2)
    ax.axhline(profit.mean(), color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated days (log scale)")
    ax.set_ylabel("Average profit (PHP)")
    ax.set_title("Convergence: the average settles as simulations increase", fontsize=11)
    show(fig)

    with st.expander("Statistics table and raw results"):
        st.dataframe(mc.output_statistics(df))
        st.dataframe(df.round(2).head(200))
        st.download_button("Download all results (CSV)", df.round(2).to_csv(index=False).encode(),
                           file_name="mc_results.csv", mime="text/csv")

# ======================================================================
# Agent-Based
# ======================================================================

with tab_abs:
    st.subheader("Agent-based simulation")
    days = st.slider("Number of simulated days", 10, 300, 50, 10)
    result = run_abs(days, seed, tables, demand, price, cost, eat_scale)
    multi, history, detail, daily = result["multi"], result["history"], result["detail"], result["daily"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Average profit", f"PHP {multi['Estimated Profit'].mean():,.0f}")
    m2.metric("Customers arriving", f"{multi['Customers Generated'].mean():.0f}")
    m3.metric("Customers served", f"{multi['Customers Served'].mean():.0f}")
    m4.metric("Average wait", f"{multi['Average Waiting Time'].mean():.1f} min")
    m5.metric("Longest wait (avg of days)", f"{multi['Maximum Waiting Time'].mean():.0f} min")

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(multi["Day"], multi["Estimated Profit"], marker="o", ms=3, color=PINK)
        ax.axhline(multi["Estimated Profit"].mean(), color="black", ls="--", lw=1)
        ax.set_xlabel("Simulated day")
        ax.set_ylabel("Estimated profit (PHP)")
        ax.set_title("Profit across days", fontsize=11)
        show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(multi["Day"], multi["Average Waiting Time"], marker="o", ms=3, color=BLUE)
        ax.set_xlabel("Simulated day")
        ax.set_ylabel("Average waiting time (minutes)")
        ax.set_title("Waiting time across days", fontsize=11)
        show(fig)

    st.markdown("**One detailed day**")
    st.caption(
        f"Day detail: {daily['Customers Generated']} customers arrived, {daily['Customers Served']} served, "
        f"{daily['Customers Still Waiting']} still waiting at closing, profit PHP {daily['Estimated Profit']:,.0f}."
    )
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(history["time"] / 60, history["waiting"], color=PINK, lw=2, label="Waiting in queue")
        ax.plot(history["time"] / 60, history["eating"], color=GREEN, lw=2, label="Eating at tables")
        ax.set_xlabel("Hour of the day")
        ax.set_ylabel("People")
        ax.set_title("Queue and tables through the day", fontsize=11)
        ax.legend(fontsize=8)
        show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(result["arrivals"], bins=20, color=BLUE, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Hour of the day")
        ax.set_ylabel("Customers arriving")
        ax.set_title("Arrivals (more at lunch and early evening)", fontsize=11)
        show(fig)

    a, b = st.columns(2)
    with a:
        if len(detail):
            show(hist(detail["Waiting Time"], "Waiting time (minutes)", ORANGE, 20, "Customer waiting times"))
    with b:
        if len(detail):
            counts = detail["Order"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.bar(counts.index, counts.values, color=[PINK, BLUE, GREEN, ORANGE][: len(counts)])
            ax.set_ylabel("Orders")
            ax.set_title("Food orders", fontsize=11)
            plt.setp(ax.get_xticklabels(), rotation=20)
            show(fig)

    st.markdown("**Animation**")
    gif_paths = [os.path.join(HERE, folder, "karinderya_day.gif") for folder in ("output_abs", "output")]
    gif_found = next((p for p in gif_paths if os.path.exists(p)), None)
    if st.button("Create animation with the current settings (about 30 seconds)"):
        with st.spinner("Drawing the animation..."):
            make_animation_gif(gif_paths[0], seed, tables, demand, price, cost, eat_scale)
        gif_found = gif_paths[0]
    if gif_found:
        st.image(gif_found, caption=f"Customer flow animation ({os.path.relpath(gif_found, HERE)})")
    else:
        st.info("No animation yet. Press the button above to create one.")

    with st.expander("Daily results table"):
        st.dataframe(multi.round(2))

# ======================================================================
# Compare
# ======================================================================

with tab_compare:
    st.subheader("Do the two models agree?")
    st.write(
        "Food waste is switched off for the Monte Carlo side here, because the agent-based model "
        "does not include it. Both use the scenario in the sidebar."
    )
    abs_result = run_abs(days, seed, tables, demand, price, cost, eat_scale)["multi"]
    mc_nowaste = run_mc(10000, seed, tables, demand, price, cost, eat_scale, False)

    rows = []
    for label, column in [("Customers arriving", "Customers Generated"), ("Customers served", "Customers Served"),
                          ("Revenue (PHP)", "Revenue"), ("Food cost (PHP)", "Food Cost"),
                          ("Estimated profit (PHP)", "Estimated Profit")]:
        a_mean, m_mean = abs_result[column].mean(), mc_nowaste[column].mean()
        rows.append({"Measure": label, "Agent-Based": round(a_mean, 1), "Monte Carlo": round(m_mean, 1),
                     "Difference (%)": round((m_mean - a_mean) / a_mean * 100, 1)})
    st.table(pd.DataFrame(rows))

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.hist(mc_nowaste["Estimated Profit"], bins=40, density=True, color=PINK, alpha=0.6,
            label="Monte Carlo (10,000 days)")
    ax.hist(abs_result["Estimated Profit"], bins=15, density=True, color=BLUE, alpha=0.6,
            label=f"Agent-Based ({days} days)")
    ax.set_xlabel("Daily profit (PHP)")
    ax.set_ylabel("Density")
    ax.set_title("Profit distribution: Monte Carlo vs. Agent-Based", fontsize=11)
    ax.legend(fontsize=8)
    show(fig)

    st.caption(
        "The agent-based model follows every customer minute by minute, while the Monte Carlo model "
        "uses a seating-capacity formula calibrated on it, so small differences are expected."
    )
