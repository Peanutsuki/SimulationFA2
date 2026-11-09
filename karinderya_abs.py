"""
Karinderya Agent-Based Simulation (ABS)
CS3110 - Modelling and Simulation

Simulates one operating day (10:00 AM - 8:00 PM) of a small karinderya.
Customers are agents with their own order, arrival time and eating time.
They arrive randomly (Poisson, time-of-day dependent), queue if no table is
free, eat, pay, and leave. The model tracks waiting time, revenue, food cost,
profit and dirty dishes.

Usage examples
--------------
    python karinderya_abs.py                      # 1 detailed day + 30-day test + GIF in output/
    python karinderya_abs.py --days 100 --seed 42
    python karinderya_abs.py --show               # also open plot windows
    python karinderya_abs.py --animate            # live animation window
    python karinderya_abs.py --no-gif             # skip the GIF (faster)

Requirements: numpy, pandas, matplotlib  (Pillow for the GIF)
"""

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.path import Path

# ======================================================================
# 1. SETTINGS
# ======================================================================

NUM_TABLES = 6
SEATS_PER_TABLE = 4
NUM_SERVERS = 2
TOTAL_SEATS = NUM_TABLES * SEATS_PER_TABLE

OPENING_TIME = 10 * 60  # 10:00 AM (minutes since midnight)
CLOSING_TIME = 20 * 60  # 8:00 PM

MENU = {
    "Chicken Adobo": {"price": 80, "food_cost": 45, "dishes": 2},
    "Pork Menudo": {"price": 75, "food_cost": 40, "dishes": 2},
    "Fried Fish": {"price": 70, "food_cost": 38, "dishes": 2},
    "Vegetable": {"price": 60, "food_cost": 30, "dishes": 2},
}

OPERATING_COSTS = {
    "daily_rent": 500,
    "utilities": 300,
    "staff_cost": 1000,
    "cleaning_supplies": 100,
}

# Expected customers per hour, by hour-of-day block: (start_hour, end_hour, rate)
ARRIVAL_RATES = [
    (10, 11, 8),
    (11, 13, 20),
    (13, 14, 25),
    (14, 16, 10),
    (16, 18, 15),
    (18, 20, 22),
]

# Eating time ~ Triangular(min, mode, max) in minutes
EATING_TIME_PARAMS = (10, 20, 35)

# Dishwashing: every N minutes, wash up to `capacity` dishes
DISHWASH_INTERVAL = 30
DISHWASH_CAPACITY = 10

# ======================================================================
# 2. AGENT
# ======================================================================


@dataclass
class Customer:
    id: int
    arrival_time: float
    order: str
    eating_time: float
    seats_needed: int = 1

    start_time: float = None
    finish_time: float = None
    waiting_time: float = 0
    total_time: float = 0
    status: str = "Waiting"


def choose_food():
    return random.choice(list(MENU.keys()))


def generate_customer(customer_id, arrival_time):
    return Customer(
        id=customer_id,
        arrival_time=arrival_time,
        order=choose_food(),
        eating_time=np.random.triangular(*EATING_TIME_PARAMS),
    )


# ======================================================================
# 3. ARRIVALS
# ======================================================================


def get_arrival_rate(time):
    """Expected customers per hour at a given time (minutes since midnight)."""
    hour = time / 60
    for start, end, rate in ARRIVAL_RATES:
        if start <= hour < end:
            return rate
    return 0


def generate_arrivals():
    """Poisson number of arrivals each hour, uniformly spread within it."""
    customers = []
    customer_id = 1

    for hour in range(OPENING_TIME // 60, CLOSING_TIME // 60):
        start_time = hour * 60
        number_of_customers = np.random.poisson(get_arrival_rate(start_time))

        for _ in range(number_of_customers):
            arrival_time = start_time + random.uniform(0, 60)
            if arrival_time < CLOSING_TIME:
                customers.append(generate_customer(customer_id, arrival_time))
                customer_id += 1

    customers.sort(key=lambda c: c.arrival_time)
    return customers


# ======================================================================
# 4. ENVIRONMENT
# ======================================================================


class Karinderya:
    def __init__(self):
        self.num_tables = NUM_TABLES
        self.seats_per_table = SEATS_PER_TABLE
        self.num_servers = NUM_SERVERS

        self.tables = [None] * self.num_tables  # customer occupying each table
        self.waiting_queue = deque()
        self.completed_customers = []

        self.revenue = 0
        self.food_cost = 0
        self.dirty_dishes = 0

        self.current_time = OPENING_TIME

    def available_table(self):
        for i in range(self.num_tables):
            if self.tables[i] is None:
                return i
        return None


def seat_customer(karinderya, customer):
    """Seat a customer at a free table. Returns the table index or None."""
    table = karinderya.available_table()
    if table is None:
        return None

    karinderya.tables[table] = customer
    customer.start_time = karinderya.current_time
    customer.waiting_time = customer.start_time - customer.arrival_time
    customer.status = "Eating"
    return table


def add_customer_to_queue(karinderya, customer):
    customer.status = "Waiting"
    karinderya.waiting_queue.append(customer)


def serve_waiting_customers(karinderya):
    """Seat people from the front of the queue while tables are free."""
    while karinderya.waiting_queue and karinderya.available_table() is not None:
        customer = karinderya.waiting_queue.popleft()
        seat_customer(karinderya, customer)


def finish_customer(karinderya, customer, table):
    food = MENU[customer.order]

    karinderya.revenue += food["price"]
    karinderya.food_cost += food["food_cost"]
    karinderya.dirty_dishes += food["dishes"]

    customer.finish_time = customer.start_time + customer.eating_time
    customer.total_time = customer.finish_time - customer.arrival_time
    customer.status = "Finished"

    karinderya.tables[table] = None
    karinderya.completed_customers.append(customer)


def process_finished_customers(karinderya):
    for table in range(karinderya.num_tables):
        customer = karinderya.tables[table]
        if customer is None:
            continue
        if customer.start_time + customer.eating_time <= karinderya.current_time:
            finish_customer(karinderya, customer, table)


def wash_dishes(karinderya, capacity=DISHWASH_CAPACITY):
    washed = min(karinderya.dirty_dishes, capacity)
    karinderya.dirty_dishes -= washed
    return washed


# ======================================================================
# 5. SIMULATION LOOP (1-minute time steps)
# ======================================================================


def run_simulation(customers, record_history=False):
    """
    Run one day. Returns (karinderya, history).
    `history` is a list of per-minute snapshots when record_history=True
    (used by the animation), otherwise an empty list.
    """
    karinderya = Karinderya()
    history = []
    current_time = OPENING_TIME
    next_customer = 0

    while current_time <= CLOSING_TIME:
        karinderya.current_time = current_time

        # 1. Customers who finished eating leave
        process_finished_customers(karinderya)

        # 2. Waiting customers take free tables
        serve_waiting_customers(karinderya)

        # 3. New arrivals: seat immediately or join the queue
        while (
            next_customer < len(customers)
            and customers[next_customer].arrival_time <= current_time
        ):
            customer = customers[next_customer]
            if seat_customer(karinderya, customer) is None:
                add_customer_to_queue(karinderya, customer)
            next_customer += 1

        # 4. Dishwashing
        if current_time % DISHWASH_INTERVAL == 0:
            wash_dishes(karinderya)

        if record_history:
            history.append(
                {
                    "time": current_time,
                    "arrived": next_customer,
                    "eating": sum(t is not None for t in karinderya.tables),
                    "waiting": len(karinderya.waiting_queue),
                    "completed": len(karinderya.completed_customers),
                    "revenue": karinderya.revenue,
                    "dirty_dishes": karinderya.dirty_dishes,
                    "tables": [t.id if t is not None else None for t in karinderya.tables],
                    "queue": [c.id for c in karinderya.waiting_queue],
                }
            )

        current_time += 1

    return karinderya, history


# ======================================================================
# 6. RESULTS
# ======================================================================


def get_daily_results(karinderya, total_customers):
    operating_cost = sum(OPERATING_COSTS.values())
    profit = karinderya.revenue - karinderya.food_cost - operating_cost

    waits = [c.waiting_time for c in karinderya.completed_customers]
    avg_wait = float(np.mean(waits)) if waits else 0
    max_wait = max(waits) if waits else 0

    return {
        "Customers Generated": total_customers,
        "Customers Served": len(karinderya.completed_customers),
        "Customers Still Eating": sum(t is not None for t in karinderya.tables),
        "Customers Still Waiting": len(karinderya.waiting_queue),
        "Average Waiting Time": round(avg_wait, 2),
        "Maximum Waiting Time": round(max_wait, 2),
        "Revenue": round(karinderya.revenue, 2),
        "Food Cost": round(karinderya.food_cost, 2),
        "Operating Cost": round(operating_cost, 2),
        "Estimated Profit": round(profit, 2),
        "Dirty Dishes": karinderya.dirty_dishes,
    }


def customer_dataframe(karinderya):
    return pd.DataFrame(
        [
            {
                "Customer ID": c.id,
                "Arrival Time": c.arrival_time,
                "Order": c.order,
                "Eating Time": round(c.eating_time, 2),
                "Waiting Time": round(c.waiting_time, 2),
                "Total Time": round(c.total_time, 2),
                "Status": c.status,
            }
            for c in karinderya.completed_customers
        ]
    )


def print_summary(daily_results):
    print("=" * 40)
    print("   KARINDERYA ABS SIMULATION RESULTS")
    print("=" * 40)
    for key, value in daily_results.items():
        if key in ("Revenue", "Food Cost", "Operating Cost", "Estimated Profit"):
            print(f"{key + ':':<26} PHP {value:,.2f}")
        elif "Time" in key:
            print(f"{key + ':':<26} {value} minutes")
        else:
            print(f"{key + ':':<26} {value}")
    print("=" * 40)


def run_many_days(n_days):
    rows = []
    for day in range(1, n_days + 1):
        customers = generate_arrivals()
        karinderya, _ = run_simulation(customers)
        row = get_daily_results(karinderya, len(customers))
        row["Day"] = day
        rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================
# 7. VISUALIZATION
# ======================================================================


def _finish_fig(fig, name, output_dir, show):
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, name), dpi=150)
    if not show:
        plt.close(fig)


def make_plots(customers, customer_df, multi_df, output_dir, show):
    # Arrivals
    fig = plt.figure(figsize=(10, 5))
    plt.hist([c.arrival_time / 60 for c in customers], bins=20)
    plt.xlabel("Time of Day (hour)")
    plt.ylabel("Number of Customers")
    plt.title("Simulated Customer Arrivals")
    _finish_fig(fig, "arrivals.png", output_dir, show)

    # Waiting time
    fig = plt.figure(figsize=(10, 5))
    plt.hist(customer_df["Waiting Time"], bins=20)
    plt.xlabel("Waiting Time (minutes)")
    plt.ylabel("Number of Customers")
    plt.title("Customer Waiting Time Distribution")
    _finish_fig(fig, "waiting_time.png", output_dir, show)

    # Eating time
    fig = plt.figure(figsize=(10, 5))
    plt.hist(customer_df["Eating Time"], bins=20)
    plt.xlabel("Eating Time (minutes)")
    plt.ylabel("Number of Customers")
    plt.title("Customer Eating Time Distribution")
    _finish_fig(fig, "eating_time.png", output_dir, show)

    # Orders and revenue by food
    food_counts = customer_df["Order"].value_counts()
    fig = plt.figure(figsize=(8, 5))
    food_counts.plot(kind="bar")
    plt.xlabel("Food")
    plt.ylabel("Number of Orders")
    plt.title("Food Orders in the Simulated Day")
    plt.xticks(rotation=45)
    _finish_fig(fig, "food_orders.png", output_dir, show)

    revenue = {food: n * MENU[food]["price"] for food, n in food_counts.items()}
    fig = plt.figure(figsize=(8, 5))
    plt.bar(list(revenue.keys()), list(revenue.values()))
    plt.xlabel("Food")
    plt.ylabel("Revenue (PHP)")
    plt.title("Revenue by Food")
    plt.xticks(rotation=45)
    _finish_fig(fig, "revenue_by_food.png", output_dir, show)

    # Multi-day profit and waiting
    fig = plt.figure(figsize=(10, 5))
    plt.plot(multi_df["Day"], multi_df["Estimated Profit"], marker="o")
    plt.xlabel("Simulation Day")
    plt.ylabel("Estimated Profit (PHP)")
    plt.title("Estimated Profit Across Simulated Days")
    _finish_fig(fig, "profit_across_days.png", output_dir, show)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(multi_df["Day"], multi_df["Average Waiting Time"], marker="o")
    plt.xlabel("Simulation Day")
    plt.ylabel("Average Waiting Time (minutes)")
    plt.title("Average Customer Waiting Time Across Simulated Days")
    _finish_fig(fig, "waiting_across_days.png", output_dir, show)

    # Profit distribution (handy when many days are run)
    fig = plt.figure(figsize=(10, 5))
    plt.hist(multi_df["Estimated Profit"], bins=min(20, max(5, len(multi_df) // 3)))
    plt.xlabel("Estimated Profit (PHP)")
    plt.ylabel("Number of Days")
    plt.title("Distribution of Daily Profit")
    _finish_fig(fig, "profit_distribution.png", output_dir, show)


# ======================================================================
# 8. ANIMATION  (cute + efficient)
# ======================================================================
# Efficiency: every position and colour is pre-computed once with numpy, and
# the drawing objects (artists) are created once and only *updated* each frame
# (no ax.clear(), no per-customer Python patches).
# Cuteness: kawaii faces, plates of food, mood colours, hearts when customers
# leave, a sun that moves across the day, and a live queue chart.

ANIMATION_FPS = 15
ANIMATION_FRAME_STEP = 4   # simulated minutes drawn per frame
LEAVE_VISIBLE = 16         # minutes a finished customer stays visible, walking out

FOOD_COLORS = {
    "Chicken Adobo": "#8B5A2B",
    "Pork Menudo": "#E4572E",
    "Fried Fish": "#F2B134",
    "Vegetable": "#6BBF59",
}

ENTRANCE_POSITION = (0.7, 8.1)
EXIT_POSITION = (0.7, 1.6)

WAIT_CALM = "#FFE17A"    # just arrived (yellow)
WAIT_ANGRY = "#FF8A80"   # waited a long time (red)
EAT_COLOR = "#FFB3C7"    # eating (pink)
LEAVE_COLOR = "#C9B6FF"  # leaving (lavender)
INK = "#5A3A2E"

EYE_L, EYE_R = np.array([-0.13, 0.07]), np.array([0.13, 0.07])
CHEEK_L, CHEEK_R = np.array([-0.25, -0.06]), np.array([0.25, -0.06])
MOUTH_SMILE, MOUTH_FROWN = np.array([0.0, -0.08]), np.array([0.0, -0.20])


def format_time(minutes):
    hour = int(minutes // 60)
    minute = int(minutes % 60)
    period = "PM" if hour >= 12 else "AM"
    display_hour = hour - 12 if hour > 12 else hour
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:{minute:02d} {period}"


def _table_centers():
    cols = 3
    rows = -(-NUM_TABLES // cols)
    ys = np.linspace(6.2, 2.9, rows) if rows > 1 else [4.6]
    return [(6.4 + 2.8 * (i % cols), float(ys[i // cols])) for i in range(NUM_TABLES)]


def _queue_slot(i):
    col, row = divmod(i, 8)
    return (1.7 + 0.9 * min(col, 3), 7.2 - 0.72 * row)


def _lerp_color(c1, c2, f):
    a, b = np.array(to_rgba(c1)), np.array(to_rgba(c2))
    return a + (b - a) * np.asarray(f)[:, None]


def _day_tint(progress):
    """Warm morning -> bright noon -> orange evening background colour."""
    keys = np.array([0.0, 0.4, 1.0])
    cols = np.array([to_rgba(c) for c in ("#FFF1CF", "#FFFBEE", "#FFD5C2")])
    return tuple(np.interp(progress, keys, cols[:, i]) for i in range(4))


def _build_frames(history, customers, frame_step):
    """Pre-compute everything that moves, once, for every frame."""
    n = len(customers)
    index = {c.id: i for i, c in enumerate(customers)}
    arrival = np.array([c.arrival_time for c in customers], dtype=float)
    finish = np.array([np.inf if c.finish_time is None else c.finish_time for c in customers])
    food_rgba = np.array([to_rgba(FOOD_COLORS.get(c.order, "#999999")) for c in customers])

    centers = _table_centers()
    seat = np.array([(x - 0.4, y + 0.05) for x, y in centers])
    plate = np.array([(x + 0.5, y - 0.05) for x, y in centers])
    entrance, exit_ = np.array(ENTRANCE_POSITION), np.array(EXIT_POSITION)
    heart = np.array(to_rgba("#FF6F91"))

    pos = np.tile(entrance, (n, 1))
    alpha = 1 - 0.45 ** (frame_step / 4)  # smooth walking, independent of frame step
    frames = []

    for k in range(0, len(history), frame_step):
        snap = history[k]
        t = snap["time"]

        eating = np.zeros(n, bool)
        waiting = np.zeros(n, bool)
        target = np.tile(entrance, (n, 1))
        table_of = np.full(n, -1)

        for ti, cid in enumerate(snap["tables"]):
            if cid is not None:
                i = index[cid]
                eating[i], target[i], table_of[i] = True, seat[ti], ti
        for qi, cid in enumerate(snap["queue"]):
            i = index[cid]
            waiting[i], target[i] = True, _queue_slot(qi)

        age = t - finish  # minutes since the customer finished (-inf if not yet)
        leaving = (age >= 0) & (age <= LEAVE_VISIBLE)
        target[leaving] = exit_
        visible = eating | waiting | leaving

        pos[visible] += (target[visible] - pos[visible]) * alpha

        wait_min = np.clip(t - arrival, 0, None)
        body = _lerp_color(WAIT_CALM, WAIT_ANGRY, np.clip(wait_min / 45, 0, 1))
        body[eating] = to_rgba(EAT_COLOR)
        body[leaving] = to_rgba(LEAVE_COLOR)
        sad = waiting & (wait_min > 20)

        # a plate appears once the diner has (almost) reached the table
        seated = eating.copy()
        seated[eating] = np.linalg.norm(pos[eating] - target[eating], axis=1) < 0.6

        heart_xy = pos[leaving] + np.column_stack(
            (np.zeros(leaving.sum()), 0.55 + 0.05 * age[leaving])
        )
        heart_rgba = np.tile(heart, (leaving.sum(), 1))
        heart_rgba[:, 3] = 1 - 0.9 * age[leaving] / LEAVE_VISIBLE

        frames.append(
            dict(
                k=k, time=t,
                xy=pos[visible].copy(), color=body[visible], sad=sad[visible],
                plate_xy=plate[table_of[seated]], plate_color=food_rgba[seated],
                heart_xy=heart_xy, heart_color=heart_rgba,
                arrived=snap["arrived"], eating=snap["eating"], waiting=snap["waiting"],
                completed=snap["completed"], revenue=snap["revenue"], dirty=snap["dirty_dishes"],
            )
        )
    return frames


def create_karinderya_animation(history, customers, frame_step=ANIMATION_FRAME_STEP):
    """Build the cute animation. Returns (figure, FuncAnimation)."""
    frames = _build_frames(history, customers, frame_step)
    span = CLOSING_TIME - OPENING_TIME

    fig = plt.figure(figsize=(11, 9), dpi=80, facecolor="#FFF8EC")
    grid = fig.add_gridspec(2, 1, height_ratios=[4.6, 1], hspace=0.12,
                            left=0.07, right=0.98, top=0.98, bottom=0.07)
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(-0.2, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---------- static scenery (drawn once) ----------
    ax.add_patch(FancyBboxPatch((0.2, 0.4), 15.6, 8.5, boxstyle="round,pad=0.02,rounding_size=0.4",
                                fc="#FFFCF5", ec="#E0B48A", lw=3, zorder=0))
    ax.text(0.3, 9.45, "Karinderya Day", fontsize=17, fontweight="bold", color="#C9703B", va="center")
    clock = ax.text(15.8, 9.45, "", fontsize=17, fontweight="bold", color="#C9703B", ha="right", va="center")

    # sun that travels along the day
    ax.plot([5.4, 11.8], [9.45, 9.45], color="#F1D9B5", lw=6, solid_capstyle="round", zorder=1)
    ax.text(5.4, 9.05, "10 AM", fontsize=7, color="#B08968", ha="center")
    ax.text(11.8, 9.05, "8 PM", fontsize=7, color="#B08968", ha="center")
    sun = ax.scatter([5.4], [9.45], s=330, c="#FFD23F", edgecolors="#FF9F1C", linewidths=2, zorder=3)

    # doors and queue sign
    ax.add_patch(FancyBboxPatch((-0.1, 7.5), 0.6, 1.3, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#C98B5A", ec="none", zorder=1))
    ax.text(0.2, 8.15, "IN", rotation=90, fontsize=8, color="white", ha="center", va="center", zorder=2)
    ax.add_patch(FancyBboxPatch((-0.1, 1.0), 0.6, 1.3, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#7FC8A9", ec="none", zorder=1))
    ax.text(0.2, 1.65, "OUT", rotation=90, fontsize=8, color="white", ha="center", va="center", zorder=2)
    ax.text(2.7, 8.35, "~ waiting line ~", fontsize=9, color="#B08968", ha="center")

    # round tables
    for i, (x, y) in enumerate(_table_centers()):
        ax.add_patch(Circle((x, y), 1.05, fc="#F6D7A7", ec="#D9A066", lw=2, zorder=1))
        ax.text(x, y - 1.3, f"Table {i + 1}", fontsize=7, color="#B08968", ha="center")

    # kitchen with the menu colours
    ax.add_patch(FancyBboxPatch((13.4, 5.3), 2.3, 3.4, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc="#FFE8C9", ec="#E0B48A", lw=2, zorder=1))
    ax.text(14.55, 8.3, "Kitchen", fontsize=10, fontweight="bold", color="#C9703B", ha="center")
    for j, (name, color) in enumerate(FOOD_COLORS.items()):
        if name in MENU:
            y = 7.5 - 0.6 * j
            ax.scatter([13.75], [y], s=90, c=color, zorder=3)
            ax.text(14.0, y, name, fontsize=7.5, color=INK, va="center")

    # sink with dirty dishes
    ax.add_patch(FancyBboxPatch((13.4, 0.6), 2.3, 4.2, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc="#E3F0FA", ec="#9CC3E0", lw=2, zorder=1))
    ax.text(14.55, 4.4, "Sink", fontsize=10, fontweight="bold", color="#4B86B4", ha="center")
    sink_text = ax.text(14.55, 3.85, "", fontsize=8, color="#4B86B4", ha="center")
    dish_grid = np.array([(13.75 + 0.38 * c, 1.0 + 0.38 * r) for r in range(6) for c in range(6)])
    dishes = ax.scatter([], [], s=75, marker="s", c="#FFFFFF", edgecolors="#8FB8D8", linewidths=1.5, zorder=3)

    # mood legend
    for x, color, label in [(5.8, WAIT_CALM, "waiting"), (8.0, EAT_COLOR, "eating"), (10.2, LEAVE_COLOR, "leaving")]:
        ax.scatter([x], [0.95], s=110, c=color, edgecolors=INK, linewidths=0.8, zorder=3)
        ax.text(x + 0.3, 0.95, label, fontsize=8, color=INK, va="center")

    stats = ax.text(9.2, 8.1, "", fontsize=10, ha="center", va="center", color=INK, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#E0B48A", alpha=0.9))

    # ---------- dynamic artists (created once, updated every frame) ----------
    smile_marker = Path(np.column_stack((np.cos(np.radians(np.linspace(215, 325, 12))),
                                         np.sin(np.radians(np.linspace(215, 325, 12))))))
    frown_marker = Path(np.column_stack((np.cos(np.radians(np.linspace(35, 145, 12))),
                                         np.sin(np.radians(np.linspace(35, 145, 12))))))
    plates = ax.scatter([], [], s=720, c="white", edgecolors="#E0B48A", linewidths=1.5, zorder=3)
    food = ax.scatter([], [], s=260, zorder=4)
    body = ax.scatter([], [], s=1350, edgecolors=INK, linewidths=1.2, zorder=5)
    cheeks = ax.scatter([], [], s=55, c="#FF9EB5", alpha=0.75, zorder=6)
    eyes = ax.scatter([], [], s=20, c=INK, zorder=7)
    smiles = ax.scatter([], [], s=130, marker=smile_marker, facecolors="none",
                        edgecolors=INK, linewidths=1.3, zorder=7)
    frowns = ax.scatter([], [], s=130, marker=frown_marker, facecolors="none",
                        edgecolors=INK, linewidths=1.3, zorder=7)
    hearts = ax.scatter([], [], s=130, marker="$\u2665$", zorder=8)

    # ---------- live queue chart ----------
    chart = fig.add_subplot(grid[1])
    hours = np.array([h["time"] for h in history]) / 60
    queue_len = np.array([h["waiting"] for h in history])
    chart.set_xlim(OPENING_TIME / 60, CLOSING_TIME / 60)
    chart.set_ylim(0, max(5, queue_len.max() + 3))
    ticks = list(range(OPENING_TIME // 60, CLOSING_TIME // 60 + 1, 2))
    chart.set_xticks(ticks)
    chart.set_xticklabels([f"{h - 12 if h > 12 else h} {'AM' if h < 12 else 'PM'}" for h in ticks], fontsize=8)
    chart.set_ylabel("people in queue", fontsize=9, color=INK)
    chart.set_facecolor("#FFFCF5")
    for side in ("top", "right"):
        chart.spines[side].set_visible(False)
    chart.grid(alpha=0.25)
    (queue_line,) = chart.plot([], [], color="#FF7FA3", lw=2.5)
    (queue_dot,) = chart.plot([], [], "o", color="#FF7FA3", ms=8)

    def update(f):
        fr = frames[f]
        xy, sad, k = fr["xy"], fr["sad"], fr["k"]

        body.set_offsets(xy)
        body.set_facecolor(fr["color"])
        eyes.set_offsets(np.vstack([xy + EYE_L, xy + EYE_R]))
        cheeks.set_offsets(np.vstack([xy + CHEEK_L, xy + CHEEK_R]))
        smiles.set_offsets(xy[~sad] + MOUTH_SMILE)
        frowns.set_offsets(xy[sad] + MOUTH_FROWN)
        plates.set_offsets(fr["plate_xy"])
        food.set_offsets(fr["plate_xy"])
        food.set_facecolor(fr["plate_color"])
        hearts.set_offsets(fr["heart_xy"])
        hearts.set_color(fr["heart_color"])

        progress = (fr["time"] - OPENING_TIME) / span
        ax.set_facecolor(_day_tint(progress))
        sun.set_offsets([[5.4 + 6.4 * progress, 9.45]])
        clock.set_text(format_time(fr["time"]))
        stats.set_text(
            f"Arrived {fr['arrived']}   Eating {fr['eating']}   Waiting {fr['waiting']}\n"
            f"Served {fr['completed']}   Revenue PHP {fr['revenue']:,.0f}"
        )
        sink_text.set_text(f"{fr['dirty']} dirty dishes")
        dishes.set_offsets(dish_grid[: min(fr["dirty"], len(dish_grid))])

        queue_line.set_data(hours[: k + 1], queue_len[: k + 1])
        queue_dot.set_data([hours[k]], [queue_len[k]])

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / ANIMATION_FPS, repeat=True)
    return fig, animation


def save_karinderya_animation(animation, path, dpi=80):
    """Save as GIF (needs Pillow) or MP4 (needs ffmpeg)."""
    if path.lower().endswith(".gif"):
        animation.save(path, writer=PillowWriter(fps=ANIMATION_FPS), dpi=dpi)
    else:
        animation.save(path, fps=ANIMATION_FPS, dpi=dpi)


# ======================================================================
# 9. MAIN
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Karinderya agent-based simulation")
    parser.add_argument("--days", type=int, default=30, help="number of days for the multi-day run")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible runs")
    parser.add_argument("--output-dir", default="output_abs", help="folder for figures and CSVs")
    parser.add_argument("--show", action="store_true", help="display plot windows")
    parser.add_argument("--animate", action="store_true", help="show the live animation window")
    parser.add_argument(
        "--save-animation", metavar="FILE", default="karinderya_day.gif",
        help="file name for the animation GIF, saved in the output folder (default: karinderya_day.gif)",
    )
    parser.add_argument("--no-gif", action="store_true", help="skip creating the GIF (saves ~30 seconds)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- One detailed day ---
    customers = generate_arrivals()
    make_gif = not args.no_gif
    karinderya, history = run_simulation(customers, record_history=args.animate or make_gif)
    daily_results = get_daily_results(karinderya, len(customers))
    customer_df = customer_dataframe(karinderya)

    print_summary(daily_results)
    customer_df.to_csv(os.path.join(args.output_dir, "customers_day1.csv"), index=False)

    # --- Many days ---
    multi_df = run_many_days(args.days)
    multi_df.to_csv(os.path.join(args.output_dir, "daily_results.csv"), index=False)
    print(f"\nSummary over {args.days} simulated days:")
    print(
        multi_df[
            ["Customers Generated", "Customers Served", "Average Waiting Time",
             "Revenue", "Food Cost", "Estimated Profit", "Dirty Dishes"]
        ].describe().round(2).to_string()
    )

    # --- Plots ---
    make_plots(customers, customer_df, multi_df, args.output_dir, args.show)
    print(f"\nFigures and CSV files saved in '{args.output_dir}/'")

    # --- Animation ---
    if args.animate or make_gif:
        fig, animation = create_karinderya_animation(history, customers)
        if make_gif:
            # A bare file name goes into the output folder with the other results
            if os.path.dirname(args.save_animation) == "":
                anim_path = os.path.join(args.output_dir, args.save_animation)
            else:
                anim_path = args.save_animation
            save_karinderya_animation(animation, anim_path)
            print(f"Animation saved to {anim_path}")
        if args.animate:
            plt.show()
            return

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()