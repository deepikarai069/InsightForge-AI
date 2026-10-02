"""Generate a realistic multi-table e-commerce sample with planted insights.

Planted for demo purposes:
  * Revenue in region 'West' via channel 'Online' collapses in the final 5 weeks (root cause demo).
  * A promo spike on 2024-11-29 (Black Friday) and a tracking glitch on 2024-08-14 (anomalies).
  * Dirty data: currency strings, duplicate rows, case variants, missing values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "sample_data"


def main(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    OUT.mkdir(exist_ok=True)

    # products
    cats = {"Electronics": (150, 900), "Home": (20, 200), "Apparel": (15, 120), "Sports": (25, 300), "Beauty": (8, 80)}
    prods = []
    for i in range(1, 61):
        cat = list(cats)[(i - 1) % 5]
        lo, hi = cats[cat]
        price = round(float(rng.uniform(lo, hi)), 2)
        prods.append({"product_id": f"P{i:03d}", "product_name": f"{cat} item {i}", "category": cat,
                      "unit_cost": round(price * float(rng.uniform(0.45, 0.7)), 2), "list_price": price})
    products = pd.DataFrame(prods)

    # customers
    n_c = 1500
    customers = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n_c + 1)],
        "segment": rng.choice(["Consumer", "Small Business", "Enterprise"], n_c, p=[0.7, 0.22, 0.08]),
        "country": rng.choice(["US", "UK", "Germany", "India", "Canada"], n_c, p=[0.45, 0.15, 0.12, 0.18, 0.10]),
        "signup_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n_c), unit="D"),
    })

    # orders (2024-01-01 → 2024-12-31)
    days = pd.date_range("2024-01-01", "2024-12-31")
    rows = []
    oid = 1000
    for d in days:
        season = 1 + 0.25 * np.sin((d.dayofyear / 365) * 2 * np.pi - 1) + 0.0008 * d.dayofyear  # mild growth
        lam = 38 * season * (1.25 if d.weekday() >= 5 else 1.0)
        if d == pd.Timestamp("2024-11-29"):
            lam *= 4.0
        n = rng.poisson(lam)
        for _ in range(n):
            oid += 1
            region = rng.choice(["North", "South", "East", "West"], p=[0.22, 0.24, 0.24, 0.30])
            channel = rng.choice(["Online", "Retail", "Partner"], p=[0.55, 0.30, 0.15])
            if region == "West" and channel == "Online" and d >= pd.Timestamp("2024-11-25"):
                if rng.random() < 0.85:  # planted outage → most orders lost
                    continue
            p = products.iloc[rng.integers(0, len(products))]
            qty = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 5]))
            disc = float(rng.choice([0, 0, 0, 0.05, 0.1, 0.2], p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1]))
            if d == pd.Timestamp("2024-11-29"):
                disc = 0.3
            price = p["list_price"] * float(rng.uniform(0.97, 1.03))
            revenue = round(qty * price * (1 - disc), 2)
            rows.append({"order_id": f"O{oid}", "order_date": d, "customer_id": customers["customer_id"].iloc[rng.integers(0, n_c)],
                         "product_id": p["product_id"], "region": region, "channel": channel, "quantity": qty,
                         "unit_price": round(price, 2), "discount": disc, "revenue": revenue,
                         "returned": int(rng.random() < (0.12 if p["category"] == "Apparel" else 0.04))})
    orders = pd.DataFrame(rows)

    # planted anomalies: tracking glitch (huge revenue) + a few extreme orders
    glitch = orders.index[orders["order_date"] == pd.Timestamp("2024-08-14")][:3]
    orders.loc[glitch, "revenue"] = orders.loc[glitch, "revenue"] * 60
    orders.loc[orders.sample(4, random_state=1).index, "quantity"] = 250

    # dirty-up
    dirty = orders.copy()
    dirty["order_date"] = dirty["order_date"].dt.strftime("%Y-%m-%d")
    dirty["revenue"] = dirty["revenue"].map(lambda v: f"${v:,.2f}")
    mask = rng.random(len(dirty)) < 0.02
    dirty.loc[mask, "channel"] = dirty.loc[mask, "channel"].str.lower()
    dirty.loc[rng.random(len(dirty)) < 0.03, "discount"] = np.nan
    dirty.loc[rng.random(len(dirty)) < 0.01, "region"] = "N/A"
    dirty = pd.concat([dirty, dirty.sample(25, random_state=3)], ignore_index=True)

    dirty.to_csv(OUT / "orders.csv", index=False)
    customers.to_csv(OUT / "customers.csv", index=False)
    products.to_csv(OUT / "products.csv", index=False)
    print(f"orders={len(dirty)} customers={len(customers)} products={len(products)} → {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
