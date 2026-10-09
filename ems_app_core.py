"""
Logic behind the EMS web app (no screen/UI code here).
Used by app_ems.py. Needs ems_microgrid.py and the two trained models
(xgb_load_model.joblib, xgb_solar_model.joblib, created by ems_full_pipeline.py).
"""
import os
import dataclasses
import numpy as np
import pandas as pd
import joblib

import ems_microgrid as ems

# Sky condition -> fraction of clear-sky sunlight
SKY = {"Clear": 1.0, "Partly cloudy": 0.6, "Cloudy": 0.3, "Rainy / heavy cloud": 0.12}


# ================================================================ MODELS
def load_models(folders=(".", os.path.dirname(os.path.abspath(__file__)))):
    for f in folders:
        lp, sp = os.path.join(f, "xgb_load_model.joblib"), os.path.join(f, "xgb_solar_model.joblib")
        if os.path.exists(lp) and os.path.exists(sp):
            return joblib.load(lp), joblib.load(sp)
    raise FileNotFoundError(
        "Trained models not found. Run  python ems_full_pipeline.py  once in the folder "
        "you start this app from (it creates xgb_load_model.joblib and xgb_solar_model.joblib)."
    )


# ====================================================== HOURLY INPUT TABLE
def make_hourly_table(start_date, days_cfg, humidity):
    """Build an editable hourly table from simple per-day settings.

    days_cfg: list of dicts: sky, tmin, tmax, holiday, people (peak occupancy)
    Columns: Time, Sun (0-1.2), OutsideTemp, Humidity, People, HVAC, Lights, Holiday
    """
    rows = []
    for i, dc in enumerate(days_cfg):
        day = pd.Timestamp(start_date).normalize() + pd.Timedelta(days=i)
        off_day = day.dayofweek >= 5 or dc["holiday"]
        tmin, tmax = min(dc["tmin"], dc["tmax"]), max(dc["tmin"], dc["tmax"])
        for h in range(24):
            sun = np.sin(np.pi * (h - 6) / 12) if 6 <= h <= 18 else 0.0     # sunrise 6, sunset 18
            sun = max(sun, 0.0) * SKY[dc["sky"]]
            temp = (tmax + tmin) / 2 + (tmax - tmin) / 2 * np.sin(2 * np.pi * (h - 9) / 24)  # peak ~15:00
            P = dc["people"]
            if off_day:
                ppl = round(0.1 * P) if 9 <= h <= 15 else 0
            else:
                ppl = P if 8 <= h < 18 else (round(P / 2) if h in (7, 18) else 0)
            rows.append(dict(
                Time=day + pd.Timedelta(hours=h), Sun=round(float(sun), 2), OutsideTemp=round(float(temp), 1),
                Humidity=float(humidity), People=int(ppl), HVAC=bool(ppl > 0),
                Lights=bool(ppl > 0 or 18 <= h < 22), Holiday=bool(dc["holiday"]),
            ))
    return pd.DataFrame(rows)


# ============================================================== PREDICTIONS
def predict_solar_per_kw(solar_b, table):
    """Solar output for each row, per 1 kW of installed PV (so any PV size can be applied later)."""
    t = pd.to_datetime(table["Time"]).reset_index(drop=True)
    irr = table["Sun"].astype(float).values
    amb = table["OutsideTemp"].astype(float).values
    a, b = solar_b["module_fit"]
    mod = amb + a * irr + b
    hour = (t.dt.hour + t.dt.minute / 60).values
    X = pd.DataFrame({
        "IRRADIATION": irr, "AMBIENT_TEMPERATURE": amb, "MODULE_TEMPERATURE": mod,
        "temp_diff": mod - amb, "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dayofyear": t.dt.dayofyear.values,
    })[solar_b["features"]]
    plant = np.clip(solar_b["model"].predict(X), 0, None)
    plant[irr <= 0] = 0.0                                  # no sunlight -> no solar power
    return plant / solar_b["plant_peak_kw"]


def predict_load(load_b, table):
    """Building load (kW) for each row of the table."""
    cols = load_b["columns"]
    t = pd.to_datetime(table["Time"]).reset_index(drop=True)
    n = len(table)
    X = pd.DataFrame({c: [load_b["defaults"].get(c, 0.0)] * n for c in cols})

    def put(name, vals):
        if name in X.columns:
            X[name] = np.asarray(vals, dtype=float)

    put("Hour", t.dt.hour); put("Day", t.dt.day); put("Month", t.dt.month)
    put("IsWeekend", t.dt.dayofweek >= 5)
    names = t.dt.day_name()
    for nm in names.unique():
        put(f"DayOfWeek_{nm}", (names == nm).astype(float))
    put("Temperature", table["OutsideTemp"].values)
    put("Humidity", table["Humidity"].values)
    put("Occupancy", table["People"].values)
    for prefix, col in (("HVACUsage", "HVAC"), ("LightingUsage", "Lights"), ("Holiday", "Holiday")):
        match = [c for c in cols if c.startswith(prefix + "_")]
        if len(match) == 1:
            X[match[0]] = table[col].astype(float).values
    return np.clip(load_b["model"].predict(X[cols]), 0, None)


def forecast_frame(load_b, solar_b, table):
    return pd.DataFrame({
        "DATE_TIME": pd.to_datetime(table["Time"]).values,
        "SOLAR_PER_KW": predict_solar_per_kw(solar_b, table),
        "LOAD_KW": predict_load(load_b, table),
    })


# ================================================================== EMS RUNS
def run_with(cfg, frame):
    df = frame.assign(SOLAR_KW=frame["SOLAR_PER_KW"] * cfg.pv_capacity_kw)[["DATE_TIME", "SOLAR_KW", "LOAD_KW"]]
    return ems.run_ems(df, cfg)


def daily_summary(res, cfg):
    d = res.copy()
    dt = cfg.dt_h
    price = d["DATE_TIME"].dt.hour.apply(lambda h: ems.price_at(h, cfg))
    imp = d.grid_to_load + d.grid_to_batt
    d["Date"] = d["DATE_TIME"].dt.date
    d["load_kwh"] = d.LOAD_KW * dt
    d["solar_kwh"] = d.SOLAR_KW * dt
    d["import_kwh"] = imp * dt
    d["load_from_grid_kwh"] = d.grid_to_load * dt
    d["export_kwh"] = d.pv_export * dt
    d["cost"] = (imp * price - d.pv_export * cfg.export_price) * dt
    d["cost_no_system"] = d.LOAD_KW * price * dt
    g = d.groupby("Date").agg(
        Load_kWh=("load_kwh", "sum"), Solar_kWh=("solar_kwh", "sum"),
        GridImport_kWh=("import_kwh", "sum"), Export_kWh=("export_kwh", "sum"),
        FromGrid_kWh=("load_from_grid_kwh", "sum"),
        EndBattery_pct=("soc", "last"), Cost=("cost", "sum"), CostWithoutSystem=("cost_no_system", "sum"),
    )
    g["CoveredBySolar+Battery_%"] = 100 * (1 - g.pop("FromGrid_kWh") / g.Load_kWh)
    g["EndBattery_pct"] = g["EndBattery_pct"] * 100
    g["Savings"] = g.CostWithoutSystem - g.Cost
    return g


def insights(res, cfg):
    msgs = []
    i = res.LOAD_KW.idxmax()
    msgs.append(f"Highest building demand: **{res.LOAD_KW[i]:.0f} kW** on {res.DATE_TIME[i]:%a %d %b at %H:%M}.")
    j = res.SOLAR_KW.idxmax()
    msgs.append(f"Best solar moment: **{res.SOLAR_KW[j]:.0f} kW** on {res.DATE_TIME[j]:%a %d %b at %H:%M}.")
    k = res.soc.idxmin()
    msgs.append(f"Lowest battery level: **{res.soc[k] * 100:.0f}%** on {res.DATE_TIME[k]:%a %d %b at %H:%M}.")
    n_grid = int((res.grid_to_load > 0.05).sum())
    msgs.append(f"The grid had to supply the building in **{n_grid} of {len(res)} hours**.")
    n_res = int((res.soc <= cfg.soc_min + 0.005).sum())
    if n_res:
        msgs.append(f"The battery sat at its minimum reserve for **{n_res} hours**. A larger battery or more PV could help.")
    if res.curtailed.sum() > 0:
        msgs.append(f"**{res.curtailed.sum() * cfg.dt_h:.0f} kWh** of solar was wasted (export is off and battery full).")
    if res.unserved.sum() > 0:
        msgs.append(f"**{res.unserved.sum() * cfg.dt_h:.0f} kWh** of load could not be served (grid import limit too low).")
    return msgs


# ============================================================ SINGLE HOUR
def explain(solar, load, r):
    if solar >= load:
        return "Solar is enough for the building; the extra goes to the battery, then the grid."
    if r["grid_to_load"] < 0.05:
        return "Solar is short, so the battery covers the gap."
    if r["batt_to_load"] > 0.05:
        return "Solar is short; the battery helps but hits its limit, so the grid covers the rest."
    return "Solar is short and the battery can't help (at its reserve level), so the grid covers it."


def single_hour(load_b, solar_b, ts, row, soc, cfg):
    """row: dict with Sun, OutsideTemp, Humidity, People, HVAC, Lights, Holiday. soc: 0-1."""
    table = pd.DataFrame([dict(Time=ts, **row)])
    solar = float(predict_solar_per_kw(solar_b, table)[0]) * cfg.pv_capacity_kw
    load = float(predict_load(load_b, table)[0])
    batt = ems.Battery(cfg)
    batt.energy = soc * cfg.batt_capacity_kwh
    r = ems.dispatch_step(solar, load, ts.hour + ts.minute / 60, batt, cfg)
    return solar, load, r, batt.soc, explain(solar, load, r)


# =============================================================== SIZING
def sizing_sweep(frame, cfg, pv_list, batt_list):
    """Re-run the same days with different PV / battery sizes."""
    rc = cfg.batt_max_charge_kw / cfg.batt_capacity_kwh
    rd = cfg.batt_max_discharge_kw / cfg.batt_capacity_kwh
    rows = []
    for pv in pv_list:
        for b in batt_list:
            c = dataclasses.replace(
                cfg, pv_capacity_kw=float(pv), batt_capacity_kwh=max(float(b), 1e-3),
                batt_max_charge_kw=b * rc, batt_max_discharge_kw=b * rd)
            s = ems.summarize(run_with(c, frame), c)
            rows.append(dict(PV_kW=pv, Battery_kWh=b, Savings=s["Savings"],
                             GridImport_kWh=s["Grid import (kWh)"],
                             Covered_pct=s["Load covered by solar+battery (%)"],
                             Export_kWh=s["Grid export (kWh)"]))
    return pd.DataFrame(rows)
