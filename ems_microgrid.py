"""
Microgrid Energy Management System (EMS) for a building.

    Solar forecast (XGBoost)  --
                                 >-->  EMS dispatch  -->  Solar / Battery / Grid  -->  Building
    Load forecast (your model) --/

Priority rules, evaluated every time step (15 min):
    1. Solar serves the building load first.
    2. Surplus solar  -> charge battery -> then export to grid (or curtail).
    3. Deficit        -> discharge battery (down to SOC_MIN) -> then import from grid.
    4. (optional) Off-peak tariff hours: top up the battery from the grid.

INPUT FILES (CSV, 15-min data, same time axis):
    solar_pred.csv : DATE_TIME, SOLAR_KW   <- output of train_xgboost_solar.py
    load_pred.csv  : DATE_TIME, LOAD_KW    <- output of your load-demand model
If these files are missing, a DEMO is run using Plant 1 data (scaled down to
building size) and a synthetic building load.

Run:  python ems_microgrid.py
"""
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================== CONFIG
@dataclass
class Config:
    dt_h: float = 0.25                 # time step in hours (15 min)

    # Solar
    pv_capacity_kw: float = 500.0      # installed PV size of YOUR building

    # Battery
    batt_capacity_kwh: float = 1000.0
    batt_max_charge_kw: float = 250.0
    batt_max_discharge_kw: float = 250.0
    batt_eff_charge: float = 0.95      # one-way efficiencies
    batt_eff_discharge: float = 0.95
    soc_min: float = 0.20              # keep 20% reserve
    soc_max: float = 0.95
    soc_init: float = 0.50

    # Grid
    grid_import_limit_kw: float = 1e9  # set e.g. 300 to model a connection limit
    allow_export: bool = True          # False -> surplus solar is curtailed
    export_price: float = 3.0          # currency / kWh (feed-in)

    # Tariff (time-of-use) - edit to your utility
    peak_hours: tuple = (18, 22)       # [start, end) hour of day
    price_peak: float = 10.0
    price_offpeak: float = 6.0

    # Optional: charge battery from grid during cheap hours
    grid_charge_enabled: bool = False
    offpeak_hours: tuple = (0, 6)
    grid_charge_target_soc: float = 0.60


# ============================================================== BATTERY
class Battery:
    def __init__(self, cfg: Config):
        self.c = cfg
        self.energy = cfg.soc_init * cfg.batt_capacity_kwh  # kWh stored

    @property
    def soc(self):
        return self.energy / self.c.batt_capacity_kwh

    def max_charge_kw(self):
        """Max power (kW, AC side) the battery can absorb this step."""
        headroom_kwh = self.c.soc_max * self.c.batt_capacity_kwh - self.energy
        p_energy = headroom_kwh / (self.c.dt_h * self.c.batt_eff_charge)
        return max(0.0, min(self.c.batt_max_charge_kw, p_energy))

    def max_discharge_kw(self):
        """Max power (kW, AC side) the battery can deliver this step."""
        avail_kwh = self.energy - self.c.soc_min * self.c.batt_capacity_kwh
        p_energy = avail_kwh * self.c.batt_eff_discharge / self.c.dt_h
        return max(0.0, min(self.c.batt_max_discharge_kw, p_energy))

    def charge(self, p_kw):
        self.energy += p_kw * self.c.dt_h * self.c.batt_eff_charge

    def discharge(self, p_kw):
        self.energy -= p_kw * self.c.dt_h / self.c.batt_eff_discharge


# ============================================================ EMS LOGIC
def price_at(hour, cfg: Config):
    return cfg.price_peak if cfg.peak_hours[0] <= hour < cfg.peak_hours[1] else cfg.price_offpeak


def dispatch_step(solar, load, hour, batt: Battery, cfg: Config):
    """Decide power flows for one time step. All values in kW (>= 0)."""
    pv_to_load = min(solar, load)
    surplus = solar - pv_to_load
    deficit = load - pv_to_load

    batt_charge = batt_discharge = grid_import = grid_export = curtailed = 0.0
    grid_to_batt = 0.0

    # --- surplus solar: battery first, then grid
    if surplus > 0:
        batt_charge = min(surplus, batt.max_charge_kw())
        surplus -= batt_charge
        if cfg.allow_export:
            grid_export = surplus
        else:
            curtailed = surplus

    # --- deficit: battery first, then grid
    if deficit > 0:
        batt_discharge = min(deficit, batt.max_discharge_kw())
        deficit -= batt_discharge
        grid_import = deficit

    # --- optional: cheap-tariff grid charging
    if (cfg.grid_charge_enabled and cfg.offpeak_hours[0] <= hour < cfg.offpeak_hours[1]
            and batt.soc < cfg.grid_charge_target_soc and batt_discharge == 0):
        target_kwh = cfg.grid_charge_target_soc * cfg.batt_capacity_kwh - batt.energy
        want = target_kwh / (cfg.dt_h * cfg.batt_eff_charge)
        room = batt.max_charge_kw() - batt_charge
        grid_to_batt = max(0.0, min(want, room))

    # --- apply grid import limit (unmet load is reported, not hidden)
    unserved = 0.0
    total_import = grid_import + grid_to_batt
    if total_import > cfg.grid_import_limit_kw:
        cut = total_import - cfg.grid_import_limit_kw
        cut_batt = min(grid_to_batt, cut)
        grid_to_batt -= cut_batt
        unserved = cut - cut_batt
        grid_import -= unserved

    # --- update battery
    total_charge = batt_charge + grid_to_batt
    if total_charge > 0:
        batt.charge(total_charge)
    if batt_discharge > 0:
        batt.discharge(batt_discharge)

    return dict(
        pv_to_load=pv_to_load, batt_to_load=batt_discharge, grid_to_load=grid_import,
        pv_to_batt=batt_charge, grid_to_batt=grid_to_batt, pv_export=grid_export,
        curtailed=curtailed, unserved=unserved, soc=batt.soc,
    )


def run_ems(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """df needs columns: DATE_TIME, SOLAR_KW, LOAD_KW."""
    if df.empty:
        raise ValueError("run_ems received no data (empty DataFrame).")
    batt = Battery(cfg)
    rows = []
    for t, s, l in zip(df["DATE_TIME"], df["SOLAR_KW"], df["LOAD_KW"]):
        r = dispatch_step(max(s, 0.0), max(l, 0.0), t.hour + t.minute / 60, batt, cfg)
        r["DATE_TIME"] = t
        rows.append(r)
    out = pd.DataFrame(rows)
    out = df.merge(out, on="DATE_TIME")

    # sanity check: load must be served = pv + battery + grid (+ unserved)
    served = out.pv_to_load + out.batt_to_load + out.grid_to_load + out.unserved
    assert np.allclose(served, out.LOAD_KW.clip(lower=0), atol=1e-6), "Energy balance broken"
    return out


# ============================================================== METRICS
def summarize(res: pd.DataFrame, cfg: Config):
    dt = cfg.dt_h
    hours = res["DATE_TIME"].dt.hour + res["DATE_TIME"].dt.minute / 60
    price = hours.apply(lambda h: price_at(h, cfg))

    load_kwh = res.LOAD_KW.sum() * dt
    solar_kwh = res.SOLAR_KW.sum() * dt
    imp = (res.grid_to_load + res.grid_to_batt)
    exp = res.pv_export

    cost = (imp * price).sum() * dt - (exp * cfg.export_price).sum() * dt
    cost_no_system = (res.LOAD_KW * price).sum() * dt
    solar_used = (res.pv_to_load + res.pv_to_batt).sum() * dt

    out = {
        "Total load (kWh)": load_kwh,
        "Solar generated (kWh)": solar_kwh,
        "Grid import (kWh)": imp.sum() * dt,
        "Grid export (kWh)": exp.sum() * dt,
        "Solar curtailed (kWh)": res.curtailed.sum() * dt,
        "Unserved load (kWh)": res.unserved.sum() * dt,
        "Load covered by solar+battery (%)": 100 * (1 - res.grid_to_load.sum() * dt / load_kwh),
        "Solar self-consumption (%)": 100 * solar_used / solar_kwh if solar_kwh else 0,
        "Battery full-cycle equivalents": (res.pv_to_batt + res.grid_to_batt).sum() * dt
                                          * cfg.batt_eff_charge / cfg.batt_capacity_kwh,
        "Net energy cost (with system)": cost,
        "Energy cost (grid only)": cost_no_system,
        "Savings": cost_no_system - cost,
    }
    return out


# =============================================================== PLOTS
def plot_results(res: pd.DataFrame, path: str, days: int = 3, dt_h: float = 0.25):
    d = res.iloc[: int(days * 24 / dt_h)]
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax[0].stackplot(
        d["DATE_TIME"], d.pv_to_load, d.batt_to_load, d.grid_to_load,
        labels=["Solar -> load", "Battery -> load", "Grid -> load"],
        colors=["#f5b700", "#2e9e5b", "#5b7fb5"], alpha=0.85,
    )
    ax[0].plot(d["DATE_TIME"], d.LOAD_KW, "k--", lw=1, label="Load demand")
    ax[0].plot(d["DATE_TIME"], d.SOLAR_KW, color="#d98200", lw=1, label="Solar forecast")
    ax[0].set_ylabel("kW"); ax[0].set_title("Who supplies the building?"); ax[0].legend(loc="upper right", ncol=3)

    ax[1].plot(d["DATE_TIME"], d.soc * 100, color="#2e9e5b")
    ax[1].set_ylabel("Battery SOC (%)"); ax[1].set_ylim(0, 100); ax[1].set_title("Battery state of charge")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


# =========================================================== DATA INPUT
def load_inputs(cfg: Config) -> pd.DataFrame:
    if os.path.exists("solar_pred.csv") and os.path.exists("load_pred.csv"):
        s = pd.read_csv("solar_pred.csv", parse_dates=["DATE_TIME"])[["DATE_TIME", "SOLAR_KW"]]
        l = pd.read_csv("load_pred.csv", parse_dates=["DATE_TIME"])[["DATE_TIME", "LOAD_KW"]]
        df = s.merge(l, on="DATE_TIME").sort_values("DATE_TIME").reset_index(drop=True)
        if df.empty:
            raise ValueError(
                "solar_pred.csv and load_pred.csv have NO timestamps in common "
                f"(solar: {s.DATE_TIME.min()} -> {s.DATE_TIME.max()}, "
                f"load: {l.DATE_TIME.min()} -> {l.DATE_TIME.max()}). "
                "Run ems_full_pipeline.py first - it aligns both onto one timeline."
            )
        # time step taken from the data itself (1 h, 15 min, ...)
        cfg.dt_h = df["DATE_TIME"].diff().median().total_seconds() / 3600
        print(f"Using solar_pred.csv and load_pred.csv ({len(df)} steps of {cfg.dt_h * 60:.0f} min)")
        return df

    # ---------- DEMO MODE ----------
    print("DEMO MODE: solar_pred.csv / load_pred.csv not found.")
    g = pd.read_csv("Plant_1_Generation_Data.csv")
    g["DATE_TIME"] = pd.to_datetime(g["DATE_TIME"], format="%d-%m-%Y %H:%M")
    plant = g.groupby("DATE_TIME")["AC_POWER"].mean().mul(22).reset_index()
    # scale the big plant down to this building's PV size
    plant["SOLAR_KW"] = plant["AC_POWER"] / plant["AC_POWER"].max() * cfg.pv_capacity_kw

    t = plant["DATE_TIME"]
    h = t.dt.hour + t.minute / 60 if False else t.dt.hour + t.dt.minute / 60
    rng = np.random.default_rng(0)
    workday = (t.dt.dayofweek < 5).astype(float)
    # office-like load: 120 kW base, up to ~380 kW in working hours, evening bump
    load = (120
            + workday * 220 * np.exp(-((h - 13) / 4.0) ** 2)
            + 40 * np.exp(-((h - 20) / 2.0) ** 2)
            + rng.normal(0, 8, len(t)))
    plant["LOAD_KW"] = load.clip(lower=40)
    return plant[["DATE_TIME", "SOLAR_KW", "LOAD_KW"]]


# ================================================================ MAIN
if __name__ == "__main__":
    cfg = Config()
    # cfg.grid_charge_enabled = True   # try cheap-tariff grid charging

    df = load_inputs(cfg)
    res = run_ems(df, cfg)

    print("\n=== EMS SUMMARY ===")
    for k, v in summarize(res, cfg).items():
        print(f"{k:<40} {v:12.1f}")

    res.to_csv("ems_results.csv", index=False)
    plot_results(res, "ems_results.png", dt_h=cfg.dt_h)
    print("\nSaved ems_results.csv and ems_results.png")
