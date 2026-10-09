"""
Microgrid EMS web app.

Setup (once):
    pip install streamlit xgboost optuna scikit-learn pandas numpy matplotlib joblib
    python ems_full_pipeline.py          # trains + saves the two models

Start the app (from the folder that contains the .joblib files):
    streamlit run app_ems.py
"""
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import ems_microgrid as ems
import ems_app_core as core

st.set_page_config(page_title="Microgrid EMS", page_icon="⚡", layout="wide")


@st.cache_resource
def get_models():
    return core.load_models()


try:
    load_b, solar_b = get_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

typ_people = max(1, int(round(load_b["defaults"].get("Occupancy", 5))))
typ_humidity = int(round(load_b["defaults"].get("Humidity", 45)))

# ============================================================== SIDEBAR
with st.sidebar:
    st.header("⚙️ System settings")
    st.subheader("Solar")
    pv = st.number_input("Solar PV size (kW)", 10.0, 5000.0, 120.0, 10.0)
    st.subheader("Battery")
    batt_kwh = st.number_input("Battery capacity (kWh)", 10.0, 10000.0, 300.0, 10.0)
    ch_kw = st.number_input("Max charge power (kW)", 1.0, 5000.0, 60.0, 5.0)
    dis_kw = st.number_input("Max discharge power (kW)", 1.0, 5000.0, 60.0, 5.0)
    soc_rng = st.slider("Allowed battery range (%)", 0, 100, (20, 95))
    soc_init = st.slider("Starting battery level (%)", 0, 100, 50)
    eff = st.slider("Battery efficiency each way (%)", 80, 100, 95) / 100
    st.subheader("Grid & prices")
    peak = st.slider("Peak-price hours", 0, 24, (18, 22))
    p_peak = st.number_input("Peak price (per kWh)", 0.0, 100.0, 10.0, 0.5)
    p_off = st.number_input("Off-peak price (per kWh)", 0.0, 100.0, 6.0, 0.5)
    allow_export = st.checkbox("Allow exporting extra solar to grid", True)
    p_exp = st.number_input("Export (feed-in) price (per kWh)", 0.0, 100.0, 3.0, 0.5)
    limit = st.number_input("Grid import limit (kW, 0 = none)", 0.0, 100000.0, 0.0, 10.0)
    gc = st.checkbox("Charge battery from grid in cheap hours", False)
    gc_win = st.slider("Cheap hours", 0, 24, (0, 6))
    gc_target = st.slider("Grid-charge up to (%)", 0, 100, 60)

cfg = ems.Config(
    dt_h=1.0, pv_capacity_kw=pv, batt_capacity_kwh=batt_kwh,
    batt_max_charge_kw=ch_kw, batt_max_discharge_kw=dis_kw,
    batt_eff_charge=eff, batt_eff_discharge=eff,
    soc_min=soc_rng[0] / 100, soc_max=soc_rng[1] / 100,
    soc_init=min(max(soc_init, soc_rng[0]), soc_rng[1]) / 100,
    grid_import_limit_kw=limit if limit > 0 else 1e9,
    allow_export=allow_export, export_price=p_exp,
    peak_hours=(peak[0], peak[1]), price_peak=p_peak, price_offpeak=p_off,
    grid_charge_enabled=gc, offpeak_hours=(gc_win[0], gc_win[1]),
    grid_charge_target_soc=gc_target / 100,
)


def plot_plan(res):
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    ax[0].plot(res.DATE_TIME, res.LOAD_KW, color="#c0392b", label="Predicted building load")
    ax[0].plot(res.DATE_TIME, res.SOLAR_KW, color="#e69f00", label="Predicted solar production")
    ax[0].fill_between(res.DATE_TIME, res.SOLAR_KW, color="#f5b700", alpha=0.25)
    ax[0].set_ylabel("kW"); ax[0].set_title("What the models predict"); ax[0].legend(loc="upper right")

    ax[1].stackplot(res.DATE_TIME, res.pv_to_load, res.batt_to_load, res.grid_to_load,
                    labels=["Solar -> building", "Battery -> building", "Grid -> building"],
                    colors=["#f5b700", "#2e9e5b", "#5b7fb5"], alpha=0.85)
    ax[1].plot(res.DATE_TIME, res.LOAD_KW, "k--", lw=1, label="Building load")
    ax[1].set_ylabel("kW"); ax[1].set_title("Who supplies the building (EMS decision)")
    ax[1].legend(loc="upper right", ncol=4)

    ax[2].plot(res.DATE_TIME, res.soc * 100, color="#2e9e5b", lw=2)
    ax[2].axhline(cfg.soc_min * 100, color="red", ls=":", lw=1, label="Minimum reserve")
    ax[2].axhline(cfg.soc_max * 100, color="gray", ls=":", lw=1, label="Maximum")
    ax[2].set_ylim(0, 100); ax[2].set_ylabel("Battery level (%)"); ax[2].set_title("Battery state of charge")
    ax[2].legend(loc="upper right")
    plt.tight_layout()
    return fig


# ================================================================== TABS
st.title("⚡ Microgrid Energy Management System")
st.caption("Predictions come from your trained XGBoost models (load demand + solar generation). "
           "The EMS then decides: solar first, then battery, then grid.")

tab1, tab2, tab3, tab4 = st.tabs(["📅 Plan future days", "⏱️ Single-hour what-if",
                                  "📐 Sizing explorer", "ℹ️ About"])

# ------------------------------------------------------------- TAB 1
with tab1:
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("First day", dt.date.today() + dt.timedelta(days=1))
    ndays = c2.slider("Number of days", 1, 7, 3)
    humidity = c3.slider("Typical humidity (%)", 20, 90, typ_humidity)

    st.subheader("1. Describe each day")
    days_cfg = []
    for i in range(ndays):
        day = pd.Timestamp(start) + pd.Timedelta(days=i)
        with st.expander(f"Day {i + 1} — {day:%A %d %b %Y}", expanded=(i == 0)):
            a, b, c, d, e = st.columns(5)
            sky = a.selectbox("Sky", list(core.SKY), key=f"sky{i}")
            tmin = b.number_input("Min temp (°C)", -10.0, 45.0, 22.0, 0.5, key=f"tmin{i}")
            tmax = c.number_input("Max temp (°C)", -10.0, 55.0, 32.0, 0.5, key=f"tmax{i}")
            ppl = d.number_input("People at peak", 0, 5000, typ_people, key=f"ppl{i}")
            hol = e.checkbox("Holiday", False, key=f"hol{i}")
        days_cfg.append(dict(sky=sky, tmin=tmin, tmax=tmax, people=int(ppl), holiday=hol))

    lo, hi = solar_b["doy_range"]
    doys = [(pd.Timestamp(start) + pd.Timedelta(days=i)).dayofyear for i in range(ndays)]
    if min(doys) < lo or max(doys) > hi:
        st.warning("The solar model was trained on mid-May to mid-June data. Your dates are outside that "
                   "period, so treat solar numbers as approximate (sunlight intensity matters most).")

    table = core.make_hourly_table(start, days_cfg, humidity)
    with st.expander("2. (Optional) Fine-tune any hour — edit the table directly"):
        st.caption("Sun: 0 = night, ~0.5 cloudy, ~1.0 clear noon. Changing the settings above resets this table.")
        sig = abs(hash(str(days_cfg) + str(start) + str(humidity)))
        table = st.data_editor(table, key=f"editor_{sig}", disabled=["Time"])

    frame = core.forecast_frame(load_b, solar_b, table)
    res = core.run_with(cfg, frame)
    s = ems.summarize(res, cfg)

    st.subheader("3. Results")
    m = st.columns(6)
    m[0].metric("Building demand", f"{s['Total load (kWh)']:,.0f} kWh")
    m[1].metric("Solar produced", f"{s['Solar generated (kWh)']:,.0f} kWh")
    m[2].metric("Bought from grid", f"{s['Grid import (kWh)']:,.0f} kWh")
    m[3].metric("Covered by solar+battery", f"{s['Load covered by solar+battery (%)']:.0f}%")
    m[4].metric("Battery at the end", f"{res.soc.iloc[-1] * 100:.0f}%")
    m[5].metric("Savings vs grid only", f"{s['Savings']:,.0f}")

    fig = plot_plan(res)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Day-by-day summary")
    st.dataframe(core.daily_summary(res, cfg).round(1))

    st.subheader("Key points")
    for msg in core.insights(res, cfg):
        st.markdown("- " + msg)

    st.download_button("⬇️ Download hourly results (CSV)", res.to_csv(index=False).encode("utf-8"),
                       "ems_plan.csv", "text/csv")

# ------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("What happens in one specific hour?")
    a, b, c = st.columns(3)
    d1 = a.date_input("Date", dt.date.today(), key="sh_date")
    hr = b.slider("Hour of day", 0, 23, 12, key="sh_hour")
    soc_sh = c.slider("Battery level right now (%)", 0, 100, int(round(cfg.soc_init * 100)), key="sh_soc")
    a, b, c, d = st.columns(4)
    sun = a.slider("Sunlight (0 = night, 1 = clear noon)", 0.0, 1.2, 0.8, 0.05, key="sh_sun")
    out_t = b.number_input("Outside temp (°C)", -10.0, 55.0, 30.0, 0.5, key="sh_t")
    hum = c.slider("Humidity (%)", 20, 90, typ_humidity, key="sh_hum")
    ppl = d.number_input("People in building", 0, 5000, typ_people, key="sh_ppl")
    a, b, c = st.columns(3)
    hv = a.checkbox("HVAC on", True, key="sh_hv")
    lt = b.checkbox("Lights on", True, key="sh_lt")
    hl = c.checkbox("Holiday", False, key="sh_hl")

    ts = pd.Timestamp(d1) + pd.Timedelta(hours=hr)
    row = dict(Sun=sun, OutsideTemp=out_t, Humidity=float(hum), People=int(ppl), HVAC=hv, Lights=lt, Holiday=hl)
    solar, load, r, soc_after, why = core.single_hour(
        load_b, solar_b, ts, row, min(max(soc_sh, cfg.soc_min * 100), cfg.soc_max * 100) / 100, cfg)

    m = st.columns(4)
    m[0].metric("Predicted solar", f"{solar:.1f} kW")
    m[1].metric("Predicted building load", f"{load:.1f} kW")
    m[2].metric("Battery level after", f"{soc_after * 100:.0f}%")
    price = ems.price_at(ts.hour, cfg)
    m[3].metric("Grid price this hour", f"{price:g} / kWh")
    st.info(why)
    flows = {"Solar → building": r["pv_to_load"], "Battery → building": r["batt_to_load"],
             "Grid → building": r["grid_to_load"], "Solar → battery": r["pv_to_batt"],
             "Grid → battery": r["grid_to_batt"], "Solar → grid (export)": r["pv_export"]}
    flows = {k: round(v, 1) for k, v in flows.items() if v > 0.05}
    if flows:
        st.bar_chart(pd.Series(flows, name="kW"))

# ------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("Which PV + battery size saves the most?")
    st.caption("Re-runs the days you planned in the first tab with different system sizes "
               "(other settings stay as in the sidebar).")
    pv_opts = st.multiselect("Solar PV sizes to test (kW)", [30, 60, 120, 180, 240, 360, 500], default=[60, 120, 240])
    b_opts = st.multiselect("Battery sizes to test (kWh, 0 = no battery)", [0, 100, 200, 300, 500, 800, 1200],
                            default=[0, 100, 300, 800])
    if pv_opts and b_opts:
        sw = core.sizing_sweep(frame, cfg, sorted(pv_opts), sorted(b_opts))
        piv = sw.pivot(index="PV_kW", columns="Battery_kWh", values="Savings")
        fig, ax = plt.subplots(figsize=(8, 1.2 + 0.8 * len(piv)))
        im = ax.imshow(piv.values, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel("Battery size (kWh)"); ax.set_ylabel("PV size (kW)"); ax.set_title("Savings vs grid only")
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, f"{v:,.0f}", ha="center", va="center")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(sw.round(1))
        best = sw.loc[sw.Savings.idxmax()]
        st.success(f"Best savings among the options tested: PV {best.PV_kW:g} kW + battery "
                   f"{best.Battery_kWh:g} kWh. Remember bigger systems cost more to buy, "
                   "which this table does not include.")
    else:
        st.info("Pick at least one PV size and one battery size.")

# ------------------------------------------------------------- TAB 4
with tab4:
    st.markdown("""
### How this app works
1. **You describe the days** (sky, temperatures, people, holiday). The app turns that into an hourly table.
2. **Solar model (XGBoost)** predicts solar production from sunlight, temperature and time of day.
3. **Load model (XGBoost)** predicts the building's demand from temperature, humidity, people, HVAC, lights, day of week.
4. **EMS rules** run hour by hour: solar serves the building first → extra solar charges the battery (then is exported) →
   shortfalls are covered by the battery (down to its reserve) → then the grid. The battery level carries over hour to hour.

### Things to keep in mind
- Predictions are only as good as the data the models were trained on. The load model learned from your
  `Energy_consumption.csv` period; the solar model from Plant 1 (mid-May to mid-June), scaled to your PV size.
- Use *People* values similar to those in your dataset's Occupancy column.
- Prices, PV and battery sizes in the sidebar are placeholders. Enter your real ones.
- The sizing explorer shows savings only. It does **not** include purchase cost.
""")
