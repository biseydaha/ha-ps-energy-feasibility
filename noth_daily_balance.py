#!/usr/bin/env python3
"""
Daily energy balance for a solar-powered UAV (Noth method)
==========================================================

First-pass feasibility check following A. Noth, "Design of Solar Powered
Airplanes for Continuous Flight" (ETH Zürich, 2008), Chapter 3: continuous
flight requires that the solar energy collected in one day covers the
energy consumed over 24 h, with night energy routed through the battery.

Irradiance is represented by Noth's sinusoidal day model, parameterised by
the noon irradiance I_max and the day length T_day:
    E_solar = (2/π) · I_max · T_day · A_sc · η_sc · η_cbr · η_mppt · η_wthr
    E_req   = P_tot · (T_day + T_night / (η_chrg · η_dchrg))

Outputs
-------
1. Day and night length vs day of year
2. Noon irradiance I_max vs day of year
3. Daily solar energy vs daily required energy
4. Daily energy margin and minimum battery mass vs day of year

The aircraft parameters below are generic placeholders for the method,
not the final DELICE design (see delice_mission_sim.py).

Usage
-----
    python noth_daily_balance.py --out figures/

Author: Şeydanur Akgün
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Inputs ────────────────────────────────────────────────────────────────────
@dataclass
class Site:
    latitude_deg: float = 39.92       # Ankara


@dataclass
class SolarModel:
    I_sc: float = 1367.0              # solar constant [W/m²]
    eta_atm: float = 0.95             # lumped atmospheric transmissivity [-]
    eta_wthr: float = 1.0             # weather factor (1 = clear sky) [-]


@dataclass
class SolarGenerator:
    A_sc_m2: float = 40.0             # solar cell area [m²]
    eta_sc: float = 0.21              # cell efficiency [-]
    eta_mppt: float = 0.97            # MPPT efficiency [-]
    eta_cbr: float = 0.95             # camber / wiring / shading losses [-]


@dataclass
class PowerModel:
    W_N: float = 1300.0               # weight [N]
    V_ms: float = 30.0                # cruise speed [m/s]
    L_D: float = 25.0                 # lift-to-drag ratio [-]
    eta_ctrl: float = 0.98            # motor controller
    eta_mot: float = 0.92             # motor
    eta_grb: float = 1.00             # gearbox (direct drive)
    eta_plr: float = 0.80             # propeller
    P_av_W: float = 40.0              # avionics [W]
    P_pld_W: float = 20.0             # payload [W]
    eta_bec: float = 0.95             # step-down converter
    eta_chrg: float = 0.97            # battery charge
    eta_dchrg: float = 0.97           # battery discharge
    batt_Wh_per_kg: float = 350.0     # battery specific energy [Wh/kg]
    usable_DoD: float = 0.8           # usable depth of discharge [-]


# ── Solar geometry ────────────────────────────────────────────────────────────
def solar_declination_deg(day_of_year: np.ndarray) -> np.ndarray:
    """Solar declination [deg]."""
    return 23.45 * np.sin(np.deg2rad((360.0 / 365.0) * (day_of_year - 81.0)))


def day_length_hours(latitude_deg: float, decl_deg: np.ndarray) -> np.ndarray:
    """Day length [h] from the sunrise hour angle (clipped for polar day/night)."""
    cosws = np.clip(-np.tan(np.deg2rad(latitude_deg)) * np.tan(np.deg2rad(decl_deg)), -1.0, 1.0)
    return 2.0 * np.arccos(cosws) * 24.0 / (2.0 * np.pi)


def i_max_noon_Wm2(I_sc: float, eta_atm: float, latitude_deg: float,
                   decl_deg: np.ndarray) -> np.ndarray:
    """Noon irradiance on a horizontal surface: I_sc · η_atm · cos(zenith_noon)."""
    lat, dec = np.deg2rad(latitude_deg), np.deg2rad(decl_deg)
    cos_zen = np.clip(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec), -1.0, 1.0)
    return I_sc * eta_atm * np.maximum(0.0, cos_zen)


# ── Energy balance ────────────────────────────────────────────────────────────
def daily_solar_energy_Wh(day_hours, I_max, A_sc_m2, eta_wthr, eta_sc, eta_cbr, eta_mppt):
    """Electrical energy collected per day [Wh] (Noth Eq. 3.8–3.9)."""
    E_density_Jm2 = (2.0 / np.pi) * I_max * day_hours * 3600.0 * eta_wthr
    return E_density_Jm2 * A_sc_m2 * eta_sc * eta_cbr * eta_mppt / 3600.0


def electrical_power_required_W(pm: PowerModel) -> float:
    """Total electrical power [W] (Noth Eq. 3.6); level power = W·V / (L/D)."""
    P_lev = pm.W_N * pm.V_ms / pm.L_D
    chain = pm.eta_ctrl * pm.eta_mot * pm.eta_grb * pm.eta_plr
    return P_lev / chain + (pm.P_av_W + pm.P_pld_W) / pm.eta_bec


def daily_required_energy_Wh(day_hours, pm: PowerModel):
    """Energy required per day [Wh] (Noth Eq. 3.7)."""
    P_tot = electrical_power_required_W(pm)
    return P_tot * (day_hours + (24.0 - day_hours) / (pm.eta_chrg * pm.eta_dchrg))


def battery_mass_kg_for_night(day_hours, pm: PowerModel):
    """Minimum battery mass [kg] to supply the night energy."""
    E_night = electrical_power_required_W(pm) * (24.0 - day_hours) / (pm.eta_chrg * pm.eta_dchrg)
    return E_night / (pm.batt_Wh_per_kg * pm.usable_DoD)


# ── Run ───────────────────────────────────────────────────────────────────────
def run(site=Site(), sm=SolarModel(), sg=SolarGenerator(), pm=PowerModel()):
    doy = np.arange(1, 366)
    decl = solar_declination_deg(doy)
    T_day = day_length_hours(site.latitude_deg, decl)
    Imax = i_max_noon_Wm2(sm.I_sc, sm.eta_atm, site.latitude_deg, decl)
    E_solar = daily_solar_energy_Wh(T_day, Imax, sg.A_sc_m2, sm.eta_wthr,
                                    sg.eta_sc, sg.eta_cbr, sg.eta_mppt)
    E_req = daily_required_energy_Wh(T_day, pm)
    return {"doy": doy, "T_day": T_day, "T_night": 24.0 - T_day, "Imax": Imax,
            "E_solar_Wh": E_solar, "E_req_Wh": E_req, "margin_Wh": E_solar - E_req,
            "m_batt_kg": battery_mass_kg_for_night(T_day, pm)}


def plot(r, out):
    def fig(name, title, ylabel, *series):
        plt.figure()
        for y, label in series:
            plt.plot(r["doy"], y, linewidth=2, label=label)
        plt.xlabel("Day of Year")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        if any(label for _, label in series):
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, name), dpi=150)
        plt.close()

    fig("noth_day_night.png", "Day / night duration", "Hours",
        (r["T_day"], "Day length [h]"), (r["T_night"], "Night length [h]"))
    fig("noth_imax.png", r"Noon irradiance $I_{\max}$", r"$I_{\max}$ [W/m$^2$]", (r["Imax"], None))
    fig("noth_energy_balance.png", "Daily energy balance", "Energy [kWh/day]",
        (r["E_solar_Wh"] / 1e3, "Solar energy"), (r["E_req_Wh"] / 1e3, "Required energy"))
    fig("noth_margin.png", "Daily energy margin", "kWh/day",
        (r["margin_Wh"] / 1e3, "Energy margin"))
    fig("noth_battery_mass.png", "Minimum battery mass (night energy)", "Battery mass [kg]",
        (r["m_batt_kg"], None))


def summary(r):
    for label, i in [("WORST DAY (minimum margin)", int(np.argmin(r["margin_Wh"]))),
                     ("BEST DAY (maximum margin)", int(np.argmax(r["margin_Wh"])))]:
        print(f"=== {label} ===")
        print(f"DOY = {r['doy'][i]}")
        print(f"T_day = {r['T_day'][i]:.2f} h, T_night = {r['T_night'][i]:.2f} h")
        print(f"Imax = {r['Imax'][i]:.1f} W/m^2")
        print(f"E_solar = {r['E_solar_Wh'][i] / 1e3:.3f} kWh/day")
        print(f"E_req   = {r['E_req_Wh'][i] / 1e3:.3f} kWh/day")
        print(f"Margin  = {r['margin_Wh'][i] / 1e3:.3f} kWh/day")
        print(f"m_batt  = {r['m_batt_kg'][i]:.2f} kg")


def main():
    ap = argparse.ArgumentParser(description="Noth daily energy balance")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    r = run()
    plot(r, args.out)
    summary(r)


if __name__ == "__main__":
    main()
