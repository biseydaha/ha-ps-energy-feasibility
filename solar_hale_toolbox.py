#!/usr/bin/env python3
"""
Solar-electric HALE design toolbox
==================================

General sizing relations for solar-electric high-altitude long-endurance
aircraft, following

    A. Bierig et al., "General Design Considerations for Solar-Electric
    High-Altitude Long-Endurance Aircraft", ICAS 2024.

Contents
--------
1. Solar geometry and daily energy density vs latitude and day of year,
   with optional Beer-Lambert attenuation (altitude-scaled air mass) and an
   optional two-panel (flat + dihedral) wing incidence model.
2. Propulsion power loading as a function of wing loading.
3. Specific power chart over a 24 h day.
4. Maximum wing loading that closes the 24 h energy balance.
5. Required battery gravimetric energy density for night flight.
6. Leg-based mission power model (climb, cruise, banked loiter, descent).
7. Battery state-of-charge simulation for a leg-based mission profile.
8. Closed-loop SOC + altitude simulation (climb on surplus, descend on deficit).

Usage
-----
    python solar_hale_toolbox.py --demo all --out figures/
    python solar_hale_toolbox.py --demo mission

Author: Şeydanur Akgün
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

trapz = getattr(np, "trapezoid", None) or np.trapz

# ── 1. Constants ──────────────────────────────────────────────────────────────
DEG2RAD = np.pi / 180.0
G = 9.81            # [m/s²]
S0 = 1367.0         # solar constant [W/m²]
RHO_0 = 1.225       # sea-level density [kg/m³]
RHO_60kFT = 0.088   # density at ~60 000 ft [kg/m³]
H_SCALE = 8434.0    # atmospheric scale height [m]


# ── 2. Solar geometry and energy ──────────────────────────────────────────────
def solar_declination(doy):
    """Solar declination [rad], Cooper approximation."""
    return 23.45 * DEG2RAD * np.sin(2 * np.pi * (284 + doy) / 365.0)


def solar_hour_angle(time_sec):
    """Hour angle [rad]: 0 at solar noon, negative in the morning."""
    return (time_sec / 3600.0 - 12.0) * 15.0 * DEG2RAD


def solar_zenith_angle(doy, time_sec, lat_deg):
    """Return (zenith angle [rad], cos zenith)."""
    phi = lat_deg * DEG2RAD
    delta = solar_declination(doy)
    tau = solar_hour_angle(time_sec)
    cos_zeta = np.clip(np.sin(phi) * np.sin(delta)
                       + np.cos(phi) * np.cos(delta) * np.cos(tau), -1.0, 1.0)
    return np.arccos(cos_zeta), cos_zeta


def solar_azimuth_angle(doy, time_sec, lat_deg):
    """Solar azimuth [rad], clockwise from north, in [0, 2π)."""
    phi = lat_deg * DEG2RAD
    delta = solar_declination(doy)
    tau = solar_hour_angle(time_sec)
    _, cos_zeta = solar_zenith_angle(doy, time_sec, lat_deg)
    sin_zeta = np.maximum(np.sin(np.arccos(cos_zeta)), 1e-6)
    sin_psi = -np.cos(delta) * np.sin(tau) / sin_zeta
    cos_psi = (np.sin(delta) - np.sin(phi) * cos_zeta) / (np.cos(phi) * sin_zeta)
    psi = np.arctan2(sin_psi, cos_psi)
    return np.where(psi < 0.0, psi + 2.0 * np.pi, psi)


def air_mass(cos_zeta, altitude_m):
    """Relative air mass 1/cos(zenith), scaled by exp(-h/H) for altitude."""
    m0 = np.zeros_like(cos_zeta)
    up = cos_zeta > 0.0
    m0[up] = 1.0 / cos_zeta[up]
    return m0 * np.exp(-altitude_m / H_SCALE)


def cos_incidence_two_panel(cos_zeta, azimuth, A1, A2, dihedral_deg=3.0):
    """
    Area-weighted incidence cosine for a wing with a flat panel (area A1) and
    a dihedral panel (area A2).
    """
    gamma = np.deg2rad(dihedral_deg)
    sin_zeta = np.sqrt(np.maximum(0.0, 1.0 - cos_zeta ** 2))
    cos_i1 = np.maximum(0.0, cos_zeta)
    cos_i2 = np.maximum(0.0, np.sin(gamma) * sin_zeta * np.cos(azimuth)
                        + np.cos(gamma) * cos_zeta)
    return (A1 * cos_i1 + A2 * cos_i2) / (A1 + A2)


def daily_energy_density(lat_deg, doy, altitude_m=18288.0, tracked=False,
                         n_steps=1440, k_atm=0.0, two_panel=None):
    """
    Daily direct solar energy density [Wh/m²].

    tracked   : True for an ideal sun-tracking receiver, False for a wing surface.
    k_atm     : Beer-Lambert extinction coefficient; flux *= exp(-k_atm * air_mass).
                0 (default) neglects attenuation, as justified above ~18 km.
    two_panel : None for a flat horizontal wing, or dict(A1=..., A2=...,
                dihedral_deg=...) for the flat + dihedral incidence model.
    """
    t = np.linspace(0, 24 * 3600, n_steps)
    _, cos_zeta = solar_zenith_angle(doy, t, lat_deg)
    up = cos_zeta > 0.0

    if tracked:
        cos_eff = np.ones_like(t)
    elif two_panel is not None:
        cos_eff = cos_incidence_two_panel(cos_zeta, solar_azimuth_angle(doy, t, lat_deg),
                                          **two_panel)
    else:
        cos_eff = cos_zeta

    flux = np.zeros_like(t)
    flux[up] = S0 * cos_eff[up] * np.exp(-k_atm * air_mass(cos_zeta, altitude_m)[up])
    return trapz(flux, t) / 3600.0


def energy_density_map(latitudes, days, tracked=False, altitude_m=18288.0,
                       n_steps=1440, **kw):
    """Daily energy density [Wh/m²] on a latitude × day-of-year grid."""
    return np.array([[daily_energy_density(lat, d, altitude_m=altitude_m,
                                           tracked=tracked, n_steps=n_steps, **kw)
                      for d in days] for lat in latitudes])


# ── 3. Aerodynamics and power required ────────────────────────────────────────
def aerodynamic_coeffs(CL, CD0, aspect_ratio, e_oswald=0.9):
    """Parabolic polar: CD = CD0 + CL² / (π Λ e)."""
    return CD0 + CL ** 2 / (np.pi * aspect_ratio * e_oswald)


def power_loading_required(WL_kgm2, rho, CL, CD0, aspect_ratio,
                           eta_prop, eta_motor, eta_wiring):
    """
    Propulsion electrical power per unit wing area [W/m²] in level flight at
    constant CL: V from L = W, P_el / S = (D/S) · V / (η_prop η_motor η_wiring).
    """
    V = np.sqrt(2.0 * WL_kgm2 * G / (rho * CL))
    D_per_area = 0.5 * rho * V ** 2 * aerodynamic_coeffs(CL, CD0, aspect_ratio)
    return D_per_area * V / (eta_prop * eta_motor * eta_wiring)


# ── 4. Solar generator and specific power ─────────────────────────────────────
def average_solar_power_per_area(lat_deg, doy, kappa_cover=0.7, eta_SG=0.28,
                                 tracked=False, n_steps=1440):
    """
    24 h average electrical power of the solar generator per wing area.
    Returns (P_avg [W/m²], E_el [Wh/m²]).
    """
    E_el = kappa_cover * eta_SG * daily_energy_density(lat_deg, doy, tracked=tracked,
                                                       n_steps=n_steps)
    return E_el / 24.0, E_el


def specific_power_chart(lat_deg, doy, WL_kgm2, rho, CL, CD0, aspect_ratio,
                         kappa_cover=0.7, eta_SG=0.28, eta_prop=0.7,
                         eta_motor=0.9, eta_wiring=0.9, n_steps=1440):
    """
    Power required vs solar power per wing area over one day.
    Returns (t [h], P_req [W/m²], P_solar [W/m²]).
    """
    t = np.linspace(0, 24 * 3600, n_steps)
    P_req = power_loading_required(WL_kgm2, rho, CL, CD0, aspect_ratio,
                                   eta_prop, eta_motor, eta_wiring)
    _, cos_zeta = solar_zenith_angle(doy, t, lat_deg)
    P_solar = np.where(cos_zeta > 0.0, kappa_cover * eta_SG * S0 * cos_zeta, 0.0)
    return t / 3600.0, np.full(t.shape, P_req), P_solar


# ── 5. Maximum feasible wing loading ──────────────────────────────────────────
def max_wing_loading_map(latitudes, days, rho, CL, CD0, aspect_ratio,
                         kappa_cover=0.7, eta_SG=0.28, eta_prop=0.7,
                         eta_motor=0.9, eta_wiring=0.9, tracked=False,
                         WL_guess_range=(0.5, 15.0), n_steps_solar=720, n_iter=50):
    """
    Maximum wing loading [kg/m²] for which 24 h average solar power equals
    propulsion power, found by bisection. NaN during polar night.
    """
    out = np.zeros((len(latitudes), len(days)))
    for i, lat in enumerate(latitudes):
        for j, d in enumerate(days):
            P_solar, _ = average_solar_power_per_area(lat, d, kappa_cover, eta_SG,
                                                      tracked, n_steps_solar)
            if P_solar <= 1e-6:
                out[i, j] = np.nan
                continue
            a, b = WL_guess_range
            for _ in range(n_iter):
                mid = 0.5 * (a + b)
                if power_loading_required(mid, rho, CL, CD0, aspect_ratio,
                                          eta_prop, eta_motor, eta_wiring) > P_solar:
                    b = mid
                else:
                    a = mid
            out[i, j] = 0.5 * (a + b)
    return out


# ── 6. Required battery energy density ────────────────────────────────────────
def required_battery_energy_density(latitudes, days, WL_kgm2, rho, CL, CD0,
                                    aspect_ratio, battery_mass_fraction=0.45,
                                    kappa_cover=0.7, eta_SG=0.28, eta_prop=0.7,
                                    eta_motor=0.9, eta_wiring=0.9, tracked=False,
                                    n_steps=1440):
    """
    Battery gravimetric energy density [Wh/kg] needed to cover every period in
    which propulsion power exceeds solar power:
        E_B = E_store/S / (W/S) / f_B
    """
    out = np.zeros((len(latitudes), len(days)))
    for i, lat in enumerate(latitudes):
        for j, d in enumerate(days):
            t_h, P_req, P_solar = specific_power_chart(
                lat, d, WL_kgm2, rho, CL, CD0, aspect_ratio, kappa_cover, eta_SG,
                eta_prop, eta_motor, eta_wiring, n_steps)
            E_store = trapz(np.maximum(0.0, P_req - P_solar), t_h)
            out[i, j] = 0.0 if E_store <= 1e-6 else E_store / WL_kgm2 / battery_mass_fraction
    return out


# ── 7. Leg-based mission power model ──────────────────────────────────────────
def leg_power_required(m, S, V, rho, CL, CD0, aspect_ratio, e_oswald,
                       eta_prop, eta_motor, eta_wiring,
                       roc=0.0, accel=0.0, bank_deg=0.0):
    """
    Electrical power for one mission leg, including climb, acceleration and
    banked-turn lift increment. The CL argument is unused: CL is recomputed
    from the lift requirement at the given airspeed.
    Returns dict with CL, CD, Drag_N, P_shaft_W, P_el_W.
    """
    q = 0.5 * rho * V ** 2
    L = m * G / np.cos(bank_deg * DEG2RAD)
    CL_req = L / (q * S)
    CD = aerodynamic_coeffs(CL_req, CD0, aspect_ratio, e_oswald)
    D = q * S * CD
    P_shaft = (D * V + m * G * roc + m * V * accel) / eta_prop
    return {"CL": CL_req, "CD": CD, "Drag_N": D, "P_shaft_W": P_shaft,
            "P_el_W": P_shaft / (eta_motor * eta_wiring)}


def leg_energy(P_el_W, duration_s):
    """Energy of one leg [Wh]."""
    return P_el_W * duration_s / 3600.0


def example_mission_case(verbose=True):
    """Energy budget of a climb / cruise / loiter / descent mission at ~60 000 ft."""
    m, S, rho = 131.6, 41.25, RHO_60kFT
    common = dict(rho=rho, CL=1.1, CD0=0.03, aspect_ratio=21, e_oswald=0.9,
                  eta_prop=0.7, eta_motor=0.9, eta_wiring=0.9)
    legs = [("Climb", dict(V=13.0, roc=0.5), 2.0),
            ("Cruise", dict(V=15.0), 6.0),
            ("Loiter", dict(V=14.0, bank_deg=20.0), 10.0),
            ("Descent", dict(V=17.0, roc=-0.6), 1.5)]
    energies = {}
    for name, kw, hours in legs:
        P = leg_power_required(m, S, **kw, **common)["P_el_W"]
        energies[name] = leg_energy(P, hours * 3600)
    if verbose:
        print("\n=== MISSION ENERGY SUMMARY ===")
        for name, E in energies.items():
            print(f"{name + ' energy':15s}: {E:8.1f} Wh")
        print("--------------------------------")
        print(f"{'TOTAL ENERGY':15s}: {sum(energies.values()):8.1f} Wh")
    return energies


# ── 8. SOC simulation for a leg-based profile ─────────────────────────────────
def solar_electrical_power_W(t_sec, lat_deg, doy, S_wing_m2, kappa_cover=0.7, eta_SG=0.28):
    """Instantaneous solar generator output [W] on a horizontal wing."""
    _, cz = solar_zenith_angle(doy, np.array([t_sec]), lat_deg)
    return max(0.0, kappa_cover * S_wing_m2 * eta_SG * S0 * cz[0])


def build_mission_power_profile(legs, dt_sec=10.0):
    """
    Concatenate legs into a power-demand time series.

    legs: list of dicts with "name", "duration_s" and "power_func"
          (callable of leg-local time [s] returning P_el [W]).
    Returns (t [s], P_req [W], leg index per sample, leg names).
    """
    t_all, P_all, id_all, names = [], [], [], []
    t0 = 0.0
    for k, leg in enumerate(legs):
        names.append(leg.get("name", f"Leg {k + 1}"))
        T = leg["duration_s"]
        t_local = np.arange(0, int(np.ceil(T / dt_sec))) * dt_sec
        t_local = t_local[t_local < T]
        p = np.array([leg["power_func"](tl) for tl in t_local], dtype=float)
        t_all.append(t0 + t_local)
        P_all.append(p)
        id_all.append(np.full(p.shape, k, dtype=int))
        t0 += T
    return np.concatenate(t_all), np.concatenate(P_all), np.concatenate(id_all), names


def simulate_soc(t_sec, P_req_W, lat_deg, doy, S_wing_m2, E_bat_max_Wh, SOC0=0.8,
                 kappa_cover=0.7, eta_SG=0.28, eta_charge=0.95, eta_discharge=0.95):
    """
    Battery energy bookkeeping for a given demand profile:
        P_net = P_solar - P_req
        charge:    E += P_net · η_ch · dt
        discharge: E += P_net / η_dis · dt
    saturated to [0, E_max]. Solar time wraps every 24 h.
    Returns (P_solar, P_net, E_bat, SOC, hit_empty, hit_full).
    """
    dt = np.diff(t_sec, prepend=t_sec[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.0
    n = len(t_sec)
    E = np.zeros(n)
    E[0] = np.clip(SOC0, 0.0, 1.0) * E_bat_max_Wh
    P_solar = np.zeros(n)
    P_net = np.zeros(n)
    hit_empty = np.zeros(n, dtype=bool)
    hit_full = np.zeros(n, dtype=bool)

    for i in range(n):
        P_solar[i] = solar_electrical_power_W(t_sec[i] % 86400, lat_deg, doy,
                                              S_wing_m2, kappa_cover, eta_SG)
        P_net[i] = P_solar[i] - P_req_W[i]
        if i == 0:
            continue
        eta = eta_charge if P_net[i] >= 0.0 else 1.0 / eta_discharge
        E[i] = E[i - 1] + P_net[i] * eta * dt[i] / 3600.0
        if E[i] <= 0.0:
            E[i], hit_empty[i] = 0.0, True
        elif E[i] >= E_bat_max_Wh:
            E[i], hit_full[i] = E_bat_max_Wh, True

    return P_solar, P_net, E, E / E_bat_max_Wh, hit_empty, hit_full


def energy_balance_check(t_sec, P_req, P_solar, P_net, E_bat, E_bat_max_Wh):
    """Compare integrated power flows with the change in stored energy."""
    t_h = t_sec / 3600.0
    print("\n=== ENERGY BALANCE CHECK ===")
    print(f"E_req   : {trapz(P_req, t_h):10.1f} Wh")
    print(f"E_solar : {trapz(P_solar, t_h):10.1f} Wh")
    print(f"E_net   : {trapz(P_net, t_h):10.1f} Wh  (solar - req)")
    print(f"dE_bat  : {E_bat[-1] - E_bat[0]:10.1f} Wh  (difference = charge/discharge losses)")
    print(f"SOC     : {E_bat[0] / E_bat_max_Wh:.3f} -> {E_bat[-1] / E_bat_max_Wh:.3f}")


# ── 9. Closed-loop SOC + altitude simulation ──────────────────────────────────
def isa_density(h):
    """ISA density [kg/m³], troposphere and isothermal layer (to ~20 km)."""
    T0, p0, g0, R, L = 288.15, 101325.0, 9.80665, 287.05, -0.0065
    if h <= 11000.0:
        T = T0 + L * h
        p = p0 * (T / T0) ** (-g0 / (L * R))
    else:
        T = T0 + L * 11000.0
        p = p0 * (T / T0) ** (-g0 / (L * R)) * np.exp(-g0 * (h - 11000.0) / (R * T))
    return p / (R * T)


def tas_from_CL(m, S, CL, rho):
    """True airspeed for level flight at constant CL [m/s]."""
    return np.sqrt(2.0 * m * G / (rho * S * CL))


def aero_power_constant_CL(m, S, CL, CD0, aspect_ratio, e_oswald, rho):
    """Return (D·V [W], V, D, CD) for level flight at constant CL."""
    V = tas_from_CL(m, S, CL, rho)
    CD = aerodynamic_coeffs(CL, CD0, aspect_ratio, e_oswald)
    D = 0.5 * rho * V ** 2 * S * CD
    return D * V, V, D, CD


def altitude_control(SOC, SOC_min, P_net, h, h_min, h_max, ROC_climb=0.3, ROC_descent=0.4):
    """Climb on solar surplus, descend on deficit or low SOC, else hold."""
    if P_net > 0.0 and SOC > SOC_min and h < h_max:
        return +ROC_climb
    if (P_net < 0.0 or SOC <= SOC_min) and h > h_min:
        return -ROC_descent
    return 0.0


def simulate_soc_with_altitude(t_sec, lat_deg, doy, m, S, CL, CD0, aspect_ratio, e_oswald,
                               h0, h_min, h_max, E_bat_max_Wh, SOC0=0.8, SOC_min=0.2,
                               kappa_cover=0.7, eta_SG=0.28, eta_prop=0.713,
                               eta_motor=0.84, eta_wiring=0.9, eta_charge=0.95,
                               eta_discharge=0.95, ROC_climb=0.99, ROC_descent=0.97,
                               t0_local_sec=0.0):
    """
    Closed-loop simulation in which surplus solar energy is stored as
    altitude (potential energy) and recovered by descending at night.
    The kinetic-energy term approximates d(½mV²)/dt with dV/V ≈ dh/(2H),
    H = 8500 m, for constant-CL flight.
    """
    dt = np.diff(t_sec, prepend=t_sec[0])
    dt[0] = dt[1]
    n = len(t_sec)
    SOC, Ebat, h, V = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    P_req, P_solar, P_net = np.zeros(n), np.zeros(n), np.zeros(n)
    SOC[0], Ebat[0], h[0] = SOC0, SOC0 * E_bat_max_Wh, h0
    eta_el = eta_motor * eta_wiring

    for i in range(1, n):
        rho = isa_density(h[i - 1])
        V[i] = tas_from_CL(m, S, CL, rho)
        P_aero = aero_power_constant_CL(m, S, CL, CD0, aspect_ratio, e_oswald, rho)[0]
        t_local = (t0_local_sec + t_sec[i]) % 86400
        P_solar[i] = solar_electrical_power_W(t_local, lat_deg, doy, S, kappa_cover, eta_SG)

        ROC = altitude_control(SOC[i - 1], SOC_min, P_solar[i] - P_aero / eta_prop / eta_el,
                               h[i - 1], h_min, h_max, ROC_climb, ROC_descent)
        P_climb = m * G * ROC
        P_kin = 0.5 * m * V[i] ** 2 * (ROC / 8500.0)
        P_req[i] = (P_aero + P_climb + P_kin) / eta_prop / eta_el
        P_net[i] = P_solar[i] - P_req[i]

        eta = eta_charge if P_net[i] >= 0 else 1.0 / eta_discharge
        Ebat[i] = np.clip(Ebat[i - 1] + P_net[i] * eta * dt[i] / 3600.0, 0.0, E_bat_max_Wh)
        SOC[i] = Ebat[i] / E_bat_max_Wh
        h[i] = np.clip(h[i - 1] + ROC * dt[i], h_min, h_max)

    return {"t_sec": t_sec, "SOC": SOC, "h": h, "V": V,
            "P_req": P_req, "P_solar": P_solar, "P_net": P_net}


# ── Demos ─────────────────────────────────────────────────────────────────────
def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {os.path.basename(path)}")


def demo_energy_maps(out):
    lats, days = np.linspace(-90, 90, 91), np.arange(1, 366, 1)
    E_h = energy_density_map(lats, days, tracked=False)
    E_t = energy_density_map(lats, days, tracked=True)
    D, L = np.meshgrid(days, lats)
    vmin, vmax = np.nanmin(E_t), np.nanmax(E_t)
    for E, label, fname in [(E_h, "direct radiation (horizontal surface)", "toolbox_energy_horizontal.png"),
                            (E_t, "normal radiation (tracking surface)", "toolbox_energy_tracking.png")]:
        plt.figure(figsize=(8, 4))
        cs = plt.contourf(D, L, E, levels=np.linspace(vmin, vmax, 90),
                          norm=Normalize(vmin=vmin, vmax=vmax), cmap="Spectral_r", extend="max")
        cb = plt.colorbar(cs)
        cb.set_label("Daily energy (Wh/m²)")
        cb.set_ticks(np.linspace(vmin, vmax, 6))
        plt.xlim(1, 365)
        plt.ylim(-90, 90)
        plt.yticks([-90, -60, -30, 0, 30, 60, 90])
        plt.xlabel("Day of year")
        plt.ylabel("Latitude [deg]")
        plt.title(f"Energy density of solar {label}")
        _save(os.path.join(out, fname))


def demo_specific_power(out):
    t_h, P_req, P_solar = specific_power_chart(30.0, 121, 3.7, RHO_60kFT, 1.1, 0.02, 22,
                                               n_steps=720)
    plt.figure(figsize=(8, 4))
    plt.plot(t_h, P_solar, label="Power from solar generator")
    plt.plot(t_h, P_req, "--", label="Power required for propulsion")
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 3))
    plt.xlabel("Time of day [h]")
    plt.ylabel("Power loading [W/m²]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Specific power chart (30°N, 1 May, W/S = 3.7 kg/m²)")
    _save(os.path.join(out, "toolbox_specific_power.png"))


def demo_wing_loading(out):
    lats, days = np.linspace(0, 90, 91), np.arange(1, 366, 1)
    cases = [("Zephyr-like (AR=21, CD0=0.02)", dict(CD0=0.02, aspect_ratio=21)),
             ("PHASA-like (AR=40, CD0=0.012)", dict(CD0=0.012, aspect_ratio=40)),
             ("DELICE (AR=22, CD0=0.01645)", dict(CD0=0.01645, aspect_ratio=22,
                                                  eta_prop=0.713, eta_motor=0.84,
                                                  eta_wiring=0.95))]
    D, L = np.meshgrid(days, lats)
    for k, (label, kw) in enumerate(cases, 1):
        WL = max_wing_loading_map(lats, days, RHO_60kFT, 1.1, n_steps_solar=360, **kw)
        plt.figure(figsize=(8, 4))
        cs = plt.contourf(D, L, WL, levels=20)
        plt.colorbar(cs, label="Max wing loading [kg/m²]")
        plt.xlabel("Day of year")
        plt.ylabel("Latitude [deg]")
        plt.title(f"Max wing loading — {label}")
        _save(os.path.join(out, f"toolbox_wl_max_{k}.png"))


def demo_battery(out):
    lats, days = np.linspace(-40, 40, 41), np.arange(1, 366, 5)
    D, L = np.meshgrid(days, lats)
    for k, (WL, CD0, AR, fB) in enumerate([(2.5, 0.02, 21, 0.45), (5.0, 0.012, 40, 0.40)], 1):
        E = required_battery_energy_density(lats, days, WL, RHO_60kFT, 1.1, CD0, AR,
                                            battery_mass_fraction=fB, n_steps=720)
        plt.figure(figsize=(8, 4))
        cs = plt.contourf(D, L, E, levels=20)
        plt.colorbar(cs, label="Required battery energy density [Wh/kg]")
        plt.xlabel("Day of year")
        plt.ylabel("Latitude [deg]")
        plt.title(f"Required battery energy density (W/S = {WL} kg/m²)")
        _save(os.path.join(out, f"toolbox_battery_{k}.png"))


def demo_mission(out):
    example_mission_case()
    out_sim = simulate_soc_with_altitude(
        t_sec=np.arange(0, 24 * 3600, 10.0), lat_deg=39.9, doy=172, m=131.6, S=41.25,
        CL=1.1, CD0=0.032, aspect_ratio=21, e_oswald=0.9, h0=18000.0, h_min=16000.0,
        h_max=22000.0, E_bat_max_Wh=20000.0, SOC0=0.8, SOC_min=0.2, t0_local_sec=6 * 3600)
    t_h = out_sim["t_sec"] / 3600.0
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t_h, out_sim["h"] / 1000.0)
    ax[0].set_ylabel("Altitude [km]")
    ax[0].set_title("Closed-loop altitude / SOC simulation (39.9°N, 21 June, start 06:00)")
    ax[1].plot(t_h, out_sim["P_req"] / 1000.0, label="P_required")
    ax[1].plot(t_h, out_sim["P_solar"] / 1000.0, label="P_solar")
    ax[1].plot(t_h, out_sim["P_net"] / 1000.0, "--", label="P_net")
    ax[1].set_ylabel("Power [kW]")
    ax[1].legend()
    ax[2].plot(t_h, out_sim["SOC"], label="SOC")
    ax[2].axhline(0.2, color="r", linestyle="--", label="SOC_min")
    ax[2].set_ylabel("SOC [-]")
    ax[2].set_xlabel("Time [h]")
    ax[2].legend()
    for a in ax:
        a.grid(True, alpha=0.3)
    _save(os.path.join(out, "toolbox_altitude_soc.png"))
    print(f"  SOC min {out_sim['SOC'].min():.3f}, final {out_sim['SOC'][-1]:.3f}")


DEMOS = {"maps": demo_energy_maps, "power": demo_specific_power,
         "wingloading": demo_wing_loading, "battery": demo_battery, "mission": demo_mission}


def main():
    ap = argparse.ArgumentParser(description="Solar-electric HALE design toolbox demos")
    ap.add_argument("--demo", choices=list(DEMOS) + ["all"], default="mission")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in (DEMOS if args.demo == "all" else [args.demo]):
        print(f"[{name}]")
        DEMOS[name](args.out)


if __name__ == "__main__":
    main()
