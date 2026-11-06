#!/usr/bin/env python3
"""
DELICE mission energy simulation
================================

Time-stepping energy simulation of DELICE, a 30 m class solar-electric
high-altitude platform designed for continuous operation at 20 km.
Used for the energy-feasibility chapter of the M.Sc. thesis and the
IAC 2026 paper (Akgün & Yaman, "Energy-Constrained Mission Design and
Power Analysis of Solar-Powered High-Altitude Platform Station").

Model chain
-----------
Atmosphere : ISA troposphere + isothermal lower stratosphere (to 20 km).
Solar      : Cooper declination, hour angle from solar time, horizontal wing
             P_pv = kappa * eta_SG * eta_MPPT * eta_wire * S0 * cos(zenith) * S_wing.
             Atmospheric attenuation is neglected at 20 km (Bierig et al.,
             ICAS 2024); see solar_hale_toolbox.py for the optional
             Beer-Lambert correction.
Aero       : Parabolic polar, CD0(h) interpolated to capture the Reynolds
             number effect with altitude.
Propulsion : Twin NeuMotor 6560/8 (KV 55) + Mejzlik 64x36 propellers.
             Propeller CT/CP tables are solved for thrust = required thrust;
             motor current and voltage follow from the torque balance.
Battery    : Energy bookkeeping with charge/discharge efficiencies,
             E_bat in [0, 22 543 Wh].

Simulated cases (21 June, 40 deg N)
-----------------------------------
M1, M2 : motor-on cruise at 20 km for 48 h, sunrise / sunset start.
G1-G3  : night motor-off glide down to a 14 km floor, re-climb at sunrise,
         with initial SOC of 100 %, 50 % and 40 %.
C1-C3  : climb from sea level to 20 km at 1 m/s, departing at sunrise,
         sunrise + 3 h and sunrise - 3 h.
Optional (--maps): global irradiation, maximum wing loading and required
battery energy density maps.

Usage
-----
    python delice_mission_sim.py              # mission + climb figures
    python delice_mission_sim.py --maps       # also the global maps (slower)
    python delice_mission_sim.py --out results/

Author: Şeydanur Akgün
"""

import argparse
import os

import numpy as np
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.20,
    "legend.fontsize": 9,
    "legend.framealpha": 0.92,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_SOL = "#FFC107"
C_REQ = "#C62828"
C_SUR = "#43A047"
C_DEF = "#EF9A9A"
C_SOC = "#1565C0"
C_PBAT = "#FF6D00"
C_NITE = "#E3F2FD"
C_ALT = "#6A1B9A"

# ── Design parameters (DELICE) ────────────────────────────────────────────────
DEG2RAD = np.pi / 180.
S0 = 1367.0               # solar constant [W/m²]
g = 9.80665               # [m/s²]
m = 135.1                 # take-off mass [kg]
S_wing = 41.874           # wing area [m²]
WL = m / S_wing           # wing loading [kg/m²]
AR = 22                   # aspect ratio
e_osw = 0.7728            # Oswald factor
CD0 = 0.01645             # zero-lift drag at 20 km
CL = 1.0917               # cruise lift coefficient
S_sol = 29.31             # solar cell area [m²]
kappa = S_sol / S_wing    # solar coverage ratio
eta_SG = 0.28             # solar cell efficiency
eta_MPPT = 0.95
eta_wire = 0.95
eta_prop = 0.778
eta_mot = 0.870
eta_wrg = 0.921
eta_sys = eta_prop * eta_mot * eta_wrg
E_bat = 22543.0           # battery capacity [Wh]
eta_ch = 0.95             # charge efficiency
eta_dis = 0.97            # discharge efficiency
P_avi = 300.0             # avionics + payload power [W]
f_bat = 0.422             # battery mass fraction
H_FLOOR = 14000.          # night glide floor [m]

# Altitude-dependent CD0 (Reynolds number effect)
_cd0_alts = np.array([0, 5000, 10000, 15000, 20000])
_cd0_vals = np.array([0.01039, 0.01133, 0.01242, 0.01323, 0.01645])


def get_CD0(h):
    """Zero-lift drag coefficient at altitude h [m]."""
    if h <= 20000:
        return float(np.interp(h, _cd0_alts, _cd0_vals))
    return 0.01645


K_ind = 1. / (np.pi * e_osw * AR)


def CL_md_at(h):
    """Lift coefficient for minimum drag (best glide) at altitude h."""
    return np.sqrt(get_CD0(h) / K_ind)


def LD_max_at(h):
    """Maximum lift-to-drag ratio at altitude h."""
    return 1. / (2. * np.sqrt(get_CD0(h) * K_ind))


# Propeller tables: Mejzlik 64x36
J_tab = np.array([0., 0.0858, 0.1717, 0.2575, 0.3433, 0.4292, 0.515,
                  0.6009, 0.6867, 0.7725, 0.8584])
CT_tab = np.array([0.0961, 0.0923, 0.0881, 0.0818, 0.0754, 0.0699, 0.0592,
                   0.0469, 0.0336, 0.0190, 0.0042])
Cp_tab = np.array([0.0367, 0.0387, 0.0403, 0.0414, 0.0421, 0.0432, 0.0404,
                   0.0357, 0.029, 0.0198, 0.0092])
D_prop = 64 * 0.0254      # propeller diameter [m]

# Motor: NeuMotor 6560/8, 12S battery bus
N_mot = 2
KV = 55.                  # [rpm/V]
Vbus = 44.2               # [V]
Rm = 0.14                 # winding resistance [Ohm]
I0 = 0.61                 # no-load current [A]
Kt = 60. / (2. * np.pi * KV)
Kv_rad = KV * 2. * np.pi / 60.
RPM_NL = (Kv_rad * (Vbus - I0 * Rm)) * 60. / (2. * np.pi)


# ── Atmosphere ────────────────────────────────────────────────────────────────
def isa(h):
    """ISA density [kg/m³] up to the lower stratosphere."""
    T0, p0, L, R = 288.15, 101325., -0.0065, 287.05
    if h <= 11000:
        T = T0 + L * h
        p = p0 * (T / T0) ** (-g / (L * R))
    else:
        T11 = T0 + L * 11000
        p11 = p0 * (T11 / T0) ** (-g / (L * R))
        T = T11
        p = p11 * np.exp(-g * (h - 11000) / (R * T))
    return p / (R * T)


rho20 = isa(20000)


# ── Solar model ───────────────────────────────────────────────────────────────
def solar_decl(doy):
    """Solar declination [rad], Cooper approximation."""
    return 23.45 * DEG2RAD * np.sin(2 * np.pi * (284 + doy) / 365.)


def cos_zenith(doy, t_sec, lat_deg):
    """Cosine of the solar zenith angle at solar time t_sec."""
    phi = lat_deg * DEG2RAD
    delta = solar_decl(doy)
    tau = (t_sec / 3600. - 12.) * 15. * DEG2RAD
    return np.clip(np.sin(phi) * np.sin(delta) +
                   np.cos(phi) * np.cos(delta) * np.cos(tau), -1., 1.)


def P_pv(t_sec, lat, doy):
    """Electrical power delivered by the PV array to the DC bus [W]."""
    cz = cos_zenith(doy, t_sec, lat)
    return max(0., kappa * eta_SG * eta_MPPT * eta_wire * S0 * cz * S_wing)


def p_pv_spec(t_sec, lat, doy):
    """Solar generator power per unit wing area [W/m²]."""
    cz = cos_zenith(doy, t_sec, lat)
    return max(0., kappa * eta_SG * S0 * cz)


def daily_E_m2(lat, doy, n=1440):
    """Daily extraterrestrial energy on a horizontal surface [Wh/m²]."""
    t = np.linspace(0, 86400, n)
    cz = np.maximum(0, [cos_zenith(doy, ti, lat) for ti in t])
    return trapezoid(S0 * cz, t) / 3600.


def sunrise_sunset(lat, doy):
    """Return (night length, sunrise, sunset) in solar hours."""
    d = solar_decl(doy)
    phi = lat * DEG2RAD
    ha = np.arccos(np.clip(-np.tan(phi) * np.tan(d), -1., 1.))
    SR = 12. - np.degrees(ha) / 15.
    SS = 12. + np.degrees(ha) / 15.
    return 24. - SS + SR, SR, SS


# ── Power required ────────────────────────────────────────────────────────────
def p_el_Wm2(rho):
    """Cruise propulsion electrical power per unit wing area [W/m²]."""
    CD = CD0 + CL ** 2 / (np.pi * AR * e_osw)
    V = np.sqrt(2. * WL * g / (rho * CL))
    return 0.5 * rho * V ** 2 * CD * V / eta_sys


p_el20 = p_el_Wm2(rho20)
P_cruise = p_el20 * S_wing + P_avi   # total cruise demand at 20 km [W]


def sink_rate(h_m):
    """Best-glide sink rate [m/s] at altitude h_m."""
    rho = isa(h_m)
    V = np.sqrt(2. * m * g / (rho * S_wing * CL_md_at(h_m)))
    return V / LD_max_at(h_m)


def WL_max_fn(lat, doy, rho=rho20):
    """Maximum wing loading [kg/m²] that closes the 24 h energy balance."""
    E = daily_E_m2(lat, doy)
    p_avg = kappa * eta_SG * E / 24.
    if p_avg <= 0:
        return 0.
    denom = np.sqrt(2. / rho) * g ** 1.5 * (CD0 / CL ** 1.5 +
                                            1. / (CL ** 0.5 * np.pi * AR * e_osw))
    return (p_avg * eta_sys / denom) ** (2. / 3.)


def e_bat_area(lat, doy, n=1440):
    """Energy to be stored per unit wing area for motor-on night cruise [Wh/m²]."""
    t = np.linspace(0, 86400, n)
    p_sg = np.array([p_pv_spec(ti, lat, doy) for ti in t])
    return trapezoid(np.maximum(0., p_el20 - p_sg), t) / 3600.


def e_bat_gr(lat, doy):
    """Required battery gravimetric energy density [Wh/kg]."""
    return e_bat_area(lat, doy) / WL / f_bat


def _prop_operating_point(V, rho, T_pp):
    """
    Solve the propeller table for the rotational speed [rev/s] that delivers
    thrust T_pp per propeller at airspeed V. Returns None if not reachable.
    """
    n_a = np.linspace(0.5, 100., 80000)
    Ct_a = np.interp(np.clip(V / (n_a * D_prop), J_tab[0], J_tab[-1]), J_tab, CT_tab)
    diff = Ct_a * rho * n_a ** 2 * D_prop ** 4 - T_pp
    cross = np.where(np.diff(np.sign(diff)))[0]
    if len(cross) == 0:
        return None
    i = cross[-1]
    return n_a[i] - diff[i] * (n_a[i + 1] - n_a[i]) / (diff[i + 1] - diff[i])


def _electrical_power(n, V, rho):
    """Electrical input power of both motors [W] at propeller speed n [rev/s]."""
    Cp = float(np.interp(V / (n * D_prop), J_tab, Cp_tab))
    Q = Cp * rho * n ** 2 * D_prop ** 5 / (2. * np.pi)
    I = Q / Kt + I0
    Ve = n * 2. * np.pi / Kv_rad
    return N_mot * (Ve + I * Rm) * I


ROC_CLIMB = 1.0  # target rate of climb [m/s]


def climb_at(h, ROC_target=ROC_CLIMB):
    """
    Constant-ROC climb at best-glide CL: the ESC throttles the motors to hold
    ROC_target. Returns (ROC, total electrical demand incl. avionics) [m/s, W].
    """
    rho = isa(h)
    cd0 = get_CD0(h)
    CL_c = np.sqrt(cd0 / K_ind)
    V = np.sqrt(2. * m * g / (rho * S_wing * CL_c))
    D = 2. * m * g * np.sqrt(cd0 * K_ind)
    T_pp = (D + m * g * ROC_target / V) / N_mot
    n = _prop_operating_point(V, rho, T_pp)
    if n is None:
        return 0., P_cruise
    return ROC_target, _electrical_power(n, V, rho) + P_avi


def P_cruise_at(h):
    """Level-cruise electrical demand (T = D) at altitude h, incl. avionics [W]."""
    rho = isa(h)
    V = np.sqrt(2. * m * g / (rho * S_wing * CL))
    CD = get_CD0(h) + CL ** 2 / (np.pi * AR * e_osw)
    D = 0.5 * rho * V ** 2 * S_wing * CD
    n = _prop_operating_point(V, rho, D / N_mot)
    if n is None:
        return P_avi
    return _electrical_power(n, V, rho) + P_avi


# ── Battery ───────────────────────────────────────────────────────────────────
def bat_step(E, Ppv, Pdem, dt=60.):
    """Advance battery energy by one step. Returns (E_new [Wh], P_bat [W])."""
    Pnet = Ppv - Pdem
    Pbat = Pnet * eta_ch if Pnet >= 0 else Pnet / eta_dis
    return np.clip(E + Pbat * dt / 3600., 0., E_bat), Pbat


def _log_entry(tm, t_UTC, h_m, E, Ppv, Pd, Pb, **extra):
    entry = {"t": tm, "t_UTC": t_UTC, "h": h_m / 1e3, "soc": E / E_bat * 100.,
             "Ppv": Ppv / 1e3, "Preq": Pd / 1e3, "Pbat": Pb / 1e3}
    entry.update(extra)
    return entry


# ── Simulators ────────────────────────────────────────────────────────────────
def sim_cruise(soc0, t0_UTC, lat=40., doy=172, dur=48., dt=60):
    """Motor-on level cruise at 20 km for `dur` hours, starting at t0_UTC."""
    E = soc0 * E_bat
    log = []
    for step in range(int(dur * 3600 / dt)):
        tm = step * dt / 3600.
        t_UTC = (tm + t0_UTC) % 24.
        Ppv = P_pv(t_UTC * 3600., lat, doy)
        E, Pb = bat_step(E, Ppv, P_cruise, dt)
        log.append(_log_entry(tm, t_UTC, 20000., E, Ppv, P_cruise, Pb))
    return log


def sim_nightoff(soc0, lat=40., doy=172, dur=48., dt=60):
    """
    Night motor-off strategy, starting at local midnight.
    Night: glide at best L/D with avionics only, down to H_FLOOR, then cruise
    at the floor. Day: climb back to 20 km, then cruise.
    """
    NH, SR, SS = sunrise_sunset(lat, doy)
    E = soc0 * E_bat
    h = 20000.
    log = []
    for step in range(int(dur * 3600 / dt)):
        tm = step * dt / 3600.
        t_UTC = tm % 24.
        if SR <= t_UTC < SS:
            Ppv = P_pv(t_UTC * 3600., lat, doy)
            if h < 20000.:
                roc, Pd = climb_at(h)
                dh = roc * dt
            else:
                Pd, dh = P_cruise, 0.
        else:
            Ppv = 0.
            if h > H_FLOOR:
                Pd, dh = P_avi, -sink_rate(h) * dt
            else:
                Pd, dh = P_cruise_at(h), 0.
        h = np.clip(h + dh, H_FLOOR, 20000.)
        E, Pb = bat_step(E, Ppv, Pd, dt)
        log.append(_log_entry(tm, t_UTC, h, E, Ppv, Pd, Pb))
    return log


def sim_climb(soc0, t0_UTC, lat=40., doy=172, dt=30):
    """Constant-ROC climb from sea level to 20 km, starting at t0_UTC."""
    E = soc0 * E_bat
    h, t = 0., 0.
    log = []
    while h < 20000. and E / E_bat > 0.01 and t < 3600. * 12.:
        roc, Pin = climb_at(h)
        if roc <= 0:
            break
        t_UTC = (t0_UTC + t / 3600.) % 24.
        Ppv = P_pv(t_UTC * 3600., lat, doy)
        E, Pb = bat_step(E, Ppv, Pin, dt)
        h += roc * dt
        t += dt
        log.append(_log_entry(t / 3600., t_UTC, h, E, Ppv, Pin, Pb, roc=roc))
    return log


def sim_full_mission(soc0, t0_UTC, lat=40., doy=172, dt=30):
    """
    End-to-end mission: climb to 20 km, cruise, glide through the night,
    re-climb at sunrise, and descend once the battery is full on day 2.
    """
    NH, SR, SS = sunrise_sunset(lat, doy)
    E = soc0 * E_bat
    h, t = 0., 0.
    log = []
    phase = "climb"
    while t < 3600. * 48.:
        t_UTC = (t0_UTC + t / 3600.) % 24.
        is_day = SR <= t_UTC < SS
        Ppv = P_pv(t_UTC * 3600., lat, doy)

        if phase == "climb":
            roc, Pin = climb_at(h)
            if roc <= 0 or h >= 20000.:
                h = min(h, 20000.)
                phase, Pd = "cruise", P_cruise
            else:
                Pd = Pin
                h += roc * dt
        elif phase == "cruise":
            Pd = P_cruise
            if not is_day:
                phase = "night_glide"
        elif phase == "night_glide":
            if h > H_FLOOR:
                Pd = P_avi
                h = max(h - sink_rate(h) * dt, H_FLOOR)
            else:
                Pd = P_cruise_at(h)
            if is_day:
                phase = "day_reclimb"
        elif phase == "day_reclimb":
            if h < 20000.:
                roc, Pd = climb_at(h)
                h = min(h + roc * dt, 20000.)
            else:
                phase = "cruise_day2"
        elif phase == "cruise_day2":
            Pd = P_cruise
            if E / E_bat > 0.99:
                phase = "descent"
        elif phase == "descent":
            Pd = P_avi
            if h > 50:
                h = max(h - sink_rate(h) * dt, 0.)
            else:
                h = 0.
                break

        E, Pb = bat_step(E, Ppv, Pd, dt)
        t += dt
        log.append(_log_entry(t / 3600., t_UTC, h, E, Ppv, Pd, Pb, phase=phase))
    return log


# ── Plot helpers ──────────────────────────────────────────────────────────────
def shade_nights(ax, SR, SS, xlim, t0):
    for day in range(int(xlim[1] / 24.) + 2):
        for ns, ne in [(0., SR), (SS, 24.)]:
            s = day * 24. + ns - t0
            e = day * 24. + ne - t0
            ax.axvspan(max(s, xlim[0]), min(e, xlim[1]),
                       alpha=0.18, color=C_NITE, zorder=0)


def add_utc_axis(ax, t0, xlim, step=6):
    ax2 = ax.twiny()
    ax2.set_xlim(xlim)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 42))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ticks = np.arange(0, xlim[1] + 1, step)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{int((p + t0) % 24):02d}:00" for p in ticks], fontsize=8.5)
    ax2.set_xlabel("Solar time  [hh:00]", fontsize=9., labelpad=6)
    return ax2


def _note_box(ax, text):
    if text:
        ax.text(0.012, 0.04, text, transform=ax.transAxes,
                fontsize=8., va="bottom", ha="left", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.45", fc="#FFFDE7",
                          ec="#F9A825", alpha=0.95, lw=1.2))


def _arrays(log):
    return {k: np.array([x[k] for x in log]) for k in ("t", "h", "soc", "Ppv", "Preq", "Pbat")}


def plot_mission(log, fig_title, path, t0_UTC, note_txt=None,
                 doy=172, lat=40., xlim=None):
    """Four panels: altitude, DC-bus power balance, battery power, SOC."""
    NH, SR, SS = sunrise_sunset(lat, doy)
    a = _arrays(log)
    t, h, soc, ppv, preq, pb = a["t"], a["h"], a["soc"], a["Ppv"], a["Preq"], a["Pbat"]
    if xlim is None:
        xlim = (0., t[-1])

    fig = plt.figure(figsize=(11, 17.5))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.5, 0.9, 2.2], hspace=0.50)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.89, bottom=0.12)
    fig.suptitle(fig_title, fontsize=10.5, fontweight="bold", y=0.97)

    ax1 = fig.add_subplot(gs[0])
    shade_nights(ax1, SR, SS, xlim, t0_UTC)
    ax1.plot(t, h, color=C_ALT, lw=2.2, label="Altitude")
    ax1.fill_between(t, 0, h, alpha=0.12, color=C_ALT)
    ax1.axhline(20., color="#1565C0", ls="--", lw=1.0, alpha=0.7, label="20 km")
    if h.min() < 19.5:
        ax1.axhline(H_FLOOR / 1e3, color="#FF8F00", ls=":", lw=1.2,
                    alpha=0.8, label=f"{H_FLOOR / 1e3:.0f} km floor")
    ax1.set_ylabel("Altitude  [km]")
    ax1.set_title("Altitude Profile", pad=10)
    ax1.set_ylim(bottom=0.)
    ax1.set_xlim(xlim)
    ax1.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.14, 0.5))
    ax1.grid(True, ls=":", alpha=0.22)

    ax2 = fig.add_subplot(gs[1])
    shade_nights(ax2, SR, SS, xlim, t0_UTC)
    ax2.fill_between(t, 0, ppv, alpha=0.70, color=C_SOL, label="$P_{pv}$  solar")
    ax2.plot(t, preq, color=C_REQ, lw=2.0, label="$P_{demand}$")
    ax2.fill_between(t, preq, ppv, where=ppv > preq,
                     alpha=0.45, color=C_SUR, label="Surplus → charging")
    ax2.fill_between(t, ppv, preq, where=ppv < preq,
                     alpha=0.30, color=C_DEF, label="Deficit ← discharge")
    ax2.set_ylabel("Power  [kW]")
    ax2.set_title("Power Balance at DC Bus", pad=10)
    ax2.set_ylim(bottom=0.)
    ax2.set_xlim(xlim)
    ax2.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.19, 0.5))
    ax2.grid(True, ls=":", alpha=0.22)

    ax3 = fig.add_subplot(gs[2])
    shade_nights(ax3, SR, SS, xlim, t0_UTC)
    ax3.fill_between(t, 0, pb, where=pb >= 0, alpha=0.75,
                     color=C_SUR, label=f"Charging  ($\\eta_{{ch}}={eta_ch}$)")
    ax3.fill_between(t, pb, 0, where=pb < 0, alpha=0.75,
                     color=C_PBAT, label=f"Discharging  ($\\eta_{{dis}}={eta_dis}$)")
    ax3.axhline(0., color="black", lw=0.8, alpha=0.5)
    ax3.set_ylabel("$P_{bat}$  [kW]")
    ax3.set_title("Battery Power Flow  (+ charge / − discharge)", pad=10)
    ax3.set_xlim(xlim)
    ax3.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.22, 0.5))
    ax3.grid(True, ls=":", alpha=0.22)

    ax4 = fig.add_subplot(gs[3])
    shade_nights(ax4, SR, SS, xlim, t0_UTC)
    ax4.plot(t, soc, color=C_SOC, lw=2.2, label="Battery SOC", zorder=5)
    ax4.axhline(20., color="red", ls="--", lw=1.5, label="20% reserve")
    ax4.axhline(100., color="gray", ls=":", lw=0.9, alpha=0.5)
    ax4.fill_between(t, 0, soc, where=soc < 20., alpha=0.28, color="red",
                     label="Below reserve", zorder=3)
    ax4.fill_between(t, 0, soc, where=soc >= 20., alpha=0.09, color=C_SOC, zorder=2)
    if soc.min() < 1.:
        dep_t = t[np.argmax(soc < 1.)]
        ax4.axvline(dep_t, color="darkred", ls="-.", lw=1.8, alpha=0.9)
        ax4.text(min(dep_t + xlim[1] * 0.015, xlim[1] * 0.80), 55,
                 f"Depleted\n@ {dep_t:.1f} h",
                 fontsize=9., color="darkred", fontweight="bold", va="center")
    else:
        idx_m = int(np.argmin(soc))
        ax4.annotate(f"Min SOC  {soc[idx_m]:.1f}%",
                     xy=(t[idx_m], soc[idx_m]),
                     xytext=(min(t[idx_m] + 2.5, xlim[1] * 0.75),
                             min(soc[idx_m] + 14., 88.)),
                     fontsize=9., color=C_SOC,
                     arrowprops=dict(arrowstyle="->", lw=1.0, color=C_SOC))
    _note_box(ax4, note_txt)
    ax4.set_ylabel("SOC  [%]")
    ax4.set_title("Battery State of Charge", pad=10)
    ax4.set_ylim(-5., 108.)
    ax4.set_xlim(xlim)
    ax4.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.15, 0.5))
    ax4.grid(True, ls=":", alpha=0.22)
    ax4.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    add_utc_axis(ax4, t0_UTC, xlim)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if soc.min() < 1.:
        return f"depleted at {t[np.argmax(soc < 1.)]:.1f} h → INFEASIBLE"
    return f"min SOC {soc.min():.1f}%, final {soc[-1]:.1f}% → FEASIBLE"


def plot_climb(log, fig_title, path, t0_UTC, note_txt=None,
               doy=172, lat=40., x_max=6.0):
    """Three panels: altitude, power balance, SOC for a climb to 20 km."""
    NH, SR, SS = sunrise_sunset(lat, doy)
    a = _arrays(log)
    t, h, soc, ppv, preq = a["t"], a["h"], a["soc"], a["Ppv"], a["Preq"]
    xlim = (0., x_max)

    fig = plt.figure(figsize=(12, 10.0))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.4, 1.0, 2.0], hspace=0.52)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.89, bottom=0.12)
    fig.suptitle(fig_title, fontsize=10.5, fontweight="bold", y=0.97)

    ax1 = fig.add_subplot(gs[0])
    shade_nights(ax1, SR, SS, xlim, t0_UTC)
    ax1.plot(t, h, color=C_ALT, lw=2.2)
    ax1.fill_between(t, 0, h, alpha=0.15, color=C_ALT)
    ax1.axhline(20., color="#1565C0", ls="--", lw=1.3, alpha=0.7, label="Target 20 km")
    ax1.set_ylabel("Altitude  [km]")
    ax1.set_title("Altitude Profile (Climb)", pad=10)
    ax1.set_ylim(bottom=0.)
    ax1.set_xlim(xlim)
    ax1.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 0.5))
    ax1.grid(True, ls=":", alpha=0.22)

    ax2 = fig.add_subplot(gs[1])
    shade_nights(ax2, SR, SS, xlim, t0_UTC)
    ax2.fill_between(t, 0, ppv, alpha=0.70, color=C_SOL, label="$P_{pv}$  solar")
    ax2.plot(t, preq, color=C_REQ, lw=2.0, label="$P_{demand}$ (climb)")
    ax2.fill_between(t, preq, ppv, where=ppv > preq, alpha=0.45, color=C_SUR, label="Surplus")
    ax2.fill_between(t, ppv, preq, where=ppv < preq, alpha=0.30, color=C_DEF, label="Deficit")
    ax2.set_ylabel("Power  [kW]")
    ax2.set_title("Power Balance", pad=10)
    ax2.set_ylim(bottom=0.)
    ax2.set_xlim(xlim)
    ax2.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.15, 0.5))
    ax2.grid(True, ls=":", alpha=0.22)

    ax3 = fig.add_subplot(gs[2])
    shade_nights(ax3, SR, SS, xlim, t0_UTC)
    ax3.plot(t, soc, color=C_SOC, lw=2.2, label="Battery SOC")
    ax3.axhline(20., color="red", ls="--", lw=1.5, label="20% reserve")
    ax3.fill_between(t, 0, soc, where=soc < 20., alpha=0.28, color="red")
    ax3.fill_between(t, 0, soc, where=soc >= 20., alpha=0.09, color=C_SOC)
    if soc.min() < 1.:
        ax3.axvline(t[-1], color="darkred", ls="-.", lw=1.8)
        ax3.text(t[-1] * 0.6, 55, f"Aborted at {h[-1]:.1f} km\nSOC={soc[-1]:.1f}%",
                 color="darkred", fontsize=9., fontweight="bold")
    else:
        idx_m = int(np.argmin(soc))
        ax3.annotate(f"Min SOC  {soc[idx_m]:.1f}%",
                     xy=(t[idx_m], soc[idx_m]),
                     xytext=(max(t[idx_m] - 0.4, 0.1), soc[idx_m] + 14.),
                     fontsize=9., color=C_SOC,
                     arrowprops=dict(arrowstyle="->", lw=1.0, color=C_SOC))
        ax3.text(xlim[1] * 0.8, soc[-1] + 5.,
                 f"Final SOC\n{soc[-1]:.1f}%\n@ 20 km",
                 ha="right", fontsize=9., color=C_SOC, fontweight="bold")
    _note_box(ax3, note_txt)
    ax3.set_ylabel("SOC  [%]")
    ax3.set_xlabel("Elapsed time  [h]", labelpad=30)
    ax3.set_title("Battery State of Charge", pad=10)
    ax3.set_ylim(-5., 108.)
    ax3.set_xlim(xlim)
    ax3.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.15, 0.5))
    ax3.grid(True, ls=":", alpha=0.22)
    ax3.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    add_utc_axis(ax3, t0_UTC, xlim)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f"reached {h[-1]:.1f} km, final SOC {soc[-1]:.1f}%"


# ── Optional global maps ──────────────────────────────────────────────────────
XTICK_POS = [1, 60, 121, 182, 244, 305, 365]
XTICK_LBL = ["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Dec"]


def _finish_map(fig, ax, path, ylim=(-90, 90), yticks=(-60, -30, 0, 30, 60)):
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Latitude  [°]")
    ax.set_xlim(1, 365)
    ax.set_ylim(*ylim)
    ax.set_xticks(XTICK_POS)
    ax.set_xticklabels(XTICK_LBL, fontsize=9)
    if yticks:
        ax.set_yticks(list(yticks))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {os.path.basename(path)}")


def make_global_maps(out):
    """Global irradiation, maximum wing loading and battery energy density maps."""
    lats = np.linspace(-90., 90., 91)
    days = np.arange(1, 366, 1)
    D_m, L_m = np.meshgrid(days, lats)

    print("Global irradiation maps...")
    E_hz = np.zeros((len(lats), len(days)))
    E_tr = np.zeros((len(lats), len(days)))
    t = np.linspace(0, 86400, 720)
    for i, lat in enumerate(lats):
        for j, doy in enumerate(days):
            cz = np.maximum(0, [cos_zenith(doy, ti, lat) for ti in t])
            E_hz[i, j] = trapezoid(S0 * cz, t) / 3600.
            E_tr[i, j] = trapezoid(S0 * (cz > 0).astype(float), t) / 3600.
    vmax = 32808.
    levels = np.linspace(0, vmax, 90)
    for E_data, subtitle, fname in [
            (E_hz, "Horizontal surface  (wing-mounted PV)", "solar_irradiation_horizontal.png"),
            (E_tr, "Tracking surface", "solar_irradiation_tracking.png")]:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        cf = ax.contourf(D_m, L_m, E_data, levels=levels, cmap="Spectral_r", extend="max")
        cb = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.95)
        cb.set_label("Daily energy  [Wh/m²]", fontsize=9.5)
        cb.set_ticks(np.linspace(0, vmax, 7))
        ax.axhline(40., color="white", ls="--", lw=2.0, alpha=0.9)
        ax.text(6, 42., "40°N (DELICE)", color="white", fontsize=9.,
                fontweight="bold", va="bottom")
        ax.set_title(f"Solar Irradiation — {subtitle}\n" r"$h = 20\ \mathrm{km}$", pad=14)
        _finish_map(fig, ax, os.path.join(out, fname))

    print("Maximum wing loading map...")
    WL_map = np.array([[WL_max_fn(lat, doy) for doy in days] for lat in lats])
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    cf = ax.contourf(D_m, L_m, WL_map, levels=np.linspace(0, 6., 60),
                     cmap="viridis", extend="max")
    cb = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.95)
    cb.set_label("Max wing loading  [kg/m²]", fontsize=9.5)
    cs = ax.contour(D_m, L_m, WL_map, levels=[WL], colors="red",
                    linewidths=2.2, linestyles="--")
    ax.clabel(cs, fmt=f"DELICE  {WL:.2f} kg/m²", fontsize=8.5, colors="red")
    ax.axhline(40., color="white", ls=":", lw=2.0, alpha=0.9)
    ax.text(6, 41.5, "40°N", color="white", fontsize=9., fontweight="bold")
    ax.set_title("Maximum Wing Loading for 24-h Energy Balance  [kg/m²]\n"
                 f"$\\Lambda={AR}$,  $C_{{D0}}={CD0}$,  "
                 f"$\\kappa={kappa:.2f}$,  $\\eta_{{SG}}={eta_SG}$", pad=14)
    _finish_map(fig, ax, os.path.join(out, "wl_max_map.png"))

    print("Required battery energy density map...")
    lats_b = np.linspace(-60., 60., 61)
    days_b = np.arange(1, 366, 3)
    Eb_map = np.array([[e_bat_gr(lat, doy) for doy in days_b] for lat in lats_b])
    D_b, L_b = np.meshgrid(days_b, lats_b)
    avail_cell = 500.                      # Amprius SA03, cell level [Wh/kg]
    avail_system = E_bat / (m * f_bat)     # system level [Wh/kg]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    cf = ax.contourf(D_b, L_b, Eb_map, levels=np.linspace(300, 650, 60),
                     cmap="YlOrRd", extend="both")
    cb = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.95)
    cb.set_label("Required battery energy density  [Wh/kg]", fontsize=9.5)
    cs1 = ax.contour(D_b, L_b, Eb_map, levels=[avail_system], colors="blue", linewidths=2.2)
    ax.clabel(cs1, fmt=f"System  {avail_system:.0f} Wh/kg", fontsize=8.5, colors="blue")
    cs2 = ax.contour(D_b, L_b, Eb_map, levels=[avail_cell], colors="#2E7D32",
                     linewidths=1.8, linestyles="--")
    ax.clabel(cs2, fmt=f"Cell  {avail_cell:.0f} Wh/kg", fontsize=8.5, colors="#2E7D32")
    ax.axhline(40., color="navy", ls=":", lw=2.0, alpha=0.85)
    ax.text(6, 38., "40°N", color="navy", fontsize=9., fontweight="bold")
    ax.set_title("Required Battery Gravimetric Energy Density  [Wh/kg]\n"
                 f"$W/S={WL:.2f}$ kg/m²,  $f_{{bat}}={f_bat}$  (motor-on overnight cruise)",
                 pad=14)
    _finish_map(fig, ax, os.path.join(out, "bat_density_map.png"),
                ylim=(-45, 45), yticks=None)


# ── Scenario runners ──────────────────────────────────────────────────────────
LAT, DOY = 40., 172   # 21 June, 40 deg N


def run_missions(out):
    NH, SR, SS = sunrise_sunset(LAT, DOY)
    head = f"21 June, φ={LAT:.0f}°N, h=20 km"
    cases = [
        ("M1", sim_cruise(1.00, SR, lat=LAT, doy=DOY), SR,
         f"Motor-ON Cruise  |  100% SOC  |  Sunrise Start\n"
         f"{head}  |  $P_{{cruise}}={P_cruise:.0f}$ W  |  48 h",
         "energy_M1_cruise_sunrise.png"),
        ("M2", sim_cruise(1.00, SS, lat=LAT, doy=DOY), SS,
         f"Motor-ON Cruise  |  100% SOC  |  Sunset Start\n"
         f"{head}  |  $P_{{cruise}}={P_cruise:.0f}$ W  |  Worst-case timing",
         "energy_M2_cruise_sunset.png"),
    ]
    for soc0, tag in [(1.00, "G1"), (0.50, "G2"), (0.40, "G3")]:
        cases.append((tag, sim_nightoff(soc0, lat=LAT, doy=DOY), 0.,
                      f"Night Motor-OFF + Glide  |  {soc0 * 100:.0f}% Initial SOC  |  Midnight Start\n"
                      f"{head}  |  Day: {P_cruise:.0f} W cruise  |  Night: {P_avi:.0f} W avionics",
                      f"energy_{tag}_glide_soc{soc0 * 100:.0f}.png"))

    results = {}
    for tag, log, t0, title, fname in cases:
        status = plot_mission(log, title, os.path.join(out, fname), t0, lat=LAT, doy=DOY)
        results[tag] = status
        print(f"  {tag}: {fname}  [{status}]")
    return results


def run_climbs(out):
    NH, SR, SS = sunrise_sunset(LAT, DOY)
    cases = [
        ("C1", SR, "Sunrise Departure", "energy_C1_climb_sunrise.png"),
        ("C2", SR + 3., "Sunrise +3h", "energy_C2_climb_sunrise_plus3h.png"),
        ("C3", SR - 3., "3h Before Sunrise", "energy_C3_climb_predawn.png"),
    ]
    results = {}
    for tag, t0, label, fname in cases:
        log = sim_climb(1.00, t0, lat=LAT, doy=DOY)
        t_arr = np.array([x["t"] for x in log])
        h_arr = np.array([x["h"] for x in log])
        soc_arr = np.array([x["soc"] for x in log])
        i20 = int(np.argmax(h_arr >= 20.0)) if np.any(h_arr >= 20.0) else len(log) - 1
        note = (f"Departure: {t0:.2f} h solar time  (SR = {SR:.2f} h)\n"
                f"Climb to 20 km: {t_arr[i20]:.2f} h elapsed\n"
                f"SOC at 20 km: {soc_arr[i20]:.1f}%")
        title = (f"Climb  |  {label}  ({t0:.2f} h)\n"
                 f"100% SOC, 21 June, φ={LAT:.0f}°N  |  ROC = {ROC_CLIMB:.1f} m/s")
        status = plot_climb(log, title, os.path.join(out, fname), t0, note_txt=note,
                            lat=LAT, doy=DOY)
        results[tag] = status
        print(f"  {tag}: {fname}  [{status}]")
    return results


def discharge_window(lat=LAT, doy=DOY):
    """Hours per day during which motor-on cruise demand exceeds PV power."""
    NH, SR, SS = sunrise_sunset(lat, doy)
    t_be = next((t for t in np.arange(SR, SR + 4., 0.01)
                 if P_pv(t * 3600., lat, doy) > P_cruise), None)
    t_ee = next((t for t in np.arange(SS - 4., SS + 0.5, 0.01)
                 if P_pv(t * 3600., lat, doy) < P_cruise), None)
    return (24. - t_ee + t_be) if t_be and t_ee else None


def print_summary():
    disc = discharge_window()
    print("\nKey numbers:")
    print(f"  p_el (20 km)      = {p_el20:.2f} W/m²")
    print(f"  P_cruise          = {P_cruise:.0f} W  (motors {P_cruise - P_avi:.0f} + avionics {P_avi:.0f})")
    print(f"  Discharge window  = {disc:.2f} h  (21 June, 40°N, motor-on)")
    print(f"  CD0(SL) / CD0(20) = {get_CD0(0):.5f} / {get_CD0(20000):.5f}")
    print(f"  L/D_max(SL / 20)  = {LD_max_at(0):.2f} / {LD_max_at(20000):.2f}")
    print(f"  Sink rate @ 20 km = {sink_rate(20000) * 100:.2f} cm/s")
    print(f"  No-load RPM       = {RPM_NL:.0f}  (Vbus = {Vbus} V)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="figures", help="output folder for figures")
    ap.add_argument("--maps", action="store_true", help="also produce global maps (slow)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Mission scenarios:")
    run_missions(args.out)
    print("Climb scenarios:")
    run_climbs(args.out)
    if args.maps:
        make_global_maps(args.out)
    print_summary()


if __name__ == "__main__":
    main()
