# HAPS Energy Feasibility

Python models for assessing the energy feasibility of solar-electric high-altitude platforms (HAPS) operating in the stratosphere. The code was developed for my M.Sc. thesis on **DELICE**, a 30 m-span, 135 kg solar-electric platform designed for continuous operation at 20 km. It produced the results presented at IAC 2026.

The models link the atmosphere, solar geometry, aerodynamics, propulsion and battery state of charge in a time-stepping simulation. They are used to answer one question: *can the aircraft stay airborne through the night, and under which operating strategy?*

## Contents

| File | Purpose |
|---|---|
| `delice_mission_sim.py` | Time-stepping mission simulation of DELICE: 48 h cruise, night glide, climb from sea level. Optional global sizing maps. |
| `solar_hale_toolbox.py` | General sizing relations after Bierig et al. (ICAS 2024): solar energy density, maximum wing loading, required battery energy density, leg-based mission energy, closed-loop altitude/SOC simulation. |
| `noth_daily_balance.py` | First-pass daily energy balance after Noth (ETH Zürich, 2008), used at the start of the conceptual design. |

## Model chain (`delice_mission_sim.py`)

- **Atmosphere:** ISA troposphere and isothermal lower stratosphere.
- **Solar input:** Cooper declination and hour angle for a horizontal wing. PV power is P = κ η_SG η_MPPT η_wire S₀ cos ζ S_wing, with κ = 0.70 cell coverage and η_SG = 0.28. Atmospheric attenuation is neglected at 20 km. The toolbox's optional Beer–Lambert term, using an altitude-scaled air mass and k = 0.12, reduces daily energy by about 1.8 % at 20 km.
- **Aerodynamics:** Parabolic polar with altitude-dependent C_D0 to capture the Reynolds number effect: C_D0 is 0.0104 at sea level and 0.0165 at 20 km.
- **Propulsion:** Two NeuMotor 6560/8 motors (KV 55) with Mejzlik 64×36 propellers. Propeller C_T/C_P tables are solved for the required thrust, and motor current and voltage follow from the torque balance.
- **Battery:** 22.5 kWh, charge efficiency 0.95 and discharge efficiency 0.97, integrated at 30–60 s steps.

## Results (21 June, 40° N)

| Case | Strategy | Result |
|---|---|---|
| M1 / M2 | Motor-on cruise at 20 km, 100 % initial SOC | Battery depleted after 23.9 h / 9.9 h (infeasible) |
| G1 | Night motor-off glide to a 14 km floor, re-climb at sunrise, 100 % SOC | Feasible, minimum SOC 26.9 % |
| G2 / G3 | Same strategy with 50 % / 40 % initial SOC | Feasible, minimum SOC 11.8 % / 1.8 % |
| C1 / C2 | Climb from sea level at 1 m/s, departing at sunrise / sunrise + 3 h | Reaches 20 km with 100 % SOC |
| C3 | Climb departing 3 h before sunrise | Reaches 20 km with 17.3 % SOC |

Cruise demand at 20 km is 2.24 kW: 1.94 kW for propulsion and 0.30 kW for avionics. On the reference day, motor-on cruise requires 11.6 h of battery discharge, more than the battery can supply. Gliding at night with only the avionics powered is what makes the mission close.

![Night glide strategy](docs/energy_G1_glide_soc100.png)

<details>
<summary>More figures</summary>

![Motor-on cruise, sunset start](docs/energy_M2_cruise_sunset.png)
![Pre-dawn climb](docs/energy_C3_climb_predawn.png)
![Maximum wing loading map, DELICE parameters](docs/toolbox_wl_max_delice.png)

</details>

## Usage

```bash
pip install -r requirements.txt

python delice_mission_sim.py                 # mission and climb figures → figures/
python delice_mission_sim.py --maps          # also global irradiation / wing loading / battery maps
python solar_hale_toolbox.py --demo all      # toolbox demos
python noth_daily_balance.py                 # first-pass daily balance
```

Each script runs on its own. All parameters are module-level constants or function arguments and can be changed for other platforms.

## Assumptions and limitations

- Time is local solar time; the equation of time is neglected.
- No wind: station-keeping and ground-track constraints are not modelled.
- Clear-sky direct irradiance only, with no diffuse or albedo component.
- The battery is modelled as an energy reservoir with constant efficiencies; temperature, ageing and voltage sag are not included.
- Level flight at constant C_L; climb and glide are treated as quasi-steady.

## References

- Akgün, S., Yaman, M. (2026). *Energy-Constrained Mission Design and Power Analysis of Solar-Powered High-Altitude Platform Station.* 77th International Astronautical Congress (IAC 2026), Antalya.
- Akgün, S. et al. (2025). *Toward Perpetual Flight: Conceptual and Preliminary Development of the Delice 1A UAV.* Journal of Physics: Conference Series, AIAC 2025.
- Bierig, A. et al. (2024). *General Design Considerations for Solar-Electric High-Altitude Long-Endurance Aircraft.* ICAS 2024.
- Noth, A. (2008). *Design of Solar Powered Airplanes for Continuous Flight.* Ph.D. thesis, ETH Zürich.

## Author

Şeydanur Akgün, University of Turkish Aeronautical Association, Ankara.
