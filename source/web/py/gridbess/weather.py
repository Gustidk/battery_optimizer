"""
Weather -> production fractions (same simple models as profiles/weather.py),
plus a synthetic example year for when real data cannot be fetched.
"""
import math
import numpy as np


def ghi_to_pv_fraction(ghi_w_m2, stc_irradiance=1000.0):
    """Linear PV model: output proportional to global irradiance relative to STC."""
    return np.clip(np.asarray(ghi_w_m2, float) / stc_irradiance, 0, 1)


def wind_speed_to_wp_fraction(v_ms, v_in=3.0, v_rated=12.0, v_out=25.0):
    """Generic turbine power curve, cubic between cut-in and rated."""
    v = np.asarray(v_ms, float)
    frac = np.zeros_like(v)
    ramp = (v >= v_in) & (v < v_rated)
    frac[ramp] = ((v[ramp] - v_in) / (v_rated - v_in)) ** 3
    frac[(v >= v_rated) & (v <= v_out)] = 1.0
    return frac


def synthetic_weather(lat, lon, start_doy, hours, seed=7, utc_offset_h=1.0):
    """Plausible but NOT measured hourly irradiance and 100 m wind speed:
    clear-sky irradiance (Haurwitz) times a day-to-day cloud factor, and a
    Weibull-like wind speed with seasonal and day-to-day variation."""
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    doy = (start_doy - 1 + t // 24) % 365 + 1
    hour = t % 24 + 0.5
    decl = math.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    solar_time = hour - utc_offset_h + lon / 15.0
    ha = np.radians(15 * (solar_time - 12))
    la = math.radians(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    clear = np.where(cosz > 0.01, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 0.01)), 0.0)
    days = hours // 24 + 2
    cloud_day = np.clip(0.55 + 0.35 * np.tanh(rng.normal(0, 1, days) * 1.2), 0.15, 1.0)
    cloud_h = np.repeat(cloud_day, 24)[:hours] * np.clip(1 + rng.normal(0, 0.12, hours), 0.5, 1.15)
    ghi = np.clip(clear * cloud_h, 0, None)
    season = 1 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365)        # windier in winter
    x = np.zeros(hours); e = rng.normal(0, 1, hours)
    for i in range(1, hours):
        x[i] = 0.97 * x[i - 1] + math.sqrt(1 - 0.97 ** 2) * e[i]
    u = 0.5 * (1 + np.tanh(x / 1.2))                                  # 0..1, slowly varying
    weib = 9.0 * (-np.log(np.clip(1 - u, 1e-6, 1))) ** (1 / 2.0)      # Weibull k=2, scale 9 m/s
    wind = np.clip(weib * season * 0.85, 0, 35)
    return ghi, wind
