# INVALID — do not use these timings

Measured while the laptop ran on battery power with the CPU throttled (Windows reported BatteryStatus=1,
~55% processor performance, ~1.97 GHz at 20:47 IST). Repetition spread was up to 2x (e.g. FLAP-20% prefill
of 512 tokens: 16.4 s to 34.0 s) and the variants were non-monotonic in size. Superseded by re-measurement
on AC power with power state recorded per variant (scripts/p1_profile.py, require_ac_power: true).
