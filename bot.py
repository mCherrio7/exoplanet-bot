import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("candidates", exist_ok=True)

print("Downloading target star data...")
# WASP-18 b is a known hot Jupiter with a deep, obvious transit
search_results = lk.search_lightcurve("WASP-18", mission="TESS")

if len(search_results) > 0:
    # download with quality bitmask to filter out momentum dumps/noise
    lc = search_results[0].download(quality_bitmask="hardest").remove_nans().flatten()
    
    # Run BLS search
    periodogram = lc.to_periodogram(method="bls", period=np.linspace(0.5, 5, 5000))
    best_period = periodogram.period_at_max_power
    best_transit_time = periodogram.transit_time_at_max_power
    
    folded = lc.fold(period=best_period, epoch_time=best_transit_time)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    folded.scatter(ax=ax, s=2)
    ax.set_title(f"Target: WASP-18 b - Period: {best_period:.4f} days")
    
    plt.savefig("candidates/candidate_transit.png")
    print("Done! Saved candidate plot.")
else:
    print("No light curve data found.")
