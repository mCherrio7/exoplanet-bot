import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("candidates", exist_ok=True)

print("Downloading target star data from NASA TESS archive...")
search_results = lk.search_lightcurve("TIC 261136674", mission="TESS")

if len(search_results) > 0:
    lc = search_results[0].download().remove_nans().flatten()
    
    # Run Box-fitting Least Squares search algorithm
    periodogram = lc.to_periodogram(method="bls", period=np.linspace(0.5, 10, 5000))
    best_period = periodogram.period_at_max_power
    best_transit_time = periodogram.transit_time_at_max_power
    
    # Fold the light curve on the detected orbital period
    folded = lc.fold(period=best_period, epoch_time=best_transit_time)
    
    # Generate plot
    fig, ax = plt.subplots(figsize=(8, 4))
    folded.scatter(ax=ax, s=2)
    ax.set_title(f"Target Candidate - Orbital Period: {best_period:.4f} days")
    
    plt.savefig("candidates/candidate_transit.png")
    print("Done! Saved candidate plot to candidates folder.")
else:
    print("No lightcurve data found.")
