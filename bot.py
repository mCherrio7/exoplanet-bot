import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("candidates", exist_ok=True)

target_star = "WASP-18"
print(f"Downloading data for {target_star}...")
search_results = lk.search_lightcurve(target_star, mission="TESS")

if len(search_results) > 0:
    lc = search_results[0].download(quality_bitmask="hardest").remove_nans().flatten()
    
    # Run BLS algorithm
    periodogram = lc.to_periodogram(method="bls", period=np.linspace(0.5, 5, 5000))
    best_period = periodogram.period_at_max_power
    best_transit_time = periodogram.transit_time_at_max_power
    
    folded = lc.fold(period=best_period, epoch_time=best_transit_time)
    
    # Plot lightcurve
    fig, ax = plt.subplots(figsize=(8, 4))
    folded.scatter(ax=ax, s=2)
    ax.set_title(f"Target: {target_star} - Period: {best_period:.4f} days")
    plt.savefig("candidates/candidate_transit.png")
    
    # Write issue body text
    report_text = f"""## 🪐 Exoplanet Transit Candidate Detected!

**Target Star:** {target_star}
**Calculated Orbital Period:** {best_period:.4f} days
**Epoch Time (T0):** {best_transit_time:.4f}

### Submission Summary:
- Light curve retrieved from public NASA TESS archives.
- Filtered out spacecraft momentum dumps and background noise.
- Candidate transit plot saved and attached to workflow artifacts.
"""
    with open("candidates/report.txt", "w") as f:
        f.write(report_text)
        
    print("Report written successfully.")
