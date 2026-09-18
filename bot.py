import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os
import random

os.makedirs("candidates", exist_ok=True)

# List of TESS Input Catalog (TIC) IDs to analyze
TARGET_POOL = [
    "TIC 100100827",  # WASP-18 (Known test case)
    "TIC 261136674",  # WASP-126
    "TIC 144548228",  # WASP-19
    "TIC 278859073",  # TOI-178
    "TIC 307210830",  # WASP-100
    "TIC 25155310"    # KELT-9
]

# Track previously scanned stars
history_file = "scanned_targets.txt"
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        scanned = [line.strip() for line in f.readlines()]
else:
    scanned = []

# Select an unscanned star from the pool
unscanned_targets = [t for t in TARGET_POOL if t not in scanned]

if not unscanned_targets:
    print("All targets in pool have been scanned! Resetting list.")
    unscanned_targets = TARGET_POOL
    scanned = []

target_star = random.choice(unscanned_targets)
print(f"Running pipeline scan for: {target_star}...")

try:
    search_results = lk.search_lightcurve(target_star, mission="TESS")
    
    if len(search_results) > 0:
        lc = search_results[0].download(quality_bitmask="hardest").remove_nans().flatten()
        
        # Run BLS algorithm
        periodogram = lc.to_periodogram(method="bls", period=np.linspace(0.5, 5, 5000))
        best_period = float(periodogram.period_at_max_power.value)
        best_transit_time = float(periodogram.transit_time_at_max_power.value)
        
        folded = lc.fold(period=best_period, epoch_time=best_transit_time)
        
        # Plot phase-folded transit
        fig, ax = plt.subplots(figsize=(8, 4))
        folded.scatter(ax=ax, s=2)
        ax.set_title(f"Target: {target_star} - Period: {best_period:.4f} days")
        plt.savefig("candidates/candidate_transit.png")
        plt.close()
        
        # Dynamic environment links
        repo = os.getenv("GITHUB_REPOSITORY", "username/repo")
        branch = os.getenv("GITHUB_REF_NAME", "main")
        raw_image_url = f"https://raw.githubusercontent.com/{repo}/{branch}/candidates/candidate_transit.png"
        
        report_text = f"""## 🪐 Exoplanet Pipeline Scan: {target_star}

**Target Star:** {target_star}
**Calculated Orbital Period:** {best_period:.4f} days
**Epoch Time (T0):** {best_transit_time:.4f}

### Transit Light Curve Plot:
![Candidate Transit Plot]({raw_image_url})

---
### Execution Status:
- TESS public archival data downloaded successfully.
- BLS transit search completed.
- Target added to search log to ensure non-repeating workflow.
"""
        # Save to scanned history
        with open(history_file, "a") as f:
            f.write(f"{target_star}\n")

    else:
        report_text = f"## ⚠️ Scan Complete\nNo public light curve data returned for {target_star}."

except Exception as e:
    report_text = f"## ⚠️ Scan Error\nAn error occurred while analyzing {target_star}: {str(e)}"

with open("candidates/report.txt", "w") as f:
    f.write(report_text)

print("Pipeline execution complete.")
