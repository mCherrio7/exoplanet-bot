import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("candidates", exist_ok=True)

target_star = "TIC 100100827"
print(f"Downloading data for {target_star}...")

try:
    search_results = lk.search_lightcurve(target_star, mission="TESS")
    
    if len(search_results) > 0:
        lc = search_results[0].download(quality_bitmask="hardest").remove_nans().flatten()
        
        # Run BLS transit search
        periodogram = lc.to_periodogram(method="bls", period=np.linspace(0.5, 5, 5000))
        best_period = float(periodogram.period_at_max_power.value)
        best_transit_time = float(periodogram.transit_time_at_max_power.value)
        
        folded = lc.fold(period=best_period, epoch_time=best_transit_time)
        
        # Save graph
        fig, ax = plt.subplots(figsize=(8, 4))
        folded.scatter(ax=ax, s=2)
        ax.set_title(f"Target: {target_star} - Period: {best_period:.4f} days")
        plt.savefig("candidates/candidate_transit.png")
        plt.close()
        
        # REPLACE 'YOUR_GITHUB_USERNAME' and 'YOUR_REPO_NAME' BELOW:
        raw_image_url = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/main/candidates/candidate_transit.png"
        
        report_text = f"""## 🪐 Exoplanet Transit Candidate Flagged!

**Target Star:** {target_star} (WASP-18 b)
**Calculated Orbital Period:** {best_period:.4f} days
**Epoch Time (T0):** {best_transit_time:.4f}

### Transit Light Curve Plot:
![Candidate Transit Plot]({raw_image_url})

---
### Submission Summary:
- Data retrieved automatically from TESS archives.
- High-precision quality filtering applied to clean instrumental noise.
- Candidate plot saved directly to repository.
"""
    else:
        report_text = f"## ⚠️ Scan Complete\nNo light curve data returned for {target_star}."

except Exception as e:
    report_text = f"## ⚠️ Scan Error\nAn error occurred while fetching target data: {str(e)}"

with open("candidates/report.txt", "w") as f:
    f.write(report_text)

print("Script execution complete. Saved report with direct raw image link.")
