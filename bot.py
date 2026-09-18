import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from astroquery.mast import Catalogs

os.makedirs("candidates", exist_ok=True)
history_file = "scanned_targets.txt"
report_file = "candidates/report.txt"

# Clean up any leftover report file from previous runs
if os.path.exists(report_file):
    os.remove(report_file)

# 1. Load historical scan log
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        scanned = [line.strip() for line in f.readlines()]
else:
    scanned = []

def get_dynamic_target():
    """Queries NASA's MAST TIC catalog for bright target stars suitable for BLS searching."""
    print("Querying MAST catalog for TESS targets...")
    catalog_data = Catalogs.query_criteria(
        catalog="TIC",
        Tmag=[8.0, 11.5],
        objType="STAR"
    )
    
    tic_ids = [f"TIC {row['ID']}" for row in catalog_data]
    unscanned = [t for t in tic_ids if t not in scanned]
    
    if not unscanned:
        print("No new targets found in query batch. Selecting random fallback from batch.")
        return random.choice(tic_ids)
        
    selected_target = random.choice(unscanned)
    print(f"Dynamically selected target: {selected_target}")
    return selected_target

target_star = get_dynamic_target()

try:
    search_results = lk.search_lightcurve(target_star, mission="TESS")
    
    if len(search_results) > 0:
        # Download and clean light curve
        lc = search_results[0].download(quality_bitmask="hardest").remove_nans().flatten()
        
        # Run Box-fitting Least Squares search
        periodogram = lc.to_periodogram(method="bls", period=np.linspace(0.5, 5, 5000))
        best_period = float(periodogram.period_at_max_power.value)
        best_transit_time = float(periodogram.transit_time_at_max_power.value)
        best_duration = float(periodogram.duration_at_max_power.value)
        best_depth = float(periodogram.depth_at_max_power.value)
        
        # Calculate Signal-to-Noise Ratio (SNR)
        # SNR = Transit Depth / Standard Error of the Light Curve
        # Scaled by square root of total in-transit points
        std_dev = np.std(lc.flux.value)
        n_points = len(lc.flux.value)
        duty_cycle = best_duration / best_period
        in_transit_points = n_points * duty_cycle
        
        if std_dev > 0 and in_transit_points > 0:
            snr = (best_depth / std_dev) * np.sqrt(in_transit_points)
        else:
            snr = 0.0
            
        print(f"Target: {target_star} | Calculated SNR: {snr:.2f} | Threshold: 8.0")
        
        # Log target as scanned regardless of SNR outcome to avoid re-scanning
        with open(history_file, "a") as f:
            f.write(f"{target_star}\n")
            
        # SNR Quality Gate: Only create candidate report if SNR > 8.0
        if snr >= 8.0:
            print(f"High-priority candidate found! (SNR {snr:.2f} >= 8.0)")
            
            folded = lc.fold(period=best_period, epoch_time=best_transit_time)
            
            # Save light curve image
            fig, ax = plt.subplots(figsize=(8, 4))
            folded.scatter(ax=ax, s=2)
            ax.set_title(f"Target: {target_star} - Period: {best_period:.4f} days - SNR: {snr:.2f}")
            plt.savefig("candidates/candidate_transit.png")
            plt.close()
            
            # Dynamic environment links
            repo = os.getenv("GITHUB_REPOSITORY", "username/repo")
            branch = os.getenv("GITHUB_REF_NAME", "main")
            raw_image_url = f"https://raw.githubusercontent.com/{repo}/{branch}/candidates/candidate_transit.png"
            
            report_text = f"""## 🚨 High-Priority Candidate Flagged: {target_star}

**Target Star:** {target_star}
**Signal-to-Noise Ratio (SNR):** {snr:.2f} (Passed Threshold >= 8.0)
**Calculated Orbital Period:** {best_period:.4f} days
**Epoch Time (T0):** {best_transit_time:.4f}
**Transit Depth:** {best_depth:.5f}
**Transit Duration:** {best_duration:.4f} days

### Phase-Folded Light Curve Plot:
![Candidate Transit Plot]({raw_image_url})

---
### Candidate Status:
- TESS public archival data analyzed via BLS.
- Strong signal detected above background noise threshold.
- Prepared for follow-up review on Planet Hunters TESS or ExoFOP.
"""
            with open(report_file, "w") as f:
                f.write(report_text)
        else:
            print(f"Target {target_star} passed scan, but SNR ({snr:.2f}) was below threshold (8.0). No Issue will be created.")

    else:
        print(f"No light curves returned for {target_star}.")

except Exception as e:
    print(f"Pipeline error while analyzing {target_star}: {str(e)}")
