import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from astroquery.mast import Catalogs

os.makedirs("candidates", exist_ok=True)
history_file = "scanned_targets.txt"

# 1. Load historical scan log
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        scanned = [line.strip() for line in f.readlines()]
else:
    scanned = []

def get_dynamic_target():
    """Queries NASA's MAST TIC catalog for bright target stars suitable for BLS searching."""
    print("Querying MAST catalog for TESS targets...")
    
    # Query TIC for stars with TESS magnitude between 8.0 and 11.5
    # (Bright enough for clear signal-to-noise transit dips)
    catalog_data = Catalogs.query_criteria(
        catalog="TIC",
        Tmag=[8.0, 11.5],
        objType="STAR"
    )
    
    # Extract TIC IDs
    tic_ids = [f"TIC {row['ID']}" for row in catalog_data]
    
    # Filter out already scanned targets
    unscanned = [t for t in tic_ids if t not in scanned]
    
    if not unscanned:
        print("No new targets found in query batch. Selecting random fallback from batch.")
        return random.choice(tic_ids)
        
    selected_target = random.choice(unscanned)
    print(f"Dynamically selected target: {selected_target}")
    return selected_target

# Pick target dynamically from NASA MAST
target_star = get_dynamic_target()

try:
    search_results = lk.search_lightcurve(target_star, mission="TESS")
    
    if len(search_results) > 0:
        # Download lightcurve (first available sector)
        lc = search_results[0].download(quality_bitmask="hardest").remove_nans().flatten()
        
        # Run Box-fitting Least Squares search
        periodogram = lc.to_periodogram(method="bls", period=np.linspace(0.5, 5, 5000))
        best_period = float(periodogram.period_at_max_power.value)
        best_transit_time = float(periodogram.transit_time_at_max_power.value)
        
        folded = lc.fold(period=best_period, epoch_time=best_transit_time)
        
        # Save lightcurve image
        fig, ax = plt.subplots(figsize=(8, 4))
        folded.scatter(ax=ax, s=2)
        ax.set_title(f"Target: {target_star} - Period: {best_period:.4f} days")
        plt.savefig("candidates/candidate_transit.png")
        plt.close()
        
        # Dynamic environment links
        repo = os.getenv("GITHUB_REPOSITORY", "username/repo")
        branch = os.getenv("GITHUB_REF_NAME", "main")
        raw_image_url = f"https://raw.githubusercontent.com/{repo}/{branch}/candidates/candidate_transit.png"
        
        report_text = f"""## 🪐 Dynamic TESS Scan Report: {target_star}

**Target Star:** {target_star}
**Calculated Orbital Period:** {best_period:.4f} days
**Epoch Time (T0):** {best_transit_time:.4f}

### Phase-Folded Light Curve Plot:
![Candidate Transit Plot]({raw_image_url})

---
### Pipeline Diagnostics:
- Query sourced live from NASA MAST TESS Input Catalog (TIC).
- BLS transit search completed successfully.
- Target logged in repository history.
"""
        # Save target to local log
        with open(history_file, "a") as f:
            f.write(f"{target_star}\n")

    else:
        report_text = f"## ⚠️ Scan Notice\nMAST catalog query returned target {target_star}, but no light curves are available in TESS archives yet."

except Exception as e:
    report_text = f"## ⚠️ Pipeline Exception\nAn error occurred while fetching or analyzing {target_star}: {str(e)}"

with open("candidates/report.txt", "w") as f:
    f.write(report_text)

print("Dynamic execution complete.")
