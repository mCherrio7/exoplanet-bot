import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from astroquery.mast import Catalogs

SNR_THRESHOLD = 8.0
MAX_TRIALS = 100  # Safety limit to avoid infinite loops per workflow run

os.makedirs("candidates", exist_ok=True)
history_file = "scanned_targets.txt"
report_file = "candidates/report.txt"

# Clean up any leftover report from previous runs
if os.path.exists(report_file):
    os.remove(report_file)

# 1. Load historical scan log
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        scanned = set(line.strip() for line in f.readlines())
else:
    scanned = set()

def fetch_unscanned_targets():
    """Queries NASA's MAST catalog for available TESS stars not yet in history."""
    print("Querying NASA MAST catalog for candidate targets...")
    try:
        catalog_data = Catalogs.query_criteria(
            catalog="TIC",
            Tmag=[8.0, 11.5],
            objType="STAR"
        )
        tic_ids = [f"TIC {row['ID']}" for row in catalog_data]
        unscanned = [t for t in tic_ids if t not in scanned]
        random.shuffle(unscanned)
        return unscanned
    except Exception as e:
        print(f"Error querying MAST: {e}")
        return []

target_pool = fetch_unscanned_targets()

if not target_pool:
    print("No unscanned targets available in query pool.")
    exit(0)

repo = os.getenv("GITHUB_REPOSITORY", "username/repo")
branch = os.getenv("GITHUB_REF_NAME", "main")

trials = 0
found_candidate = False

print(f"\n--- Searching until SNR >= {SNR_THRESHOLD} (Max Trials: {MAX_TRIALS}) ---")

while target_pool and trials < MAX_TRIALS and not found_candidate:
    trials += 1
    target_star = target_pool.pop()
    
    print(f"[Trial {trials}/{MAX_TRIALS}] Scanning {target_star}...", end=" ", flush=True)
    
    # Immediately log to history so target is never re-scanned
    with open(history_file, "a") as f:
        f.write(f"{target_star}\n")
    scanned.add(target_star)

    try:
        # Fetch official SPOC light curves
        search_results = lk.search_lightcurve(target_star, mission="TESS", author="SPOC")
        
        if len(search_results) == 0:
            print("❌ No SPOC light curves available.")
            continue

        # Download first sector
        lc = search_results[0].download(quality_bitmask="hardest").remove_nans().flatten()
        
        # BLS periodogram search
        periodogram = lc.to_periodogram(method="bls", period=np.linspace(0.5, 5, 2000))
        best_period = float(periodogram.period_at_max_power.value)
        best_transit_time = float(periodogram.transit_time_at_max_power.value)
        best_duration = float(periodogram.duration_at_max_power.value)
        best_depth = float(periodogram.depth_at_max_power.value)
        
        # Calculate Signal-to-Noise Ratio (SNR)
        std_dev = np.std(lc.flux.value)
        n_points = len(lc.flux.value)
        duty_cycle = best_duration / best_period
        in_transit_points = n_points * duty_cycle
        
        snr = (best_depth / std_dev) * np.sqrt(in_transit_points) if (std_dev > 0 and in_transit_points > 0) else 0.0
        print(f"Done. (SNR: {snr:.2f} | P: {best_period:.4f}d)")
        
        # Check if quality gate is met
        if snr >= SNR_THRESHOLD:
            print(f"\n🎉 HIGH-CONFIDENCE CANDIDATE FOUND on trial {trials}! ({target_star})")
            found_candidate = True
            
            clean_name = target_star.replace(" ", "_")
            image_filename = f"candidates/{clean_name}_transit.png"
            
            folded = lc.fold(period=best_period, epoch_time=best_transit_time)
            fig, ax = plt.subplots(figsize=(8, 4))
            folded.scatter(ax=ax, s=2)
            ax.set_title(f"{target_star} - P: {best_period:.4f}d - SNR: {snr:.2f}")
            plt.savefig(image_filename)
            plt.close()
            
            raw_image_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{image_filename}"
            
            report_text = f"""## 🚨 High-Priority Exoplanet Candidate Flagged

**Target Star:** `{target_star}`
**Signal-to-Noise Ratio (SNR):** `{snr:.2f}` (Threshold: ≥ {SNR_THRESHOLD})
**Calculated Orbital Period ($P$):** `{best_period:.4f}` days
**Epoch ($T_0$):** `{best_transit_time:.4f}`
**Transit Depth:** `{best_depth:.5f}`
**Transit Duration:** `{best_duration:.4f}` days
**Scanned Sequence Trial:** #{trials}

### Phase-Folded Light Curve Plot:
![Candidate Transit Plot]({raw_image_url})

---
### Diagnostics:
- Identified after evaluating {trials} target stars in sequence.
- Target logged in `scanned_targets.txt`.
- Ready for manual follow-up verification on Planet Hunters TESS / ExoFOP.
"""
            with open(report_file, "w") as f:
                f.write(report_text)
                
    except Exception as e:
        print(f"⚠️ Error: {str(e)}")

print("\n--- Pipeline Execution Summary ---")
print(f"Total targets scanned in run: {trials}")
if found_candidate:
    print("Status: Candidate found and report.txt written.")
else:
    print(f"Status: No target met SNR >= {SNR_THRESHOLD} within {MAX_TRIALS} trials.")
