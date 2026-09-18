import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from astroquery.mast import Catalogs

BATCH_SIZE = 20
SNR_THRESHOLD = 8.0

os.makedirs("candidates", exist_ok=True)
history_file = "scanned_targets.txt"
report_file = "candidates/report.txt"

if os.path.exists(report_file):
    os.remove(report_file)

if os.path.exists(history_file):
    with open(history_file, "r") as f:
        scanned = set(line.strip() for line in f.readlines())
else:
    scanned = set()

def fetch_target_batch(batch_size):
    print("Querying MAST catalog for upcoming target batch...")
    catalog_data = Catalogs.query_criteria(
        catalog="TIC",
        Tmag=[8.0, 11.5],
        objType="STAR"
    )
    
    tic_ids = [f"TIC {row['ID']}" for row in catalog_data]
    unscanned = [t for t in tic_ids if t not in scanned]
    random.shuffle(unscanned)
    return unscanned[:batch_size]

targets = fetch_target_batch(BATCH_SIZE)
flagged_candidates = []

print(f"\n--- Starting Optimized Batch Execution ({len(targets)} targets) ---")

for idx, target_star in enumerate(targets, 1):
    print(f"[{idx}/{len(targets)}] Processing {target_star}...", end=" ", flush=True)
    
    # Log target to prevent retries
    with open(history_file, "a") as f:
        f.write(f"{target_star}\n")
    scanned.add(target_star)

    try:
        # 1. Search official NASA SPOC pipeline products only (MUCH faster)
        search_results = lk.search_lightcurve(target_star, mission="TESS", author="SPOC")
        
        if len(search_results) == 0:
            print("❌ No official SPOC light curves.")
            continue

        # 2. Download ONLY the first available sector to keep downloads lightweight (<5MB)
        lc = search_results[0].download(quality_bitmask="hardest").remove_nans().flatten()
        
        # 3. High-precision BLS periodogram
        periodogram = lc.to_periodogram(method="bls", period=np.linspace(0.5, 5, 3000))
        best_period = float(periodogram.period_at_max_power.value)
        best_transit_time = float(periodogram.transit_time_at_max_power.value)
        best_duration = float(periodogram.duration_at_max_power.value)
        best_depth = float(periodogram.depth_at_max_power.value)
        
        # Calculate SNR
        std_dev = np.std(lc.flux.value)
        n_points = len(lc.flux.value)
        duty_cycle = best_duration / best_period
        in_transit_points = n_points * duty_cycle
        
        snr = (best_depth / std_dev) * np.sqrt(in_transit_points) if (std_dev > 0 and in_transit_points > 0) else 0.0
        print(f"Done. (SNR: {snr:.2f} | P: {best_period:.4f}d)")
        
        # Quality Filter
        if snr >= SNR_THRESHOLD:
            print(f"  🚨 HIGH-CONFIDENCE CANDIDATE FLAGGED! ({target_star})")
            
            clean_name = target_star.replace(" ", "_")
            image_filename = f"candidates/{clean_name}_transit.png"
            
            folded = lc.fold(period=best_period, epoch_time=best_transit_time)
            fig, ax = plt.subplots(figsize=(8, 4))
            folded.scatter(ax=ax, s=2)
            ax.set_title(f"{target_star} - P: {best_period:.4f}d - SNR: {snr:.2f}")
            plt.savefig(image_filename)
            plt.close()
            
            repo = os.getenv("GITHUB_REPOSITORY", "username/repo")
            branch = os.getenv("GITHUB_REF_NAME", "main")
            raw_image_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{image_filename}"
            
            flagged_candidates.append({
                "target": target_star,
                "period": best_period,
                "epoch": best_transit_time,
                "depth": best_depth,
                "duration": best_duration,
                "snr": snr,
                "image_url": raw_image_url
            })

    except Exception as e:
        print(f"⚠️ Error: {str(e)}")

print("\n--- Batch Processing Complete ---")

# Save Issue report if candidates found
if flagged_candidates:
    report_text = f"## 🪐 Batch Execution Report: {len(flagged_candidates)} Candidate(s) Flagged\n\n"
    report_text += f"**Batch Size:** {len(targets)} stars scanned\n"
    report_text += f"**SNR Filter Threshold:** ≥ {SNR_THRESHOLD}\n\n---\n\n"
    
    for cand in flagged_candidates:
        report_text += f"### 🚨 Candidate: {cand['target']}\n"
        report_text += f"- **Signal-to-Noise Ratio (SNR):** `{cand['snr']:.2f}`\n"
        report_text += f"- **Orbital Period ($P$):** `{cand['period']:.4f}` days\n"
        report_text += f"- **Epoch ($T_0$):** `{cand['epoch']:.4f}`\n"
        report_text += f"- **Transit Depth:** `{cand['depth']:.5f}`\n\n"
        report_text += f"![Transit Plot]({cand['image_url']})\n\n---\n\n"
        
    with open(report_file, "w") as f:
        f.write(report_text)
