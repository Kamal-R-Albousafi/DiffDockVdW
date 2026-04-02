import os
import pandas as pd

# def extract_rank1_confidences(molecules_dir, output_csv="rank1_confidences.csv", output_dir = "confidence_csvs"):
#     records = []

#     for fname in os.listdir(molecules_dir):
#         if "rank1_confidence-" in fname and fname.endswith(".sdf"):
#             parts = fname.split("_")
            
#             cid = parts[2]
            
#             conf_str = fname.split("rank1_confidence")[1].replace(".sdf", "")
#             confidence = float(conf_str)

#             records.append({
#                 "CID": cid,
#                 "rank1_confidence": confidence
#             })
    
#     os.makedirs(output_dir, exist_ok=True)
#     output_csv = os.path.join(output_dir, "rank1_confidences.csv")
#     df = pd.DataFrame(records)
#     df.to_csv(output_csv, index=False)

#     print(f"Saved {len(df)} entries to {output_csv}")

def extract_rank1_confidences(molecules_dir,output_csv="rank1_confidences.csv", output_dir="confidence_csvs"):
    records = []

    for fname in os.listdir(molecules_dir):
        if "rank1_confidence-" in fname and fname.endswith(".sdf"):
            try:
                cid = fname.split("CID_")[1].split("_")[0]

                conf_str = fname.split("rank1_confidence")[1].replace(".sdf", "")
                confidence = float(conf_str)

                records.append({
                    "CID": cid,
                    "rank1_confidence": confidence
                })
            except (IndexError, ValueError):
                continue

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, output_csv)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} entries to {output_csv}")



extract_rank1_confidences(
    molecules_dir="VS_DD_known_activators2_2025_12_21/molecules",
    output_csv="rank1_confidencesKnownActivators2.csv"
)
