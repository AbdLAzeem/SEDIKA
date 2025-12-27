import os
import shutil

# Configuration
ARTIFACT_DIR = "C:/Users/mizal/.gemini/antigravity/brain/13ac5187-4bb2-406a-9545-e0fbed5bb711"
OUTPUT_DIR = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/reports_txt"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

reports = [
    "data_comparison_report.md",
    "processed_data_report.md",
    "advanced_analysis_report.md",
    "supervised_ml_report.md",
    "dl_performance_report.md",
    "anomaly_detection_report.md",
    "expert_discussion_report.md"
]

print(f"Exporting reports to {OUTPUT_DIR}...")

for report in reports:
    src_path = os.path.join(ARTIFACT_DIR, report)
    # Change extension to .txt
    dst_filename = report.replace(".md", ".txt")
    dst_path = os.path.join(OUTPUT_DIR, dst_filename)
    
    if os.path.exists(src_path):
        try:
            # Read and write to ensure clean copy, maybe strip strictly if needed but MD is text compatible
            with open(src_path, 'r', encoding='utf-8') as f_src:
                content = f_src.read()
            
            with open(dst_path, 'w', encoding='utf-8') as f_dst:
                f_dst.write(content)
                
            print(f"[OK] Exported {report} -> {dst_filename}")
        except Exception as e:
            print(f"[FAIL] Could not export {report}: {e}")
    else:
        print(f"[WARN] Source file not found: {src_path}")

print("\nExport completed.")
