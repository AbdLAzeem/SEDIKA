import os
import shutil
import markdown
from xhtml2pdf import pisa

# Configuration
ARTIFACT_DIR = "C:/Users/mizal/.gemini/antigravity/brain/a108dff8-7a57-4d4f-9dbd-5f002b136790"
OUTPUT_DIR = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/final_reports"

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

def convert_md_to_pdf(md_content, output_path):
    # Convert MD to HTML
    html = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    # Add some basic styling
    full_html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: sans-serif; font-size: 10pt; }}
        h1 {{ color: #2E4053; border-bottom: 2px solid #2E4053; padding-bottom: 5px; }}
        h2 {{ color: #2874A6; margin-top: 20px; }}
        h3 {{ color: #1F618D; }}
        pre {{ background-color: #F4F6F6; padding: 10px; border: 1px solid #D5D8DC; font-family: monospace; white-space: pre-wrap; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #D5D8DC; padding: 8px; text-align: left; }}
        th {{ background-color: #EAEDED; }}
    </style>
    </head>
    <body>
    {html}
    </body>
    </html>
    """
    
    # Write PDF
    with open(output_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)
        
    return not pisa_status.err

print(f"Exporting final reports to {OUTPUT_DIR}...")

for report in reports:
    src_path = os.path.join(ARTIFACT_DIR, report)
    
    if os.path.exists(src_path):
        try:
            # 1. Read Content
            with open(src_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 2. Save .md
            dst_md_path = os.path.join(OUTPUT_DIR, report)
            with open(dst_md_path, 'w', encoding='utf-8') as f_md:
                f_md.write(content)
            print(f"[OK] Saved MD: {report}")
            
            # 3. Save .pdf
            dst_pdf_path = os.path.join(OUTPUT_DIR, report.replace(".md", ".pdf"))
            success = convert_md_to_pdf(content, dst_pdf_path)
            
            if success:
                print(f"[OK] Saved PDF: {os.path.basename(dst_pdf_path)}")
            else:
                print(f"[FAIL] PDF conversion error for {report}")
                
        except Exception as e:
            print(f"[FAIL] Error processing {report}: {e}")
    else:
        print(f"[WARN] Report not found: {report}")

print("\nFinal Export Complete.")
