from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


class ClinicalReportGenerator:
    def generate_report(self, image, detections, explanations, output_path):
        from fpdf import FPDF

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "GallStone AI Detection Report", ln=True, align="C")
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detection Summary", ln=True)
        pdf.set_font("Arial", size=10)
        for i, det in enumerate(detections):
            confidence = det.get("confidence", 0)
            uncertainty = det.get("uncertainty", 0)
            pdf.cell(0, 8, f"Detection {i + 1}: Confidence={confidence:.1%}, Uncertainty=+/-{uncertainty:.3f}", ln=True)

        tmp_dir = output_path.parent / "_report_images"
        tmp_dir.mkdir(exist_ok=True)
        for name, img_array in explanations.items():
            temp_path = tmp_dir / f"{name}.png"
            plt.imsave(temp_path, img_array)
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Explainability: {name}", ln=True)
            pdf.image(str(temp_path), x=10, y=30, w=190)
        pdf.output(str(output_path))
        return str(output_path)
