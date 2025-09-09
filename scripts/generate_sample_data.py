#!/usr/bin/env python3
"""Generate sample data for testing the bloodwork AI platform."""

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def generate_sample_tabular_data(
    output_dir: Path,
    num_patients: int = 1000,
    num_measurements_per_patient: int = 3
) -> None:
    """Generate sample tabular lab data."""
    print("Generating sample tabular data...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate patient data
    patients = []
    measurements = []
    
    for patient_id in range(num_patients):
        # Patient demographics
        age = random.randint(18, 80)
        sex = random.choice(["M", "F"])
        
        # Generate multiple measurements per patient
        for measurement_id in range(num_measurements_per_patient):
            # CBC data
            cbc_data = {
                "patient_id": f"P{patient_id:04d}",
                "date": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "age": age,
                "sex": sex,
                "wbc": random.uniform(3.0, 15.0),
                "rbc": random.uniform(3.5, 6.0),
                "hb": random.uniform(10.0, 18.0),
                "hct": random.uniform(30.0, 55.0),
                "plt": random.uniform(150, 450),
                "mcv": random.uniform(80, 100),
                "mch": random.uniform(25, 35),
                "mchc": random.uniform(30, 36),
                "rdw": random.uniform(11, 16),
                "neut_pct": random.uniform(40, 80),
                "lymph_pct": random.uniform(20, 50),
                "mono_pct": random.uniform(2, 12),
                "eos_pct": random.uniform(0, 8),
                "baso_pct": random.uniform(0, 2)
            }
            
            # CMP data
            cmp_data = {
                "na": random.uniform(135, 145),
                "k": random.uniform(3.5, 5.5),
                "cl": random.uniform(95, 110),
                "co2": random.uniform(20, 30),
                "bun": random.uniform(5, 25),
                "cr": random.uniform(0.5, 1.5),
                "glucose": random.uniform(70, 140),
                "ast": random.uniform(10, 40),
                "alt": random.uniform(10, 40),
                "alp": random.uniform(30, 120),
                "bilirubin": random.uniform(0.2, 1.2),
                "protein": random.uniform(6.0, 8.5),
                "albumin": random.uniform(3.5, 5.5),
                "ca": random.uniform(8.5, 10.5)
            }
            
            # Lipid data
            lipid_data = {
                "total_chol": random.uniform(150, 250),
                "hdl": random.uniform(40, 80),
                "triglycerides": random.uniform(50, 200)
            }
            lipid_data["ldl"] = lipid_data["total_chol"] - lipid_data["hdl"] - (lipid_data["triglycerides"] / 5)
            
            # Thyroid data
            thyroid_data = {
                "tsh": random.uniform(0.4, 4.0),
                "ft4": random.uniform(0.8, 1.8),
                "ft3": random.uniform(2.0, 4.5)
            }
            
            # Coagulation data
            coag_data = {
                "pt": random.uniform(10, 15),
                "inr": random.uniform(0.8, 1.2),
                "aptt": random.uniform(25, 35),
                "d_dimer": random.uniform(0, 500),
                "fibrinogen": random.uniform(200, 400),
                "platelets": cbc_data["plt"]  # Use same as CBC
            }
            
            # Vitamins/Iron data
            vitamins_data = {
                "ferritin": random.uniform(15, 300),
                "iron": random.uniform(50, 200),
                "tibc": random.uniform(250, 450),
                "transferrin": random.uniform(200, 400),
                "b12": random.uniform(200, 1000),
                "folate": random.uniform(3, 20),
                "crp": random.uniform(0, 10)
            }
            vitamins_data["transferrin_sat"] = (vitamins_data["iron"] / vitamins_data["tibc"]) * 100
            
            # Combine all data
            measurement = {**cbc_data, **cmp_data, **lipid_data, **thyroid_data, **coag_data, **vitamins_data}
            
            # Add clinical outcomes (simplified)
            measurement["anemia_type"] = random.choice(["normal", "iron_deficiency", "thalassemia_trait", "b12_deficiency"])
            measurement["thyroid_dysfunction"] = random.choice(["normal", "hypothyroidism", "hyperthyroidism"])
            measurement["metabolic_syndrome"] = random.choice([True, False])
            measurement["cvd_risk"] = random.uniform(0.05, 0.3)
            measurement["dic_stage"] = random.choice(["normal", "early", "moderate", "severe"])
            measurement["b12_deficiency"] = random.choice([True, False])
            measurement["ferritin_low"] = random.choice([True, False])
            
            measurements.append(measurement)
    
    # Save data
    df = pd.DataFrame(measurements)
    
    # Save by panel type
    panels = {
        "cbc": ["patient_id", "date", "age", "sex", "wbc", "rbc", "hb", "hct", "plt", "mcv", "mch", "mchc", "rdw", "neut_pct", "lymph_pct", "mono_pct", "eos_pct", "baso_pct", "anemia_type"],
        "cmp": ["patient_id", "date", "age", "sex", "na", "k", "cl", "co2", "bun", "cr", "glucose", "ast", "alt", "alp", "bilirubin", "protein", "albumin", "ca", "metabolic_syndrome"],
        "lipid": ["patient_id", "date", "age", "sex", "total_chol", "ldl", "hdl", "triglycerides", "cvd_risk"],
        "thyroid": ["patient_id", "date", "age", "sex", "tsh", "ft4", "ft3", "thyroid_dysfunction"],
        "coag": ["patient_id", "date", "age", "sex", "pt", "inr", "aptt", "d_dimer", "fibrinogen", "platelets", "dic_stage"],
        "vitamins_iron": ["patient_id", "date", "age", "sex", "ferritin", "iron", "tibc", "transferrin", "transferrin_sat", "b12", "folate", "crp", "b12_deficiency", "ferritin_low"]
    }
    
    for panel_name, columns in panels.items():
        panel_df = df[columns].copy()
        panel_df.to_csv(output_dir / f"{panel_name}_data.csv", index=False)
        print(f"Saved {panel_name} data: {len(panel_df)} records")
    
    # Save combined data
    df.to_csv(output_dir / "combined_data.csv", index=False)
    print(f"Saved combined data: {len(df)} records")


def generate_sample_image_data(
    output_dir: Path,
    num_images: int = 100,
    image_size: tuple = (640, 640)
) -> None:
    """Generate sample microscopy images with annotations."""
    print("Generating sample image data...")
    
    # Create output directories
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Cell types and colors
    cell_types = {
        "rbc": (255, 0, 0),      # Red
        "wbc": (0, 255, 0),      # Green
        "platelet": (0, 0, 255), # Blue
        "neutrophil": (255, 255, 0),  # Yellow
        "lymphocyte": (255, 0, 255),  # Magenta
        "monocyte": (0, 255, 255),    # Cyan
        "eosinophil": (128, 0, 128),  # Purple
        "basophil": (255, 165, 0)     # Orange
    }
    
    # COCO format annotations
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for i, (cell_type, color) in enumerate(cell_types.items()):
        coco_annotations["categories"].append({
            "id": i,
            "name": cell_type,
            "supercategory": "blood_cell"
        })
    
    annotation_id = 1
    
    for image_id in range(num_images):
        # Create image
        image = Image.new("RGB", image_size, (240, 240, 240))  # Light gray background
        draw = ImageDraw.Draw(image)
        
        # Generate random cells
        num_cells = random.randint(5, 20)
        image_annotations = []
        
        for _ in range(num_cells):
            # Random cell type
            cell_type = random.choice(list(cell_types.keys()))
            color = cell_types[cell_type]
            
            # Random position and size
            cell_size = random.randint(20, 60)
            x = random.randint(0, image_size[0] - cell_size)
            y = random.randint(0, image_size[1] - cell_size)
            
            # Draw cell (circle)
            draw.ellipse([x, y, x + cell_size, y + cell_size], fill=color, outline="black", width=2)
            
            # Add annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id + 1,
                "category_id": list(cell_types.keys()).index(cell_type),
                "bbox": [x, y, cell_size, cell_size],
                "area": cell_size * cell_size,
                "iscrowd": 0
            }
            image_annotations.append(annotation)
            annotation_id += 1
        
        # Save image
        image_filename = f"blood_smear_{image_id:04d}.jpg"
        image_path = images_dir / image_filename
        image.save(image_path, "JPEG", quality=95)
        
        # Add image info
        coco_annotations["images"].append({
            "id": image_id + 1,
            "file_name": image_filename,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": "2023-01-01T00:00:00Z"
        })
        
        # Add annotations
        coco_annotations["annotations"].extend(image_annotations)
        
        # Save YOLO format annotation
        yolo_annotation = []
        for ann in image_annotations:
            bbox = ann["bbox"]
            x_center = (bbox[0] + bbox[2] / 2) / image_size[0]
            y_center = (bbox[1] + bbox[3] / 2) / image_size[1]
            width = bbox[2] / image_size[0]
            height = bbox[3] / image_size[1]
            
            yolo_line = f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_annotation.append(yolo_line)
        
        yolo_file = annotations_dir / f"blood_smear_{image_id:04d}.txt"
        with open(yolo_file, "w") as f:
            f.write("\n".join(yolo_annotation))
    
    # Save COCO annotations
    coco_file = annotations_dir / "annotations.json"
    with open(coco_file, "w") as f:
        json.dump(coco_annotations, f, indent=2)
    
    print(f"Generated {num_images} sample images with annotations")


def generate_sample_configs(output_dir: Path) -> None:
    """Generate sample configuration files."""
    print("Generating sample configuration files...")
    
    # Create config directory
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample data configuration
    data_config = {
        "data": {
            "root_path": "./data",
            "raw_path": "./data/raw",
            "interim_path": "./data/interim",
            "processed_path": "./data/processed"
        },
        "preprocessing": {
            "imputation_strategy": "median",
            "outlier_method": "winsorize",
            "scaling_method": "standard"
        },
        "feature_engineering": {
            "enable_ratios": True,
            "enable_trends": True,
            "enable_interactions": True
        }
    }
    
    with open(config_dir / "data_config.yaml", "w") as f:
        import yaml
        yaml.dump(data_config, f, default_flow_style=False)
    
    # Sample model configuration
    model_config = {
        "model": {
            "algorithm": "xgboost",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        },
        "training": {
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42
        },
        "evaluation": {
            "metrics": ["accuracy", "f1", "roc_auc"]
        }
    }
    
    with open(config_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)
    
    print("Generated sample configuration files")


def main():
    """Generate all sample data."""
    print("Generating sample data for bloodwork AI platform...")
    
    # Create output directory
    output_dir = Path("sample_data")
    output_dir.mkdir(exist_ok=True)
    
    # Generate tabular data
    tabular_dir = output_dir / "tabular"
    generate_sample_tabular_data(tabular_dir, num_patients=500, num_measurements_per_patient=2)
    
    # Generate image data
    image_dir = output_dir / "images"
    generate_sample_image_data(image_dir, num_images=50, image_size=(640, 640))
    
    # Generate configs
    generate_sample_configs(output_dir)
    
    print("Sample data generation completed!")
    print(f"Data saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
