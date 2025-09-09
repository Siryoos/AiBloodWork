# Bloodwork AI: Multi-Modal Bloodwork Analysis Platform

A comprehensive machine learning platform for analyzing bloodwork data using both microscopy images and tabular lab data to predict various clinical conditions.

## 🚀 Features

### Multi-Modal Analysis
- **Tabular Data**: Complete Blood Count (CBC), Comprehensive Metabolic Panel (CMP), Lipid Panel, Thyroid Panel, Coagulation Panel, Vitamins & Iron Panel
- **Image Data**: Blood smear microscopy analysis with YOLO-based cell detection and classification

### Clinical Predictions
- **Anemia Classification**: Iron deficiency, thalassemia trait, B12 deficiency, and other types
- **Metabolic Syndrome**: Early detection and risk assessment
- **Cardiovascular Disease Risk**: Lipid-based risk scoring
- **Thyroid Dysfunction**: Hypothyroidism, hyperthyroidism, subclinical conditions
- **DIC Early Warning**: Disseminated Intravascular Coagulation detection
- **Vitamin Deficiencies**: B12 and iron deficiency prediction

### Advanced ML Capabilities
- **Multiple Algorithms**: XGBoost, LightGBM, Neural Networks, Logistic Regression
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Feature Engineering**: Automated ratio creation, trend analysis, interaction features
- **Model Calibration**: Probability calibration for reliable predictions
- **Explainability**: SHAP analysis and Grad-CAM visualizations

### Production Ready
- **FastAPI Serving**: RESTful API with comprehensive endpoints
- **Docker Support**: Containerized deployment
- **Kubernetes**: Scalable orchestration
- **MLflow Integration**: Experiment tracking and model registry
- **Monitoring**: Prometheus metrics and structured logging

## 📁 Project Structure

```
bloodwork-ai/
├── README.md
├── pyproject.toml
├── setup.cfg
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── dvc.yaml
├── params.yaml
├── mlflow.yaml
├── configs/
│   ├── global.yaml
│   ├── data/
│   │   ├── images.yaml
│   │   └── tabular.yaml
│   ├── models/
│   │   ├── vision/
│   │   │   ├── yolo_rbc.yaml
│   │   │   ├── yolo_wbc.yaml
│   │   │   └── yolo_platelet.yaml
│   │   └── tabular/
│   │       ├── cbc_anemia.yaml
│   │       ├── cmp_mets.yaml
│   │       ├── lipid_cvd.yaml
│   │       ├── thyroid_dysfx.yaml
│   │       ├── coag_dic.yaml
│   │       ├── vitamins_b12.yaml
│   │       └── iron_ferritin.yaml
│   └── serving/
│       ├── fastapi.yaml
│       └── thresholds.yaml
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── tabular/
│   ├── interim/
│   └── processed/
│       ├── images_coco/
│       └── tabular_featured/
├── src/bloodwork_ai/
│   ├── __init__.py
│   ├── utils/
│   │   ├── io.py
│   │   ├── log.py
│   │   ├── seed.py
│   │   └── metrics.py
│   ├── schemas/
│   │   ├── tabular_schema.py
│   │   └── label_taxonomies.py
│   ├── ingestion/
│   │   ├── tabular_loader.py
│   │   └── image_loader.py
│   ├── preprocessing/
│   │   ├── tabular_clean.py
│   │   ├── tabular_feature_engineering.py
│   │   ├── time_series.py
│   │   ├── image_stain_norm.py
│   │   └── image_augment.py
│   ├── features/
│   │   ├── tabular_selectors.py
│   │   └── vision_postproc.py
│   ├── models/
│   │   ├── registry.py
│   │   ├── vision/
│   │   │   └── yolo_detector.py
│   │   └── tabular/
│   │       ├── xgboost_model.py
│   │       ├── lightgbm_model.py
│   │       ├── mlp_model.py
│   │       └── logistic_model.py
│   ├── tasks/
│   │   ├── cbc/
│   │   │   └── anemia_classifier.py
│   │   ├── cmp/
│   │   │   └── mets_predictor.py
│   │   ├── lipid/
│   │   │   └── cvd_predictor.py
│   │   ├── thyroid/
│   │   │   └── thyroid_classifier.py
│   │   ├── coag/
│   │   │   └── dic_early_warning.py
│   │   ├── vitamins/
│   │   │   └── b12_deficiency.py
│   │   └── iron/
│   │       └── ferritin_low_predictor.py
│   ├── training/
│   │   ├── train_tabular.py
│   │   ├── train_vision.py
│   │   └── optuna_search.py
│   ├── evaluation/
│   │   ├── eval_tabular.py
│   │   ├── eval_vision.py
│   │   └── report_builder.py
│   ├── explainability/
│   │   ├── shap_tabular.py
│   │   └── gradcam_vision.py
│   └── serving/
│       ├── fastapi_app.py
│       ├── batch_infer.py
│       └── pydantic_models.py
├── scripts/
│   ├── generate_sample_data.py
│   ├── prepare_coco.py
│   ├── export_to_yolo.py
│   ├── sync_registry.py
│   └── calibrate_thresholds.py
├── tests/
│   ├── unit/
│   │   ├── test_schemas.py
│   │   ├── test_preprocessing.py
│   │   └── test_models.py
│   └── e2e/
│       └── test_full_pipeline.py
├── model_card_templates/
│   ├── anemia.md
│   ├── mets.md
│   ├── dic.md
│   └── thyroid.md
├── docker/
│   ├── Dockerfile.training
│   ├── Dockerfile.serving
│   └── compose.yaml
├── k8s/
│   ├── deployment-serving.yaml
│   ├── hpa.yaml
│   └── servicemonitor.yaml
├── airflow_dags/
│   └── bloodwork_pipeline.py
└── .github/
    ├── workflows/
    │   ├── ci.yaml
    │   └── train_eval_register.yaml
    └── ISSUE_TEMPLATE.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, for vision models)
- Docker (optional, for containerized deployment)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bloodwork-ai.git
   cd bloodwork-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Generate sample data**
   ```bash
   python scripts/generate_sample_data.py
   ```

5. **Run the pipeline**
   ```bash
   make dev-setup
   dvc repro
   ```

### Development Setup

1. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Run tests**
   ```bash
   make test
   ```

## 🚀 Quick Start

### 1. Generate Sample Data
```bash
python scripts/generate_sample_data.py
```

### 2. Train Models
```bash
# Train anemia classifier
make train-anemia

# Train WBC detection model
make train-wbc

# Train all models
make train-all
```

### 3. Start API Server
```bash
make serve
```

### 4. Make Predictions
```bash
# Tabular prediction
curl -X POST "http://localhost:8080/predict/cbc/anemia" \
  -H "Content-Type: application/json" \
  -d '{
    "wbc": 7.5,
    "rbc": 4.2,
    "hb": 12.5,
    "hct": 38.0,
    "mcv": 90.5
  }'

# Image prediction
curl -X POST "http://localhost:8080/predict/vision/wbc" \
  -F "image=@blood_smear.jpg"
```

## 📊 Usage Examples

### Tabular Data Analysis

```python
from bloodwork_ai.ingestion import TabularLoader
from bloodwork_ai.preprocessing import TabularCleaner, FeatureEngineer
from bloodwork_ai.models.tabular import XGBoostModel

# Load data
loader = TabularLoader()
data = loader.load_panel_data("data/raw/tabular/cbc_data.csv", "cbc")

# Clean and engineer features
cleaner = TabularCleaner()
clean_data = cleaner.clean_data(data, "cbc")

engineer = FeatureEngineer()
features = engineer.engineer_features(clean_data)

# Train model
model = XGBoostModel()
model.fit(features, data["anemia_type"])

# Make predictions
predictions = model.predict(features)
```

### Image Analysis

```python
from bloodwork_ai.ingestion import ImageLoader
from bloodwork_ai.models.vision import YOLODetector

# Load image
loader = ImageLoader()
image = loader.load_image("data/raw/images/blood_smear.jpg")

# Detect cells
detector = YOLODetector()
detector.load_model("models/yolo_wbc.pt")
detections = detector.predict(image)

# Count cells
cell_counts = detector.count_cells(image)
print(f"WBC count: {cell_counts.get('wbc', 0)}")
```

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=bloodwork-ai

# Data Paths
DATA_ROOT_PATH=./data
ARTIFACTS_PATH=./artifacts

# Model Configuration
MODEL_REGISTRY_PATH=./artifacts/models
```

### Model Configuration
Edit configuration files in `configs/models/` to customize:
- Hyperparameters
- Training settings
- Evaluation metrics
- Feature engineering options

## 📈 Model Performance

### Tabular Models
- **Anemia Classification**: 85%+ accuracy, 0.90+ AUROC
- **Metabolic Syndrome**: 80%+ accuracy, 0.85+ AUROC
- **CVD Risk Prediction**: 0.85+ AUROC, <0.05 calibration error
- **Thyroid Dysfunction**: 85%+ accuracy, 0.90+ AUROC

### Vision Models
- **WBC Detection**: 90%+ mAP@0.5
- **RBC Detection**: 95%+ mAP@0.5
- **Platelet Detection**: 85%+ mAP@0.5
- **Cell Counting**: <10% MAE

## 🐳 Docker Deployment

### Build Images
```bash
make docker-build
```

### Run with Docker Compose
```bash
docker-compose -f docker/compose.yaml up
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

## 📊 Monitoring

### MLflow UI
```bash
make mlflow-ui
```
Access at http://localhost:5000

### Prometheus Metrics
- Model performance metrics
- API request metrics
- System resource usage

### Logging
- Structured JSON logging
- Request/response logging
- Error tracking and alerting

## 🧪 Testing

### Run Tests
```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# End-to-end tests
make test-e2e
```

### Test Coverage
```bash
make test-cov
```

## 📚 API Documentation

### Tabular Endpoints
- `POST /predict/cbc/anemia` - Anemia classification
- `POST /predict/cmp/mets` - Metabolic syndrome prediction
- `POST /predict/lipid/cvd` - CVD risk prediction
- `POST /predict/thyroid` - Thyroid dysfunction classification
- `POST /predict/coag/dic` - DIC early warning
- `POST /predict/vitamins/b12` - B12 deficiency prediction
- `POST /predict/iron/ferritin` - Ferritin low prediction

### Vision Endpoints
- `POST /predict/vision/rbc` - RBC detection
- `POST /predict/vision/wbc` - WBC detection
- `POST /predict/vision/platelet` - Platelet detection

### Utility Endpoints
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - API documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use type hints
- Add docstrings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Medical Disclaimer

This software is for research and educational purposes only. It is not intended for clinical use or medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [MLflow](https://mlflow.org/) for experiment tracking
- [DVC](https://dvc.org/) for data version control
- [FastAPI](https://fastapi.tiangolo.com/) for API framework
- [Optuna](https://optuna.org/) for hyperparameter optimization

## 📞 Support

- 📧 Email: support@bloodwork-ai.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/bloodwork-ai/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/bloodwork-ai/wiki)

## 🔮 Roadmap

- [ ] Real-time streaming analysis
- [ ] Multi-site data integration
- [ ] Advanced explainability features
- [ ] Mobile app integration
- [ ] Cloud deployment templates
- [ ] Additional clinical conditions
- [ ] Federated learning support