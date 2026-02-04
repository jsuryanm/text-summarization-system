# End-to-End Text Summarization System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A production-ready, end-to-end text summarization system built with **BART-base**, deployed on AWS with FastAPI and Streamlit. This project demonstrates MLOps best practices with modular architecture, configuration-driven pipelines, and containerized deployment.

## 🌟 Highlights

- **State-of-the-art NLP**: Fine-tuned Facebook's BART-base model for abstractive summarization
- **Production Architecture**: Clean, modular design following software engineering best practices
- **MLOps Integration**: Configuration-driven pipelines with reproducible experiments
- **Dual Interface**: REST API (FastAPI) + Interactive Web UI (Streamlit)
- **Cloud Deployment**: Fully deployed on AWS with Docker containers
- **Performance**: Achieved ROUGE-1: 42.73, ROUGE-2: 20.29, ROUGE-L: 39.70

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [AWS Deployment](#-aws-deployment)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Features

### Core Capabilities
- **Abstractive Summarization**: Generates human-like summaries using transformer architecture
- **Configurable Pipeline**: YAML-based configuration for easy experimentation
- **Artifact Management**: Smart training skip logic based on existing artifacts
- **Comprehensive Evaluation**: ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
- **REST API**: Production-ready FastAPI endpoints
- **Interactive UI**: User-friendly Streamlit interface with latency monitoring

### MLOps Features
- Modular component-based architecture
- Configuration-driven training and inference
- Automated data validation
- Model versioning and artifact tracking
- Docker containerization for reproducibility
- CI/CD ready structure

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  ┌──────────────────────┐      ┌───────────────────────┐   │
│  │  Streamlit Web UI    │      │   REST API (FastAPI)  │   │
│  │  (Port 8501)         │◄────►│   (Port 8000)         │   │
│  └──────────────────────┘      └───────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Prediction  │  │  Training    │  │  Evaluation  │     │
│  │  Pipeline    │  │  Pipeline    │  │  Pipeline    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Component Layer                          │
│  ┌──────────┐  ┌────────────┐  ┌─────────┐  ┌──────────┐  │
│  │  Data    │  │   Data     │  │  Model  │  │  Model   │  │
│  │ Ingestion│─►│Validation  │─►│ Trainer │─►│Evaluation│  │
│  └──────────┘  └────────────┘  └─────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Model Layer                              │
│         BART-base (facebook/bart-base) Fine-tuned           │
│              on CNN/DailyMail Dataset                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠 Technology Stack

### Core Technologies
- **Framework**: PyTorch
- **Model**: Hugging Face Transformers (BART-base)
- **API**: FastAPI
- **UI**: Streamlit
- **Containerization**: Docker, Docker Compose

### ML/NLP Stack
- transformers (Hugging Face)
- datasets (Hugging Face)
- evaluate (ROUGE metrics)
- torch
- tokenizers

### DevOps & Cloud
- AWS EC2 (compute)
- AWS S3 (model storage)
- Docker (containerization)
- GitHub Actions (CI/CD)

### Data & Utilities
- pandas
- numpy
- PyYAML (configuration)
- python-box (config access)
- ensure (data validation)

---

## 📁 Project Structure

```
text-summarization-system/
│
├── config/
│   └── config.yaml              # Pipeline configuration
│
├── src/summarizer/
│   ├── components/              # Core components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/                # Pipeline orchestration
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   ├── stage_05_model_evaluation.py
│   │   └── prediction.py
│   │
│   ├── config/                  # Configuration management
│   │   └── configuration.py
│   │
│   ├── entity/                  # Data models
│   │   └── config_entity.py
│   │
│   ├── utils/                   # Utilities
│   │   └── common.py
│   │
│   ├── constants/               # Constants
│   │   └── __init__.py
│   │
│   └── logging/                 # Logging setup
│       └── __init__.py
│
├── artifacts/                   # Generated artifacts (gitignored)
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
│
├── research/                    # Jupyter notebooks for experimentation
│
├── .github/workflows/           # CI/CD pipelines
│
├── app.py                       # FastAPI application
├── streamlit_app.py             # Streamlit application
├── main.py                      # Training pipeline entry point
├── params.yaml                  # Training hyperparameters
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── pyproject.toml               # Project metadata
│
├── api.Dockerfile               # FastAPI container
├── streamlit.Dockerfile         # Streamlit container
├── docker-compose.yml           # Multi-container orchestration
│
└── README.md                    # This file
```

---

## 💻 Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended, NVIDIA RTX 4060 or better)
- 8GB+ RAM
- Docker (for containerized deployment)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/jsuryanm/text-summarization-system.git
cd text-summarization-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package in editable mode**
```bash
pip install -e .
```

---

## 🎯 Usage

### Training Pipeline

Run the complete training pipeline:

```bash
python main.py
```

This executes all stages sequentially:
1. **Data Ingestion**: Downloads CNN/DailyMail dataset
2. **Data Validation**: Validates dataset structure
3. **Data Transformation**: Tokenizes and prepares data
4. **Model Training**: Fine-tunes BART-base model
5. **Model Evaluation**: Computes ROUGE metrics

The pipeline includes smart artifact checking - if a stage has already been completed, it will be skipped automatically.

### API Server

Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access API documentation at: `http://localhost:8000/docs`

### Streamlit UI

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

Access the UI at: `http://localhost:8501`

### Programmatic Usage

```python
from summarizer.pipeline.prediction import PredictionPipeline

# Initialize pipeline
predictor = PredictionPipeline()

# Generate summary
text = """
Your long article text here...
"""

summary = predictor.predict(text)
print(f"Summary: {summary}")
```

---

## 📊 Model Performance

### Evaluation Metrics

The model was evaluated on the CNN/DailyMail test set using ROUGE metrics:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 39.43 | Strong unigram overlap - good content coverage |
| **ROUGE-2** | 17.65 | Solid bigram matching - maintains fluency |
| **ROUGE-L** | 26.89 | Good structural similarity |
| **ROUGE-Lsum** | 36.34 | High summary-level coherence |

### Understanding ROUGE Metrics

- **ROUGE-1**: Measures word-level overlap between generated and reference summaries
- **ROUGE-2**: Evaluates phrase-level (bigram) similarity and fluency
- **ROUGE-L**: Based on longest common subsequence, captures word order
- **ROUGE-Lsum**: Summary-level ROUGE-L, standard for CNN/DailyMail

### Training Details

- **Base Model**: facebook/bart-base (139M parameters)
- **Dataset**: CNN/DailyMail (10% used due to hardware constraints)
- **Hardware**: NVIDIA RTX 4060
- **Training Epochs**: Configurable via `params.yaml`
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup with decay

> **Note**: Training was performed on 10% of the dataset due to GPU memory limitations. With a more powerful GPU (e.g., A100, V100), you can train on the full dataset for improved performance.

---

## 🔌 API Documentation

### Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Text Summarization API is running"
}
```

#### Predict Summary
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "Your long article or document text here..."
}
```

**Response:**
```json
{
  "summary": "Concise generated summary of the input text",
  "inference_time_ms": 234.5
}
```

**Error Response:**
```json
{
  "detail": "Error message"
}
```

### Example using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text here"}'
```

### Example using Python

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "text": "Your long article text here..."
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Summary: {result['summary']}")
print(f"Inference Time: {result['inference_time_ms']}ms")
```

---

## 🐳 Docker Deployment

### Build Images

**Build FastAPI container:**
```bash
docker build -f api.Dockerfile -t summarization-api:latest .
```

**Build Streamlit container:**
```bash
docker build -f streamlit.Dockerfile -t summarization-ui:latest .
```

### Run Containers

**Run API server:**
```bash
docker run -d -p 8000:8000 --name api-server summarization-api:latest
```

**Run Streamlit UI:**
```bash
docker run -d -p 8501:8501 --name ui-server summarization-ui:latest
```

### Docker Compose

For running both services together:

```bash
docker-compose up -d
```

This will start:
- FastAPI server on `http://localhost:8000`
- Streamlit UI on `http://localhost:8501`

**Stop services:**
```bash
docker-compose down
```

---

## ☁️ AWS Deployment

### Architecture Overview

```
Internet
   │
   ├─► AWS EC2 Instance (FastAPI) → Port 8000
   │
   └─► AWS EC2 Instance (Streamlit) → Port 8501
        │
        └─► AWS S3 (Model Artifacts)
```

### Deployment Steps

1. **Prepare EC2 Instances**
   - Launch 2 EC2 instances (t2.medium or better)
   - Configure security groups (allow ports 8000, 8501, 22)
   - Install Docker and Docker Compose

2. **Configure GitHub Actions self-hosted runner in EC2

3. **Deploy Containers**
   - SSH into EC2 instances
   - Pull Docker images or build from source

4. **Configure Load Balancer** (Optional)
   - Set up Application Load Balancer
   - Configure health checks
   - Enable auto-scaling

### Environment Variables

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET=your-bucket-name
MODEL_PATH=summarization-artifacts/model_trainer/
```

---

## ⚙️ Configuration

### config/config.yaml

Controls pipeline behavior:

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/entbappy/Branching-tutorial/raw/master/summarizer-data.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

data_validation:
  root_dir: artifacts/data_validation
  STATUS_FILE: artifacts/data_validation/status.txt
  ALL_REQUIRED_FILES: ["train", "test", "validation"]

# ... additional configuration
```

### params.yaml
The example given below for training is a rough example. I used a lot more parameters for the TrainingArguments function please refer my params.yaml for the exact parameters. 

Defines training hyperparameters:

```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  save_steps: 1e6
  gradient_accumulation_steps: 16
  fp16: true  # Mixed precision training
```

> **Tip**: Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` based on your GPU memory.

---

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📝 Future Enhancements

- [ ] Upload model artifacts to S3 for persistence
- [ ] Implement A/B testing for model versions
- [ ] Add support for multi-document summarization
- [ ] Integrate monitoring with Prometheus/Grafana
- [ ] Add support for multiple languages
- [ ] Implement user feedback loop
- [ ] Create mobile-friendly UI
- [ ] Add batch processing endpoints

---

## 🐛 Known Issues

- Training on full dataset requires high-memory GPU (16GB+ VRAM)
- Initial model loading takes 10-15 seconds
- Large input texts (>1000 tokens) may have longer inference times

---

## 📚 Resources

- [BART Paper](https://arxiv.org/abs/1910.13461)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
- [ROUGE Metric Documentation](https://huggingface.co/spaces/evaluate-metric/rouge)

---

## 📧 Contact

**Jayasuryan Mutyala** - [@jsuryanm](https://github.com/jsuryanm)

Project Link: [https://github.com/jsuryanm/text-summarization-system](https://github.com/jsuryanm/text-summarization-system)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Hugging Face team for the Transformers library
- Facebook AI Research for the BART model
- CNN/DailyMail dataset creators
- FastAPI and Streamlit communities

---


If you find this project helpful, please consider giving it a star! ⭐


