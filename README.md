# End-to-End Text Summarization System 

This project is an **end-to-end implementation of text summarization system** built using **Hugging Face Transformers**, **PyTorch**, **FastAPI**, and **Streamlit**, following **clean architecture and MLOps best practices**.

---

##  Key Features

- Transformer-based abstractive summarization (Seq2Seq)
- Modular, configuration-driven pipeline architecture
- Training, evaluation, and inference separation
- ROUGE-based model evaluation
- REST API inference using FastAPI
- Interactive UI using Streamlit
- Dockerized services for deployment
- Training skip logic using artifact checks
- Reproducible experiments via YAML configs

---

##  Model & Dataset

- **Model**: Hugging Face Seq2Seq model ideal to use facebook/bart-large-cnn but due to device constraints I used to `facebook/bart-base`
- **Dataset**: CNN / DailyMail (Hugging Face Datasets)
- **Task**: Abstractive text summarization
- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum

---

##  Project Structure

```
text-summarization-system/
├── config/
│   └── config.yaml
├── params.yaml
├── src/summarizer/
│   ├── components/
│   ├── pipeline/
│   ├── config/
│   ├── entity/
│   ├── utils/
│   ├── constants/
│   └── logging/
├── artifacts/
├── app.py
├── streamlit_app.py
├── api.Dockerfile
├── streamlit.Dockerfile
├── main.py
└── README.md
```

---

##  Configuration-Driven Design

All pipeline behavior is controlled through YAML files:

- `config/config.yaml` – dataset paths, artifact locations, tokenizer/model paths  
- `params.yaml` – training hyperparameters, batch sizes, optimizer, scheduler, precision settings

This enables **reproducibility**, **clean experimentation**, and **easy tuning**.

---

## Training Pipeline

Run the full pipeline:

```bash
python main.py
```

Pipeline stages:
1. Data Ingestion  
2. Data Validation  
3. Data Transformation  
4. Model Training (with artifact-based skip logic)  
5. Model Evaluation  

---

## Evaluation Metrics (ROUGE)

The model is evaluated using **ROUGE**, the standard metric suite for summarization.

### ROUGE-1
- Measures unigram (word-level) overlap  
- Indicates content coverage and key information capture

### ROUGE-2
- Measures bigram overlap  
- Reflects phrase-level fluency and coherence

### ROUGE-L
- Based on longest common subsequence  
- Evaluates structural similarity and word order

### ROUGE-Lsum
- Summary-level ROUGE-L  
- Commonly reported for CNN/DailyMail benchmarks

### Generation Length
- Average number of tokens generated per summary  
- Helps detect overly short or verbose outputs

#### Final fine-tuning Results

```
ROUGE-1     : 42.73
ROUGE-2     : 20.29
ROUGE-L     : 29.34
ROUGE-Lsum  : 39.70
```

These scores indicate strong content coverage, good fluency, and solid structural alignment with reference summaries.

> Note: I had trained the bart-base model on 10% of the dataset due to hardware and time constraints with respect to this project

> Note: ROUGE measures lexical overlap and does not guarantee factual correctness.

---

### Inference API (FastAPI)

Start the API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoint

**POST** `/predict`

```json
{
  "text": "Long article text here"
}
```

Response:

```json
{
  "summary": "Generated concise summary"
}
```

---

##  Streamlit UI

```bash
streamlit run streamlit_app.py
```

Provides an interactive frontend connected to the FastAPI backend with inference latency display.

---

##  Docker Support

```bash
docker build -f api.Dockerfile -t summarization-api .
docker build -f streamlit.Dockerfile -t summarization-ui .
```

```bash
docker run -p 8000:8000 summarization-api
docker run -p 8501:8501 summarization-ui
```

---

## 📌 Future Improvements

- Experiment tracking (MLflow / W&B)
- Model registry
- Quantized inference
- CI/CD pipelines
- Cloud deployment (AWS ECS / EC2)
- Monitoring and drift detection

---

## License

MIT License
