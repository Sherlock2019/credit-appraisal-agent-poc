# 💳 Credit Appraisal Agent — AI That Learns from the Real World

> ⚡ *Open Source • Privacy-First • Self-Improving • Production-Ready*

---

## 🌍 Overview

The **Credit Appraisal Agent** is an open-source, explainable AI system that automates credit risk evaluation.
It analyzes income, collateral, and debt information to produce **transparent, bias-free, and legally compliant** lending decisions — instantly.

This Proof-of-Concept (PoC) is part of a broader **AI Agent Library** initiative: a collection of modular, self-improving agents designed to learn continuously from real users for real-world applications.

---

## 🧩 Current PoC Architecture

### 🧱 Components and Tools (Today)

| Layer                   | Purpose                                                      | Tools / Frameworks                   |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------ |
| **Frontend**            | Web interface for uploading data and viewing results         | Streamlit                            |
| **API Layer**           | REST endpoints for model inference, feedback, and retraining | FastAPI                              |
| **Model**               | Logistic Regression / Random Forest                          | scikit-learn + joblib                |
| **Feature Engineering** | DTI, LTV, CCR, ITI, affordability ratios                     | Pandas + NumPy                       |
| **Retraining Loop**     | Human review → feedback CSV → automatic retrain              | FastAPI `/v1/training/train`         |
| **Storage**             | Local JSON and joblib model registry                         | File system                          |
| **Visualization**       | Approval rates, collateral mix, reasons for denial           | Streamlit + Plotly                   |
| **Deployment (PoC)**    | Local or Docker-based single node                            | Linux + Python venv / Docker Compose |

### 🔁 Data Flow (PoC)

```
User Uploads Dataset
   ↓
FastAPI processes and appraises via model
   ↓
Dashboard displays explainable results
   ↓
Human feedback exported to CSV
   ↓
Retraining improves next model version
```

---

## 🚀 From PoC to Production — Scaling Vision

Transitioning from a single proof-of-concept into a **production-grade, multi-agent AI platform** capable of integrating structured data, documents, and real-time feedback.

### ⚙️ Scalable Production Architecture

| Layer                | Tool / Platform                       | Description                                          |
| -------------------- | ------------------------------------- | ---------------------------------------------------- |
| **Frontend**         | Streamlit / React                     | Secure web portal for credit officers                |
| **API Gateway**      | FastAPI + Nginx + Gunicorn            | Load-balanced REST API                               |
| **Orchestration**    | Flowwise / Dataiku / Kubeflow         | Multi-agent workflows and model retraining pipelines |
| **AI Engine**        | scikit-learn + Hugging Face + OpenLLM | Mix of classical ML and transformer-based reasoning  |
| **Data Management**  | PostgreSQL + MinIO / Ceph / S3        | Structured + unstructured data storage               |
| **Model Registry**   | MLflow + Helm + ArgoCD (GitOps)       | Versioning, promotion, and rollback                  |
| **Monitoring**       | Prometheus + Grafana + Loki           | Telemetry and model drift analytics                  |
| **Security**         | Vault + OAuth2                        | Encryption, identity, and audit control              |
| **Compute Platform** | Kubernetes / OpenStack / GPU Cloud    | Hybrid deployment across any environment             |

---

## 🤝 Future Integrations

### 🧠 Hugging Face & OpenLLM

Integrate transformer models to interpret unstructured documents (bank statements, payslips) for deeper scoring accuracy.

### 📊 Dataiku & Flowwise

Visual orchestration for automated retraining and multi-model decision pipelines (credit + fraud + risk).

### 📚 Open Datasets

Incorporate **UCI**, **Kaggle**, and **World Bank** datasets blended with anonymized local data to train models responsibly.

---

## 🤖 Hugging Face Sandbox Library (HF-kaggle branch)

The `HF-kaggle` branch ships a reusable sandbox library that pairs pre-trained Hugging Face checkpoints with Kaggle-ready fine-tuning utilities. Each agent includes a lightweight wrapper so you can bootstrap experiments without wiring up pipelines from scratch.

| Agent Type         | Task                                   | Default Hugging Face Model            | Example Kaggle Dataset                | Output Goal                    |
| ------------------ | -------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------ |
| Credit Appraisal   | Credit scoring & risk classification   | `roberta-base`                        | Give Me Some Credit                   | Predict creditworthiness       |
| Asset Appraisal    | Property valuation (text + imagery)    | `distilbert-base-uncased` + `google/vit-base-patch16-224` | House Prices – Advanced Regression | Estimate property valuation    |
| KYC Agent          | ID verification & OCR                  | `microsoft/layoutlm-base-uncased` + `microsoft/trocr-base-stage1` | IDR Dataset (OCR/Doc)              | Extract and classify identity  |
| Customer Support   | Chat intent classification             | `distilbert-base-uncased`             | Bank Customer Complaints              | Route intents / responses      |
| Fraud Detection    | Transaction anomaly detection (text)   | `bert-base-uncased`                   | Credit Card Fraud Detection           | Flag suspicious transactions   |

### Loading agents in Python

```python
from services.api.agents import CreditAppraisalTextAgent, AssetAppraisalAgent, KYCAgent

credit_agent = CreditAppraisalTextAgent()
scores = credit_agent.score_single("Applicant has stable income and low DTI")

asset_agent = AssetAppraisalAgent()
image_scores = asset_agent.score_images(["datasets/house_photo.jpg"])

kyc_agent = KYCAgent()
entities = kyc_agent.extract_entities("Name: Jane Doe\nPassport: X123456")
```

### Fine-tune with Kaggle datasets

Use the CLI entry point to fine-tune any registered task on a CSV/TSV dataset:

```bash
python -m services.api.agents.trainer \
  --task_name credit_appraisal \
  --dataset_path datasets/give_me_some_credit.csv \
  --text_col description \
  --label_col target \
  --num_train_epochs 3
```

The trainer persists weights and tokenizer artifacts under `~/credit-appraisal-agent-poc/models/<task>_trained/` along with a `training_meta.json` audit trail.

### REST endpoints for HF training

The FastAPI service exposes lightweight endpoints to orchestrate training runs remotely:

| Endpoint                | Description                                  |
| ----------------------- | -------------------------------------------- |
| `GET /v1/training/hf/tasks` | List registered Hugging Face tasks/models |
| `POST /v1/training/hf/train` | Trigger fine-tuning given dataset + columns |

`POST /v1/training/hf/train` accepts the same parameters as the CLI (`task_name`, `dataset_path`, `text_col`, `label_col`, etc.) and returns the output directory containing the newly trained checkpoint.

### 🌐 Multi-Cloud, Multi-Region

Deploy seamlessly across **Kubernetes**, **OpenStack**, or **GPU-for-Rent** sandboxes while respecting data-sovereignty laws.

---

## 🧬 The AI Agent Library Vision — Self-Improving by Design

The **Credit Appraisal Agent** is one node in a larger ecosystem of **AI micro-agents** (Credit, Fraud, KYC, AML, Real-Estate, Education, etc.).
Each agent continuously learns from real user feedback and retrains autonomously — forming a living, evolving **AI Factory** for practical business AI.

> 💡 *Each decision and user correction becomes a new training signal — a feedback loop that keeps improving the system organically.*

### 🧩 Agent Evolution Workflow

```mermaid
graph TD
A[User Interaction] --> B[Credit Appraisal Agent]
B --> C[Feedback Collected]
C --> D[Flowwise Orchestrator]
D --> E[MLflow Retraining + Model Registry]
E --> F[Updated Model Deployed]
F --> G[Monitoring & Metrics]
G -->|Insights| A
```

---

## 🌍 Compliance and Data Sovereignty

✅ 100 % compliant with **GDPR**, **Vietnam Data Law 2025**, and other global privacy regulations.
🔒 Trains only on **synthetic or anonymized data**, ensuring full legality on cross-border GPU infrastructure.
⚙️ Designed for **data localization**: only model weights, never personal data, are transmitted externally.

---

## 🧱 Deployment Flexibility

```bash
# Example: Deploy via Helm + Kustomize
helm upgrade --install credit-appraisal ./deployment \
  -f deployment/values.yaml -n credit-ai --create-namespace
```

Accessible via:

```
https://credit-appraisal.local
```

---

## 🚀 Summary — Why It Matters

| Value                        | Description                            |
| ---------------------------- | -------------------------------------- |
| 🕒 Faster Lending Decisions  | From days to seconds                   |
| 🧠 Transparent & Explainable | Each result includes reason codes      |
| 🔐 Private & Compliant       | Works under any jurisdiction           |
| ☁️ Cloud-Agnostic            | Runs on any cloud or local environment |
| 🔁 Continuously Learning     | Improves with real user feedback       |

---

> 🧠 *“From sandbox to self-learning , self improving Agent models factory, to production — AI that learns from the people it serves.”*
