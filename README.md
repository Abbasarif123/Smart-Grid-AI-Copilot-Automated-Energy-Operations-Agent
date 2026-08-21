# Smart Grid AI Copilot & Automated Energy Operations Agent

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1E90FF?style=for-the-badge&logo=xgboost&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

A stateful, multi-agent operations engine designed to automate real-time triage for electricity grid incidents (like negative spot prices). This project combines a deterministic Machine Learning forecasting model with an autonomous Large Language Model (LLM) agent, ensuring that any proposed actions are legally compliant and require human approval before execution.

## The Business Problem & Solution

Energy markets are highly volatile. When spot prices drop to negative values due to excess renewable output, grid operators must react quickly to store energy, curtail generation, or adjust loads. 

This copilot automates the initial investigation and drafting of a response plan:
1. **Data Ingestion:** Pulls 15-minute resolution electricity market data from the SMARD.de API.
2. **ML Forecasting:** Uses a trained XGBoost model to predict prices for the next 4 to 24 hours.
3. **Regulatory Compliance (RAG):** Queries a local ChromaDB vector database to ensure proposed actions comply with German grid regulations.
4. **Human-in-the-Loop (HITL):** Uses LangGraph to pause execution and present a drafted action plan to a human operator via a Streamlit dashboard. The agent cannot proceed without explicit approval.

## Architecture Overview

The system is built on a 4-layer architecture:

1.  **Data & ML Layer (`ml_pipeline/`):** Handles the ingestion of SMARD API data and the training of the XGBoost forecasting model.
2.  **Agent Core (`agent/`):** The LangGraph state machine. It routes decisions, maintains state (memory), and defines the agent's available tools (Data Query, ML Forecast, RAG Search).
3.  **API Layer (`app/api.py`):** FastAPI endpoints that trigger the agent, pause it for human review, and resume execution upon approval.
4.  **UI Layer (`app/dashboard.py`):** A Streamlit frontend dashboard for operators to interact with the system and review/approve drafted actions.

## Quickstart Guide

Follow these steps to run the complete system locally.

### 1. Clone & Install
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/Abbasarif123/Smart-Grid-AI-Copilot-Automated-Energy-Operations-Agent.git](https://github.com/Abbasarif123/Smart-Grid-AI-Copilot-Automated-Energy-Operations-Agent.git)
cd Smart-Grid-AI-Copilot-Automated-Energy-Operations-Agent
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add your free Groq API key:
```env
GROQ_API_KEY="gsk_your_api_key_here"
```

### 3. Generate Data, ML, and RAG Artifacts
Run the pipeline scripts to download the sample data, train the model, and build the local vector database:
```bash
# 1. Download SMARD.de data
python ml_pipeline/ingest_smard.py

# 2. Train the XGBoost forecasting model
python ml_pipeline/train.py

# 3. Build the ChromaDB regulatory knowledge base
python ml_pipeline/ingest_regulations.py
```

### 4. Start the Application
You need to run both the FastAPI backend and the Streamlit frontend. Open two separate terminals:

**Terminal 1 (Backend):**
```bash
uvicorn app.api:api --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
streamlit run app/dashboard.py --server.port 8501
```

Navigate to `http://localhost:8501` in your browser. Click **"Investigate Incident"** to watch the agent analyze the data, run the forecast, check regulations, and pause for your approval!

---

## Prompting Guidelines & Sample Prompts

Because this agent uses a Human-in-the-Loop (HITL) architecture, it needs to know *when* to pause and ask for your approval. 

**⚠️ Critical Rule:** You must include the exact phrase **`PROPOSE ACTION`** in your prompt. This acts as a trigger word for the LangGraph state machine to halt execution and present the draft plan to the Streamlit UI.

### Sample Prompts to Try:
Copy and paste these into the Streamlit dashboard:

* **Full Pipeline Test (Data + Forecast + RAG):**
  > "Prices dropped to negative values in early March. Query the historical data from 2024-03-03 to 2024-03-05, run a 4-hour forecast, check the regulations for what we must do during negative price events, and PROPOSE ACTION."

* **Forecast & Strategy Focus:**
  > "What is the 12-hour ML forecast for grid prices? Based on the trend and our regulatory guidelines, PROPOSE ACTION to minimize financial loss."

* **Data & Regulation Check:**
  > "Query the spot prices for 2024-03-04. Are there any BNetzA guidelines about baseload plants during these price drops? PROPOSE ACTION."

---

## Testing

This project includes a Pytest suite to verify the deterministic components of the pipeline (feature engineering and tool logic).

To run the tests:
```bash
pytest -v
```

---
