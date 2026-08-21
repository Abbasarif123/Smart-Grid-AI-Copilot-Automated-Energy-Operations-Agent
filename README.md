work in progress
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