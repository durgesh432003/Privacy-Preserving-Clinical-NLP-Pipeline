# Privacy-Preserving Clinical NLP Pipeline

This project is a fully local, privacy-preserving multi-agent AI pipeline that converts unstructured hospital discharge summaries into structured HL7 FHIR R4 clinical records.

## Architecture

- **Extractor Agent**: Extracts structured entities from text.
- **Validator Agent**: Verifies clinical plausibility and data formatting.
- **Formatter Agent**: Maps data to FHIR R4 resources.
- **Orchestration**: LangGraph StateGraph.
- **Model**: Phi-4 Mini served locally via Ollama.

## Setup Instructions

### Prerequisites
- Python 3.10+ installed
- Ollama installed and running locally
- Port `11434` (Ollama), `8000` (FastAPI), and `8501` (Streamlit) available

### Running the System
1. Clone the repository and navigate into the root directory.
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Ensure you have an `.env` file configured (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```
4. Pull the required Ollama model:
   ```bash
   ollama pull phi4-mini
   ```
5. Start the pipeline natively:
   ```bash
   ./start.sh
   ```
6. Access the Streamlit UI at [http://localhost:8501](http://localhost:8501).

### Troubleshooting
- If Ollama fails to start or respond, verify the daemon is running (`systemctl status ollama` on Linux).
- If the model is slow or using CPU, ensure `OLLAMA_NUM_GPU=999` is set in your `.env` to offload all layers to your GPU.
- If the Streamlit UI shows API connection errors, verify that `API_KEY` in your `.env` matches the one loaded by the backend.
