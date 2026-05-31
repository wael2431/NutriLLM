# 🥗 NutriLLM

A fine-tuned nutrition assistant built on **Llama 3.2 1B**, trained on the Epicurious recipe dataset and deployed locally via **Ollama** with a **Streamlit** chat interface.

---

## 📋 Overview

NutriLLM is an end-to-end LLM fine-tuning project that:

1. **Prepares** a conversational dataset from the Epicurious recipe CSV (`epi_r.csv`)
2. **Fine-tunes** Llama 3.2 1B using LoRA + Unsloth on Google Colab
3. **Exports** the model to GGUF format (4-bit quantized) for CPU-friendly inference
4. **Runs** a local chat interface via Ollama + Streamlit

---

## 🗂️ Project Structure

```
NutriLLM/
├── epi_r.csv          # Epicurious recipe dataset (source data)
├── data.py            # Dataset preparation (Google Colab)
├── training.py        # Fine-tuning script with Unsloth + LoRA (Google Colab)
├── ollama_chat.py     # Streamlit chat UI (local inference via Ollama)
└── requirements.txt   # Python dependencies
```

---

## ⚙️ Pipeline

### Step 1 — Data Preparation (`data.py`)

Run in **Google Colab**. Reads `epi_r.csv` from the Epicurious dataset and generates a conversational JSON file (`votre_dataset_epicurious.json`) saved to Google Drive.

Each row in the CSV becomes a question/answer pair:
- **Question:** `"Propose-moi une idée de repas sain qui contient ces éléments : [ingredients]"`
- **Answer:** Recipe name + nutritional profile (calories, protein, fat, sodium)

### Step 2 — Fine-Tuning (`training.py`)

Run in **Google Colab** (GPU required). Uses [Unsloth](https://github.com/unslothai/unsloth) for fast, memory-efficient training.

| Setting | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-1B-Instruct` |
| Method | LoRA (rank 16) |
| Quantization | 4-bit (`load_in_4bit=True`) |
| Target modules | `q_proj`, `v_proj` |
| Optimizer | AdamW 8-bit |
| Learning rate | 2e-4 |
| Export format | GGUF `q4_k_m` |

The fine-tuned model is saved to Google Drive as `NutriLLM/` in GGUF format.

### Step 3 — Local Inference (`ollama_chat.py`)

A Streamlit chat app that sends messages to a locally running Ollama instance serving the `nutrillm-v1` model.

---

## 🚀 Getting Started

### Prerequisites

- [Ollama](https://ollama.com/) installed and running
- Python 3.9+
- A fine-tuned GGUF model (produced by `training.py`)

### 1. Load the model into Ollama

After downloading the folder that contain the GGUF file from Google Drive`:

```bash
ollama create nutrillm-v1 -f Modelfile
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the chat UI

```bash
streamlit run ollama_chat.py
```

Open your browser at `http://localhost:8501`.

---

## 🛠️ Training Environment

- Google Colab (T4 GPU )
- Libraries: `unsloth`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, `transformers`

---

## 📦 Dependencies

See `requirements.txt`. Key packages for local inference:

```
streamlit
requests
```

---

## 📄 Dataset

The project uses the **Epicurious Recipes** dataset (`epi_r.csv`), which contains recipe titles, nutritional information (calories, protein, fat, sodium), and binary ingredient columns.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

