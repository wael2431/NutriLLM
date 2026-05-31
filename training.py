#code executé en Google colab
#cellue1
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install datasets

#cellule2

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 1. CONFIGURATION INITIALE
# ==========================================
max_seq_length = 2048 # Longueur maximale du contexte (mots/tokens)
dtype = None # Détection automatique (Float16 ou Bfloat16)
load_in_4bit = True # Quantification en 4-bit pour économiser la VRAM

# Chargement du modèle Llama 3.2 1B (version optimisée par Unsloth)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = torch.float16,
    load_in_4bit = load_in_4bit,
)

# ==========================================
# 2. CONFIGURATION DE L'ADAPTATEUR LORA
# ==========================================
# Au lieu d'entraîner tout le modèle, on entraîne une petite couche "LoRA"
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rang LoRA (16 est un bon équilibre entre performance et vitesse)
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth optimise le dropout à 0
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Réduit massivement la VRAM utilisée
)

#cellule3

from google.colab import drive
import os

drive.mount('/content/drive')

# --- VÉRIFICATION DES CHEMINS ---
# Puisque tu les as importés directement dans Colab, ils sont dans /content/

import glob
train_files = glob.glob('/content/**/dataset_epicurious.json', recursive=True)
# ==========================================
# 3. PRÉPARATION DU DATASET
# ==========================================
# On définit le format de prompt attendu par Llama 3 (System, User, Assistant)
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Tu es NutriLLM, un assistant expert en nutrition saine. Propose toujours des alternatives équilibrées.<|eot_id|><|start_header_id|>user<|end_header_id|>
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}<|eot_id|>"""

# Fonction pour formater chaque ligne de votre dataset
def format_prompts(examples):
    inputs = examples["question"]   # La question de l'utilisateur (ex: "Fais un burger")
    outputs = examples["reponse"]   # La réponse idéale (ex: "Voici un burger sain...")
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        text = prompt_template.format(input_text, output_text)
        texts.append(text)
    return { "text" : texts }

# Remplacer "votre_dataset.json" par votre fichier préparé à partir d'Epicurious
dataset = load_dataset("json", data_files=train_files[0], split="train")
dataset = dataset.map(format_prompts, batched = True)

# ==========================================
# 4. CONFIGURATION DE L'ENTRAÎNEMENT (SFTTrainer)
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100, # Ajustez selon la taille du dataset (ex: 1 epoch complet)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#cellule4



# ==========================================
# 5. LANCEMENT DE L'ENTRAÎNEMENT
# ==========================================
print("Démarrage de l'entraînement...")
trainer_stats = trainer.train()
print("Entraînement terminé avec succès !")

# ==========================================
# 6. SAUVEGARDE ET EXPORT (GGUF pour CPU)
# ==========================================
# On exporte le modèle au format GGUF (quantifié en 4-bit) pour pouvoir l'utiliser avec Ollama ou Llama.cpp sans GPU.

model.save_pretrained_gguf(
    '/content/drive/MyDrive/NutriLLM',
    tokenizer,
    quantization_method="q4_k_m"
)


