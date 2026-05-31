# code executé en google colab
#cellule1

import pandas as pd
import json
from google.colab import drive
import os

drive.mount('/content/drive')

# --- VÉRIFICATION DES CHEMINS ---
# Puisque tu les as importés directement dans Colab, ils sont dans /content/

import glob

#cellule2

train_files = glob.glob('/content/**/epi_r.csv', recursive=True)

if train_files:
    # On prend le premier chemin trouvé pour chaque fichier
    df = pd.read_csv(train_files[0])
    print("chemin trouvé")
else:
    print("chemin non trouvé")
dataset_ia = []

# 2. Boucler sur chaque ligne du tableau pour "rédiger" des phrases
for index, row in df.iterrows():
    # 1. On récupère la liste des ingrédients qui ont la valeur 1 pour cette recette
    # On exclut les colonnes de base (title, rating, calories, protein, fat, sodium)
    colonnes_ingredients = row.index[6:]
    ingredients_recette = [col for col in colonnes_ingredients if row[col] == 1.0]

    # On prend les 3 ou 4 premiers ingrédients pour simuler la demande de l'utilisateur
    liste_mots_cles = ", ".join(ingredients_recette[:4])

    # 2. On rédige la question artificielle
    question = f"Propose-moi une idée de repas sain qui contient ces éléments : {liste_mots_cles}."

    # 3. On rédige la réponse (Sans les étapes de cuisson, puisqu'elles n'existent pas)
    reponse = (
        f"Je vous suggère la recette suivante : **{row['title']}**.\n\n"
        f"📊 **Profil Nutritionnel :**\n"
        f"- Calories : {row['calories']} kcal\n"
        f"- Protéines : {row['protein']}g\n"
        f"- Lipides : {row['fat']}g\n"
        f"- Sodium : {row['sodium']}mg"
    )

    dataset_ia.append({"question": question, "reponse": reponse})

# Sauvegarde

chemin_final_drive = os.path.join("/content/drive/MyDrive", "votre_dataset_epicurious.json")
with open(chemin_final_drive, "w", encoding="utf-8") as f:
    json.dump(dataset_ia, f, ensure_ascii=False, indent=4)

print(f"✅ Succès ! Le dataset pour l'IA a été enregistré directement dans votre Drive sous : {chemin_final_drive}")
