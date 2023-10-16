from flask import Flask, request, jsonify
import os
from PyPDF2 import PdfReader
import string
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from Models.skills import skills2
# Charger le modèle linguistique français de spaCy
nlp = spacy.load("fr_core_news_sm")

app = Flask(__name__)

# Définir une liste de stopwords personnalisée
custom_stopwords = ["le", "la", "de", "des", "du", "et", "en", "pour", "avec", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles"]

def nettoyer_texte(texte):
    # Supprimer la ponctuation
    texte = texte.translate(str.maketrans('', '', string.punctuation))

    # Mettre en minuscules
    texte = texte.lower()

    # Supprimer les numéros de téléphone
    texte = re.sub(r'\d{10,}', '', texte)

    # Supprimer les adresses e-mail
    texte = re.sub(r'\S+@\S+', '', texte)

    # Supprimer les stopwords personnalisés
    mots = texte.split()  # Diviser le texte en mots
    mots = [mot for mot in mots if mot not in custom_stopwords]  # Supprimer les stopwords

    # Utiliser spaCy pour normaliser les mots et supprimer les mots répétés
    doc = nlp(" ".join(mots))
    mots_norm = [token.lemma_ for token in doc if not token.is_stop]

    # Filtrer les mots pertinents pour les compétences
    mots_pertinents = [mot for mot in mots_norm if mot in skills2]

    # Réassembler les mots pertinents pour les compétences
    texte_skills = ' '.join(mots_pertinents)

    return texte_skills

# Route pour l'extraction de données depuis les fichiers PDF, nettoyage et extraction de compétences
@app.route('/extract_clean_and_extract_skills', methods=['GET'])
def extract_clean_and_extract_skills():
    # Spécifiez le chemin du répertoire contenant les CV
    dossier_cv = "CV"

    # Utilisez os.listdir pour récupérer la liste des fichiers dans le répertoire
    fichiers_dossier = os.listdir(dossier_cv)

    # Créez une liste vide pour stocker les compétences extraites des CV
    extracted_skills_list = []

    # Parcourez la liste des fichiers
    for fichier in fichiers_dossier:
        if fichier.endswith(".pdf"):
            chemin_pdf = os.path.join(dossier_cv, fichier)
            pdf_reader = PdfReader(chemin_pdf)

            cv_texte = ""
            # Parcourez chaque page du PDF et extrayez le texte
            for page in pdf_reader.pages:
                cv_texte += page.extract_text()

            # Appliquez la fonction de nettoyage et d'extraction de compétences
            cleaned_cv = nettoyer_texte(cv_texte)
            extracted_skills_list.append({"file": fichier, "skills": cleaned_cv})

    # Créer un objet TfidfVectorizer avec sublinear_tf
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True)

    # Extraire uniquement les compétences des CV pour la vectorisation TF-IDF
    skills_text = [cv["skills"] for cv in extracted_skills_list]

    # Effectuer la vectorisation TF-IDF sur les compétences des CV
    tfidf_matrixCV = tfidf_vectorizer.fit_transform(skills_text)

    # Récupérer les noms de fonction
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Boucler sur tous les CV
    for cv_index, cv in enumerate(extracted_skills_list):
        tfidf_vector_for_cv = tfidf_matrixCV[cv_index]

        # Récupérer les scores TF-IDF sous forme de tableau
        tfidf_scores_for_cv = tfidf_vector_for_cv.toarray()[0]

        print(f"CV {cv_index + 1} - Valeurs TF-IDF :")
        for word, tfidf_score in zip(feature_names, tfidf_scores_for_cv):
            print(f"Mot : {word}, TF-IDF : {tfidf_score}")
        print("\n")


    return jsonify(extracted_skills_list)

if __name__ == '__main__':
    app.run(debug=True)
