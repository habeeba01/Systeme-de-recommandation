from flask import Flask, request, jsonify
import os
from PyPDF2 import PdfReader
import string
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Models.skills import skills  # Importez vos compétences
from BD.Connexion import client, db
from Models.OffreEmploi import OffreEmploi
import pandas as pd

# Charger le modèle linguistique français de spaCy
nlp = spacy.load("fr_core_news_sm")
tfidf_matrixCV = None

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
    mots_pertinents = [mot for mot in mots_norm if mot in skills]

    # Réassembler les mots pertinents pour les compétences
    texte_skills = ' '.join(mots_pertinents)

    return texte_skills

def mots_communs(texte, liste):
    mots = texte.split()
    mots_communs = [mot for mot in mots if mot in liste]
    return ' '.join(mots_communs)

def lemmatize(text):
    doc = nlp(text)
    # Convertir le document en une chaîne de caractères
    text_str = ' '.join([token.text for token in doc])
    # Supprimer les chiffres
    text_str = re.sub(r'\d+', '', text_str)
    # Supprimer les caractères isolés (un seul alphabet)
    text_str = re.sub(r'\b\w\b', '', text_str)
    # Supprimer les espaces
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    # Supprimer les caractères spéciaux avec une expression régulière
    cleaned_text = re.sub(r"[,./?\":;+=\[\](){}]", "", text_str)
    cleaned_doc = nlp(cleaned_text)
    cleaned_tokens = [token.lemma_ for token in cleaned_doc if not (token.is_stop or token.is_punct or token.pos_ == "VERB")]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

# Créer un objet TfidfVectorizer avec sublinear_tf
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words=custom_stopwords)

@app.route('/extract_clean_and_extract_skills', methods=['POST'])
def extract_clean_and_extract_skills():
    global tfidf_matrixCV
    
    if 'cv_file' not in request.files:
        return jsonify({"error": "No file part"})

    cv_file = request.files['cv_file']

    if cv_file.filename == '':
        return jsonify({"error": "No selected file"})

    # Lire le contenu du CV
    cv_texte = ""
    with cv_file as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            cv_texte += page.extract_text()

    # Appliquez la fonction de nettoyage et d'extraction de compétences sur le CV de l'utilisateur
    cleaned_cv = nettoyer_texte(cv_texte)
    
    # Effectuer la vectorisation TF-IDF sur le CV de l'utilisateur
    tfidf_matrixCV = tfidf_vectorizer.transform([cleaned_cv])
    
    return jsonify({"message": "CV processed successfully"})


@app.route('/vectorize_offres', methods=['GET'])
def vectorize_offres():
    global tfidf_matrixCV
    if tfidf_matrixCV is None:
        return jsonify({"error": "No CV data available. Please process a CV first."})

    data = []  # Créez une liste pour stocker les données
    collection = db["offres_emploi"]

    # Récupération de toutes les offres
    cursor = collection.find({})

    # Création d'objets OffreEmploi à partir des données de MongoDB
    for document in cursor:
        name = document['name']
        combined_text = document['combined_text']
        lien = document['lien']
        offre = OffreEmploi(name, combined_text, lien)

        # Ajoutez chaque offre à la liste "data"
        data.append({
            'name': offre.name,
            'combined_text': offre.combined_text,
            'lien': offre.lien
        })

    df = pd.DataFrame(data)

    # Effectuer la vectorisation TF-IDF sur les compétences des offres
    comp = df['combined_text'].apply(lambda x: mots_communs(x, skills))
    tfidf_matrixOffre = tfidf_vectorizer.transform(comp)  # Utilisez transform ici, pas fit_transform
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Calculer la similarité entre le CV de l'utilisateur et toutes les offres
    similarity_scores = cosine_similarity(tfidf_matrixCV, tfidf_matrixOffre)

    recommendations = []

    sorted_scores = sorted(enumerate(similarity_scores[0]), key=lambda x: x[1], reverse=True)
    cv_recommendations = []
    for idx, score in sorted_scores:
        cv_recommendations.append({
            'offre_index': idx,
            'lien_offre': data[idx]['lien'],
            'similarity_score': score
        })
    recommendations.append({
        'cv_index': 0,
        'recommendations': cv_recommendations
    })

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
