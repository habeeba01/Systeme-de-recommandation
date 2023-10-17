from flask import Flask, request,render_template,redirect, url_for
import os
from PyPDF2 import PdfReader 
import string
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Models.skills import skills  # Importez vos compétences
from BD.Connexion import db
from Models.OffreEmploi import OffreEmploi
import pandas as pd
from Models.similarityOffre import SimilarityOffre
from Models.User import User
import bcrypt

#######
app = Flask(__name__,template_folder='../templates')

########HOME###################
@app.route('/')
def home():
    return render_template('index.html')

###########COLLECTION############
collection = db['user']
######LOGIN############################
@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None
    if request.method == 'POST' or request.method == 'GET':
        email = request.form.get('email')
        password = request.form.get('password')

        user = collection.find_one({'email': email})

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return redirect('index_cv')
        else :
             error_message = "Email ou mot de passe incorrect"

    return render_template('index.html',error_message=error_message)



########SIGNUP#########################
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error_message = None
    if request.method == 'POST':
        nom = request.form.get('nom')
        prenom = request.form.get('prenom')
        email = request.form.get('email')
        password = request.form.get('password')

        existing_user = collection.find_one({'email': email})

        if existing_user:
            error_message = "Cet email existe deja"
        else:

            # Créez un nouvel utilisateur en fournissant tous les arguments nécessaires
            new_user = User(nom, prenom, email, password)

            # Hachez le mot de passe avant de le stocker
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Enregistrez l'utilisateur dans la base de données
            collection.insert_one({'nom': new_user.nom, 'prenom': new_user.prenom, 'email': new_user.email, 'password': hashed_password})

            return redirect(url_for('index_cv'))
    return render_template('index_inscrire.html', error_message=error_message)


###############MAIN#################
@app.route('/index_cv')
def index_cv():
    return render_template('index_cv.html')



##############MODELE#########################
nlp = spacy.load("fr_core_news_sm")
tfidf_matrixCV = None

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
################
def clean_and_preprocess(text):
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Mettre en minuscules
    text = text.lower()

    # Supprimer les numéros de téléphone (10 chiffres ou plus)
    text = re.sub(r'\d{10,}', '', text)

    # Supprimer les adresses e-mail
    text = re.sub(r'\S+@\S+', '', text)

    # Supprimer les stopwords personnalisés
    custom_stopwords = ["le", "la", "de", "des", "du", "et", "en", "pour", "avec", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles"]
    words = text.split()
    words = [word for word in words if word not in custom_stopwords]

    # Réassembler les mots nettoyés en une seule chaîne
    cleaned_text = ' '.join(words)

    return cleaned_text
################
@app.route('/process_uploaded_cv', methods=['GET', 'POST'])
def process_uploaded_cv():
    if 'pdfFile' not in request.files:
        return "Aucun fichier CV n'a été téléchargé."

    uploaded_cv = request.files['pdfFile']

    if uploaded_cv.filename == '':
        return "Le nom du fichier est vide."

    cv_text = ""

    if uploaded_cv:
        pdf_reader = PdfReader(uploaded_cv)
        if pdf_reader:
            for page in pdf_reader.pages:
                cv_text += page.extract_text()

    # Clean and preprocess the CV text
    cleaned_cv = clean_and_preprocess(cv_text)

    # Fit the TF-IDF vectorizer to the skills
    tfidf_vectorizer.fit(skills)

    # Transform the user's CV with the fitted vectorizer
    tfidf_vector_for_user_cv = tfidf_vectorizer.transform([cleaned_cv])

    data = []

    collection = db["offres_emploi"]
    cursor = collection.find({})

    for document in cursor:
        name = document['name']
        combined_text = document['combined_text']
        lien = document['lien']
        offre = OffreEmploi(name, combined_text, lien)

        # Clean and preprocess the combined text of the job offer
        cleaned_offer_text = clean_and_preprocess(offre.combined_text)

        # Transform the offer text with the fitted vectorizer
        tfidf_vector_for_offer = tfidf_vectorizer.transform([cleaned_offer_text])

        # Calculate the similarity between the user's CV and the job offer
        similarity_score = cosine_similarity(tfidf_vector_for_user_cv, tfidf_vector_for_offer)

        similarity_offre = SimilarityOffre(lien, similarity_score[0][0])

        data.append(similarity_offre)

    data.sort(key=lambda x: x.similarity, reverse=True)

    recommendations = [{
        'lienOffre': offre.lien,
        'similarity': offre.similarity
    } for offre in data]

    cv_recommendations = {
        'recommendations': recommendations
    }

    return render_template('offre.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True,port=5003)