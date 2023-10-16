from flask import Flask, jsonify
from Models.OffreEmploi import OffreEmploi
from BD.Connexion import db, client
import spacy
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from Models.skills import skills


app = Flask(__name__)

@app.route('/offres', methods=['GET'])
def get_offres_from_mongodb():
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
    
    # Fermeture de la connexion à MongoDB
    client.close()
    ####################
    df = pd.DataFrame(data)

    #pretraitement
    nlp = spacy.load("fr_core_news_sm")

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

    df['combined_text']= df["combined_text"].apply(lambda txt: lemmatize(txt))

    # affichage
    pd.set_option('display.max_colwidth', None)


    def mots_communs(texte, liste):
        mots = texte.split()
        mots_communs = [mot for mot in mots if mot in liste]
        return ' '.join(mots_communs)

    # Appliquer la fonction à chaque ligne du DataFrame
    comp = df['combined_text'].apply(lambda x: mots_communs(x, skills))

    print(comp)

    # Créer un objet TfidfVectorizer avec sublinear_tf
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True)

    # Effectuer la vectorisation TF-IDF sur l'ensemble des CV nettoyés
    tfidf_matrixOffre = tfidf_vectorizer.fit_transform(comp)
    #print(tfidf_matrix)
    # Récupérer les noms de fonction
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Boucler 
    for offre_index in range(len(data)):
        tfidf_vector_for_cv = tfidf_matrixOffre[offre_index]

        # Récupérer les scores TF-IDF sous forme de tableau
        tfidf_scores_for_cv = tfidf_vector_for_cv.toarray()[0]

        print(f"offre {offre_index + 1} - Valeurs TF-IDF :")
        for word, tfidf_score in zip(feature_names, tfidf_scores_for_cv):
            print(f"Mot : {word}, TF-IDF : {tfidf_score}")
        print("\n")
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
