import requests
from bs4 import BeautifulSoup
import time
from pymongo import MongoClient
from BD.Connexion import db, client
from Models.OffreEmploi import OffreEmploi

# Sélectionnez la collection dans la base de données
collection = db["offres_emploi"]

# Récupérez les liens des offres déjà enregistrées
existing_links = set(doc["lien"] for doc in collection.find({}, {"lien": 1}))

# Start with the initial URL
url = "https://www.stagiaires.ma/offres-stages"
data = []

while url:
    response = None
    while response is None:
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion : {e}")
            time.sleep(5)  # Attendre pendant 5 secondes avant de réessayer

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        target_classes = ["offer-container p-xxs m-b-xs", "offer-container p-xxs m-b-xs bg-muted"]
        for x in soup.find_all(class_=target_classes):
            div = x.find('a')
            lien = div['href']
            if lien not in existing_links:  # Vérifiez si le lien n'existe pas déjà
                response3 = requests.get(lien)
                if response3.status_code == 200:
                    soup3 = BeautifulSoup(response3.text, 'html.parser')
                    name = soup3.find('h4', class_="inline").text
                    div_element = soup3.find('div', class_='well well-sm m-b-none')  # Localisez la <div> cible
                    if div_element:
                        paragraphs = div_element.find_all('p')  # Trouvez tous les éléments <p> dans la <div>
                        # Initialisez une chaîne vide pour stocker le texte combiné
                        combined_text = ''
                        for paragraph in paragraphs:
                            combined_text += paragraph.text + ' '

                        # Créez un dictionnaire pour les données extraites
                        entry = {
                            "Entreprise": name,
                            "Description": combined_text,
                            "Lien": lien
                        }

                        data.append(entry)
                else:
                    print(f"Erreur de connexion lors de la récupération de {lien}")

        # Trouvez l'élément <ul> avec la classe "pagination"
        pagination_ul = soup.find('ul', class_='pagination')

        # Vérifiez s'il y a un lien "suivant"
        next_link = pagination_ul.find('li', class_='next')
        if next_link:
            link = next_link.find('a')
            if link:
                url = link['href']
                print(f"Next Link Href: {url}")
        else:
            url = None

# Parcourez les données filtrées et insérez-les dans la collection MongoDB
for entry in data:
    offre_emploi = OffreEmploi(name=entry["Entreprise"], combined_text=entry["Description"], lien=entry["Lien"])
    collection.insert_one(offre_emploi.__dict__)

# Fermez la connexion MongoDB
client.close()
