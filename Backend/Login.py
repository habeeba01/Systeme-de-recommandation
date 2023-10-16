from flask import Flask, request, jsonify
from Models.User import User
import bcrypt
from BD.Connexion import db

app = Flask(__name__)
  
collection = db['user']

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    nom = data.get('nom')
    prenom = data.get('prenom')
    email = data.get('email')
    password = data.get('password')

    if not nom or not prenom or not email or not password:
        return jsonify({'message': 'remplissez tous les champs'}), 400

    existing_user = collection.find_one({'email': email})

    if existing_user:
        return jsonify({'message': 'Cet email est déjà pris'}), 400

    # Créez un nouvel utilisateur en fournissant tous les arguments nécessaires
    new_user = User(nom, prenom, email, password)

    # Hachez le mot de passe avant de le stocker
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Enregistrez l'utilisateur dans la base de données
    collection.insert_one({'nom': new_user.nom, 'prenom': new_user.prenom, 'email': new_user.email, 'password': hashed_password})

    return jsonify({'message': 'Inscription réussie'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Les champs email et password sont obligatoires'}), 400

    user = collection.find_one({'email': email})

    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({'message': 'Connexion réussie'}), 200

    return jsonify({'message': 'email ou mot de passe incorrect'}), 401

if __name__ == '__main__':
    app.run(debug=True )
