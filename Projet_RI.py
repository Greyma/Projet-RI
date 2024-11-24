import os
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

os.system('python -m spacy download en_core_web_sm')

# Charger le modèle linguistique spaCy
nlp = spacy.load("en_core_web_sm")


# Parcourir le dossier Collection_TIME
folder_path = "Collection_TIME"
files = os.listdir(folder_path)

corpus_files = []
for filename in files:
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r", encoding="utf-8") as file:
        corpus_files.append(file.read())


# Fonction pour extraire les entités nommées d'un document
def extract_named_entities(doc): 
    spacy_doc = nlp(doc)
    entities = [ent.text for ent in spacy_doc.ents]  # Extraire uniquement les entités nommées
    return " ".join(entities)  # Retourner les entités sous forme de texte concaténé

# Étape 2 : Extraire les entités nommées pour chaque document
entity_documents = [extract_named_entities(doc) for doc in corpus_files]

# Étape 3 : Calculer la pondération TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(entity_documents)

# Étape 4 : Appariement des requêtes
def search_query(query):
    # Extraire les entités nommées de la requête
    query_entities = extract_named_entities(query)
    # Transformer la requête en vecteur TF-IDF
    query_vector = vectorizer.transform([query_entities])
    # Calculer la similarité cosinus avec les documents
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # Classer les documents par similarité
    ranked_indices = similarities.argsort()[::-1]
    return [(i, similarities[i]) for i in ranked_indices]

    # Initialiser l'application Flask
app = Flask(__name__)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', default='', type=str)
    if not query:
        return jsonify({"error": "A query parameter is required."}), 400
    
    results = search_query(query)
    response = []
    for idx, score in results:
        if score >= 0.1 :
            response.append({
                "document_id": int(idx) + 1,  # Convertir en int standard
                "similarity": round(float(score), 2),  # Convertir en float standard
                "content": corpus_files[int(idx)]  # Assurez-vous que idx est un entier natif
            })

    return jsonify(response)

# Lancer le serveur Flask
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=False)