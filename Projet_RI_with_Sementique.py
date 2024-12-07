import os
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Télécharger et charger le modèle SpaCy
os.system('python -m spacy download en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

# Parcourir le dossier Collection_TIME
folder_path = "Collection_TIME"
files = os.listdir(folder_path)

corpus_files = []
for filename in files:
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r", encoding="utf-8") as file:
        corpus_files.append(file.read())

# Fonction pour extraire les entités nommées d'un document avec leurs types
def extract_named_entities(doc):
    spacy_doc = nlp(doc)
    entities = {ent.label_: [] for ent in spacy_doc.ents}
    for ent in spacy_doc.ents:
        entities[ent.label_].append(ent.text)
    return entities

# Fonction pour convertir les entités en texte concaténé (par type)
def entities_to_text(entities):
    return " ".join([" ".join(ents) for ents in entities.values()])

# Extraction des entités nommées et transformation en texte concaténé
entity_documents = [entities_to_text(extract_named_entities(doc)) for doc in corpus_files]

# Calcul de la pondération TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(entity_documents)

# Fonction de recherche avec amélioration sémantique
def search_query(query):
    # Extraire les entités nommées de la requête
    query_entities = extract_named_entities(query)
    # Transformer les entités de la requête en texte concaténé
    query_text = entities_to_text(query_entities)
    # Transformer la requête en vecteur TF-IDF
    query_vector = vectorizer.transform([query_text])
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
        if score >= 0.1:  # Seulement les documents pertinents
            response.append({
                "document_id": int(idx) + 1,
                "similarity": round(float(score), 2),
                "content": corpus_files[int(idx)],
                "named_entities": extract_named_entities(corpus_files[int(idx)])  # Ajouter les entités extraites
            })

    return jsonify(response)

# Lancer le serveur Flask
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)
