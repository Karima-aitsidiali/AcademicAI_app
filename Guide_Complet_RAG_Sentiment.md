# Guide Complet : Système RAG et Analyse de Sentiment

## Table des Matières
1. [Introduction au Système](#introduction)
2. [Architecture Générale](#architecture)
3. [Système RAG (Retrieval Augmented Generation)](#rag)
4. [Analyse de Sentiment](#sentiment)
5. [Intégration et Workflow Complet](#workflow)

---

## 1. Introduction au Système {#introduction}

Votre application AcaBot est un **chatbot éducatif intelligent** qui utilise deux technologies avancées pour offrir une expérience d'apprentissage personnalisée :

- **RAG (Retrieval Augmented Generation)** : Pour récupérer et générer des réponses basées sur des documents pédagogiques
- **Analyse de Sentiment** : Pour analyser les retours des utilisateurs et améliorer l'expérience

Cette application permet aux étudiants de poser des questions sur leurs cours et de recevoir des réponses précises basées sur les documents de leur formation.

---

## 2. Architecture Générale {#architecture}

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Système RAG    │───▶│   Réponses      │
│   Pédagogiques  │    │   (FAISS+LLM)    │    │   Générées      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Feedbacks     │───▶│   Analyse de     │───▶│   Amélioration  │
│   Utilisateurs  │    │   Sentiment      │    │   Continue      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 3. Système RAG (Retrieval Augmented Generation) {#rag}

### 3.1 Qu'est-ce que le RAG ?

Le RAG est une technique qui **combine la recherche d'informations avec la génération de texte**. Au lieu de se fier uniquement à la mémoire d'un modèle de langage, le RAG :

1. **Recherche** des passages pertinents dans une base de données
2. **Utilise** ces passages comme contexte pour générer une réponse

### 3.2 Code Complet de la Classe RAGChatbot

📁 **Fichier : `rag_chatbot.py`**

Voici le code complet de votre système RAG avec explications détaillées :

```python
import sys
import time
import os, re
import pickle
import faiss
import numpy as np
from colorama import Fore, Style
from sentence_transformers import SentenceTransformer

class RAGChatbot:
    def __init__(self, ollama_api, chunk_size=384, chunk_overlap=96, 
                 faiss_index_file='./vector_store/faiss_index.faiss', 
                 metadata_file='./vector_store/metadata.pickle', 
                 hashes_file='./vector_store/hashes.pickle'):
        """
        Initialise le chatbot RAG avec tous ses composants.
        
        Ligne 1: ollama_api = Interface pour communiquer avec le modèle de langage
        Ligne 2: embedding_model = Modèle pour convertir le texte en vecteurs numériques
        Ligne 3: file_processor = Gestionnaire pour traiter différents types de fichiers
        Ligne 4: dimension = Taille des vecteurs (768 pour BGE-base-en-v1.5)
        Lignes 5-7: Configuration de l'algorithme HNSW pour FAISS
        Lignes 8-10: Chemins des fichiers de sauvegarde
        """
        self.ollama_api = ollama_api                                           # Ligne 1
        self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')    # Ligne 2
        self.file_processor = FileProcessor(chunk_size, chunk_overlap)         # Ligne 3
        self.dimension = 768  # Correct pour BAAI/bge-base-en-v1.5             # Ligne 4
        self.MCNoeud = 32                                                      # Ligne 5
        self.efSearch = 100                                                    # Ligne 6
        self.efConstruction = 80                                               # Ligne 7
        self.faiss_index_file = faiss_index_file                              # Ligne 8
        self.metadata_file = metadata_file                                     # Ligne 9
        self.hashes_file = hashes_file                                         # Ligne 10

        # Chargement ou initialisation des composants
        self.index = self.load_or_initialize_index()                          # Ligne 11
        self.metadata = self.load_or_initialize_metadata()                    # Ligne 12
        self.load_processed_hashes()                                          # Ligne 13
```

### 3.3 Explication Détaillée : Initialisation

**Lignes 1-4 : Composants essentiels**
- `ollama_api` : Permet d'envoyer des requêtes au modèle de langage local
- `embedding_model` : Transforme le texte en vecteurs de 768 dimensions
- `file_processor` : Gère la lecture et le découpage des fichiers
- `dimension = 768` : Chaque mot/phrase devient un vecteur de 768 nombres

**Lignes 5-7 : Configuration HNSW (Hierarchical Navigable Small World)**
- `MCNoeud = 32` : Nombre de connexions par nœud dans le graphe
- `efSearch = 100` : Nombre de candidats explorés lors de la recherche
- `efConstruction = 80` : Paramètre de construction de l'index

**Lignes 8-13 : Gestion de la persistance**
- Les données sont sauvegardées sur disque pour éviter de tout recalculer
- `index` : Structure FAISS contenant les vecteurs
- `metadata` : Informations sur chaque chunk (texte, fichier source, etc.)
- `hashes` : Évite de traiter deux fois le même fichier

### 3.4 Code Complet : Chargement et Initialisation des Index

📁 **Fichier : `rag_chatbot.py` (méthodes de la classe RAGChatbot)**

```python
def load_or_initialize_index(self):
    """
    Charge l'index FAISS depuis le disque ou crée un nouveau si inexistant.
    
    Ligne 1: Vérification si le fichier d'index existe déjà
    Ligne 2: Si existe, chargement de l'index sauvegardé
    Ligne 4-6: Sinon, création d'un nouvel index HNSW avec produit scalaire
    Ligne 7-8: Configuration des paramètres de recherche et construction
    """
    if os.path.exists(self.faiss_index_file):                                # Ligne 1
        return faiss.read_index(self.faiss_index_file)                       # Ligne 2
    else:
        index = faiss.IndexHNSWFlat(self.dimension, self.MCNoeud,            # Ligne 4
                                   faiss.METRIC_INNER_PRODUCT)               # Ligne 5
        index.hnsw.efSearch = self.efSearch                                  # Ligne 6
        index.hnsw.efConstruction = self.efConstruction                      # Ligne 7
        return index                                                         # Ligne 8

def load_or_initialize_metadata(self):
    """
    Charge les métadonnées (informations sur chaque chunk) depuis le disque.
    
    Ligne 1: Vérification si le fichier de métadonnées existe
    Ligne 2-3: Si existe, chargement avec pickle (désérialisation)
    Ligne 5: Sinon, création d'une liste vide pour stocker les futures métadonnées
    """
    if os.path.exists(self.metadata_file):                                  # Ligne 1
        with open(self.metadata_file, 'rb') as f:                           # Ligne 2
            return pickle.load(f)                                           # Ligne 3
    else:
        return []                                                           # Ligne 5
```

### 3.5 Code Complet : Processus d'Ingestion des Documents

📁 **Fichier : `rag_chatbot.py` (méthode `ingestion_file` de la classe RAGChatbot)**

```python
def ingestion_file(self, base_filename, file_content, departement_id, filiere_id, 
                  module_id, activite_id, profile_id, user_id):
    """
    Traite un fichier complet : extraction, chunking, vectorisation, stockage.
    """
    try:
        # Étape 1: Traitement du fichier par le FileProcessor
        chunks, file_hash = self.file_processor.process_file(base_filename, file_content)  # Ligne 1
        
        # Vérifications de base
        if chunks is None:                                                   # Ligne 2
            print(f"Traitement de {base_filename} annulé (déjà traité).")    # Ligne 3
            if file_hash:                                                    # Ligne 4
                raise ValueError(f"Le fichier {base_filename} a déjà été traité.")  # Ligne 5
            else:
                raise ValueError(f"Le contenu du fichier {base_filename} n'a pas pu être traité.")  # Ligne 7
        
        if not chunks:                                                       # Ligne 8
            print(f"Aucun chunk extrait de {base_filename}.")               # Ligne 9
            return                                                           # Ligne 10

        # Étape 2: Génération des embeddings pour chaque chunk
        embeddings = []                                                      # Ligne 11
        for chunk_text_content in chunks:                                   # Ligne 12
            # Conversion du texte en vecteur numérique normalisé
            embedding = self.embedding_model.encode([chunk_text_content],    # Ligne 13
                                                  normalize_embeddings=True)[0]  # Ligne 14
            embeddings.append(embedding)                                    # Ligne 15

        if not embeddings:                                                   # Ligne 16
            print(f"Aucun embedding généré pour {base_filename}.")          # Ligne 17
            raise ValueError(f"Échec de la génération d'embeddings.")       # Ligne 18

        # Étape 3: Ajout des vecteurs à l'index FAISS
        embeddings_np = np.array(embeddings, dtype='float32')               # Ligne 19
        start_index = self.index.ntotal                                     # Ligne 20
        self.index.add(embeddings_np)                                       # Ligne 21

        # Étape 4: Sauvegarde des métadonnées pour chaque chunk
        for i, chunk_text_content in enumerate(chunks):                     # Ligne 22
            current_global_chunk_index = start_index + i                    # Ligne 23
            self.metadata.append({                                          # Ligne 24
                "file_hash": file_hash,                                     # Ligne 25
                "original_filename": base_filename,                         # Ligne 26
                "faiss_index": current_global_chunk_index,                  # Ligne 27
                "chunk_text": chunk_text_content,                          # Ligne 28
                "departement_id": departement_id,                          # Ligne 29
                "filiere_id": filiere_id,                                  # Ligne 30
                "module_id": module_id,                                    # Ligne 31
                "activite_id": activite_id,                                # Ligne 32
                "profile_id": profile_id,                                  # Ligne 33
                "user_id": user_id                                         # Ligne 34
            })
            
            # Sauvegarde dans la base de données SQLite
            FilterManager.insert_metadata_sqlite(                          # Ligne 35
                base_filename=base_filename, file_hash=file_hash,           # Ligne 36
                chunk_index=current_global_chunk_index,                     # Ligne 37
                chunk_text=chunk_text_content,                             # Ligne 38
                departement_id=departement_id, filiere_id=filiere_id,       # Ligne 39
                module_id=module_id, activite_id=activite_id,               # Ligne 40
                profile_id=profile_id, user_id=user_id                      # Ligne 41
            )
        
        # Sauvegarde de l'état complet
        self.save_state()                                                   # Ligne 42
        print(f"Fichier {base_filename} indexé. {len(chunks)} chunks ajoutés.")  # Ligne 43
        print(f"L'index Faiss contient maintenant {self.index.ntotal} vecteurs.")  # Ligne 44
        
    except Exception as e:
        print(f"Erreur lors de l'indexation de {base_filename} : {str(e)}")  # Ligne 45
        raise                                                               # Ligne 46
```

### 3.6 Explication Détaillée : Ingestion

**Lignes 1-10 : Traitement et vérifications**
- `process_file()` extrait le texte et le découpe en chunks
- `file_hash` permet de détecter les doublons
- Si `chunks is None`, le fichier a déjà été traité ou est vide

**Lignes 11-18 : Vectorisation**
- Chaque chunk de texte est transformé en vecteur de 768 dimensions
- `normalize_embeddings=True` normalise les vecteurs (longueur = 1)
- Cette normalisation améliore la qualité de la recherche par similarité

**Lignes 19-21 : Ajout à FAISS**
- Conversion en array NumPy avec type float32 (optimisé pour FAISS)
- `start_index` = position actuelle dans l'index pour numéroter les nouveaux vecteurs
- `add()` insère tous les vecteurs d'un coup (plus efficace)

**Lignes 22-41 : Métadonnées et traçabilité**
- Pour chaque chunk, on sauvegarde son texte et ses informations contextuelles
- `current_global_chunk_index` = identifiant unique dans FAISS
- Double sauvegarde : mémoire (pour rapidité) + SQLite (pour persistance)

### 3.7 Les Étapes du Processus RAG (suite)

#### 📥 **Étape 1 : Ingestion des Documents** ✅

Complétée ci-dessus avec le code détaillé.

#### ✂️ **Étape 2 : Chunking (Découpage en Segments)**

📁 **Fichier : `file_processor.py` (méthode `split_into_chunks` de la classe FileProcessor)**

```python
def split_into_chunks(self, text):
    """
    Découpe un texte long en segments plus petits avec chevauchement.
    
    Ligne 1: Vérification que le texte n'est pas vide
    Ligne 2: Initialisation de la liste des chunks
    Ligne 3: Position de début du découpage
    Ligne 4: Longueur totale du texte à traiter
    Ligne 5-10: Boucle de découpage avec chevauchement
    """
    if not text: return []                                                   # Ligne 1
    chunks = []                                                              # Ligne 2
    start = 0                                                                # Ligne 3
    text_length = len(text)                                                  # Ligne 4
    while start < text_length:                                               # Ligne 5
        end = start + self.chunk_size                                        # Ligne 6
        chunk = text[start:end].strip()                                      # Ligne 7
        if chunk:                                                            # Ligne 8
            chunks.append(chunk)                                             # Ligne 9
        start += self.chunk_size - self.chunk_overlap                        # Ligne 10
    return chunks                                                            # Ligne 11
```

### 3.8 Explication Détaillée : Chunking

**Ligne 6 : Calcul de la fin du chunk**
- `self.chunk_size = 384` caractères par défaut
- Chaque segment fait donc maximum 384 caractères

**Ligne 7 : Extraction et nettoyage**
- `text[start:end]` extrait la portion de texte
- `.strip()` supprime les espaces en début/fin

**Ligne 10 : Chevauchement intelligent**
- `self.chunk_overlap = 96` caractères
- Au lieu d'avancer de 384, on avance seulement de 384-96 = 288
- Les 96 derniers caractères du chunk précédent se retrouvent dans le suivant

**Pourquoi ce chevauchement ?**
1. **Préservation du contexte** : Évite de couper au milieu d'une phrase importante
2. **Meilleure recherche** : Une information à cheval sur deux chunks reste accessible
3. **Robustesse** : Compense les imperfections du découpage automatique

**Exemple concret :**
```
Document : "L'intelligence artificielle est une technologie révolutionnaire. Elle permet d'automatiser de nombreuses tâches complexes. Les applications sont multiples..."

Chunk 1 (0-384) : "L'intelligence artificielle est une technologie révolutionnaire. Elle permet d'automatiser de nombreuses tâches complexes. Les applications..."

Chunk 2 (288-672) : "...tâches complexes. Les applications sont multiples dans de nombreux domaines comme la médecine, la finance..."
                     ↑ Chevauchement de 96 caractères ↑
```

#### 🔢 **Étape 3 : Vectorisation (Embeddings)**

📁 **Fichier : `rag_chatbot.py` (dans la méthode `ingestion_file`)**

```python
# Dans la méthode ingestion_file, lignes 11-18
embeddings = []                                                      # Ligne 11
for chunk_text_content in chunks:                                   # Ligne 12
    # Conversion du texte en vecteur numérique normalisé
    embedding = self.embedding_model.encode([chunk_text_content],    # Ligne 13
                                          normalize_embeddings=True)[0]  # Ligne 14
    embeddings.append(embedding)                                    # Ligne 15
```

### 3.9 Explication Détaillée : Vectorisation (Embeddings)

**Ligne 11 : Initialisation de la liste**
- `embeddings = []` crée une liste vide pour stocker tous les vecteurs

**Ligne 12 : Boucle sur chaque chunk**
- Pour chaque segment de texte, nous allons créer un vecteur numérique

**Lignes 13-14 : Transformation en vecteur**
- `self.embedding_model` utilise le modèle `BAAI/bge-base-en-v1.5`
- `encode([chunk_text_content])` convertit le texte en vecteur
- `normalize_embeddings=True` normalise le vecteur (longueur = 1)
- `[0]` récupère le premier (et unique) vecteur du résultat

**Ligne 15 : Stockage**
- Ajoute le vecteur à la liste pour traitement ultérieur

**Qu'est-ce qu'un embedding concrètement ?**
```
Texte : "L'intelligence artificielle"
Vecteur : [0.23, -0.15, 0.67, 0.12, 0.45, ..., 0.34]  (768 nombres)

Texte similaire : "L'IA et machine learning"  
Vecteur : [0.21, -0.18, 0.69, 0.09, 0.48, ..., 0.29]  (proche du précédent)
```

### 3.10 Code Complet : Recherche avec MMR

📁 **Fichier : `rag_chatbot.py` (méthode `find_relevant_context` de la classe RAGChatbot)**

```python
def find_relevant_context(self, user_query, departement_id=None, filiere_id=None,
                          top_k=3, similarity_threshold=0.65, use_mmr=False, mmr_lambda=0.5):
    """
    Recherche les chunks les plus pertinents pour une requête utilisateur.
    Utilise le filtrage académique et optionnellement MMR pour la diversité.
    """
    try:
        # Étape 1: Conversion de la question en vecteur
        query_embedding = self.embedding_model.encode([user_query],          # Ligne 1
                                                    normalize_embeddings=True)[0]  # Ligne 2
        normalized_query = query_embedding.reshape(1, -1)                   # Ligne 3
    except Exception as e:
        print(f"Erreur lors de la génération de l'embedding: {e}")          # Ligne 4
        return None                                                         # Ligne 5

    # Étape 2: Vérification que l'index FAISS contient des données
    if not hasattr(self.index, 'ntotal') or self.index.ntotal == 0:        # Ligne 6
        print("Aucun contexte indexé dans Faiss.")                         # Ligne 7
        return None                                                         # Ligne 8

    try:
        # Étape 3: Récupération des chunks autorisés selon les filtres académiques
        allowed_faiss_ids = FilterManager.get_allowed_indices(              # Ligne 9
            departement_id, filiere_id                                      # Ligne 10
        )
    except Exception as e:
        print(f"Erreur lors de la récupération des IDs autorisés: {e}")    # Ligne 11
        return None                                                         # Ligne 12

    if not allowed_faiss_ids:                                               # Ligne 13
        print("Aucun chunk autorisé trouvé.")                              # Ligne 14
        return None                                                         # Ligne 15

    # Étape 4: Préparation des IDs valides pour la reconstruction
    allowed_faiss_ids_array = np.array(list(allowed_faiss_ids), dtype='int64')  # Ligne 16
    valid_ids_for_reconstruction = np.array(                               # Ligne 17
        [fid for fid in allowed_faiss_ids_array if 0 <= fid < self.index.ntotal],  # Ligne 18
        dtype='int64'                                                       # Ligne 19
    )

    if len(valid_ids_for_reconstruction) == 0:                              # Ligne 20
        print("Aucun ID Faiss valide dans l'index principal.")             # Ligne 21
        return None                                                         # Ligne 22

    # Étape 5: Reconstruction des vecteurs depuis FAISS
    sub_vectors_list = []                                                   # Ligne 23
    try:
        reconstruction_source = self.index.storage                         # Ligne 24
        for vector_id in valid_ids_for_reconstruction:                      # Ligne 25
            reconstructed_vec = reconstruction_source.reconstruct(int(vector_id))  # Ligne 26
            sub_vectors_list.append(reconstructed_vec)                      # Ligne 27
        
        sub_vectors = np.array(sub_vectors_list, dtype='float32')           # Ligne 28
    except Exception as e:
        print(f"Erreur lors de la reconstruction des vecteurs: {e}")       # Ligne 29
        return None                                                         # Ligne 30

    # Étape 6: Création d'un sous-index pour la recherche
    try:
        sub_index = faiss.IndexFlatIP(self.dimension)                       # Ligne 31
        sub_index.add(sub_vectors)                                          # Ligne 32
    except Exception as e:
        print(f"Erreur lors de la création du sous-index: {e}")            # Ligne 33
        return None                                                         # Ligne 34

    # Étape 7: Recherche par similarité
    try:
        k_search = min(top_k * 5 if use_mmr else top_k, sub_index.ntotal)   # Ligne 35
        distances, local_indices_in_sub = sub_index.search(normalized_query, k_search)  # Ligne 36
    except Exception as e:
        print(f"Erreur lors de la recherche: {e}")                         # Ligne 37
        return None                                                         # Ligne 38

    # Étape 8: Filtrage par seuil de similarité et application de MMR si demandé
    candidate_embeddings = [sub_vectors[i] for i in local_indices_in_sub[0]  # Ligne 39
                            if i >= 0 and distances[0][list(local_indices_in_sub[0]).index(i)] >= similarity_threshold]  # Ligne 40
    candidate_indices = [i for i in local_indices_in_sub[0]                 # Ligne 41
                        if i >= 0 and distances[0][list(local_indices_in_sub[0]).index(i)] >= similarity_threshold]  # Ligne 42

    if use_mmr and candidate_embeddings:                                    # Ligne 43
        mmr_indices = self.mmr(normalized_query.flatten(), candidate_embeddings,  # Ligne 44
                              k=top_k, lambda_param=mmr_lambda)            # Ligne 45
        selected_indices = [candidate_indices[i] for i in mmr_indices]     # Ligne 46
    else:
        selected_indices = candidate_indices[:top_k]                        # Ligne 47

    # Étape 9: Récupération des textes correspondants
    relevant_chunks_texts = []                                              # Ligne 48
    for i in selected_indices:                                              # Ligne 49
        global_faiss_id = valid_ids_for_reconstruction[i]                   # Ligne 50
        chunk_data = None                                                   # Ligne 51
        for meta_item in self.metadata:                                     # Ligne 52
            if meta_item.get("faiss_index") == global_faiss_id:             # Ligne 53
                chunk_data = meta_item                                      # Ligne 54
                break                                                       # Ligne 55
        if chunk_data and 'chunk_text' in chunk_data:                       # Ligne 56
            relevant_chunks_texts.append(chunk_data['chunk_text'])          # Ligne 57

    if not relevant_chunks_texts:                                           # Ligne 58
        print("Aucun chunk pertinent trouvé.")                             # Ligne 59
        return None                                                         # Ligne 60
    
    return relevant_chunks_texts                                            # Ligne 61
```

### 3.11 Explication Détaillée : Recherche avec Filtrage Académique

**Lignes 1-5 : Conversion de la question en vecteur**
- `user_query` (ex: "Qu'est-ce que l'IA?") → vecteur de 768 dimensions
- `normalize_embeddings=True` normalise le vecteur pour une meilleure recherche
- `reshape(1, -1)` transforme en format attendu par FAISS (matrice 1×768)

**Lignes 6-8 : Vérification de l'index**
- `self.index.ntotal` = nombre total de vecteurs stockés
- Si 0, aucun document n'a été indexé → impossible de répondre

**Lignes 9-15 : Filtrage académique**
- `FilterManager.get_allowed_indices()` récupère les IDs autorisés
- Selon le département (ex: Informatique) et filière (ex: Génie Logiciel)
- Seuls les chunks pertinents pour l'étudiant sont considérés

**Lignes 16-22 : Validation des IDs**
- Conversion en array NumPy pour optimisation
- Filtrage des IDs valides (0 ≤ ID < nombre total de vecteurs)
- Protection contre les erreurs d'index

**Lignes 23-30 : Reconstruction des vecteurs**
- `self.index.storage` = zone de stockage des vecteurs dans FAISS
- `reconstruct(vector_id)` récupère le vecteur original depuis son ID
- Nécessaire car on ne peut pas chercher directement dans un sous-ensemble

**Lignes 31-34 : Création du sous-index**
- `IndexFlatIP` = index plat avec produit scalaire (Inner Product)
- Contient uniquement les vecteurs autorisés pour l'étudiant
- Plus rapide que filtrer après la recherche

**Lignes 35-38 : Recherche par similarité**
- `k_search` = nombre de candidats (×5 si MMR activé)
- `search()` trouve les vecteurs les plus similaires à la question
- Retourne distances (scores de similarité) et indices

**Lignes 39-47 : Application de MMR (optionnelle)**
- Filtrage par seuil (`similarity_threshold = 0.65`)
- Si MMR activé : diversification des résultats
- Sinon : prise des `top_k` plus pertinents

**Lignes 48-61 : Récupération des textes**
- Pour chaque indice sélectionné → récupération du texte original
- Recherche dans `self.metadata` pour retrouver le contenu textuel
- Retour de la liste des chunks pertinents

### 3.12 Code Complet : Algorithme MMR (Maximal Marginal Relevance)

📁 **Fichier : `rag_chatbot.py` (méthode `mmr` de la classe RAGChatbot)**

```python
def mmr(self, query_embedding, candidate_embeddings, k=3, lambda_param=0.5):
    """
    Maximal Marginal Relevance pour diversifier les chunks sélectionnés.
    
    query_embedding: vecteur de la question (shape: dim,)
    candidate_embeddings: liste des vecteurs candidats (shape: n, dim)
    k: nombre de chunks à retourner
    lambda_param: balance pertinence/diversité (0.0=diversité max, 1.0=pertinence max)
    """
    selected = []                                                           # Ligne 1
    selected_indices = []                                                   # Ligne 2
    candidate_indices = list(range(len(candidate_embeddings)))             # Ligne 3
    query_embedding = query_embedding.reshape(1, -1)                       # Ligne 4
    candidate_embeddings = np.array(candidate_embeddings)                  # Ligne 5
    
    # Calcul de la similarité entre la question et chaque candidat
    sim_to_query = np.dot(candidate_embeddings, query_embedding.T).flatten()  # Ligne 6
    
    # Sélection itérative des meilleurs candidats
    for _ in range(min(k, len(candidate_embeddings))):                      # Ligne 7
        if not selected:                                                    # Ligne 8
            # Premier candidat : le plus pertinent
            idx = int(np.argmax(sim_to_query))                              # Ligne 9
            selected.append(candidate_embeddings[idx])                      # Ligne 10
            selected_indices.append(idx)                                    # Ligne 11
            candidate_indices.remove(idx)                                   # Ligne 12
        else:
            # Candidats suivants : équilibre pertinence/diversité
            max_score = -np.inf                                             # Ligne 13
            max_idx = -1                                                    # Ligne 14
            
            for idx in candidate_indices:                                   # Ligne 15
                # Calcul de la pertinence par rapport à la question
                relevance = sim_to_query[idx]                               # Ligne 16
                
                # Calcul de la diversité (max similarité avec les sélectionnés)
                diversity = max([np.dot(candidate_embeddings[idx], s)       # Ligne 17
                               for s in selected])                          # Ligne 18
                
                # Score MMR = équilibre pertinence - diversité
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity  # Ligne 19
                
                # Mémoriser le meilleur candidat
                if mmr_score > max_score:                                   # Ligne 20
                    max_score = mmr_score                                   # Ligne 21
                    max_idx = idx                                           # Ligne 22
            
            # Ajouter le meilleur candidat à la sélection
            selected.append(candidate_embeddings[max_idx])                  # Ligne 23
            selected_indices.append(max_idx)                               # Ligne 24
            candidate_indices.remove(max_idx)                              # Ligne 25
    
    return selected_indices                                                 # Ligne 26
```

### 3.13 Explication Détaillée : Algorithme MMR

**Lignes 1-6 : Initialisation**
- `selected` : liste des vecteurs déjà choisis
- `selected_indices` : indices des vecteurs choisis 
- `candidate_indices` : liste des candidats restants (0, 1, 2, ...)
- `sim_to_query` : similarité de chaque candidat avec la question

**Ligne 6 : Calcul des similarités**
- `np.dot()` calcule le produit scalaire (mesure de similarité)
- Plus le score est élevé, plus le candidat est pertinent

**Lignes 8-12 : Premier candidat**
- Pour le premier choix, on prend simplement le plus pertinent
- `np.argmax()` trouve l'indice du maximum
- On l'ajoute aux sélectionnés et le retire des candidats

**Lignes 13-25 : Candidats suivants**
- Pour chaque candidat restant, calcul du score MMR
- `relevance` = à quel point il répond à la question
- `diversity` = à quel point il est différent des déjà sélectionnés
- `mmr_score` = équilibre entre les deux selon `lambda_param`

**La formule MMR :**
```
Score = λ × Pertinence - (1-λ) × Diversité

λ = 1.0 : Seulement la pertinence compte
λ = 0.0 : Seulement la diversité compte  
λ = 0.5 : Équilibre 50/50
```

**Exemple pratique :**

Question : "Qu'est-ce que le machine learning ?"

Candidats trouvés :
1. "Le machine learning est une branche de l'IA..." (Pertinence: 0.9)
2. "Le ML utilise des algorithmes d'apprentissage..." (Pertinence: 0.85, Similarité avec 1: 0.8)
3. "Les applications ML incluent la vision par ordinateur..." (Pertinence: 0.7, Similarité avec 1: 0.3)

Avec λ = 0.5 :
- Candidat 1 sélectionné en premier (plus pertinent)
- Candidat 2 : Score = 0.5×0.85 - 0.5×0.8 = 0.025
- Candidat 3 : Score = 0.5×0.7 - 0.5×0.3 = 0.2

→ Candidat 3 sélectionné (plus diversifié)

### 3.14 Code Complet : Génération de Réponse

📁 **Fichier : `rag_chatbot.py` (méthode `generate_response` de la classe RAGChatbot)**

```python
def generate_response(self, user_query, user_id, profile_id, departement_id, filiere_id, use_mmr=False):
    """
    Génère une réponse complète à partir de la question utilisateur.
    Intègre recherche RAG, construction de prompt et génération LLM.
    """
    INVITE_PROFILE_ID = 5                                                   # Ligne 1

    # Gestion du profil invité avec restrictions
    is_invite = (profile_id == INVITE_PROFILE_ID)                          # Ligne 2
    if is_invite:                                                           # Ligne 3
        departement_id = 1  # Département Scolarité uniquement             # Ligne 4
        filiere_id = 3                                                      # Ligne 5

    # Étape 1: Recherche des chunks pertinents avec RAG
    context_chunks = self.find_relevant_context(                           # Ligne 6
        user_query, departement_id, filiere_id,                            # Ligne 7
        top_k=3, similarity_threshold=0.65, use_mmr=use_mmr                # Ligne 8
    )

    # Étape 2: Construction du prompt selon le type de demande
    if context_chunks:                                                      # Ligne 9
        context_text = "\n".join(context_chunks)                           # Ligne 10
        
        # Détection du type de demande
        if PromptBuilder.is_qcm_request(user_query):                        # Ligne 11
            prompt_text = PromptBuilder.build_qcm_prompt(context_text, user_query)  # Ligne 12
        elif "résumé" in user_query.lower() or "synthèse" in user_query.lower():   # Ligne 13
            prompt_text = PromptBuilder.build_summary_prompt(context_text, user_query)  # Ligne 14
        else:
            prompt_text = PromptBuilder.build_standard_prompt(context_text, user_query)  # Ligne 15

        # Gestion spéciale pour les invités
        if is_invite:                                                       # Ligne 16
            prompt_text = (                                                 # Ligne 17
                "⚠️ **Mode invité activé :**\n"                            # Ligne 18
                "Tu dois répondre **uniquement** à partir des informations fournies par le département Scolarité.\n"  # Ligne 19
                "Si la question ne concerne pas le département Scolarité, ou si tu n'as pas l'information dans le contexte, réponds exactement :\n"  # Ligne 20
                "\"Je ne peux répondre qu'aux questions relatives au département Scolarité.\"\n"  # Ligne 21
                "N'ajoute rien d'autre, n'explique pas, ne propose pas de ressources, ne donne aucune information supplémentaire.\n"  # Ligne 22
                "N'utilise **aucune autre information**, même si elle est présente dans le contexte.\n"  # Ligne 23
                "N'intègre pas de ressources supplémentaires extérieures au département Scolarité.\n\n"  # Ligne 24
                "**Gestion des Ressources :**\n"                           # Ligne 25
                "La section « 📚 Pour Aller Plus Loin » doit apparaître **uniquement si des ressources pertinentes sont disponibles** dans le contexte du département Scolarité.\n"  # Ligne 26
                "Si aucune ressource n'est pertinente, **n'inclus pas cette section**.\n\n"  # Ligne 27
                "**Important :**\n"                                         # Ligne 28
                "Ne réponds à aucune question qui ne concerne pas le département Scolarité. Ignore toute demande hors de ce périmètre.\n"  # Ligne 29
                "Exemple :\n"                                               # Ligne 30
                "Q : Qu'est-ce que PySpark ?\n"                            # Ligne 31
                "R : Je ne peux répondre qu'aux questions relatives au département Scolarité.\n"  # Ligne 32
            ) + prompt_text                                                 # Ligne 33

    else:
        # Aucun contexte trouvé
        prompt_text = (                                                     # Ligne 34
            f"Question : {user_query}\n"                                    # Ligne 35
            "Réponds que tu n'as pas d'informations pertinentes pour cette question."  # Ligne 36
        )

    # Étape 3: Génération de la réponse avec le LLM
    llm_raw_response = self.ollama_api.chat_with_ollama(prompt_text)        # Ligne 37
    cleaned_llm_response = self.clean_llm_response(llm_raw_response)        # Ligne 38

    # Étape 4: Sauvegarde dans l'historique
    chat_id = FilterManager.save_chat_history(                             # Ligne 39
        user_id=user_id,                                                    # Ligne 40
        question=user_query,                                                # Ligne 41
        answer=cleaned_llm_response,                                        # Ligne 42
        departement_id=departement_id,                                      # Ligne 43
        filiere_id=filiere_id,                                              # Ligne 44
        profile_id=profile_id,                                              # Ligne 45
        db_path=self.SQLITE_DB_PATH                                         # Ligne 46
    )

    return cleaned_llm_response, chat_id                                    # Ligne 47
```

### 3.15 Code Complet : Nettoyage des Réponses LLM

📁 **Fichier : `rag_chatbot.py` (méthode `clean_llm_response` de la classe RAGChatbot)**

```python
def clean_llm_response(self, response: str) -> str:
    """
    Nettoie la réponse brute du LLM en supprimant les balises indésirables.
    
    Ligne 1: Extraction après la dernière balise de fermeture
    Lignes 2-3: Suppression des balises <think> (mode réflexion du LLM)
    Ligne 4: Suppression des espaces en début/fin
    """
    # Extraire le contenu après la dernière balise de fermeture
    if '</' in response:                                                    # Ligne 1
        response = response.split('</')[-1]                                 # Ligne 2

    # Nettoyage des balises de réflexion
    cleaned = re.sub(r"<?think>", "", response, flags=re.DOTALL | re.IGNORECASE)  # Ligne 3
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)       # Ligne 4
    
    return cleaned.strip()                                                  # Ligne 5
```

### 3.16 Explication Détaillée : Génération de Réponse

**Lignes 1-5 : Gestion des profils utilisateur**
- `INVITE_PROFILE_ID = 5` : identifiant du profil invité
- Les invités sont limités au département Scolarité uniquement
- Empêche l'accès aux contenus techniques ou spécialisés

**Lignes 6-8 : Recherche RAG**
- Appel de `find_relevant_context()` avec tous les paramètres
- `top_k=3` : maximum 3 chunks retournés
- `similarity_threshold=0.65` : seuil de pertinence
- `use_mmr` : activation optionnelle de la diversification

**Lignes 9-15 : Construction du prompt adaptatif**
- `context_text` : concaténation des chunks trouvés
- Détection automatique du type de demande :
  - QCM : mots-clés "qcm", "quiz", "choix multiple"
  - Résumé : mots "résumé", "synthèse"
  - Standard : autres demandes

**Lignes 16-33 : Restrictions pour invités**
- Prompt spécialisé avec consignes strictes
- Limitation au département Scolarité uniquement
- Réponse standardisée si hors périmètre
- Contrôle de l'affichage des ressources

**Lignes 34-36 : Gestion absence de contexte**
- Si aucun chunk pertinent trouvé
- Réponse polie indiquant l'absence d'information

**Lignes 37-38 : Génération et nettoyage**
- `chat_with_ollama()` : appel au modèle de langage
- `clean_llm_response()` : suppression des artéfacts

**Lignes 39-47 : Traçabilité**
- Sauvegarde de la conversation en base
- Retour de la réponse et de l'ID de la conversation

### 3.17 Code Complet : Traitement des Fichiers

📁 **Fichier : `file_processor.py` (classe FileProcessor)**

```python
import os
import hashlib
import json
import PyPDF2
from docx import Document
import io
from pptx import Presentation
from Utilitaire.EDA_Cleaner import TextPipeline, TextCleaner

class FileProcessor:
    def __init__(self, chunk_size=384, chunk_overlap=96):
        """
        Initialise le processeur de fichiers.
        
        chunk_size: taille maximum d'un segment en caractères
        chunk_overlap: chevauchement entre segments en caractères
        processed_hashes: ensemble des fichiers déjà traités
        """
        self.chunk_size = chunk_size                                        # Ligne 1
        self.chunk_overlap = chunk_overlap                                  # Ligne 2
        self.processed_hashes = set()                                       # Ligne 3

    def calculate_hash(self, content_string):
        """
        Calcule un hash SHA-256 du contenu pour détecter les doublons.
        
        Ligne 1: Encodage du texte en UTF-8
        Ligne 2: Calcul du hash et conversion en hexadécimal
        """
        return hashlib.sha256(content_string.encode('utf-8')).hexdigest()   # Ligne 1

    def read_content_from_bytes(self, base_filename, file_content_bytes):
        """
        Convertit le contenu d'un fichier (bytes) en string selon son extension.
        
        Ligne 1: Extraction de l'extension du fichier
        Lignes 3-8: Traitement des fichiers texte
        Lignes 9-15: Traitement des PDF
        Lignes 16-20: Traitement des DOCX
        Lignes 21-24: Traitement des JSON
        Lignes 25-31: Traitement des PPTX
        """
        file_extension = os.path.splitext(base_filename)[1].lower()         # Ligne 1
        content_str = ""                                                    # Ligne 2

        try:
            if file_extension == '.txt':                                    # Ligne 3
                content_str = file_content_bytes.decode('utf-8')            # Ligne 4
                
            elif file_extension == '.pdf':                                  # Ligne 5
                pdf_stream = io.BytesIO(file_content_bytes)                 # Ligne 6
                pdf_reader = PyPDF2.PdfReader(pdf_stream)                   # Ligne 7
                for page in pdf_reader.pages:                               # Ligne 8
                    page_text = page.extract_text()                        # Ligne 9
                    if page_text:                                           # Ligne 10
                        content_str += page_text + "\n"                    # Ligne 11
                        
            elif file_extension == '.docx':                                 # Ligne 12
                docx_stream = io.BytesIO(file_content_bytes)                # Ligne 13
                doc = Document(docx_stream)                                 # Ligne 14
                for para in doc.paragraphs:                                 # Ligne 15
                    content_str += para.text + "\n"                        # Ligne 16
                    
            elif file_extension == '.json':                                 # Ligne 17
                json_string = file_content_bytes.decode('utf-8')            # Ligne 18
                data = json.loads(json_string)                              # Ligne 19
                content_str = json.dumps(data, ensure_ascii=False)          # Ligne 20
                
            elif file_extension == '.pptx':                                 # Ligne 21
                pptx_stream = io.BytesIO(file_content_bytes)                # Ligne 22
                prs = Presentation(pptx_stream)                             # Ligne 23
                for slide in prs.slides:                                    # Ligne 24
                    for shape in slide.shapes:                              # Ligne 25
                        if hasattr(shape, "text"):                          # Ligne 26
                            content_str += shape.text + "\n"               # Ligne 27
            else:
                raise ValueError(f"Type de fichier non supporté : {file_extension}")  # Ligne 28

            return content_str.strip()                                      # Ligne 29
            
        except Exception as e:
            raise Exception(f"Erreur lecture {base_filename} ({file_extension}) : {str(e)}")  # Ligne 30

    def process_file(self, base_filename, file_content_bytes):
        """
        Traite complètement un fichier : lecture, hash, nettoyage, chunking.
        
        Retourne (chunks, file_hash) ou (None, file_hash) si déjà traité
        """
        try:
            # Étape 1: Extraction du contenu textuel
            content_string = self.read_content_from_bytes(base_filename, file_content_bytes)  # Ligne 1

            if not content_string:                                          # Ligne 2
                print(f"Aucun contenu textuel extrait de {base_filename}")  # Ligne 3
                return None, None                                           # Ligne 4

            # Étape 2: Calcul du hash pour détecter les doublons
            file_hash = self.calculate_hash(content_string)                 # Ligne 5

            if file_hash in self.processed_hashes:                          # Ligne 6
                print(f"Fichier {base_filename} déjà traité (hash: {file_hash})")  # Ligne 7
                return None, file_hash                                      # Ligne 8

            # Étape 3: Nettoyage du texte avec pipeline EDA
            cleaner = TextCleaner()                                         # Ligne 9
            pipeline = TextPipeline(cleaner)                                # Ligne 10
            cleaned_content = pipeline.process(content_string)              # Ligne 11

            if not cleaned_content:                                         # Ligne 12
                print(f"Contenu vide après nettoyage pour {base_filename}") # Ligne 13
                return None, file_hash                                      # Ligne 14

            # Étape 4: Découpage en chunks
            chunks = self.split_into_chunks(cleaned_content)                # Ligne 15

            if not chunks:                                                  # Ligne 16
                print(f"Aucun chunk généré pour {base_filename}")          # Ligne 17
                return None, file_hash                                      # Ligne 18

            # Si tout s'est bien passé
            self.processed_hashes.add(file_hash)                            # Ligne 19
            return chunks, file_hash                                        # Ligne 20

        except Exception as e:
            print(f"Erreur dans FileProcessor pour {base_filename}: {str(e)}")  # Ligne 21
            raise                                                           # Ligne 22
```

---

## 4. Analyse de Sentiment {#sentiment}

### 4.1 Objectif de l'Analyse de Sentiment

L'analyse de sentiment permet de :
- **Comprendre** la satisfaction des étudiants
- **Identifier** les points d'amélioration
- **Adapter** le système selon les retours

### 4.2 Code Complet : Classe SimpleSentimentAnalyzer

📁 **Fichier : `sentiment_analyzer.py` (classe principale)**

```python
import pandas as pd
from ollama_api import OllamaAPI
from sqlalchemy import create_engine, text
import json
import os
import time
import logging
from dotenv import load_dotenv
from textblob import TextBlob
from langchain_groq import ChatGroq

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = "sqlite:///./bdd/chatbot_metadata.db"

SQL_QUERY = """
SELECT id as ID, Département, Filière, Profile, "User", FeedBack, "timestamp"
FROM "V_FeedBack"
WHERE polarity IS NULL
"""

class SimpleSentimentAnalyzer:
    def __init__(self):
        """
        Initialise l'analyseur de sentiment avec support multi-méthodes.
        
        Ligne 1: Initialisation de la variable LLM
        Lignes 2-14: Configuration du LLM Groq si disponible
        """
        self.llm = None                                                         # Ligne 1
        if GROQ_API_KEY:                                                        # Ligne 2
            try:
                self.llm = ChatGroq(                                            # Ligne 3
                    model_name="llama3-8b-8192",                               # Ligne 4
                    temperature=0,                                              # Ligne 5
                    api_key=GROQ_API_KEY,                                       # Ligne 6
                    max_retries=2,                                              # Ligne 7
                    request_timeout=30,                                         # Ligne 8
                    max_tokens=200                                              # Ligne 9
                )
                logger.info("LLM Groq initialisé avec succès")                 # Ligne 10
            except Exception as e:
                logger.warning(f"Impossible d'initialiser Groq: {e}")          # Ligne 11
                self.llm = None                                                 # Ligne 12

    def analyze_sentiment_rule_based(self, feedback_text):
        """
        Analyse de sentiment basée sur des règles linguistiques.
        Méthode principale et la plus fiable.
        """
        feedback_lower = feedback_text.lower()                                 # Ligne 1
        
        # Définition des mots-clés pour détecter le contenu pédagogique
        content_keywords = [                                                    # Ligne 2
            'contenu', 'cours', 'leçon', 'résumé', 'explication', 'pédagogique',
            'matière', 'sujet', 'chapitre', 'module', 'formation', 'apprentissage',
            'réponse', 'solution', 'information', 'connaissance', 'enseignement'
        ]
        
        # Mots-clés positifs
        positive_keywords = [                                                   # Ligne 3
            'génial', 'super', 'magnifique', 'excellent', 'parfait', 'bien', 'bon',
            'merci', 'formidable', 'fantastique', 'top', 'bravo', 'impressionnant',
            'efficace', 'utile', 'clair', 'précis', 'complet', 'satisfait', 
            'agréable', 'intéressant', 'captivant', 'engageant', 'instructif', 'pratique',
            'facile à comprendre', 'bien structuré', 'bien expliqué',
            'très utile', 'très clair', 'très complet', 'très intéressant', 'très engageant'
        ]
        
        # Mots-clés négatifs
        negative_keywords = [                                                   # Ligne 4
            'mauvais', 'nul', 'horrible', 'décevant', 'problème', 'erreur',
            'difficile', 'compliqué', 'confus', 'incompréhensible', 'long',
            'court', 'bref', 'insuffisant', 'pas convaincu',
            'peu clair', 'peu précis', 'peu complet', 'peu intéressant',
            'peu engageant', 'peu utile', 'peu pratique', 'pas utile',
            'pas pratique', 'pas efficace', 'pas satisfaisant', 'pas agréable',
        ]
        
        # Mots-clés neutres (critiques constructives)
        neutral_keywords = [                                                    # Ligne 5
            'correcte', 'acceptable', 'moyen', 'peut mieux faire', 'améliorer',
            'manque de', 'besoin de', 'pourrait être mieux', 'manque de clarté',
            'manque de détails', "manque d'exemples", 'manque de structure'
        ]
        
        # Vérification si le contenu pédagogique est mentionné
        has_content = any(keyword in feedback_lower for keyword in content_keywords)  # Ligne 6
        if not has_content:                                                     # Ligne 7
            return {"aspect": "Contenu pédagogique", "polarity": "non mentionné"}  # Ligne 8
        
        # Comptage des occurrences
        positive_count = sum(1 for keyword in positive_keywords                 # Ligne 9
                           if keyword in feedback_lower)                        # Ligne 10
        negative_count = sum(1 for keyword in negative_keywords                 # Ligne 11
                           if keyword in feedback_lower)                        # Ligne 12
        neutral_count = sum(1 for keyword in neutral_keywords                   # Ligne 13
                          if keyword in feedback_lower)                         # Ligne 14
        
        # Gestion des cas spéciaux
        if 'trop long' in feedback_lower or 'trop brève' in feedback_lower:     # Ligne 15
            negative_count += 1                                                 # Ligne 16
        
        if 'préfère' in feedback_lower and 'résumé' in feedback_lower:          # Ligne 17
            neutral_count += 1                                                  # Ligne 18
        
        # Décision finale basée sur les comptages
        if positive_count > negative_count and positive_count > neutral_count:  # Ligne 19
            polarity = "positive"                                               # Ligne 20
        elif negative_count > positive_count and negative_count > neutral_count:  # Ligne 21
            polarity = "négative"                                               # Ligne 22
        elif neutral_count > 0:                                                 # Ligne 23
            polarity = "neutre"                                                 # Ligne 24
        else:
            polarity = "neutre"                                                 # Ligne 25
        
        logger.debug(f"Analyse: pos={positive_count}, neg={negative_count}, neu={neutral_count} -> {polarity}")  # Ligne 26
        return {"aspect": "Contenu pédagogique", "polarity": polarity}          # Ligne 27

    def analyze_sentiment_textblob(self, feedback_text):
        """
        Analyse de sentiment avec TextBlob (méthode de backup).
        Utilisée si les règles ne donnent pas de résultat satisfaisant.
        """
        try:
            blob = TextBlob(feedback_text)                                      # Ligne 1
            sentiment_score = blob.sentiment.polarity                          # Ligne 2
            
            # Vérification du contenu pédagogique
            content_keywords = ['contenu', 'cours', 'réponse', 'explication', 'résumé']  # Ligne 3
            has_content = any(keyword in feedback_text.lower()                  # Ligne 4
                            for keyword in content_keywords)                    # Ligne 5
            
            if not has_content:                                                 # Ligne 6
                return {"aspect": "Contenu pédagogique", "polarity": "non mentionné"}  # Ligne 7
            
            # Classification selon le score
            if sentiment_score > 0.1:                                           # Ligne 8
                polarity = "positive"                                           # Ligne 9
            elif sentiment_score < -0.1:                                        # Ligne 10
                polarity = "négative"                                           # Ligne 11
            else:
                polarity = "neutre"                                             # Ligne 12
            
            return {"aspect": "Contenu pédagogique", "polarity": polarity}      # Ligne 13
            
        except:
            return self.analyze_sentiment_rule_based(feedback_text)             # Ligne 14

    def analyze_sentiment_llm(self, feedback_text):
        """
        Analyse avec LLM externe (Groq/Ollama) si disponible.
        Méthode la plus sophistiquée mais optionnelle.
        """
        if not self.llm:                                                        # Ligne 1
            return None                                                         # Ligne 2
        
        try:
            prompt = f"""Analyse le sentiment de ce feedback concernant le contenu pédagogique.

Feedback: "{feedback_text}"

Réponds UNIQUEMENT avec un JSON valide:
{{"aspect": "Contenu pédagogique", "polarity": "positive/négative/neutre/non mentionné"}}

Si le feedback ne mentionne pas le contenu pédagogique, utilise "non mentionné".
"""                                                                             # Ligne 3
            
            ollama_api = OllamaAPI()                                            # Ligne 4
            response = ollama_api.chat_with_ollama(prompt)                      # Ligne 5
            response_text = response.content if hasattr(response, 'content') else str(response)  # Ligne 6
            
            # Extraction du JSON de la réponse
            start_idx = response_text.find('{')                                 # Ligne 7
            end_idx = response_text.rfind('}')                                  # Ligne 8
            
            if start_idx != -1 and end_idx != -1:                              # Ligne 9
                json_str = response_text[start_idx:end_idx+1]                   # Ligne 10
                result = json.loads(json_str)                                   # Ligne 11
                
                if 'aspect' in result and 'polarity' in result:                 # Ligne 12
                    return result                                               # Ligne 13
        
        except Exception as e:
            logger.warning(f"Erreur LLM: {e}")                                  # Ligne 14
        
        return None                                                             # Ligne 15

    def analyze_single_feedback(self, feedback_text):
        """
        Analyse un feedback avec cascade de méthodes (LLM → Règles → TextBlob).
        
        Ligne 1-2: Vérification que le feedback n'est pas vide
        Ligne 3: Nettoyage et limitation à 500 caractères
        Lignes 4-8: Tentative d'analyse avec LLM
        Ligne 9: Fallback sur analyse par règles
        """
        if not feedback_text or not feedback_text.strip():                     # Ligne 1
            return {"aspect": "Contenu pédagogique", "polarity": "vide"}        # Ligne 2
        
        cleaned_feedback = feedback_text.strip()[:500]                         # Ligne 3
        
        # Méthode 1: LLM (si disponible)
        if self.llm:                                                            # Ligne 4
            llm_result = self.analyze_sentiment_llm(cleaned_feedback)           # Ligne 5
            if llm_result:                                                      # Ligne 6
                logger.info(f"Analyse LLM réussie: {llm_result}")               # Ligne 7
                return llm_result                                               # Ligne 8
        
        # Méthode 2: Règles (principale et plus fiable)
        rule_result = self.analyze_sentiment_rule_based(cleaned_feedback)       # Ligne 9
        logger.info(f"Analyse par règles: {rule_result}")                       # Ligne 10
        return rule_result                                                      # Ligne 11
```

### 4.3 Explication Détaillée : Architecture de l'Analyse

**Initialisation (Lignes 1-12) :**
- Configuration optionnelle du LLM Groq pour analyse avancée
- `temperature=0` : réponses déterministes (pas de créativité)
- `max_tokens=200` : limitation pour éviter les réponses trop longues
- Gestion gracieuse si l'API n'est pas disponible

**Analyse par Règles (Lignes 1-27) :**
- **Ligne 1** : Conversion en minuscules pour uniformiser
- **Lignes 2-5** : Définition de 4 catégories de mots-clés
- **Ligne 6** : Vérification que le feedback parle bien de contenu pédagogique
- **Lignes 9-14** : Comptage des occurrences dans chaque catégorie
- **Lignes 15-18** : Gestion de cas spéciaux ("trop long", préférences)
- **Lignes 19-25** : Logique de décision par majorité

**Analyse TextBlob (Lignes 1-14) :**
- **Ligne 1** : Création de l'objet TextBlob
- **Ligne 2** : Score de -1 (très négatif) à +1 (très positif)
- **Lignes 8-12** : Classification avec seuils (±0.1)
- **Ligne 14** : Fallback sur règles en cas d'erreur

**Analyse LLM (Lignes 1-15) :**
- **Ligne 3** : Prompt structuré demandant une réponse JSON
- **Lignes 4-6** : Appel au modèle via Ollama
- **Lignes 7-13** : Extraction et validation du JSON retourné

### 4.4 Code Complet : Gestion des Données et Traitement en Lot

📁 **Fichier : `sentiment_analyzer.py` (méthodes de traitement)**

```python
def fetch_feedbacks(self):
    """
    Récupère les feedbacks non analysés depuis la base de données.
    
    Ligne 1: Connexion à la base SQLite
    Ligne 2-4: Exécution de la requête pour les feedbacks sans polarité
    Ligne 5: Log du nombre de feedbacks récupérés
    """
    engine = create_engine(DATABASE_URL)                                   # Ligne 1
    try:
        with engine.connect() as connection:                               # Ligne 2
            df = pd.read_sql_query(SQL_QUERY, connection)                  # Ligne 3
        logger.info(f"{len(df)} feedbacks à traiter récupérés")            # Ligne 4
        return df                                                          # Ligne 5
    except Exception as e:
        logger.error(f"Erreur base de données: {e}")                       # Ligne 6
        return pd.DataFrame()                                              # Ligne 7

def update_feedback_polarity(self, feedback_id, polarity):
    """
    Met à jour la polarité d'un feedback en base de données.
    
    Ligne 1: Connexion avec commit automatique
    Lignes 2-5: Exécution de la mise à jour
    Ligne 6: Log de confirmation
    """
    engine = create_engine(DATABASE_URL)                                   # Ligne 1
    try:
        with engine.begin() as connection:                                 # Ligne 2
            connection.execute(                                            # Ligne 3
                text("UPDATE feedbacks SET polarity = :polarity WHERE id = :id"),  # Ligne 4
                {"polarity": polarity, "id": feedback_id}                  # Ligne 5
            )
        logger.info(f"Feedback ID {feedback_id} mis à jour avec '{polarity}'")  # Ligne 6
    except Exception as e:
        logger.error(f"Erreur mise à jour feedback ID {feedback_id}: {e}")  # Ligne 7

def analyze_and_update_feedbacks(self):
    """
    Traite tous les feedbacks non analysés et met à jour la base.
    Méthode principale pour l'analyse en lot.
    """
    feedbacks_df = self.fetch_feedbacks()                                  # Ligne 1
    if feedbacks_df.empty:                                                 # Ligne 2
        logger.warning("Aucun feedback à traiter")                         # Ligne 3
        return self.reload_feedbacks_for_dashboard()                       # Ligne 4
    
    total_feedbacks = len(feedbacks_df)                                    # Ligne 5
    logger.info(f"Début de l'analyse de {total_feedbacks} feedbacks")      # Ligne 6
    
    # Traitement de chaque feedback
    for idx, row in feedbacks_df.iterrows():                              # Ligne 7
        logger.info(f"{'='*50}")                                           # Ligne 8
        logger.info(f"Analyse du feedback n°{idx + 1}/{total_feedbacks} de {row['User']}")  # Ligne 9
        
        # Vérification du contenu
        if pd.isna(row['FeedBack']) or not row['FeedBack'].strip():         # Ligne 10
            sentiment = {"aspect": "Contenu pédagogique", "polarity": "vide"}  # Ligne 11
        else:
            feedback = row['FeedBack']                                     # Ligne 12
            logger.info(f"Feedback: '{feedback[:100]}{'...' if len(feedback) > 100 else ''}'")  # Ligne 13
            sentiment = self.analyze_single_feedback(feedback)             # Ligne 14
        
        # Mise à jour en base
        self.update_feedback_polarity(row['ID'], sentiment["polarity"])    # Ligne 15
        logger.info(f"-> Résultat: {sentiment}")                           # Ligne 16
        
        # Pause pour éviter la surcharge
        time.sleep(0.1)                                                    # Ligne 17
    
    # Rechargement pour le dashboard
    return self.reload_feedbacks_for_dashboard()                          # Ligne 18

def reload_feedbacks_for_dashboard(self):
    """
    Recharge tous les feedbacks avec leur polarité pour affichage dashboard.
    
    Lignes 1-9: Requête pour récupérer tous les feedbacks analysés
    Ligne 10: Remplacement des valeurs nulles par 'non mentionné'
    """
    engine = create_engine(DATABASE_URL)                                   # Ligne 1
    with engine.connect() as connection:                                   # Ligne 2
        analyzed_df = pd.read_sql_query(                                   # Ligne 3
            """
            SELECT "ID", Département, Filière, Profile, "User", polarity, "timestamp"
            FROM "V_FeedBack"
            """,                                                           # Ligne 4
            connection                                                     # Ligne 5
        )
    analyzed_df['polarity'] = analyzed_df['polarity'].fillna('non mentionné')  # Ligne 6
    return analyzed_df                                                     # Ligne 7
```

### 4.5 Explication Détaillée : Traitement en Lot

**Récupération des données (Lignes 1-7) :**
- `DATABASE_URL` pointe vers la base SQLite locale
- `SQL_QUERY` sélectionne uniquement les feedbacks sans polarité (`WHERE polarity IS NULL`)
- Gestion d'erreur avec retour DataFrame vide si problème

**Mise à jour individuelle (Lignes 1-7) :**
- `engine.begin()` démarre une transaction avec commit automatique
- `text()` permet d'utiliser du SQL paramétré sécurisé
- Évite les injections SQL avec les paramètres liés

**Traitement en lot (Lignes 1-18) :**
- **Ligne 4** : Si pas de feedbacks, recharge quand même pour le dashboard
- **Lignes 7-9** : Boucle avec affichage du progrès
- **Lignes 10-14** : Gestion des feedbacks vides vs avec contenu
- **Ligne 17** : Pause de 0.1s pour éviter la surcharge système

### 4.6 Exemples Pratiques d'Analyse

**Exemple 1 : Feedback Positif**
```
Input: "Le cours sur l'IA était vraiment excellent, très clair et bien expliqué !"

Analyse par règles:
- content_keywords trouvés: ['cours'] → has_content = True
- positive_keywords trouvés: ['excellent', 'très clair', 'bien expliqué'] → positive_count = 3
- negative_keywords trouvés: [] → negative_count = 0
- neutral_keywords trouvés: [] → neutral_count = 0

Résultat: {"aspect": "Contenu pédagogique", "polarity": "positive"}
```

**Exemple 2 : Feedback Négatif**
```
Input: "La réponse était confuse et peu claire, difficile à comprendre"

Analyse par règles:
- content_keywords trouvés: ['réponse'] → has_content = True
- positive_keywords trouvés: [] → positive_count = 0
- negative_keywords trouvés: ['confuse', 'peu claire', 'difficile'] → negative_count = 3
- neutral_keywords trouvés: [] → neutral_count = 0

Résultat: {"aspect": "Contenu pédagogique", "polarity": "négative"}
```

**Exemple 3 : Feedback Hors Sujet**
```
Input: "Bonjour, comment allez-vous aujourd'hui ?"

Analyse par règles:
- content_keywords trouvés: [] → has_content = False

Résultat: {"aspect": "Contenu pédagogique", "polarity": "non mentionné"}
```

**Exemple 4 : Feedback avec LLM**
```
Input: "Le contenu était correct mais manquait d'exemples pratiques"

LLM Response: {"aspect": "Contenu pédagogique", "polarity": "neutre"}

Justification: Mélange de satisfaction ("correct") et de critique constructive ("manquait d'exemples")
```

---

## 5. Intégration et Workflow Complet {#workflow}

### 5.1 Flux Utilisateur Type

```
┌─────────────────┐
│ 1. Étudiant     │
│ pose question   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 2. Authentif.   │ ──── Département/Filière
│ & Filtrage      │      déterminés
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 3. RAG Search   │ ──── Recherche dans
│ (avec MMR)      │      documents autorisés
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 4. Génération   │ ──── Prompt adapté
│ LLM Response    │      (standard/QCM/résumé)
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 5. Réponse      │ ──── Sauvegarde en
│ + Historique    │      historique
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 6. Feedback     │ ──── Analyse sentiment
│ (optionnel)     │      pour amélioration
└─────────────────┘
```

### 5.2 Code Complet : Point d'Entrée Principal

📁 **Fichier : `main.py` (point d'entrée de l'application)**

```python
import os
from ollama_api import OllamaAPI
from rag_chatbot import RAGChatbot

def main():
    """
    Point d'entrée principal du système RAG.
    Gère l'interface utilisateur et orchestre les opérations.
    """
    # Initialisation des composants principaux
    ollama_api = OllamaAPI()                                                # Ligne 1
    chatbot = RAGChatbot(ollama_api)                                        # Ligne 2

    # Affichage du menu principal
    print("Bienvenue dans le système RAG !")                               # Ligne 3
    print("Commandes disponibles :")                                       # Ligne 4
    print("- 'index chemin_du_fichier' : pour indexer un fichier")         # Ligne 5
    print("- 'chat' : pour démarrer une conversation")                     # Ligne 6
    print("- 'exit' : pour quitter")                                       # Ligne 7

    # Boucle principale d'interaction
    while True:                                                             # Ligne 8
        command = input("\nEntrez une commande : ").strip()                # Ligne 9

        if command.lower() == "exit":                                       # Ligne 10
            print("Au revoir !")                                           # Ligne 11
            break                                                           # Ligne 12
            
        elif command.lower() == "chat":                                     # Ligne 13
            try:
                # Configuration du contexte académique
                print("=== Configuration du contexte académique ===")       # Ligne 14
                departement_id = int(input("ID du département : "))         # Ligne 15
                filiere_id = int(input("ID de la filière : "))              # Ligne 16
                module_id = int(input("ID du module : "))                   # Ligne 17
                activite_id = int(input("ID de l'activité : "))             # Ligne 18
                profile_id = int(input("ID du profil (1=Admin, 2=Prof, 3=Etudiant, 4=Invité) : "))  # Ligne 19
                user_id = int(input("Votre identifiant utilisateur : "))    # Ligne 20
                print("Contexte enregistré.\n")                            # Ligne 21

                # Démarrage de la conversation
                chatbot.chat(                                               # Ligne 22
                    departement_id=departement_id,                         # Ligne 23
                    filiere_id=filiere_id,                                  # Ligne 24
                    module_id=module_id,                                    # Ligne 25
                    activite_id=activite_id,                                # Ligne 26
                    profile_id=profile_id,                                  # Ligne 27
                    user_id=user_id                                         # Ligne 28
                )

            except ValueError:
                print("Erreur : tous les identifiants doivent être des entiers valides.")  # Ligne 29
                
        elif command.lower().startswith("index "):                         # Ligne 30
            file_path = command[6:].strip()                                 # Ligne 31
            base_filename = os.path.basename(file_path)                     # Ligne 32
            
            if not os.path.exists(file_path):                              # Ligne 33
                print(f"Le fichier {file_path} n'existe pas.")             # Ligne 34
                continue                                                    # Ligne 35
            
            try:
                # Collecte des métadonnées du fichier
                departement_id = int(input("ID du département : "))         # Ligne 36
                filiere_id = int(input("ID de la filière : "))              # Ligne 37
                module_id = int(input("ID du module : "))                   # Ligne 38
                activite_id = int(input("ID de l'activité (1=Inscription, 2=Cours, 3=TD, 4=TP...) : "))  # Ligne 39
                profile_id = int(input("ID du profil (1=Admin, 2=Prof, 3=Etudiant, 4=Invité) : "))  # Ligne 40
                user_id = int(input("ID de l'utilisateur : "))              # Ligne 41

                # Indexation du fichier
                chatbot.ingestion_file(                                     # Ligne 42
                    base_filename,                                          # Ligne 43
                    file_path,                                              # Ligne 44
                    departement_id=departement_id,                         # Ligne 45
                    filiere_id=filiere_id,                                  # Ligne 46
                    module_id=module_id,                                    # Ligne 47
                    activite_id=activite_id,                                # Ligne 48
                    profile_id=profile_id,                                  # Ligne 49
                    user_id=user_id                                         # Ligne 50
                )

            except ValueError:
                print("Erreur : tous les identifiants doivent être des entiers valides.")  # Ligne 51
            except Exception as e:
                print(f"Erreur lors de l'indexation : {e}")                # Ligne 52
                
        else:
            print("Commande non reconnue. Veuillez réessayer.")            # Ligne 53

if __name__ == "__main__":
    main()                                                                  # Ligne 54
```

### 5.3 Explication Détaillée : Interface Principale

**Lignes 1-2 : Initialisation**
- `OllamaAPI()` : Interface avec le modèle de langage local
- `RAGChatbot()` : Orchestrateur principal du système RAG

**Lignes 3-7 : Interface utilisateur**
- Menu simple et clair pour l'utilisateur
- Trois commandes principales : index, chat, exit

**Lignes 8-12 : Boucle principale**
- `while True` : fonctionnement continu jusqu'à 'exit'
- `input().strip()` : récupération et nettoyage des commandes

**Lignes 13-29 : Mode conversation**
- Configuration complète du contexte académique
- Identification de l'utilisateur et de ses permissions
- Gestion d'erreur pour les entrées invalides

**Lignes 30-53 : Mode indexation**
- Extraction du chemin depuis la commande
- Vérification que le fichier existe
- Collecte des métadonnées pour le contexte académique
- Appel de l'ingestion avec tous les paramètres

### 5.4 Architecture Complète du Système

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SYSTÈME RAG COMPLET                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   main.py       │    │  rag_chatbot.py │    │file_processor.py│         │
│  │                 │    │                 │    │                 │         │
│  │ • Interface CLI │────│ • Orchestration │────│ • Traitement    │         │
│  │ • Gestion users │    │ • Recherche RAG │    │   fichiers      │         │
│  │ • Menu commandes│    │ • Génération LLM│    │ • Chunking      │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                │
│           │                       │                       │                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ filter_manager. │    │   ollama_api.py │    │sentiment_analyzer│        │
│  │      py         │    │                 │    │      .py        │         │
│  │                 │    │ • Interface LLM │    │                 │         │
│  │ • Filtres       │────│ • Communication│────│ • Analyse       │         │
│  │   académiques   │    │   modèle        │    │   feedbacks     │         │
│  │ • Base SQLite   │    │ • Prompts       │    │ • 3 méthodes    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                │
│           └───────────────────────┼───────────────────────┘                │
│                                   │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         COMPOSANTS EXTERNES                         │   │
│  │                                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │    FAISS    │  │   SQLite    │  │   Ollama    │  │  SentenceT. │ │   │
│  │  │             │  │             │  │             │  │             │ │   │
│  │  │ • Index     │  │ • Metadata  │  │ • LLM Local │  │ • Embeddings│ │   │
│  │  │   vectoriel │  │ • History   │  │ • Génération│  │ • BGE Model │ │   │
│  │  │ • Recherche │  │ • Users     │  │   texte     │  │ • Vectors   │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Points Clés pour l'Optimisation

1. **Qualité des Embeddings** : Le modèle BGE-base-en-v1.5 est optimisé pour la langue française
2. **Taille des Chunks** : 384 caractères avec 96 de chevauchement = bon équilibre
3. **MMR Lambda** : 0.5 offre un bon compromis pertinence/diversité
4. **Seuil de Similarité** : 0.65 filtre les résultats peu pertinents
5. **Prompts Structurés** : Garantissent la qualité et la cohérence des réponses

### 5.6 Monitoring et Métriques

**Performance RAG :**
- Temps de recherche vectorielle
- Nombre de chunks récupérés
- Scores de similarité moyens
- Utilisation du MMR (diversité vs pertinence)

**Analyse de Sentiment :**
- Distribution des polarités (positive/négative/neutre)
- Taux de feedbacks "non mentionné"
- Performance des différentes méthodes d'analyse
- Évolution temporelle de la satisfaction

**Utilisation Système :**
- Nombre de documents indexés
- Volume de l'index FAISS
- Fréquence des requêtes par utilisateur
- Répartition par département/filière

---

## Conclusion

Ce système RAG avec analyse de sentiment représente une solution complète pour l'éducation numérique :

✅ **Réponses précises** grâce à la recherche vectorielle FAISS  
✅ **Diversité des contenus** avec l'algorithme MMR  
✅ **Personnalisation** par profil utilisateur et filtrage académique  
✅ **Amélioration continue** via l'analyse des feedbacks  
✅ **Sécurité** et contrôle d'accès par département/filière  
✅ **Architecture modulaire** permettant l'évolution et la maintenance  

L'architecture modulaire permet facilement d'ajouter de nouvelles fonctionnalités ou d'améliorer les composants existants sans affecter le reste du système.

#### 🔍 **Recherche de Similarité Standard**

```python
query_embedding = self.embedding_model.encode([user_query], normalize_embeddings=True)[0]
distances, indices = self.index.search(normalized_query, k=5)
```

Le système :
1. **Convertit** la question en vecteur
2. **Cherche** les vecteurs les plus proches dans FAISS
3. **Récupère** les chunks correspondants

#### 🎯 **MMR (Maximal Marginal Relevance)**

**Le problème :** La recherche standard peut retourner des passages très similaires (redondants).

**La solution MMR :**

```python
def mmr(self, query_embedding, candidate_embeddings, k=3, lambda_param=0.5):
    selected = []
    for _ in range(k):
        max_score = -np.inf
        for idx in candidate_indices:
            # Pertinence par rapport à la question
            relevance = similarity(query, candidate[idx])
            
            # Diversité par rapport aux déjà sélectionnés  
            diversity = max([similarity(candidate[idx], selected_doc) for selected_doc in selected])
            
            # Score MMR = équilibre pertinence/diversité
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            
            if mmr_score > max_score:
                best_candidate = idx
```

**Comment MMR fonctionne :**

1. **Pertinence** : À quel point ce passage répond à la question ?
2. **Diversité** : À quel point ce passage apporte des informations nouvelles ?
3. **Équilibre** : `lambda_param` contrôle le compromis
   - `lambda = 1.0` : Priorité totale à la pertinence
   - `lambda = 0.0` : Priorité totale à la diversité
   - `lambda = 0.5` : Équilibre

**Exemple pratique :**

Question : "Qu'est-ce que le machine learning ?"

Sans MMR :
- Passage 1 : "Le machine learning est une branche de l'IA..."
- Passage 2 : "Le machine learning, ou apprentissage automatique..."
- Passage 3 : "Le ML fait partie de l'intelligence artificielle..."

Avec MMR :
- Passage 1 : "Le machine learning est une branche de l'IA..."
- Passage 2 : "Les algorithmes supervisés utilisent des données étiquetées..."
- Passage 3 : "Les applications incluent la reconnaissance vocale..."

#### 🔄 **Filtrage Académique**

```python
allowed_faiss_ids = FilterManager.get_allowed_indices(departement_id, filiere_id)
```

Le système s'assure que l'étudiant accède **uniquement** aux documents de :
- Son **département** (ex: Informatique, Mathématiques)
- Sa **filière** (ex: Génie Logiciel, Data Science)
- Ses **modules** et **activités**

### 3.4 Génération de la Réponse

#### 📝 **Construction du Prompt**

```python
if PromptBuilder.is_qcm_request(user_query):
    prompt_text = PromptBuilder.build_qcm_prompt(context_text, user_query)
elif "résumé" in user_query.lower():
    prompt_text = PromptBuilder.build_summary_prompt(context_text, user_query)
else:
    prompt_text = PromptBuilder.build_standard_prompt(context_text, user_query)
```

Le système adapte son comportement selon le type de demande :

**Prompt Standard :**
```
# Contexte de l'Apprentissage
[Passages pertinents trouvés]

# Ta Mission
Tu es un assistant pédagogique expert...

# Question de l'Étudiant
[Question de l'utilisateur]

## Tes Directives :
1. Source unique : Réponds uniquement à partir du contexte
2. Ton engageant et motivant
3. Réponse concise et claire
4. Uniquement en français
```

#### 🤖 **Appel au LLM (Ollama)**

```python
llm_raw_response = self.ollama_api.chat_with_ollama(prompt_text)
cleaned_response = self.clean_llm_response(llm_raw_response)
```

Le **Large Language Model** (via Ollama) :
- Analyse le contexte fourni
- Génère une réponse adaptée
- Respecte les consignes pédagogiques

---

## 4. Analyse de Sentiment {#sentiment}

### 4.1 Objectif de l'Analyse de Sentiment

L'analyse de sentiment permet de :
- **Comprendre** la satisfaction des étudiants
- **Identifier** les points d'amélioration
- **Adapter** le système selon les retours

### 4.2 Architecture du Système d'Analyse

```python
class SimpleSentimentAnalyzer:
    def __init__(self):
        self.llm = ChatGroq(model_name="llama3-8b-8192")  # LLM externe (optionnel)
```

### 4.3 Méthodes d'Analyse (en Cascade)

#### 🎯 **Méthode 1 : Analyse par LLM (Prioritaire)**

```python
def analyze_sentiment_llm(self, feedback_text):
    prompt = f"""Analyse le sentiment de ce feedback concernant le contenu pédagogique.
    
    Feedback: "{feedback_text}"
    
    Réponds UNIQUEMENT avec un JSON valide:
    {{"aspect": "Contenu pédagogique", "polarity": "positive/négative/neutre/non mentionné"}}
    """
    response = ollama_api.chat_with_ollama(prompt)
```

**Avantages :**
- **Compréhension contextuelle** avancée
- **Nuances** linguistiques
- **Adaptabilité** aux expressions variées

#### 📊 **Méthode 2 : Analyse par Règles (Fallback principal)**

```python
def analyze_sentiment_rule_based(self, feedback_text):
    feedback_lower = feedback_text.lower()
    
    # Détection du contenu pédagogique
    content_keywords = [
        'contenu', 'cours', 'leçon', 'résumé', 'explication', 'pédagogique',
        'matière', 'sujet', 'chapitre', 'module', 'formation', 'apprentissage'
    ]
    
    # Mots positifs
    positive_keywords = [
        'génial', 'super', 'magnifique', 'excellent', 'parfait', 'bien', 'bon',
        'efficace', 'utile', 'clair', 'précis', 'complet', 'satisfait',
        'très utile', 'très clair', 'bien expliqué'
    ]
    
    # Mots négatifs
    negative_keywords = [
        'mauvais', 'nul', 'horrible', 'décevant', 'problème', 'erreur',
        'difficile', 'compliqué', 'confus', 'incompréhensible',
        'peu clair', 'pas utile', 'pas efficace'
    ]
```

**Algorithme :**

1. **Vérification de pertinence :**
   ```python
   has_content = any(keyword in feedback_lower for keyword in content_keywords)
   if not has_content:
       return {"aspect": "Contenu pédagogique", "polarity": "non mentionné"}
   ```

2. **Comptage des occurrences :**
   ```python
   positive_count = sum(1 for keyword in positive_keywords if keyword in feedback_lower)
   negative_count = sum(1 for keyword in negative_keywords if keyword in feedback_lower)
   ```

3. **Décision finale :**
   ```python
   if positive_count > negative_count:
       polarity = "positive"
   elif negative_count > positive_count:
       polarity = "négative" 
   else:
       polarity = "neutre"
   ```

#### 📈 **Méthode 3 : TextBlob (Backup)**

```python
def analyze_sentiment_textblob(self, feedback_text):
    blob = TextBlob(feedback_text)
    sentiment_score = blob.sentiment.polarity  # Entre -1 (négatif) et +1 (positif)
    
    if sentiment_score > 0.1:
        polarity = "positive"
    elif sentiment_score < -0.1:
        polarity = "négative"
    else:
        polarity = "neutre"
```

### 4.4 Processus Complet d'Analyse

```python
def analyze_single_feedback(self, feedback_text):
    # Vérification et nettoyage
    if not feedback_text or not feedback_text.strip():
        return {"aspect": "Contenu pédagogique", "polarity": "vide"}
    
    cleaned_feedback = feedback_text.strip()[:500]  # Limitation à 500 caractères
    
    # Méthode 1 : LLM (si disponible)
    if self.llm:
        llm_result = self.analyze_sentiment_llm(cleaned_feedback)
        if llm_result:
            return llm_result
    
    # Méthode 2 : Règles (principale)
    return self.analyze_sentiment_rule_based(cleaned_feedback)
```

### 4.5 Gestion des Données et Mises à Jour

#### 💾 **Récupération des Feedbacks**

```python
def fetch_feedbacks(self):
    SQL_QUERY = """
    SELECT id as ID, Département, Filière, Profile, "User", FeedBack, "timestamp"
    FROM "V_FeedBack"
    WHERE polarity IS NULL  -- Seulement les feedbacks non analysés
    """
```

#### 🔄 **Mise à Jour en Base**

```python
def update_feedback_polarity(self, feedback_id, polarity):
    connection.execute(
        text("UPDATE feedbacks SET polarity = :polarity WHERE id = :id"),
        {"polarity": polarity, "id": feedback_id}
    )
```

**Workflow :**
1. Récupérer les feedbacks non analysés
2. Analyser chaque feedback
3. Stocker le résultat en base
4. Générer des statistiques pour le dashboard

---

## 5. Intégration et Workflow Complet {#workflow}

### 5.1 Flux Utilisateur Type

```
┌─────────────────┐
│ 1. Étudiant     │
│ pose question   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 2. Authentif.   │ ──── Département/Filière
│ & Filtrage      │      déterminés
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 3. RAG Search   │ ──── Recherche dans
│ (avec MMR)      │      documents autorisés
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 4. Génération   │ ──── Prompt adapté
│ LLM Response    │      (standard/QCM/résumé)
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 5. Réponse      │ ──── Sauvegarde en
│ + Historique    │      historique
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ 6. Feedback     │ ──── Analyse sentiment
│ (optionnel)     │      pour amélioration
└─────────────────┘
```

### 5.2 Gestion des Profils Utilisateur

```python
INVITE_PROFILE_ID = 5

if profile_id == INVITE_PROFILE_ID:
    # Restrictions spéciales pour les invités
    departement_id = 1  # Seulement Scolarité
    filiere_id = 3
    
    prompt_text = (
        "⚠️ Mode invité activé :\n"
        "Tu dois répondre uniquement à partir des informations du département Scolarité.\n"
        "Si la question ne concerne pas le département Scolarité, réponds :\n"
        "'Je ne peux répondre qu'aux questions relatives au département Scolarité.'"
    )
```

### 5.3 Optimisations et Bonnes Pratiques

#### ⚡ **Performance**
- **Cache FAISS** : Les index sont sauvegardés sur disque
- **Hashes de fichiers** : Évitent le retraitement des documents identiques
- **Pagination** : Limitation des résultats pour les grosses requêtes

#### 🔒 **Sécurité**
- **Filtrage strict** par département/filière
- **Validation des inputs** utilisateur
- **Hachage des mots de passe**
- **Sanitization** des réponses LLM

#### 📊 **Monitoring**
- **Logging détaillé** de toutes les opérations
- **Métriques de performance** (temps de réponse, précision)
- **Dashboard d'analyse** des feedbacks

### 5.4 Points Clés pour l'Optimisation

1. **Qualité des Embeddings** : Le modèle BGE-base-en-v1.5 est optimisé pour la langue française
2. **Taille des Chunks** : 384 caractères avec 96 de chevauchement = bon équilibre
3. **MMR Lambda** : 0.5 offre un bon compromis pertinence/diversité
4. **Seuil de Similarité** : 0.65 filtre les résultats peu pertinents
5. **Prompts Structurés** : Garantissent la qualité et la cohérence des réponses

---

## Conclusion

Ce système RAG avec analyse de sentiment représente une solution complète pour l'éducation numérique :

✅ **Réponses précises** grâce à la recherche vectorielle  
✅ **Diversité des contenus** avec MMR  
✅ **Personnalisation** par profil utilisateur  
✅ **Amélioration continue** via l'analyse des feedbacks  
✅ **Sécurité** et filtrage académique  

L'architecture modulaire permet facilement d'ajouter de nouvelles fonctionnalités ou d'améliorer les composants existants.