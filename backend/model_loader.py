import pickle
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_PATH = Path("./models")

# ----- FAQ Model -----
FAQ_PATH = BASE_PATH / "faqs_model"

faq_df = pickle.load(open(FAQ_PATH / "faq_data.pkl", "rb"))
faq_index = faiss.read_index(str(FAQ_PATH / "faiss.index"))
faq_embeddings = np.load(FAQ_PATH / "faq_embeddings.npy")

# Sentence embedding model for FAQ queries
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----- Agri QA Model -----
agri_model_data = pickle.load(open(BASE_PATH / "agri_qa_model.pkl", "rb"))
agri_model = agri_model_data["model"]
tfidf_vectorizer = agri_model_data["tfidf_vectorizer"]
label_encoder = agri_model_data["label_encoder"]
