import numpy as np
from model_loader import faq_index, faq_df, embedding_model
import faiss

def retrieve_faq_answer(query: str, top_k: int = 3):
    """
    Retrieve top FAQ answer from FAISS index
    """
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    D, I = faq_index.search(query_vec, top_k)
    
    # Return the first match (highest similarity)
    idx = I[0][0]
    answer = faq_df.iloc[idx]["answers"]
    
    return answer  # <-- must be indented properly
