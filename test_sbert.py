from sentence_transformers import SentenceTransformer

print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Semantic plagiarism detection is powerful.",
    "Meaning-based similarity improves plagiarism detection."
]

embeddings = model.encode(sentences)

print("SBERT working successfully ✅")
print("Number of sentences:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))
