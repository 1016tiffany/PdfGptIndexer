from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from langchain_community.embeddings import OllamaEmbeddings

# Initialize your embedding model
MODEL = 'mxbai-embed-large'
embeddings_model = OllamaEmbeddings(model=MODEL)

def generate_embedding(text):
    """Calculate cosine similarity between two embeddings."""

    return embeddings_model._embed(text)

def calculate_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

def plot_embeddings(embeddings, labels):
    """Visualize embeddings using PCA."""

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', label=labels)
    plt.title('PCA of Document Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

# test
# Assuming you have a list of embeddings from different documents
embeddings = [embeddings_model._embed(text) for text in ["Text from doc 1", "Text from doc 2", "Text from doc 3"]]
labels = ['Doc1', 'Doc2', 'Doc3']
plot_embeddings(embeddings, labels)
# # Perform PCA
# pca = PCA(n_components=2)
# reduced_embeddings = pca.fit_transform(embeddings)

# # Plot
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
# plt.title('PCA of Document Embeddings')
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.show()

