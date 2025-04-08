import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import seaborn as sns


# Function for text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = text.strip()  # Remove extra spaces
    return text


text_column = "Abstract"

df = pd.read_csv("Arxiv_Resources.csv")
df = df.dropna(subset=[text_column])
df = df.drop_duplicates()

# Handle missing values
df[text_column] = (
    df[text_column].fillna("").apply(preprocess_text)
)  # Fill NaN & clean text

# Apply TF-IDF after preprocessing
tfidf_vectorizer = TfidfVectorizer(
    max_features=700, ngram_range=(1, 2), stop_words="english"
)
tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column]).toarray()

# 3. Normalize TF-IDF values to [0,1] range for the RBM
# (RBMs work better with binary or values in [0,1])
min_max_scaler = MinMaxScaler()
tfidf_scaled = min_max_scaler.fit_transform(tfidf_matrix)

# Dimensionality Reduction using RBM.
rbm = BernoulliRBM(
    n_components=10,  # Number of hidden units
    learning_rate=0.1,
    n_iter=20,
    verbose=True,
    random_state=42,
)

X_rbm = rbm.fit_transform(tfidf_scaled)

print("Original TF-IDF shape:", tfidf_matrix.shape)
print("RBM features shape:", X_rbm.shape)

# Apply PCA with 2 components
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_rbm)

# Visualize using PCA.
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# GMM
k = 5
model = GaussianMixture(n_components=k, random_state=42)
labels_model = model.fit_predict(X_rbm)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_model, palette="viridis")
plt.show()

# Sample
sample_X, sample_y = model.sample(100) # change this as needed
samples_text = tfidf_vectorizer.inverse_transform(sample_X)

print(samples_text)