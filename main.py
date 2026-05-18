import numpy as np
from sklearn.decomposition import PCA

# 1. Generate synthetic high-dimensional data
# We'll create 3D data that inherently has a 2D structure with some noise.
# Imagine points mostly lying on a plane in 3D space.
np.random.seed(42)
n_samples = 100

# Create two underlying 'latent' components
c1 = np.random.rand(n_samples) * 10
c2 = np.random.rand(n_samples) * 5

# Create 3 features (dimensions) that are linear combinations of c1, c2, plus noise.
# This simulates high-dimensional data where the 'true' underlying structure is lower-dimensional.
X_high_dim = np.zeros((n_samples, 3))
X_high_dim[:, 0] = c1 + 0.5 * c2 + np.random.randn(n_samples) * 0.5  # Feature 1
X_high_dim[:, 1] = 0.8 * c1 - 0.2 * c2 + np.random.randn(n_samples) * 0.3 # Feature 2
X_high_dim[:, 2] = 0.3 * c1 + 0.9 * c2 + np.random.randn(n_samples) * 0.7 # Feature 3

print("Original high-dimensional data shape:", X_high_dim.shape)
print("First 5 rows of original data:\n", X_high_dim[:5])

# 2. Initialize PCA
# We want to reduce the 3-dimensional data to 2 principal components.
# This aims to capture most of the variance in the data with fewer dimensions.
pca = PCA(n_components=2)

# 3. Fit PCA to the data and transform it
# PCA learns the principal components (new axes) from the data and then
# projects the data onto these new, lower-dimensional axes.
X_reduced = pca.fit_transform(X_high_dim)

print("\nTransformed lower-dimensional data shape:", X_reduced.shape)
print("First 5 rows of transformed data:\n", X_reduced[:5])

# 4. Analyze explained variance
# This shows how much of the original data's variance is retained by each principal component.
# A high total explained variance (e.g., >90%) indicates that the lower-dimensional
# representation captures most of the original information.
print("\nExplained variance ratio by each principal component:", pca.explained_variance_ratio_)
print("Total explained variance by 2 components: {:.2f}%".format(pca.explained_variance_ratio_.sum() * 100))

# 5. Access principal components (the new axes)
# These are the directions in the original feature space along which the data varies most.
# Each row represents a principal component, showing its contribution from each original feature.
print("\nPrincipal Components (eigenvectors):\n", pca.components_)
