import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

# =============================================================================
# LOAD REAL FACE DATASET
# =============================================================================
print("Loading faces...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data  # Each row is one flattened face image
y = lfw_people.target  # Person labels
names = lfw_people.target_names  # Person names

n_samples, n_features = X.shape  # n_samples faces, n_features pixels each
h, w = lfw_people.images.shape[1:]  # height and width of images

print(f"Dataset: {n_samples} faces from {len(names)} people")
print(f"Each face: {h}x{w} = {n_features} pixels (dimensions)")

# =============================================================================
# THE MAGIC: PCA USES EIGENVALUES & EIGENVECTORS
# =============================================================================
n_components = 150  # Number of eigenfaces to use (K in the equations)

# PCA does the following internally:
# 1. Centers the data: X̃ = X - μ
# 2. Computes covariance: C = X̃ᵀX̃
# 3. Finds eigenvectors v and eigenvalues λ: Cv = λv
# 4. Sorts by eigenvalue and keeps top K
pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized')
pca.fit(X)

# The eigenvectors are the "eigenfaces"
# Each eigenvector is reshaped from 1D array back to 2D image
eigenfaces = pca.components_.reshape((n_components, h, w))

# The eigenvalues tell us how much variance each eigenface captures
variance_explained = pca.explained_variance_ratio_

print(f"\n🎯 With only {n_components} eigenfaces we capture "
      f"{sum(variance_explained)*100:.1f}% of the information")
print(f"   That's {n_components}/{n_features} = "
      f"{100*n_components/n_features:.1f}% of the dimensions!")

# =============================================================================
# VISUALIZATION 1: THE EIGENFACES (Eigenvectors)
# =============================================================================
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
fig.suptitle('First 18 Eigenfaces (eigenvectors vᵢ)', 
             fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < len(eigenfaces):
        # Display the i-th eigenface
        ax.imshow(eigenfaces[i], cmap='gray')
        # Show its eigenvalue (variance it explains)
        ax.set_title(f'λ{i+1}={variance_explained[i]:.3f}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('eigenfaces.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eigenfaces.png")

# =============================================================================
# VISUALIZATION 2: FACE RECONSTRUCTION
# Showing: x ≈ μ + Σ wᵢvᵢ for different values of K
# =============================================================================
original_face = X[0]  # Take one face from the dataset

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Reconstructing a face with different numbers of eigenfaces', 
             fontsize=14, fontweight='bold')

# Try reconstructing with different numbers of components
components_to_try = [5, 10, 25, 50, 100, 150, 200, 300, 500, n_features]

for idx, n_comp in enumerate(components_to_try):
    if idx >= 10:
        break
    
    if n_comp > n_components:
        # For the last one, use all available components
        pca_temp = PCA(n_components=min(n_comp, X.shape[0], X.shape[1]))
        pca_temp.fit(X)
        coefficients = pca_temp.transform([original_face])
        reconstructed_face = pca_temp.inverse_transform(coefficients)
    else:
        # Project onto first n_comp eigenfaces: w = Vᵀ(x - μ)
        coefficients = pca.transform([original_face])[:, :n_comp]
        
        # Reconstruct: x ≈ μ + Σ wᵢvᵢ
        # We do this by filling unused components with zeros
        coef_full = np.zeros((1, n_components))
        coef_full[0, :n_comp] = coefficients
        reconstructed_face = pca.inverse_transform(coef_full)
    
    # Plot the reconstruction
    ax = axes[idx // 5, idx % 5]
    ax.imshow(reconstructed_face.reshape(h, w), cmap='gray')
    ax.set_title(f'{n_comp} eigenfaces')
    ax.axis('off')

plt.tight_layout()
plt.savefig('reconstruction.png', dpi=150, bbox_inches='tight')
print("✅ Saved: reconstruction.png")

# =============================================================================
# MATHEMATICAL EXPLANATION FOR STUDENTS
# =============================================================================
print("\n" + "="*60)
print("📚 MATHEMATICAL BREAKDOWN")
print("="*60)
print(f"""
1. Each face is a vector: x ∈ ℝ^{n_features}

2. Compute mean face: μ = (1/{n_samples}) Σ xᵢ

3. Center the data: X̃ = X - μ

4. Compute covariance matrix: C = (1/{n_samples}) X̃ᵀX̃
   (This is a {n_features}×{n_features} matrix!)

5. Solve eigenvalue problem: Cv = λv
   - Eigenvectors v₁, v₂, ..., vₙ are the "eigenfaces"
   - Eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ rank their importance

6. Keep top K={n_components} eigenvectors: V = [v₁, v₂, ..., v_K]

7. Any face can be approximated:
   x ≈ μ + Σ wᵢvᵢ  where w = Vᵀ(x - μ)
       i=1 to K

8. COMPRESSION ACHIEVED:
   • Original: {n_features:,} numbers per face
   • Compressed: {n_components} numbers per face
   • Ratio: {100*n_components/n_features:.1f}% of original
   • Information kept: {sum(variance_explained)*100:.1f}%
""")

print("\n🎓 REAL-WORLD APPLICATIONS:")
print("   • Face recognition (Face ID, security systems)")
print("   • Image compression")
print("   • Data visualization (reducing 10,000D to 2D)")
print("   • Anomaly detection")
print("   • Machine learning preprocessing")

# =============================================================================
# BONUS: SHOW THE COEFFICIENTS (weights wᵢ)
# =============================================================================
# These are the coefficients w in: x ≈ μ + Σ wᵢvᵢ
coefs = pca.transform([original_face])[0]

plt.figure(figsize=(12, 4))
plt.bar(range(len(coefs)), np.abs(coefs))
plt.xlabel('Eigenface number (i)')
plt.ylabel('Coefficient magnitude |wᵢ|')
plt.title('Coefficients wᵢ needed to reconstruct this face\n(x ≈ μ + Σ wᵢvᵢ)')
plt.grid(True, alpha=0.3)
plt.savefig('coefficients.png', dpi=150, bbox_inches='tight')
print("✅ Saved: coefficients.png")

plt.show()

print("\n" + "="*60)
print("✨ SUCCESS! Show these images to your students:")
print("   1. eigenfaces.png - The eigenvectors (basis faces)")
print("   2. reconstruction.png - How quality improves with more eigenfaces")
print("   3. coefficients.png - The weights wᵢ for one specific face")
print("="*60)

# =============================================================================
# TEACHING TIPS
# =============================================================================
print("\n💡 TEACHING TIPS:")
print("   1. Start by showing eigenfaces.png - they look spooky/ghostly!")
print("   2. Explain: these are the 'building blocks' (basis vectors)")
print("   3. Show reconstruction.png - watch quality improve")
print("   4. Key insight: Just 50-100 eigenfaces ≈ original quality")
print("   5. Connect to linear algebra: span, basis, projection")
print(f"   6. Math: going from {n_features:,}D to {n_components}D space!")