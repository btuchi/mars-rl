#!/usr/bin/env python3
"""
Check diversity of reference features using multiple metrics
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from pathlib import Path

def load_reference_features(npz_path):
    """Load reference features from NPZ file"""
    print(f"Loading reference features from: {npz_path}")
    ref_features = np.load(npz_path)
    ref_array = np.stack([ref_features[key] for key in ref_features.keys()])
    print(f"Loaded {len(ref_array)} reference images with {ref_array.shape[1]}D features")
    return ref_array

def calculate_cosine_diversity(features):
    """Calculate diversity using cosine similarity"""
    similarities = cosine_similarity(features)
    
    # Remove diagonal (self-similarity = 1.0)
    mask = ~np.eye(similarities.shape[0], dtype=bool)
    off_diagonal_similarities = similarities[mask]
    
    return {
        'mean_similarity': off_diagonal_similarities.mean(),
        'std_similarity': off_diagonal_similarities.std(),
        'min_similarity': off_diagonal_similarities.min(),
        'max_similarity': off_diagonal_similarities.max(),
        'diversity_score': off_diagonal_similarities.std()  # Higher std = more diverse
    }

def calculate_euclidean_diversity(features):
    """Calculate diversity using Euclidean distance"""
    distances = euclidean_distances(features)
    
    # Remove diagonal (self-distance = 0.0)
    mask = ~np.eye(distances.shape[0], dtype=bool)
    off_diagonal_distances = distances[mask]
    
    return {
        'mean_distance': off_diagonal_distances.mean(),
        'std_distance': off_diagonal_distances.std(),
        'min_distance': off_diagonal_distances.min(),
        'max_distance': off_diagonal_distances.max(),
        'diversity_score': off_diagonal_distances.mean()  # Higher mean distance = more diverse
    }

def calculate_feature_variance_diversity(features):
    """Calculate diversity using feature variance"""
    # Calculate variance across each feature dimension
    feature_variances = features.var(axis=0)
    
    return {
        'mean_variance': feature_variances.mean(),
        'std_variance': feature_variances.std(),
        'min_variance': feature_variances.min(),
        'max_variance': feature_variances.max(),
        'diversity_score': feature_variances.mean()  # Higher variance = more diverse
    }

def calculate_silhouette_diversity(features, n_clusters_range=[3, 5, 8]):
    """Calculate diversity using silhouette analysis"""
    best_score = -1
    best_n_clusters = None
    results = {}
    
    for n_clusters in n_clusters_range:
        if len(features) < n_clusters:
            continue
            
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            
            results[n_clusters] = score
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                
        except Exception as e:
            print(f"Warning: Could not calculate silhouette for {n_clusters} clusters: {e}")
            continue
    
    return {
        'best_n_clusters': best_n_clusters,
        'best_score': best_score,
        'all_scores': results,
        'diversity_score': best_score  # Higher silhouette = better separated clusters
    }

def suggest_images_to_remove(features, filenames, similarity_threshold=0.95, distance_threshold=0.1):
    """Suggest which images to remove to improve diversity"""
    print("\n" + "="*60)
    print("IMAGE REMOVAL SUGGESTIONS")
    print("="*60)
    
    # Calculate pairwise similarities and distances
    similarities = cosine_similarity(features)
    distances = euclidean_distances(features)
    
    # Find highly similar image pairs
    similar_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            cosine_sim = similarities[i, j]
            euclidean_dist = distances[i, j]
            
            # Flag as similar if high cosine similarity OR low euclidean distance
            if cosine_sim > similarity_threshold or euclidean_dist < distance_threshold:
                similar_pairs.append((i, j, cosine_sim, euclidean_dist))
    
    if not similar_pairs:
        print("✅ No highly similar images found - good diversity!")
        return []
    
    print(f"🔍 Found {len(similar_pairs)} highly similar image pairs:")
    print(f"   (Using thresholds: cosine_sim > {similarity_threshold} OR euclidean_dist < {distance_threshold})")
    
    # Count how many times each image appears in similar pairs
    image_similarity_count = {}
    for i, j, cos_sim, euc_dist in similar_pairs:
        image_similarity_count[i] = image_similarity_count.get(i, 0) + 1
        image_similarity_count[j] = image_similarity_count.get(j, 0) + 1
    
    # Sort by similarity count (images that are similar to many others)
    candidates_to_remove = sorted(image_similarity_count.items(), 
                                key=lambda x: x[1], reverse=True)
    
    print(f"\n📋 Similar image pairs:")
    for i, (idx1, idx2, cos_sim, euc_dist) in enumerate(similar_pairs[:10]):  # Show top 10
        name1 = filenames[idx1] if idx1 < len(filenames) else f"image_{idx1}"
        name2 = filenames[idx2] if idx2 < len(filenames) else f"image_{idx2}"
        print(f"  {i+1:2d}. {name1[:30]:<30} ↔ {name2[:30]:<30} "
              f"(cos: {cos_sim:.3f}, dist: {euc_dist:.3f})")
    
    if len(similar_pairs) > 10:
        print(f"  ... and {len(similar_pairs) - 10} more pairs")
    
    print(f"\n🎯 REMOVAL SUGGESTIONS:")
    print(f"   (Images that appear in multiple similar pairs)")
    
    removal_suggestions = []
    for idx, count in candidates_to_remove[:10]:  # Top 10 candidates
        filename = filenames[idx] if idx < len(filenames) else f"image_{idx}"
        print(f"  🗑️  {filename[:50]:<50} (similar to {count} other images)")
        removal_suggestions.append((idx, filename, count))
    
    # Calculate potential diversity improvement
    if removal_suggestions:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Start by removing the top 2-3 most redundant images")
        print(f"  2. Re-run diversity analysis after each removal")
        print(f"  3. Stop when diversity metrics improve significantly")
        print(f"  4. Keep at least 50+ images for good reference coverage")
    
    return removal_suggestions

def analyze_cluster_distribution(features, filenames, n_clusters=5):
    """Analyze how images are distributed in clusters"""
    print("\n" + "="*60)
    print("CLUSTER DISTRIBUTION ANALYSIS")
    print("="*60)
    
    if len(features) < n_clusters:
        print(f"⚠️ Too few images ({len(features)}) for {n_clusters} clusters")
        return
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Count images per cluster
        cluster_counts = {}
        cluster_images = {}
        for i, label in enumerate(labels):
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
            if label not in cluster_images:
                cluster_images[label] = []
            filename = filenames[i] if i < len(filenames) else f"image_{i}"
            cluster_images[label].append((i, filename))
        
        print(f"📊 Distribution across {n_clusters} clusters:")
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            percentage = (count / len(features)) * 100
            print(f"  Cluster {cluster_id}: {count:2d} images ({percentage:5.1f}%)")
        
        # Identify overcrowded clusters
        avg_size = len(features) / n_clusters
        overcrowded_clusters = [cid for cid, count in cluster_counts.items() 
                              if count > avg_size * 1.5]
        
        if overcrowded_clusters:
            print(f"\n🎯 OVERCROWDED CLUSTERS (>150% of average size):")
            for cluster_id in overcrowded_clusters:
                count = cluster_counts[cluster_id]
                print(f"\n  📁 Cluster {cluster_id} ({count} images):")
                
                # Show a few example images from this cluster
                for i, (img_idx, filename) in enumerate(cluster_images[cluster_id][:5]):
                    print(f"    - {filename[:50]}")
                if len(cluster_images[cluster_id]) > 5:
                    remaining = len(cluster_images[cluster_id]) - 5
                    print(f"    ... and {remaining} more images")
                
                print(f"  💡 Consider removing some images from this cluster to improve balance")
        else:
            print(f"\n✅ Clusters are well-balanced!")
        
        return cluster_images, cluster_counts
        
    except Exception as e:
        print(f"❌ Error in cluster analysis: {e}")
        return None, None

def interpret_diversity_scores(cosine_div, euclidean_div, variance_div, silhouette_div):
    """Provide interpretation of diversity scores"""
    print("\n" + "="*60)
    print("DIVERSITY INTERPRETATION")
    print("="*60)
    
    # Cosine similarity interpretation
    cosine_score = cosine_div['diversity_score']
    if cosine_score > 0.2:
        cosine_rating = "HIGH"
    elif cosine_score > 0.1:
        cosine_rating = "MODERATE"
    else:
        cosine_rating = "LOW"
    
    # Euclidean distance interpretation (depends on feature magnitude)
    euclidean_score = euclidean_div['diversity_score']
    if euclidean_score > 1.0:
        euclidean_rating = "HIGH"
    elif euclidean_score > 0.5:
        euclidean_rating = "MODERATE"
    else:
        euclidean_rating = "LOW"
    
    # Feature variance interpretation
    variance_score = variance_div['diversity_score']
    if variance_score > 0.1:
        variance_rating = "HIGH"
    elif variance_score > 0.05:
        variance_rating = "MODERATE"
    else:
        variance_rating = "LOW"
    
    # Silhouette interpretation
    silhouette_score = silhouette_div['diversity_score']
    if silhouette_score > 0.5:
        silhouette_rating = "EXCELLENT (well-separated clusters)"
    elif silhouette_score > 0.3:
        silhouette_rating = "GOOD (distinct clusters)"
    elif silhouette_score > 0.1:
        silhouette_rating = "FAIR (some clustering)"
    else:
        silhouette_rating = "POOR (no clear clusters)"
    
    print(f"📊 Cosine Similarity Diversity: {cosine_rating} ({cosine_score:.4f})")
    print(f"📏 Euclidean Distance Diversity: {euclidean_rating} ({euclidean_score:.4f})")
    print(f"📈 Feature Variance Diversity: {variance_rating} ({variance_score:.4f})")
    print(f"🎯 Silhouette Cluster Diversity: {silhouette_rating} ({silhouette_score:.4f})")
    
    # Overall recommendation
    high_count = sum([r == "HIGH" for r in [cosine_rating, euclidean_rating, variance_rating]]) + \
                 (1 if silhouette_score > 0.3 else 0)
    
    print(f"\n🔍 OVERALL ASSESSMENT:")
    if high_count >= 3:
        print("✅ EXCELLENT diversity - Great for RL training!")
    elif high_count >= 2:
        print("✅ GOOD diversity - Should work well for RL training")
    elif high_count >= 1:
        print("⚠️ MODERATE diversity - May work but consider adding more varied images")
    else:
        print("❌ LOW diversity - Strongly recommend adding more varied reference images")

def main():
    """Main function to check reference feature diversity"""
    print("🔍 Reference Feature Diversity Analysis")
    print("="*60)
    
    # Find the NPZ file
    npz_path = Path("ppo_diffusion/reference_features/reference_crater_features_v2.npz")
    if not npz_path.exists():
        print(f"❌ Reference features file not found: {npz_path}")
        return
    
    # Load features and get filenames
    features = load_reference_features(npz_path)
    
    # Extract filenames from NPZ keys
    ref_features_npz = np.load(npz_path)
    filenames = []
    for key in ref_features_npz.keys():
        # Extract original filename from key (format: image_XXX_filename)
        if '_' in key:
            parts = key.split('_', 2)  # Split into ['image', 'XXX', 'filename']
            if len(parts) >= 3:
                filenames.append(parts[2])  # Original filename
            else:
                filenames.append(key)
        else:
            filenames.append(key)
    ref_features_npz.close()
    
    print(f"\n📋 Analyzing diversity with {len(features)} reference images...")
    
    # Calculate all diversity metrics
    print("\n1️⃣ Calculating Cosine Similarity Diversity...")
    cosine_div = calculate_cosine_diversity(features)
    
    print("2️⃣ Calculating Euclidean Distance Diversity...")
    euclidean_div = calculate_euclidean_diversity(features)
    
    print("3️⃣ Calculating Feature Variance Diversity...")
    variance_div = calculate_feature_variance_diversity(features)
    
    print("4️⃣ Calculating Silhouette Cluster Diversity...")
    silhouette_div = calculate_silhouette_diversity(features)
    
    # Display results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    print(f"\n📐 COSINE SIMILARITY METRICS:")
    print(f"  Mean similarity: {cosine_div['mean_similarity']:.4f}")
    print(f"  Std deviation: {cosine_div['std_similarity']:.4f}")
    print(f"  Range: [{cosine_div['min_similarity']:.4f}, {cosine_div['max_similarity']:.4f}]")
    print(f"  Diversity score: {cosine_div['diversity_score']:.4f}")
    
    print(f"\n📏 EUCLIDEAN DISTANCE METRICS:")
    print(f"  Mean distance: {euclidean_div['mean_distance']:.4f}")
    print(f"  Std deviation: {euclidean_div['std_distance']:.4f}")
    print(f"  Range: [{euclidean_div['min_distance']:.4f}, {euclidean_div['max_distance']:.4f}]")
    print(f"  Diversity score: {euclidean_div['diversity_score']:.4f}")
    
    print(f"\n📈 FEATURE VARIANCE METRICS:")
    print(f"  Mean variance: {variance_div['mean_variance']:.4f}")
    print(f"  Std deviation: {variance_div['std_variance']:.4f}")
    print(f"  Range: [{variance_div['min_variance']:.4f}, {variance_div['max_variance']:.4f}]")
    print(f"  Diversity score: {variance_div['diversity_score']:.4f}")
    
    print(f"\n🎯 SILHOUETTE CLUSTER METRICS:")
    print(f"  Best clustering: {silhouette_div['best_n_clusters']} clusters")
    print(f"  Best silhouette score: {silhouette_div['best_score']:.4f}")
    print(f"  All cluster scores: {silhouette_div['all_scores']}")
    
    # Provide interpretation
    interpret_diversity_scores(cosine_div, euclidean_div, variance_div, silhouette_div)
    
    # Suggest images to remove for better diversity
    print("\n5️⃣ Analyzing Image Removal Suggestions...")
    removal_suggestions = suggest_images_to_remove(features, filenames)
    
    # Analyze cluster distribution
    print("\n6️⃣ Analyzing Cluster Distribution...")
    cluster_images, cluster_counts = analyze_cluster_distribution(features, filenames)

if __name__ == "__main__":
    main()