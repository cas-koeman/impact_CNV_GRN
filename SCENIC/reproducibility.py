import os
import loompy as lp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import spearmanr

def _get_target_genes(loom_connection):
    """Extract target genes for each regulon"""
    regulons_binary = np.array(loom_connection.ra['Regulons'].tolist())
    target_genes = {}
    for i, name in enumerate(loom_connection.ca['RegulonsAUC'].dtype.names):
        target_genes[name] = set(loom_connection.ra['Gene'][np.where(regulons_binary[:, i] == 1)[0]])
    return target_genes

def _get_all_edges(target_genes_dict):
    """Convert target genes dictionary to set of edges (TF-TG pairs)"""
    edges = set()
    for tf, tgs in target_genes_dict.items():
        for tg in tgs:
            edges.add((tf, tg))
    return edges

def compare_tumor_level_regulons(sample_ids, dataset_id, output_base_dir):
    """Compare regulons and their edges across samples using the main tumor-level regulon files"""
    base_scenic_dir = "/work/project/ladcol_020/scGRNi/RNA/SCENIC"
    sample_regulons = {}
    regulon_counts = {}
    sample_edges = {}  # Store edges for each sample
    all_edges = set()   # Store union of all edges across samples

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Load regulons and edges from the main tumor loom file for each sample
    for sample_id in sample_ids:
        sample_path = os.path.join(base_scenic_dir, dataset_id, sample_id)
        tumor_loom_path = os.path.join(sample_path, "Tumor_pyscenic_output.loom")

        if not os.path.exists(tumor_loom_path):
            print(f"Warning: Tumor loom file not found - {tumor_loom_path}")
            continue

        try:
            with lp.connect(tumor_loom_path, mode='r', validate=False) as lf:
                # Get regulon names
                regulon_names = set(lf.ca['RegulonsAUC'].dtype.names)
                sample_regulons[sample_id] = regulon_names
                regulon_counts[sample_id] = len(regulon_names)
                
                # Get edges (TF-TG pairs)
                target_genes = _get_target_genes(lf)
                edges = _get_all_edges(target_genes)
                sample_edges[sample_id] = edges
                all_edges.update(edges)
                
                print(f"Sample {sample_id}: {len(regulon_names)} regulons, {len(edges)} edges")
        except Exception as e:
            print(f"Error processing {tumor_loom_path}: {e}")

    samples = sorted(sample_regulons.keys())

    # ===== Regulon-level analysis (your original code) =====
    # Absolute overlap matrix
    overlap_matrix = np.zeros((len(samples), len(samples)), dtype=int)
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i <= j:
                intersection = len(sample_regulons[s1] & sample_regulons[s2])
                overlap_matrix[i, j] = intersection
                overlap_matrix[j, i] = intersection

    overlap_df = pd.DataFrame(overlap_matrix, index=samples, columns=samples)

    # Jaccard similarity matrix
    similarity_matrix = np.zeros((len(samples), len(samples)))
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i <= j:
                intersection = len(sample_regulons[s1] & sample_regulons[s2])
                union = len(sample_regulons[s1] | sample_regulons[s2])
                similarity = intersection / union if union > 0 else 0
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    similarity_df = pd.DataFrame(similarity_matrix, index=samples, columns=samples)

    # Spearman correlation matrix
    correlation_matrix = np.zeros((len(samples), len(samples)))
    all_regulons = sorted(set.union(*sample_regulons.values()))

    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            vec1 = np.array([1 if reg in sample_regulons[s1] else 0 for reg in all_regulons])
            vec2 = np.array([1 if reg in sample_regulons[s2] else 0 for reg in all_regulons])
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                correlation_matrix[i, j] = spearmanr(vec1, vec2)[0]

    correlation_df = pd.DataFrame(correlation_matrix, index=samples, columns=samples)

    # ===== Edge-level analysis (new code) =====
    # Edge overlap matrix
    edge_overlap_matrix = np.zeros((len(samples), len(samples)), dtype=int)
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i <= j:
                intersection = len(sample_edges[s1] & sample_edges[s2])
                edge_overlap_matrix[i, j] = intersection
                edge_overlap_matrix[j, i] = intersection

    edge_overlap_df = pd.DataFrame(edge_overlap_matrix, index=samples, columns=samples)

    # Edge Jaccard similarity matrix
    edge_similarity_matrix = np.zeros((len(samples), len(samples)))
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i <= j:
                intersection = len(sample_edges[s1] & sample_edges[s2])
                union = len(sample_edges[s1] | sample_edges[s2])
                similarity = intersection / union if union > 0 else 0
                edge_similarity_matrix[i, j] = similarity
                edge_similarity_matrix[j, i] = similarity

    edge_similarity_df = pd.DataFrame(edge_similarity_matrix, index=samples, columns=samples)

    # Edge Spearman correlation matrix
    edge_correlation_matrix = np.zeros((len(samples), len(samples)))
    all_edges_list = sorted(all_edges)  # Convert to list for consistent ordering

    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            vec1 = np.array([1 if edge in sample_edges[s1] else 0 for edge in all_edges_list])
            vec2 = np.array([1 if edge in sample_edges[s2] else 0 for edge in all_edges_list])
            if i == j:
                edge_correlation_matrix[i, j] = 1.0
            else:
                edge_correlation_matrix[i, j] = spearmanr(vec1, vec2)[0]

    edge_correlation_df = pd.DataFrame(edge_correlation_matrix, index=samples, columns=samples)

    # Rename samples for better visualization
    pretty_names = {sid: f"Run {i+1}" for i, sid in enumerate(samples)}
    overlap_df.rename(index=pretty_names, columns=pretty_names, inplace=True)
    similarity_df.rename(index=pretty_names, columns=pretty_names, inplace=True)
    correlation_df.rename(index=pretty_names, columns=pretty_names, inplace=True)
    edge_overlap_df.rename(index=pretty_names, columns=pretty_names, inplace=True)
    edge_similarity_df.rename(index=pretty_names, columns=pretty_names, inplace=True)
    edge_correlation_df.rename(index=pretty_names, columns=pretty_names, inplace=True)

    # Permanent Intersection (PerInt) matrix
    perint_matrix = np.zeros((len(samples), len(samples)))
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i <= j:
                intersection = len(sample_edges[s1] & sample_edges[s2])
                min_size = min(len(sample_edges[s1]), len(sample_edges[s2]))
                perint = intersection / min_size if min_size > 0 else 0
                perint_matrix[i, j] = perint
                perint_matrix[j, i] = perint

    perint_df = pd.DataFrame(perint_matrix, index=samples, columns=samples)
    perint_df.rename(index=pretty_names, columns=pretty_names, inplace=True)

    # Print results
    print("\nAbsolute Regulon Overlap Matrix Between Samples (Tumor-level):")
    print(overlap_df)
    print("\nJaccard Similarity Matrix Between Samples (Regulons):")
    print(similarity_df)
    print("\nSpearman Correlation Matrix Between Samples (Regulons):")
    print(correlation_df)
    print("\nAbsolute Edge Overlap Matrix Between Samples:")
    print(edge_overlap_df)
    print("\nJaccard Similarity Matrix Between Samples (Edges):")
    print(edge_similarity_df)
    print("\nSpearman Correlation Matrix Between Samples (Edges):")
    print(edge_correlation_df)
    print("\nperINT Matrix Between Samples (Edges):")
    print(perint_df)

    # Create visualizations
    output_prefix = os.path.join(output_base_dir, "tumor_level_regulon_comparison")

    # Save all dataframes
    overlap_df.to_csv(f"{output_prefix}_absolute.csv")
    similarity_df.to_csv(f"{output_prefix}_jaccard.csv")
    edge_overlap_df.to_csv(f"{output_prefix}_edge_absolute.csv")
    edge_similarity_df.to_csv(f"{output_prefix}_edge_jaccard.csv")

    # Heatmap for absolute edge overlap
    plt.figure(figsize=(12, 10))
    sns.heatmap(edge_overlap_df, annot=True, fmt="d", cmap="coolwarm")
    plt.title("Edge Overlap Between Samples (Absolute Counts)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_edge_absolute_heatmap.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_prefix}_edge_absolute_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Lower-triangle heatmap for edge Jaccard similarity
    mask = np.triu(np.ones_like(edge_similarity_df, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(edge_similarity_df, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1,
                cbar_kws={"orientation": "horizontal", "shrink": 0.6, "pad": 0.25, "label": "Jaccard index"}, ax=ax)
    plt.title("Edge Similarity Between Samples", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_edge_jaccard_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Lower-triangle heatmap for edge Spearman correlation
    mask = np.triu(np.ones_like(edge_correlation_df, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(edge_correlation_df, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1,
                cbar_kws={"orientation": "horizontal", "shrink": 0.6, "pad": 0.25, "label": "Spearman coefficient"}, ax=ax)
    plt.title("Edge Correlation Between Samples", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_edge_correlation_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Lower-triangle heatmap for edge perINT correlation
    mask = np.triu(np.ones_like(perint_df, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(perint_df, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1,
                cbar_kws={"orientation": "horizontal", "shrink": 0.6, "pad": 0.25, "label": "perINT score"}, ax=ax)
    plt.title("Edge Correlation Between Samples", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_perINT_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nSaved all tumor-level comparison results to: {output_prefix}.*")

    return (overlap_df, similarity_df, correlation_df, 
            edge_overlap_df, edge_similarity_df, edge_correlation_df)


def main():
    """Run tumor-level regulon analysis across samples"""
    dataset_id = "ccRCC_GBM/repr"
    output_base_dir = "/work/project/ladcol_020/scGRNi/RNA/SCENIC/ccRCC_GBM/repr"

    sample_ids = [
        "run_0",
        "run_1",
        "run_2",
        "run_3",
        "run_4",
        "run_5",
        "run_6",
        "run_7",
        "run_8"
    ]

    print(f"\n{'='*80}")
    print("COMPARING TUMOR-LEVEL REGULONS AND EDGES ACROSS SAMPLES")
    print(f"{'='*80}")

    results = compare_tumor_level_regulons(
        sample_ids,
        dataset_id,
        output_base_dir
    )

    print("\n[Complete] Tumor-level regulon and edge analysis finished")


if __name__ == "__main__":
    main()