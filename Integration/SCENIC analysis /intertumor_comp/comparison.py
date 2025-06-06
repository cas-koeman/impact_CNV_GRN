import os
import loompy as lp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def compare_tumor_level_regulons(sample_ids, dataset_id, output_base_dir):
    """Compare regulons across samples using the main tumor-level regulon files

    This function uses the main Tumor_pyscenic_output.loom file for each sample instead of
    aggregating from subclusters, which avoids any duplication issues.
    """

    base_scenic_dir = "/work/project/ladcol_020/scGRNi/RNA/SCENIC"
    sample_regulons = {}
    regulon_counts = {}

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Load regulons from the main tumor loom file for each sample
    for sample_id in sample_ids:
        sample_path = os.path.join(base_scenic_dir, dataset_id, sample_id)
        tumor_loom_path = os.path.join(sample_path, "Tumor_pyscenic_output.loom")

        if not os.path.exists(tumor_loom_path):
            print(f"Warning: Tumor loom file not found - {tumor_loom_path}")
            continue

        try:
            with lp.connect(tumor_loom_path, mode='r', validate=False) as lf:
                regulon_names = set(lf.ca['RegulonsAUC'].dtype.names)
                sample_regulons[sample_id] = regulon_names
                regulon_counts[sample_id] = len(regulon_names)
                print(f"Sample {sample_id}: {len(regulon_names)} regulons")
        except Exception as e:
            print(f"Error processing {tumor_loom_path}: {e}")

    samples = sorted(sample_regulons.keys())

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

    # Pearson correlation matrix
    correlation_matrix = np.zeros((len(samples), len(samples)))
    all_regulons = sorted(set.union(*sample_regulons.values()))

    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            vec1 = np.array([1 if reg in sample_regulons[s1] else 0 for reg in all_regulons])
            vec2 = np.array([1 if reg in sample_regulons[s2] else 0 for reg in all_regulons])
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                correlation_matrix[i, j] = np.corrcoef(vec1, vec2)[0, 1]

    correlation_df = pd.DataFrame(correlation_matrix, index=samples, columns=samples)

    # Rename samples for better visualization
    pretty_names = {sid: f"Run {i+1}" for i, sid in enumerate(samples)}
    overlap_df.rename(index=pretty_names, columns=pretty_names, inplace=True)
    similarity_df.rename(index=pretty_names, columns=pretty_names, inplace=True)
    correlation_df.rename(index=pretty_names, columns=pretty_names, inplace=True)

    # Print results
    print("\nAbsolute Regulon Overlap Matrix Between Samples (Tumor-level):")
    print(overlap_df)
    print("\nJaccard Similarity Matrix Between Samples:")
    print(similarity_df)
    print("\nPearson Correlation Matrix Between Samples:")
    print(correlation_df)

    # Create visualizations
    output_prefix = os.path.join(output_base_dir, "tumor_level_regulon_comparison")

    # Save only overlap and Jaccard dataframes
    overlap_df.to_csv(f"{output_prefix}_absolute.csv")
    similarity_df.to_csv(f"{output_prefix}_jaccard.csv")

    # Heatmap for absolute overlap
    plt.figure(figsize=(12, 10))
    sns.heatmap(overlap_df, annot=True, fmt="d", cmap="coolwarm")
    plt.title("Regulon Overlap Between Samples (Absolute Counts)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_absolute_heatmap.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_prefix}_absolute_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Lower-triangle heatmap for Jaccard similarity
    mask = np.triu(np.ones_like(similarity_df, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(similarity_df, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1,
                cbar_kws={"orientation": "horizontal", "shrink": 0.6, "pad": 0.25, "label": "Jaccard index"}, ax=ax)
    plt.title("Regulon Similarity Between Samples", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_jaccard_heatmap.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_prefix}_jaccard_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Lower-triangle heatmap for Pearson correlation
    mask = np.triu(np.ones_like(correlation_df, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(correlation_df, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1,
                cbar_kws={"orientation": "horizontal", "shrink": 0.6, "pad": 0.25, "label": "Pearson coefficient"}, ax=ax)
    plt.title("Regulon Correlation Between Samples", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_correlation_heatmap.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_prefix}_correlation_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nSaved all tumor-level comparison results to: {output_prefix}.*")

    return overlap_df, similarity_df, correlation_df


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
    ]

    print(f"\n{'='*80}")
    print("COMPARING TUMOR-LEVEL REGULONS ACROSS SAMPLES")
    print(f"{'='*80}")

    overlap_df, similarity_df, correlation_df = compare_tumor_level_regulons(
        sample_ids,
        dataset_id,
        output_base_dir
    )

    print("\n[Complete] Tumor-level regulon analysis finished")


if __name__ == "__main__":
    main()
