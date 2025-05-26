#!/usr/bin/env python3
"""
Script to create boxplot of Cramér's V for subclusters per sample (simplified analysis only)
with subcluster-colored points and smaller fonts for slide presentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
CRAMERS_V_FILE = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/all_samples_results.csv"
OUTPUT_FILE = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/full_cramers_v_boxplot.png"

def load_and_prepare_data():
    """Load and prepare the data for plotting."""
    try:
        cramers_df = pd.read_csv(CRAMERS_V_FILE)
        cramers_df = cramers_df[
            (cramers_df['analysis_type'] == 'full') &
            (cramers_df['population'].str.startswith('Subcluster_Tumor_'))
        ].copy()

        cramers_df['Sample'] = cramers_df['sample'].str.split('_').str[0]
        subcluster_numbers = cramers_df['population'].str.extract(r's(\d+)')[0].astype(int)
        cramers_df['Subcluster'] = 'Subcluster ' + subcluster_numbers.astype(str)
        return cramers_df

    except Exception as e:
        print(f"Error loading or processing data: {str(e)}")
        return None

def create_boxplot_plot(plot_df):
    """Create the boxplot visualization with adjusted styling."""
    sns.set_style("ticks")
    plt.figure(figsize=(16, 9))  # Slide-friendly dimensions

    # Font sizes reduced ~30%
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 17

    # Order samples by decreasing median Cramér’s V
    sample_order = (
        plot_df.groupby('Sample')['cramers_v']
        .median()
        .sort_values(ascending=False)
        .index
    )

    ax = sns.boxplot(
        data=plot_df,
        x='Sample',
        y='cramers_v',
        color='white',
        width=0.6,
        linewidth=1,
        fliersize=0,
        order=sample_order
    )

    unique_subclusters = sorted(plot_df['Subcluster'].unique())
    palette = sns.color_palette('coolwarm', n_colors=len(unique_subclusters))

    sns.stripplot(
        data=plot_df,
        x='Sample',
        y='cramers_v',
        hue='Subcluster',
        palette=palette,
        size=8,
        alpha=0.6,
        jitter=0.2,
        ax=ax,
        edgecolor='none',
        order=sample_order
    )

    ax.axhline(0.05, color='gray', linestyle='--', linewidth=1.5, alpha=0.3)
    ax.axhline(0.15, color='gray', linestyle='--', linewidth=1.5, alpha=0.3)

    ax.set_ylabel("Cramer's V", labelpad=15)
    ax.set_xlabel("")

    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    legend = plt.legend(bbox_to_anchor=(1, 0.6), loc='upper left', borderaxespad=0., frameon=False,)
    sns.despine(top=True, right=True)
    plt.tight_layout()

    try:
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Boxplot saved to:\n{OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
    finally:
        plt.close()

def main():
    print("Creating Cramér's V boxplot for subclusters...")
    plot_df = load_and_prepare_data()

    if plot_df is not None and not plot_df.empty:
        create_boxplot_plot(plot_df)
    else:
        print("No valid data to plot.")

    print("Script completed.")

if __name__ == "__main__":
    main()
