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
CRAMERS_V_FILE = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/allresults_scenic_analysis.csv"
OUTPUT_FILE = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/cramers_v_boxplot.png"

def load_and_prepare_data():
    """Load and prepare the data for plotting."""
    try:
        cramers_df = pd.read_csv(CRAMERS_V_FILE)
        cramers_df = cramers_df[
            (cramers_df['analysis_type'] == 'simplified') &
            (cramers_df['population'].str.startswith('Subcluster_Tumor_'))
        ].copy()

        cramers_df['Sample'] = cramers_df['sample'].str.split('_').str[0]
        cramers_df['Subcluster'] = 's' + cramers_df['population'].str.extract(r's(\d+)')[0]
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
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 17

    ax = sns.boxplot(
        data=plot_df,
        x='Sample',
        y='cramers_v',
        color='white',
        width=0.6,
        linewidth=2.5,
        fliersize=0
    )

    unique_subclusters = sorted(plot_df['Subcluster'].unique())
    palette = sns.color_palette('icefire', n_colors=len(unique_subclusters))

    sns.stripplot(
        data=plot_df,
        x='Sample',
        y='cramers_v',
        hue='Subcluster',
        palette=palette,
        size=8,
        alpha=0.9,
        jitter=0.2,
        ax=ax,
        edgecolor='none'
    )

    ax.axhline(0.2, color='gray', linestyle='--', linewidth=2, alpha=0.3)

    ax.set_xlabel("Sample", labelpad=15)
    ax.set_ylabel("Cramer's V", labelpad=15)

    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Subcluster')
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