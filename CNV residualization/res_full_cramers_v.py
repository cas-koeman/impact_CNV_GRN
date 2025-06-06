#!/usr/bin/env python3
"""
Script to create paired boxplots of Cramér's V for subclusters per sample comparing
Normal and Residualized expression, with expression-type-colored points.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D

# Define file paths
NORMAL_FILE = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/all_samples_results.csv"
RESIDUALIZED_FILE = "/work/project/ladcol_020/residual_CNV/ccRCC_GBM/merged_subclusters_results.csv"
OUTPUT_FILE = "/work/project/ladcol_020/residual_CNV/ccRCC_GBM/full_paired_cramers_v_boxplot.png"

def load_and_prepare_data():
    """Load and prepare the data from both files, finding shared samples."""
    try:
        # Load normal expression data
        normal_df = pd.read_csv(NORMAL_FILE)
        normal_df = normal_df[
            (normal_df['analysis_type'] == 'full') &
            (normal_df['population'].str.startswith('Subcluster_Tumor_'))
        ].copy()
        normal_df['Sample'] = normal_df['sample'].str.split('_').str[0]
        normal_df['Expression_Type'] = 'Normal'
        
        # Load residualized expression data
        residual_df = pd.read_csv(RESIDUALIZED_FILE)
        residual_df = residual_df[
            (residual_df['analysis_type'] == 'full') &
            (residual_df['population'].str.startswith('Subcluster_Tumor_'))
        ].copy()
        residual_df['Sample'] = residual_df['sample'].str.split('_').str[0]
        residual_df['Expression_Type'] = 'Residualized'
        
        # Find shared samples
        shared_samples = set(normal_df['Sample']).intersection(set(residual_df['Sample']))
        print(f"Found {len(shared_samples)} shared samples")
        
        # Filter to only shared samples and combine
        normal_df = normal_df[normal_df['Sample'].isin(shared_samples)]
        residual_df = residual_df[residual_df['Sample'].isin(shared_samples)]
        
        combined_df = pd.concat([normal_df, residual_df])
        
        return combined_df

    except Exception as e:
        print(f"Error loading or processing data: {str(e)}")
        return None

def create_boxplot_plot(plot_df):
    """Create the paired boxplot visualization with adjusted styling."""
    print("\nCreating paired Cramér's V plot...")
    
    sns.set_style("ticks")
    plt.figure(figsize=(16, 9))  # Slide-friendly dimensions

    # Font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20
    })

    # Order samples by decreasing median Cramér's V (using Normal expression)
    sample_order = (
        plot_df[plot_df['Expression_Type'] == 'Normal']
        .groupby('Sample')['cramers_v']
        .median()
        .sort_values(ascending=False)
        .index
    )

    # Create boxplot with white boxes
    ax = sns.boxplot(
        data=plot_df,
        x='Sample',
        y='cramers_v',
        hue='Expression_Type',
        palette=['white', 'white'],  # Both boxplots white
        width=0.6,
        linewidth=1.5,
        fliersize=0,
        order=sample_order,
        dodge=True,
        gap=0.1
    )
    
    # Set edge colors for the boxplots (both black)
    for artist in ax.artists:
        artist.set_edgecolor('black')
    
    # Add stripplot with expression type colors
    expr_palette = {'Normal': '#1f77b4', 'Residualized': '#d62728'}  # Blue for Normal, Red for Residualized
    sns.stripplot(
        data=plot_df,
        x='Sample',
        y='cramers_v',
        hue='Expression_Type',
        palette=expr_palette,
        size=8,
        alpha=0.5,
        jitter=0.25,
        ax=ax,
        edgecolor='none',
        order=sample_order,
        dodge=True
    )

    # Add reference lines
    ax.axhline(0.05, color='gray', linestyle='--', linewidth=1.5, alpha=0.3)
    ax.axhline(0.15, color='gray', linestyle='--', linewidth=1.5, alpha=0.3)

    # Style plot
    ax.set_ylabel("Cramér's V", labelpad=15)
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    
    # Create custom legend with colored dots and title
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal',
              markerfacecolor='#1f77b4', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Residualized',
              markerfacecolor='#d62728', markersize=10)
    ]
    
    # Add legend with title and specified font sizes
    legend = ax.legend(handles=legend_elements, 
                      bbox_to_anchor=(1, 0.6), 
                      loc='upper left', 
                      borderaxespad=0., 
                      frameon=False,
                      title='Expression matrix')
    
    # Set the legend title font size
    plt.setp(legend.get_title(), fontsize=20)

    sns.despine(top=True, right=True)
    plt.tight_layout()

    # Save plot
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Saved to {OUTPUT_FILE}")
    plt.close()

def main():
    print("Creating paired Cramér's V boxplot for subclusters...")
    plot_df = load_and_prepare_data()

    if plot_df is not None and not plot_df.empty:
        create_boxplot_plot(plot_df)
    else:
        print("No valid data to plot.")

    print("Script completed.")

if __name__ == "__main__":
    main()