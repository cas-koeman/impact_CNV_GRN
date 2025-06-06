import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D
from scipy import stats

# Define file paths
NORMAL_FILE = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/all_samples_results.csv"
RESIDUALIZED_FILE = "/work/project/ladcol_020/residual_CNV/ccRCC_GBM/merged_subclusters_results.csv"
OUTPUT_FILE = "/work/project/ladcol_020/residual_CNV/ccRCC_GBM/simplified_paired_cramers_v_boxplot.png"

def load_and_prepare_data():
    """Load and prepare the data from both files, finding shared samples for simplified analysis."""
    try:
        # Load normal expression data (simplified)
        normal_df = pd.read_csv(NORMAL_FILE)
        normal_df = normal_df[
            (normal_df['analysis_type'] == 'simplified') &
            (normal_df['population'].str.startswith('Subcluster_Tumor_'))
        ].copy()
        normal_df['Sample'] = normal_df['sample'].str.split('_').str[0]
        normal_df['Expression_Type'] = 'Normal'
        
        # Load residualized expression data (simplified)
        residual_df = pd.read_csv(RESIDUALIZED_FILE)
        residual_df = residual_df[
            (residual_df['analysis_type'] == 'simplified') &
            (residual_df['population'].str.startswith('Subcluster_Tumor_'))
        ].copy()
        residual_df['Sample'] = residual_df['sample'].str.split('_').str[0]
        residual_df['Expression_Type'] = 'Residualized'
        
        # Find shared samples
        shared_samples = set(normal_df['Sample']).intersection(set(residual_df['Sample']))
        print(f"Found {len(shared_samples)} shared samples for simplified analysis")
        
        # Filter to only shared samples and combine
        normal_df = normal_df[normal_df['Sample'].isin(shared_samples)]
        residual_df = residual_df[residual_df['Sample'].isin(shared_samples)]
        
        # Combine and reset index to avoid duplicate index issues
        combined_df = pd.concat([normal_df, residual_df]).reset_index(drop=True)
        
        return combined_df

    except Exception as e:
        print(f"Error loading or processing data: {str(e)}")
        return None

def create_boxplot_plot(plot_df):
    """Create the paired boxplot visualization with adjusted styling."""
    print("\nCreating paired Cramér's V plot for simplified analysis...")
    
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

    # Create the figure and axis first
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Create boxplot with white boxes
    sns.boxplot(
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
        gap=0.1,
        ax=ax
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
        alpha=0.7,
        jitter=0.25,
        ax=ax,
        edgecolor='none',
        order=sample_order,
        dodge=True
    )

    # Add reference lines
    ax.axhline(0.1, color='gray', linestyle='--', linewidth=1.5, alpha=0.3)
    ax.axhline(0.3, color='gray', linestyle='--', linewidth=1.5, alpha=0.3)

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

def perform_paired_test(combined_df):
    """Perform Wilcoxon signed-rank test on Cramér's V values."""
    try:
        # Calculate mean Cramér's V per sample for each expression type
        grouped = combined_df.groupby(['Sample', 'Expression_Type'])['cramers_v'].mean().unstack()
        
        # Extract the paired values
        normal_values = grouped['Normal'].values
        residual_values = grouped['Residualized'].values
        
        # Perform Wilcoxon signed-rank test
        wilcoxon_result = stats.wilcoxon(normal_values, residual_values)
        
        # Calculate median differences (more appropriate for non-parametric test)
        median_diff = np.median(normal_values - residual_values)
        
        print("\nWilcoxon signed-rank test results:")
        print(f"Number of samples: {len(normal_values)}")
        print(f"Median Cramér's V (Normal): {np.median(normal_values):.4f}")
        print(f"Median Cramér's V (Residualized): {np.median(residual_values):.4f}")
        print(f"Median difference (Normal - Residualized): {median_diff:.4f}")
        print(f"Test statistic: {wilcoxon_result.statistic:.4f}")
        print(f"p-value: {wilcoxon_result.pvalue:.4g}")
        
        # Also calculate effect size (matched-pairs rank-biserial correlation)
        effect_size = calculate_wilcoxon_effect_size(normal_values, residual_values)
        print(f"Effect size (rank-biserial correlation): {effect_size:.4f}")
        
        return wilcoxon_result, median_diff, effect_size
        
    except Exception as e:
        print(f"Error performing statistical test: {str(e)}")
        return None, None, None

def calculate_wilcoxon_effect_size(a, b):
    """Calculate matched-pairs rank-biserial correlation as effect size."""
    # Get the differences
    diffs = np.array(a) - np.array(b)
    
    # Remove zero differences if any
    diffs = diffs[diffs != 0]
    
    # Calculate effect size
    n = len(diffs)
    if n == 0:
        return 0.0
    r_plus = sum(diffs > 0)
    effect_size = (2 * r_plus / n) - 1
    return effect_size

def main():
    print("Creating paired Cramér's V boxplot for simplified analysis subclusters...")
    plot_df = load_and_prepare_data()

    if plot_df is not None and not plot_df.empty:
        # Perform the statistical test
        test_result, median_diff, effect_size = perform_paired_test(plot_df)
        
        # Get the grouped data properly
        grouped = plot_df.groupby(['Sample', 'Expression_Type'])['cramers_v'].mean().unstack()
        normal_vals = grouped['Normal'].values
        residual_vals = grouped['Residualized'].values
        
        # Print results in table format
        print("\n" + "="*60)
        print("STATISTICAL TEST RESULTS".center(60))
        print("="*60)
        
        # Prepare table data using the reliable values
        table_data = [
            ["Number of samples", f"{len(normal_vals)}"],
            ["Median Cramér's V (Normal)", f"{np.median(normal_vals):.4f}"],
            ["Median Cramér's V (Residualized)", f"{np.median(residual_vals):.4f}"],
            ["Median difference", f"{median_diff:.4f}"],
            ["Wilcoxon statistic", f"{test_result.statistic:.4f}"],
            ["p-value", f"{test_result.pvalue:.4g}"],
            ["Effect size (rank-biserial)", f"{effect_size:.4f}"]
        ]
        
        # Print the table
        max_label_length = max(len(row[0]) for row in table_data)
        for label, value in table_data:
            print(f"{label.ljust(max_label_length)} : {value}")
        
        print("="*60 + "\n")
        
        # Create the visualization
        create_boxplot_plot(plot_df)
    else:
        print("No valid data to plot.")

    print("Script completed.")

if __name__ == "__main__":
    main()