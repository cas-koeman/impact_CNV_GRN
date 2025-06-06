#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.collections import PolyCollection
from scipy import stats

def load_and_prepare_data():
    """Load and prepare the comparison data for plotting."""
    print("Loading Cramer's V and CNA data...")
    try:
        # Load data files
        cramers_df = pd.read_csv("/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/allresults_scenic_analysis.csv")
        burden_df = pd.read_csv("/work/project/ladcol_020/integration_GRN_CNV/file_preparation/cna_burden_subtype_summary.csv")

        # Filter and clean data
        cramers_df = cramers_df[cramers_df['analysis_type'] == 'simplified'].copy()
        
        # Process burden data
        results = {}
        for _, row in burden_df.iterrows():
            sample_id = row['Sample_ID']
            subtype = row['Subtype'].replace('Tumor_', '') if 'Tumor_' in row['Subtype'] else row['Subtype']
            if sample_id not in results:
                results[sample_id] = {}
            results[sample_id][subtype] = {
                'mean_burden': row['Mean_Burden(%)'],
                'mean_cnh': row['Mean_CNH'],
                'total_cells': row['Number_of_Cells']
            }

        # Combine datasets
        plot_data = []
        for sample_id, sample_results in results.items():
            short_sample_id = sample_id.split('_')[0]
            for subtype, result in sample_results.items():
                if subtype == 'overall' and len(sample_results) > 1:
                    continue
                cramers_row = cramers_df[
                    (cramers_df['sample'] == sample_id) & 
                    (cramers_df['population'] == f"Subcluster_Tumor_{subtype}" if subtype != 'overall' else False)
                ]
                if not cramers_row.empty:
                    plot_data.append({
                        'Sample': short_sample_id,
                        'Subtype': subtype,
                        'Mean_CNA_Burden': result['mean_burden'],
                        'Mean_CNH': result['mean_cnh'],  
                        'Cramers_V': cramers_row.iloc[0]['cramers_v'],
                        'Number_of_Cells': result['total_cells']
                    })
        
        plot_df = pd.DataFrame(plot_data).dropna()
        print(f"Loaded data with {len(plot_df)} entries across {len(plot_df['Sample'].unique())} samples")
        return plot_df

    except Exception as e:
        print(f"Error loading or processing data: {str(e)}")
        return None

def create_comparison_plot(plot_df):
    """Create the dual comparison plot with consistent styling."""
    print("\nCreating comparison plot...")    
    # Set consistent styling
    sns.set_style("ticks")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20
    })

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    # Define colors
    main_color = '#CCCCCC'  # Medium gray for regression lines
    fill_color = '#DDDDDD'  # Light gray for confidence intervals
    
    # Create color mapping for samples using coolwarm colormap
    unique_samples = plot_df['Sample'].unique()
    sample_colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_samples)))
    color_dict = dict(zip(unique_samples, sample_colors))

    # Plot 1: Cramér's V vs CNA Burden
    # First plot the regression line
    sns.regplot(
        data=plot_df,
        x='Mean_CNA_Burden',
        y='Cramers_V',
        scatter=False,
        line_kws={'color': main_color, 'linewidth': 2.5},
        ci=95,
        ax=ax1
    )
    
    # Then add the scatter points colored by sample
    for sample, group in plot_df.groupby('Sample'):
        ax1.scatter(
            x=group['Mean_CNA_Burden'],
            y=group['Cramers_V'],
            s=100,
            alpha=0.6,
            color=color_dict[sample],
            edgecolor='white',
            label=None  # No label to avoid legend
        )
    
    # Calculate and print regression stats for plot 1
    valid_data = plot_df[['Mean_CNA_Burden', 'Cramers_V']].dropna()
    if len(valid_data) > 1:
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(
            valid_data['Mean_CNA_Burden'], 
            valid_data['Cramers_V']
        )
        r_squared1 = r_value1**2
        print(f"\nCNA Burden Regression (n={len(valid_data)}):")
        print(f"R² = {r_squared1:.3f}")
        print(f"Slope = {slope1:.5f}")
        print(f"Intercept = {intercept1:.5f}")
        print(f"p-value = {p_value1:.5f}")
    else:
        print("\nInsufficient data for CNA Burden regression")
    
    # Style plot 1
    ax1.set_xlabel('Mean CNA Burden (%)', labelpad=10)
    ax1.set_ylabel("Cramér's V", labelpad=10)
    
    # Plot 2: Cramér's V vs CNH
    # First plot the regression line
    sns.regplot(
        data=plot_df,
        x='Mean_CNH',
        y='Cramers_V',
        scatter=False,
        line_kws={'color': main_color, 'linewidth': 2.5},
        ci=95,
        ax=ax2
    )
    
    # Then add the scatter points colored by sample
    for sample, group in plot_df.groupby('Sample'):
        ax2.scatter(
            x=group['Mean_CNH'],
            y=group['Cramers_V'],
            s=100,
            alpha=0.6,
            color=color_dict[sample],
            edgecolor='white',
            label=None  # No label to avoid legend
        )
    
    # Calculate and print regression stats for plot 2
    valid_data = plot_df[['Mean_CNH', 'Cramers_V']].dropna()
    if len(valid_data) > 1:
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
            valid_data['Mean_CNH'], 
            valid_data['Cramers_V']
        )
        r_squared2 = r_value2**2
        print(f"\nCNH Regression (n={len(valid_data)}):")
        print(f"R² = {r_squared2:.3f}")
        print(f"Slope = {slope2:.5f}")
        print(f"Intercept = {intercept2:.5f}")
        print(f"p-value = {p_value2:.5f}")
    else:
        print("\nInsufficient data for CNH regression")
    
    # Style plot 2
    ax2.set_xlabel('Mean Copy Number Heterogeneity', labelpad=10)
    ax2.yaxis.label.set_visible(False)
    ax2.ticklabel_format(style='plain', axis='x')

    # Adjust confidence intervals and colors
    for ax in [ax1, ax2]:
        for collection in ax.collections:
            if isinstance(collection, PolyCollection):
                collection.set_alpha(0.2)
                collection.set_facecolor(fill_color)
        sns.despine(ax=ax, top=True, right=True)

    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    output_path = os.path.join("cramers_v_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()

def main():
    """Main execution flow"""
    print("Starting Cramer's V comparison analysis...")
    
    # Load and process data
    plot_df = load_and_prepare_data()

    if plot_df is not None and not plot_df.empty:
        create_comparison_plot(plot_df)
    else:
        print("No valid data to plot.")

    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()