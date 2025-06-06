#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def clean_data(df):
    """Clean and prepare the plotting data"""
    print("Cleaning data...")
    
    # Remove 'overall' entries
    df = df[df['Subtype'] != 'overall'].copy()
    
    # Clean sample names (remove everything after first underscore)
    df['Sample'] = df['Sample_ID'].str.split('_').str[0]
    
    # Rename subclusters to "Subcluster 1", "Subcluster 2" etc.
    unique_subtypes = df['Subtype'].unique()
    subtype_mapping = {st: f"Subcluster {i+1}" for i, st in enumerate(sorted(unique_subtypes))}
    df['Subcluster'] = df['Subtype'].map(subtype_mapping)
    
    return df

def create_plots(plot_df, output_dir="plots"):
    """Create both CNA burden and CNH plots in specified style"""
    print("\nCreating plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set consistent styling
    sns.set_style("ticks")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20
    })

    # Create both plots
    for metric, ylabel in [('Mean_Burden(%)', "CNA Burden (%)"), 
                          ('Mean_CNH', "Copy Number Heterogeneity")]:
        
        print(f"Generating {metric} plot...")
        plt.figure(figsize=(16, 9))
    
        
        # Create boxplot
        ax = sns.boxplot(
            data=plot_df,
            x='Sample',
            y=metric,
            color='white',
            width=0.6,
            linewidth=1.5,
            fliersize=0,
        )
        
        # Add stripplot
        palette = sns.color_palette('coolwarm', n_colors=len(plot_df['Subcluster'].unique()))
        sns.stripplot(
            data=plot_df,
            x='Sample',
            y=metric,
            hue='Subcluster',
            palette=palette,
            size=10,
            alpha=0.9,
            jitter=0.2,
            ax=ax,
            edgecolor='none',
        )
        
        # Style plot
        ax.set_ylabel(ylabel, labelpad=15)
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1, 0.6), frameon=False, title='Subcluster', fontsize=18, title_fontsize=20)

        sns.despine(top=True, right=True)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f"{metric.split('(')[0].lower()}_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()

def main():
    """Main execution flow"""
    print("Starting CNA plotting...")
    
    # Configuration
    input_csv = "cna_burden_subtype_summary.csv"
    output_dir = "cna_plots"
    
    # Load and process data
    df = pd.read_csv(input_csv)
    plot_data = clean_data(df)
    print(f"Final dataset contains {len(plot_data)} entries across {len(plot_data['Sample'].unique())} samples")
    
    # Generate plots
    create_plots(plot_data, output_dir)
    
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()