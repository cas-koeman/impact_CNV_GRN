#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D

def load_and_prepare_data():
    """Load and prepare the correlation data for plotting."""
    print("Loading data files...")
    
    # Read the three TSV files
    within = pd.read_csv("C3L-00448-T1_CPT0010160004_within_region_correlations.tsv", sep='\t')
    intra = pd.read_csv("C3L-00448-T1_CPT0010160004_intrachromosomal_correlations_same_cnv.tsv", sep='\t')
    inter = pd.read_csv("C3L-00448-T1_CPT0010160004_interchromosomal_correlations_same_cnv.tsv", sep='\t')

    # Prepare the data for each file
    def prepare_data(df, region_type):
        melted = df.melt(value_vars=['Mean_Correlation_Altered', 'Mean_Correlation_Neutral'],
                        var_name='CNV_Status', value_name='Correlation')
        
        melted['CNV_Status'] = melted['CNV_Status'].replace({
            'Mean_Correlation_Altered': 'With CNV',
            'Mean_Correlation_Neutral': 'No CNV'
        })
        
        melted['Region_Type'] = region_type
        return melted

    # Prepare and combine data
    within_data = prepare_data(within, 'Within Region')
    intra_data = prepare_data(intra, 'Intrachromosomal')
    inter_data = prepare_data(inter, 'Interchromosomal')
    combined_data = pd.concat([within_data, intra_data, inter_data])
    
    print(f"Loaded data with {len(combined_data)} entries")
    return combined_data

def create_boxplot_plot(plot_df):
    """Create the boxplot visualization with adjusted styling."""
    print("\nCreating correlation plot...")    
    sns.set_style("ticks")
    plt.figure(figsize=(16, 9))  # Slide-friendly dimensions

    # Font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,  # Legend item size
        'legend.title_fontsize': 20  # Legend title size
    })

    # Set the order for the x-axis
    region_order = ['Within Region', 'Intrachromosomal', 'Interchromosomal']
    
    # Color palette
    cnv_palette = {'With CNV': '#d62728', 'No CNV': '#1f77b4'}

    # Create boxplot with increased spacing
    ax = sns.boxplot(
        data=plot_df,
        x='Region_Type',
        y='Correlation',
        hue='CNV_Status',
        palette={'With CNV': 'white', 'No CNV': 'white'},
        width=0.6,
        linewidth=1.5,
        fliersize=0,
        order=region_order,
        dodge=True,
        gap=0.1
    )
    
    # Set edge colors
    for i, artist in enumerate(ax.artists):
        artist.set_edgecolor(cnv_palette['With CNV'] if i % 2 == 0 else cnv_palette['No CNV'])
    
    # Add colored points with slightly increased jitter separation
    sns.stripplot(
        data=plot_df,
        x='Region_Type',
        y='Correlation',
        hue='CNV_Status',
        palette=cnv_palette,
        size=8,
        alpha=0.5,  # Increased alpha for better visibility
        jitter=0.3,
        ax=ax,
        edgecolor='none',
        order=region_order,
        dodge=True
    )

    ax.set_ylabel("Mean Correlation Coefficient", labelpad=15)
    ax.set_xlabel("")

    # Create custom legend with colored dots and title
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Altered (Gain/Loss)',
              markerfacecolor=cnv_palette['With CNV'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Neutral',
              markerfacecolor=cnv_palette['No CNV'], markersize=10)
    ]
    
    # Add legend with title and specified font sizes
    legend = ax.legend(handles=legend_elements, 
                      bbox_to_anchor=(1, 0.6), 
                      loc='upper left', 
                      borderaxespad=0., 
                      frameon=False,
                      title='CNV Status')
    
    # Ensure the title font size is applied (sometimes needs explicit setting)
    plt.setp(legend.get_title(), fontsize=20)
    
    sns.despine(top=True, right=True)
    plt.tight_layout()

    # Save plot
    output_path = os.path.join("correlation_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()

def main():
    """Main execution flow"""
    print("Starting correlation analysis...")
    
    # Load and process data
    plot_df = load_and_prepare_data()

    if plot_df is not None and not plot_df.empty:
        create_boxplot_plot(plot_df)
    else:
        print("No valid data to plot.")

    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()