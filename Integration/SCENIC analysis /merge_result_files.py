#!/usr/bin/env python3
import os
import glob
import pandas as pd
import argparse

def merge_result_csvs(base_dir="/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM", 
                      output_file="merged_analysis_results.csv"):
    """
    Loop through all sample-specific results CSV files and merge them into a single file,
    excluding any subclusters result files.
    
    Args:
        base_dir: Base directory containing sample folders
        output_file: Name of the output merged CSV file
    """
    # Find all sample result CSV files
    result_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if (file.endswith("_results.csv") 
                and "scenic_analysis" in root 
                and "_subclusters" not in file):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print(f"No result CSV files found in {base_dir}")
        return
    
    print(f"Found {len(result_files)} result files (excluding subclusters)")
    
    # Initialize an empty DataFrame to store the merged data
    merged_data = None
    
    # Process each file
    for i, file_path in enumerate(result_files):
        try:
            sample_name = os.path.basename(file_path).replace("_results.csv", "")
            print(f"Processing file {i+1}/{len(result_files)}: {sample_name}")
            df = pd.read_csv(file_path)
            if merged_data is None:
                merged_data = df
            else:
                merged_data = pd.concat([merged_data, df], ignore_index=True)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if merged_data is not None:
        output_path = os.path.join(base_dir, output_file)
        merged_data.to_csv(output_path, index=False)
        print(f"Successfully merged {len(result_files)} files into {output_path}")
        print(f"Total rows in merged file: {len(merged_data)}")
    else:
        print("No data was successfully loaded.")

def main():
    parser = argparse.ArgumentParser(description="Merge sample results CSV files into a single file.")
    parser.add_argument("--base_dir", type=str, 
                        default="/work/project/ladcol_020/residual_CNV/ccRCC_GBM",
                        help="Base directory containing sample folders")
    parser.add_argument("--output", type=str, 
                        default="all_samples_results.csv",
                        help="Name of the output merged CSV file")
    
    args = parser.parse_args()
    merge_result_csvs(args.base_dir, args.output)

if __name__ == "__main__":
    main()