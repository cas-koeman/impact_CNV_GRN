#!/bin/bash

# Define the remote server
REMOTE_SERVER="ckoeman@MBIOHW30.bio.med.uni-muenchen.de"

# Base remote path
REMOTE_BASE_PATH="/work/project/ladcol_020"

# Base local path
LOCAL_BASE_PATH="/Users/caskoeman/Documents/LMU MCB/Master's Thesis/Results"

# SSH control path
CONTROL_PATH="/tmp/ssh-control-%r@%h:%p"

# Array of aliquots
ALIQUOTS=(
        "C3L-00004-T1_CPT0001540013"
        "C3L-00026-T1_CPT0001500003"
        "C3L-00088-T1_CPT0000870003"
        "C3L-00416-T2_CPT0010100001"
        "C3L-00448-T1_CPT0010160004"
        "C3L-00917-T1_CPT0023690004"
        "C3L-01313-T1_CPT0086820004"
        "C3N-00317-T1_CPT0012280004"
        "C3N-00495-T1_CPT0078510004"
)

# Files to download for each aliquot (relative paths)
declare -A FILE_PATHS
FILE_PATHS[0]="scGRNi/RNA/SCENIC/ccRCC_GBM/ALIQUOT/figures/regulon_heatmap.png"
FILE_PATHS[1]="scGRNi/RNA/SCENIC/ccRCC_GBM/ALIQUOT/figures/tumor_regulon_heatmap.png"
FILE_PATHS[2]="scGRNi/RNA/SCENIC/ccRCC_GBM/ALIQUOT/figures/umap_expr_cell_type.png"
FILE_PATHS[3]="scCNV/copyKAT/ccRCC_GBM/ALIQUOT/ALIQUOT_copykat_chromosome_heatmap_cell_types.jpg"
FILE_PATHS[4]="scCNV/inferCNV/ccRCC_GBM/ALIQUOT/infercnv.20_HMM_predHMMi6.leiden.hmm_mode-subclusters.Pnorm_0.5.repr_intensities.png"
FILE_PATHS[5]="scCNV/inferCNV/ccRCC_GBM/ALIQUOT/infercnv.png"
#FILE_PATHS[6]="integration_visualization/ccRCC_GBM/ALIQUOT/all_cnv_mosaic.png"
#FILE_PATHS[7]="integration_visualization/ccRCC_GBM/ALIQUOT/tf_overlap_venn.png"
#FILE_PATHS[8]="integration_visualization/ccRCC_GBM/ALIQUOT/tumor_cnv_mosaic.png"
FILE_PATHS[9]="integration_GRN_CNV/ccRCC_GBM/ALIQUOT/zscore_cnv_boxplot.png"
FILE_PATHS[10]="integration_GRN_CNV/ccRCC_GBM/ALIQUOT/zscore_cnv_violinplot.png"

# Start SSH control master connection
echo "Establishing SSH connection to $REMOTE_SERVER..."
ssh -M -S "$CONTROL_PATH" -o ControlPersist=yes "$REMOTE_SERVER" "echo Connection established"

# Loop through each aliquot
for aliquot in "${ALIQUOTS[@]}"; do
    echo "Processing aliquot: $aliquot"

    # Extract the CPT ID for files that need it
    cpt_id=$(echo $aliquot | cut -d'_' -f2)

    # Create local directory for this aliquot if it doesn't exist
    local_dir="${LOCAL_BASE_PATH}/${aliquot}"
    mkdir -p "$local_dir"

    # Download each file for this aliquot
    for i in "${!FILE_PATHS[@]}"; do
        # Replace ALIQUOT placeholder with actual aliquot ID
        file_path="${FILE_PATHS[$i]//ALIQUOT/$aliquot}"

        # Replace ALIQUOT_ID placeholder with CPT ID for files that need it
        file_path="${file_path//ALIQUOT_ID/$cpt_id}"

        # Full remote path for this file
        remote_path="${REMOTE_BASE_PATH}/${file_path}"

        echo "Downloading: $remote_path"

        # Execute SCP command with control path option
        scp -o ControlPath="$CONTROL_PATH" "${REMOTE_SERVER}:${remote_path}" "${local_dir}/"

        # Check if the download was successful
        if [ $? -eq 0 ]; then
            echo "Successfully downloaded to ${local_dir}/$(basename ${file_path})"
        else
            echo "Failed to download ${remote_path}"
        fi
    done

    echo "Completed downloads for $aliquot"
    echo "----------------------------------------"
done

# Close the SSH control master connection
echo "Closing SSH connection..."
ssh -S "$CONTROL_PATH" -O exit "$REMOTE_SERVER"

echo "All downloads completed!"