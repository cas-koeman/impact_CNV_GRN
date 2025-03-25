# Assessing the Impact of Copy Number Variation on Gene Regulatory Network Inference in scRNA-seq
## Description
In this project, I am to understand how copy number variations (CNVs) influence gene regulatory network (GRN) inference in single-cell RNA sequencing (scRNA-seq). 
There are two outcomes as of now: 
- Quantitative analysis of CNVs' impact on scRNA GRN inference, highlighting the limitations and robustness of current scRNA GRN inference methods under genomic instability
- Identification of biological pathways and GO terms that are especially affected by CNVs in scRNA datasets, providing insights into potential biases or current scGRN methods.
  
## Installation
To ensure reproducibility and ease of setup, YAML files have been provided to configure the necessary dependencies. Simply use your preferred environment manager to create the required setup:

```
conda env create -f environment.yaml
conda activate <env_name>
```
## Pipeline  
The analysis consists of three main components:

1. **CNV calling** - 2 methods are used to call the CNVs from the scRNA-seq data (the output of **inferCNV** is used for downstream visualization)
   - [copyKAT](https://github.com/cas-koeman/Master_Thesis/tree/main/copyKAT)
   - [inferCNV](https://github.com/cas-koeman/Master_Thesis/tree/main/inferCNV)
2. **GRN inference** – GRNs are created based on the scRNA-seq data
   - [SCENIC](https://github.com/cas-koeman/Master_Thesis/tree/main/SCENIC)
3. **Integration and visualization**

## Usage  

To run the integrated pipeline for copy number variation (CNV) and gene regulatory network (GRN) analysis on scRNA-seq data, follow these steps:  

### **Prerequisites**  
Ensure that you have access to the necessary computational resources and that the required software dependencies are installed. You will need:
- **Conda environment**: Install and activate the appropriate Conda environments (e.g., `copyKAT`, `r_env2`, `pyscenic`).
- **Input Data**: The script assumes that the required input data is already available in the specified directory paths.

### **Running the Pipeline**  
1. **Clone the repository** (if applicable):  
   ```
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
2. **Set up the conda environments**  
   The required environments are listed in the Conda YAML files provided (e.g., environment.yaml). Use the following command to set them up:
   ```
   conda env create -f environment.yaml
   conda activate <env_name>
3. **Prepare the sample and dataset configuration**
   Modify the variables DATASET_ID and SAMPLE_ID in the script to match your data. For example:
   ```
   DATASET_ID="ccRCC_GBM"  
   SAMPLE_ID="C3L-00004-T1_CPT0001540013"
4. **Submit the job to SLURM**
   This script is intended to be run on a SLURM-based cluster. You can submit the job using the following command:
   ```
   sbatch sc_analysis_pipeline.sh
   ```
5. **Monitor progress**  
   Monitor the progress of the job by checking the output and error logs:
    - Standard output: ``` /path/to/your/dir/utilities/logs/integrated_pipeline.out ```
    - Error output: ``` /path/to/your/dir/utilities/logs/integrated_pipeline.err ```
      
6. **Output directories**
   The results from each pipeline will be saved in the corresponding output directories:
    - copyKAT results: ```${BASE_DIR}/scCNV/copyKAT/${DATASET_ID}/${SAMPLE_ID}```
    - inferCNV results: ```${BASE_DIR}/scCNV/inferCNV/${DATASET_ID}/${SAMPLE_ID}```
    - pySCENIC results: ```${BASE_DIR}/scGRNi/RNA/SCENIC/${DATASET_ID}/${SAMPLE_ID}```

### Customization
You can modify the following parameters within the script to adjust the pipeline for different datasets or analysis configurations:
- CELL_TYPES: Specifies the cell types to be used in pySCENIC analysis (e.g., "None", "Tumor", "Non-Tumor").
- PRUNE_FLAGS: Specifies pruning options for pySCENIC (e.g., "None").
    - The default is that pruning is turned on, with no special folders being made.
    - Once pruning is specifically turned on or off (e.g. "True" or "False"), specific folders will be made where you can find the output ("pruned" or "unpruned")

