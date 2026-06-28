# Isorhamnetin-Pyroptosis-MultiOmics

A reproducible bioinformatics and computational pathology workflow for investigating the protective mechanism of **isorhamnetin (Isorhy)** in **Pasteurella multocida (P. multocida)-induced lung injury**, with a focus on pyroptosis-related molecular mechanisms and **NLRP2-centered evidence integration**.

The project integrates molecular docking, machine-learning recovery scoring, H&E image-based lung pathology quantification, transcriptomic differential expression analysis, enrichment analysis, GSEA, WGCNA, PPI network reconstruction, and final report generation.

## Project overview

This repository contains scripts used to build a multi-layer evidence chain:

```text
Animal phenotype → Histopathology → Inflammatory markers → Transcriptomics → Network mechanism
```

The workflow is organized around four analytical components:

1. **Molecular mechanism prediction**
   - AlphaFold2-based protein structure preparation
   - DiffDock-based ligand-protein docking
   - Pyroptosis target prioritization for GSDMD, GSDMD-NT, NLRP3, pro-Caspase-1, and cleaved-Caspase-1

2. **Animal-level recovery modelling**
   - Random forest-based recovery probability estimation
   - Feature importance analysis using inflammatory, pathological, bacterial-load, and apoptosis-related indicators
   - Group-level recovery trajectory assessment across Control, P. multocida, and Isorhy dose groups

3. **Histopathology and report asset reconstruction**
   - H&E image overlay panel assembly
   - Publication-style figure rebuilding
   - Pathology visualization and integrated report asset preparation

4. **Transcriptomic and network-level mechanism analysis**
   - Differential expression analysis visualization
   - Volcano plot correction and relabeling
   - GO / KEGG / Reactome enrichment visualization
   - Formal or exploratory GSEA
   - WGCNA / WGCNA-like co-expression analysis
   - STRING-enhanced or fallback PPI construction
   - AI-inspired PPI node prioritization
   - NLRP2-centered integrated evidence summary

## Main features

- Automatic detection of input expression, DEG, enrichment, candidate gene, and PPI-related files
- DEG summary tables, volcano plots, and heatmaps
- Recalculation of adjusted p-values for DEG result workbooks when needed
- GSEA focused on pyroptosis, inflammasome, innate immune, NOD-like receptor, and IL-1-related pathways
- WGCNA publication upgrade when the R `WGCNA` package is available
- Exploratory Python WGCNA-like fallback when formal WGCNA is unavailable
- PPI network generation using local evidence and optional STRING enhancement
- AI-inspired weighted PPI interpretation for candidate regulator prioritization
- Automated generation of English and Chinese reports
- Word document generation for full methods and results narrative

## Repository structure

```text
Isorhamnetin-Pyroptosis-MultiOmics/
├── README.md
├── scripts/
   ├── run_full_analysis.py
   ├── run_second_round_enhancement.py
   ├── run_publication_upgrade.py
   ├── run_formal_gsea.R
   ├── run_wgcna_publication.R
   ├── recalc_padj_and_plot_volcano.py
   ├── part2_recovery_rf.py
   ├── fix_report_assets.py
   └── build_full_story_doc.py

```

## Data requirements

The full workflow expects local input files such as:

- Differential expression result workbook
- Gene expression matrices
- Sample group information
- GO / KEGG / Reactome enrichment tables
- Candidate gene lists
- Optional local PPI files
- Optional H&E image overlays and report assets

Large raw data files are not included in this repository. Users should place their own data files under `input/` or the corresponding local asset folders.

## Installation

Create a clean Python environment:

```bash
conda create -n isorhy-pm python=3.11 -y
conda activate isorhy-pm
```

Install Python dependencies:

```bash
pip install pandas openpyxl numpy scipy matplotlib seaborn networkx requests gseapy statsmodels scikit-learn adjustText pillow python-docx
```

Optional R dependencies:

```r
install.packages(c("WGCNA", "jsonlite"))
BiocManager::install(c("fgsea", "msigdbr"))
```

## Main outputs

Typical outputs include:

```text
output/
├── tables/
│   ├── DEG_summary.csv
│   ├── GSEA_results.csv
│   ├── WGCNA_module_assignment.csv
│   ├── AI_PPI_node_importance.csv
│   └── NLRP2_integrated_evidence.csv
├── figures/
│   ├── volcano_plot.png
│   ├── expression_heatmap.png
│   ├── GSEA_pyroptosis_like_pathway.png
│   ├── WGCNA_module_trait_heatmap.png
│   └── AI_enhanced_PPI_network.png
├── figures_pub/
│   ├── volcano_Pm_vs_control_publication.png
│   ├── volcano_PmIroshy_vs_Pm_publication.png
│   ├── PPI_Nlrp2_centered_publication.png
│   └── AI_PPI_Nlrp2_centered_publication.png
├── networks/
│   ├── PPI_edges.csv
│   ├── PPI_nodes.csv
│   ├── PPI_edges_STRING_enhanced.csv
│   └── PPI_nodes_STRING_enhanced.csv
├── logs/
│   ├── analysis.log
│   ├── file_detection_summary.tsv
│   └── network_access.log
├── final_report.md
├── final_report_zh.md
└── completion_summary.txt
```

## Notes

This repository contains an analysis workflow rather than a finalized biological claim. Some modules can be formal or exploratory depending on the available data and local software environment.

In particular:

- GSEA is formal only when a valid ranked gene list and compatible gene-set resources are available.
- WGCNA is formal only when the R `WGCNA` package is installed and the sample size is suitable.
- Python WGCNA-like outputs should be treated as exploratory evidence.
- AI-enhanced PPI is an AI-inspired weighted network prioritization method, not a trained graph neural network unless explicitly replaced by a real GNN model.
- NLRP2 should be described as a candidate regulator unless supported by further experimental validation.
