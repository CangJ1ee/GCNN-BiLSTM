# SHAP-guided GCNN-BiLSTM

**An explainable intrusion detection model for smart transportation networks**

This repository contains the source code accompanying the manuscript submitted to *PeerJ Computer Science*:

> SHAP-guided GCNN-BiLSTM: An explainable intrusion detection model for smart transportation networks.

**Authors:** Yuan Liu, Huige Li (Corresponding Author), Zhenmin Shi, Yutao Lu, Dongsheng Wang  
**Affiliation:** Jiangsu University of Science & Technology, Zhenjiang, China

---

## Description

This project implements a hybrid deep learning intrusion detection system that combines:

- **Graph Neural Networks** — GCN (Graph Convolutional Network) and GAT (Graph Attention Network) to capture relational structure among traffic/node samples;
- **CNN–BiLSTM** — convolutional and bidirectional LSTM layers to extract local and sequential patterns from feature representations;
- **SHAP** — KernelSHAP-based post-hoc explanations of feature importance for predicted attack types.

The pipeline supports (1) multi-class network intrusion detection on the **CIC-IDS2017** traffic dataset and (2) malicious node detection in a **VANET / smart transportation** scenario. Ablation scripts are provided to evaluate the contribution of graph, CNN, and BiLSTM components.

---

## Dataset Information

### 1. CIC-IDS2017 (main experiments)

| Item | Detail |
|------|--------|
| Dataset | CIC-IDS2017 (ISCX Traffic Labelling CSV files) |
| Source | Canadian Institute for Cybersecurity — [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) |
| Files used | `Monday`–`Friday` WorkingHours `*.pcap_ISCX.csv` (8 files) |
| Label column | `Label` |
| Example classes | BENIGN, DDoS, DoS GoldenEye/Hulk/slowloris, PortScan, FTP-Patator, SSH-Patator, Heartbleed, etc. |

Copies of these CSV files may be placed in the project root. By default, `Main.py` reads absolute paths under `E:/NSLKdd/SHAP/TrafficLabelling/` — **update `dataset_paths` in `main()` to your local paths before running**.

### 2. VANET malicious nodes (smart transportation scenario)

| Item | Detail |
|------|--------|
| File | `vanet_malicious_nodes.csv` |
| Label column | `is_malicious` |
| Features | Node position/speed, packet statistics, trust score, attack-related counters, etc. |

Used by `VANET.py`. Update the path in `VANET.py` → `main()` if needed.

### Dataset citation

Please cite the CIC-IDS2017 dataset when using this code:

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

---

## Code Information

| File | Role |
|------|------|
| `Main.py` | **Main entry**: CIC-IDS2017 loading, preprocessing, similarity-graph construction, Hybrid GCN–GAT–CNN–BiLSTM training/evaluation, SHAP analysis, security advice demo |
| `VANET.py` | VANET malicious-node detection (Focal Loss, class weights, early stopping) |
| `noGraph.py` | Ablation: remove GCN/GAT (CNN–BiLSTM only) |
| `noCNN.py` | Ablation: remove CNN |
| `noBiLSTM.py` | Ablation: remove BiLSTM |
| `noCNN-BiLSTM.py` | Ablation: remove the CNN–BiLSTM block |

### Model overview (`HybridGCNGATCNNBiLSTM`)

```
Input features
    → Cosine-similarity graph (threshold 0.8, max 50 edges/node)
    → GCN × 2 → GAT × 2 (4 heads)
    → CNN (Conv1d) → BiLSTM
    → Fully connected classifier
```

For SHAP and sample-wise detection, a non-graph path (`use_graph=False`) feeds raw features into a parallel CNN–BiLSTM branch.

### Key hyperparameters (Main.py)

| Parameter | Value |
|-----------|--------|
| Train / test split | 70% / 30%, stratified, `random_state=42` |
| Optimizer | Adam, learning rate `0.001` |
| Loss | CrossEntropyLoss |
| Epochs | 100 (ablation scripts: 50) |
| Graph similarity threshold | 0.8 |
| Hidden sizes (GCN/GAT/LSTM) | 128 |
| GAT heads | 4 |
| Dropout | 0.1 |
| Device | CUDA if available, else CPU |

---

## Requirements

- Python 3.8+ (recommended)
- PyTorch
- PyTorch Geometric (`torch_geometric`)
- pandas, numpy
- scikit-learn
- shap
- matplotlib, seaborn
- transformers (optional; used for DialoGPT-enhanced security advice; rule-based advice works without it)

Install example (adjust PyTorch / PyG wheels for your CUDA version):

```bash
pip install torch pandas numpy scikit-learn shap matplotlib seaborn transformers
pip install torch-geometric
```

GPU is optional but recommended for full training and SHAP analysis.

---

## Usage Instructions

### 1. Prepare data

1. Download CIC-IDS2017 Traffic Labelling CSV files from the official CIC website (or use the CSV files provided with this repository).
2. Place them in a local directory.
3. Edit `dataset_paths` in `Main.py` (function `main()`) so each path points to your files. Example:

```python
dataset_paths = [
    "./Monday-WorkingHours.pcap_ISCX.csv",
    "./Tuesday-WorkingHours.pcap_ISCX.csv",
    "./Wednesday-workingHours.pcap_ISCX.csv",
    "./Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "./Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "./Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "./Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "./Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]
```

For VANET experiments, set the path to `vanet_malicious_nodes.csv` in `VANET.py`.

### 2. Run the main model (CIC-IDS2017 + SHAP)

```bash
python Main.py
```

Pipeline steps executed by `Main.py`:

1. Load and merge CSV files  
2. Preprocess (column cleaning, numeric conversion, inf/NaN handling, scaling, label encoding)  
3. Stratified train/test split  
4. Build cosine-similarity graphs  
5. Train Hybrid GCN–GAT–CNN–BiLSTM  
6. Evaluate (Accuracy, Precision, Recall, F1, confusion matrix)  
7. SHAP feature-importance analysis  
8. Optional attack-detection and security-advice demonstration  
9. Write reports / figures  

### 3. Run the VANET scenario

```bash
python VANET.py
```

### 4. Run ablation studies

```bash
python noGraph.py
python noCNN.py
python noBiLSTM.py
python noCNN-BiLSTM.py
```

Update dataset paths inside each ablation script the same way as in `Main.py`.

### 5. Outputs (typical)

| Output | Description |
|--------|-------------|
| `confusion_matrix.png` | Classification confusion matrix |
| `training_loss.png` | Training loss curve |
| `shap_summary_bar.png` | SHAP summary bar chart |
| `shap_beeswarm_*.png` | Per-attack SHAP beeswarm plots |
| `attack_feature_importance.png` | Feature importance by attack type |
| `shap_analysis_report.json` | Structured SHAP analysis results |
| `security_report.json` | Detection / advice demonstration report |
| `best_vanet_model.pth` | Best checkpoint (VANET script) |

---

## Methodology

1. **Data loading** — Merge daily CIC-IDS2017 CSV files (`latin1` encoding).  
2. **Preprocessing** — Normalize column names; drop identifiers (`Flow_ID`, IPs, `Timestamp`, etc.); convert to numeric; replace inf; impute NaN; clip extremes; `LabelEncoder` + `StandardScaler`.  
3. **Graph construction** — Cosine similarity between samples; connect pairs above threshold (default 0.8), capped per node; build PyTorch Geometric `Data` objects. Large sets may be subsampled for tractable graph building.  
4. **Modeling** — Stack GCN and GAT layers, then CNN–BiLSTM and a MLP classifier for multi-class prediction.  
5. **Training** — Adam optimizer, CrossEntropyLoss (Main) or Focal Loss with class weights (VANET), gradient clipping.  
6. **Evaluation** — Accuracy, Precision, Recall, F1; confusion matrix visualization.  
7. **Explainability** — SHAP `KernelExplainer` on a background set; aggregate absolute SHAP values by attack type; export plots and JSON reports.  
8. **Ablation** — Remove graph / CNN / BiLSTM modules to quantify each component’s contribution.

---

## Citations

If you use this code or the associated method in your research, please cite:

```
Liu, Y., Li, H., Shi, Z., Lu, Y., & Wang, D. SHAP-guided GCNN-BiLSTM: An explainable
intrusion detection model for smart transportation networks. Submitted to PeerJ
Computer Science. (under review)
```

Corresponding author: Huige Li, Jiangsu University of Science & Technology, Zhenjiang, China.

A journal DOI will be added after the article is accepted and published. Do not invent or placeholder a DOI before then.

Also cite the CIC-IDS2017 dataset as listed in the **Dataset Information** section.

---

## License & Contribution

This code is provided for academic research and reproducibility of the associated PeerJ submission.

For questions related to the manuscript or code, please contact the corresponding author Huige Li via the contact information listed on the manuscript / PeerJ submission.

Contributions or issue reports related to reproducibility are welcome; please describe your environment (OS, Python, PyTorch, CUDA versions) when reporting problems.

---

## Project Structure

```
GCNN-BiLSTM/
├── Main.py                 # Main CIC-IDS2017 + SHAP pipeline
├── VANET.py                # Smart transportation / VANET experiment
├── noGraph.py              # Ablation: without graph modules
├── noCNN.py                # Ablation: without CNN
├── noBiLSTM.py             # Ablation: without BiLSTM
├── noCNN-BiLSTM.py         # Ablation: without CNN–BiLSTM
├── vanet_malicious_nodes.csv
├── *.pcap_ISCX.csv         # CIC-IDS2017 CSV files (if bundled)
├── shap_analysis_report.json
├── security_report.json
└── README.md
```

---

## Notes

- Default data paths in the scripts are absolute Windows paths used during development; **you must edit them** before running on another machine.
- Full SHAP analysis can be time-consuming; `max_samples` in `run_shap_analysis` can be reduced for quicker trials.
- The optional `transformers` / DialoGPT component is not required for training, evaluation, or SHAP; if LLM loading fails, the code falls back to rule-based security recommendations.
