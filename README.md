# Predictive Maintenance & Anomaly Detection Pipeline using Graph Transformer Autoencoder

An end-to-end **Machine Learning pipeline** for **predictive maintenance** and **anomaly detection** in smart home appliances using a **Graph Transformer Autoencoder (GTAE)**. The system leverages the **REFIT Smart Home Dataset** to learn normal appliance behavior and identify early signs of hardware degradation through graph-based representation learning and reconstruction error analysis.

---

## 📌 Project Overview

Modern smart homes generate continuous streams of appliance power consumption data. Detecting subtle behavioral changes before hardware failure can significantly reduce maintenance costs and improve appliance reliability.

This project introduces a **Graph Transformer Autoencoder (GTAE)** that models relationships between household appliances as behavioral graphs. During inference, deviations from learned normal behavior are identified using reconstruction error, enabling early fault detection.

---

## ✨ Key Features

* End-to-end automated ML pipeline
* REFIT Smart Home Dataset preprocessing
* Synthetic dataset generation (fallback)
* Graph construction using **Jaccard Similarity**
* Graph Transformer Autoencoder implementation in PyTorch
* Multi-task learning:

  * Feature reconstruction
  * Graph structure reconstruction
* Synthetic fault injection for evaluation
* Real-time anomaly prediction
* Interactive HTML dashboard
* REST API for CSV uploads
* Automated report generation

---

# 📂 Project Structure

```text
Predictive-Maintenance-GTAE/
│
├── 1_raw_data/                             # Raw REFIT smart home dataset
│   ├── CLEAN_House1.csv                    # House 1 appliance power readings
│   ├── CLEAN_House2.csv                    # House 2 appliance power readings
│   └── ...                                 # Remaining house datasets
│
├── 3_processed_outputs/                    # Automatically generated outputs
│   ├── House_[ID]_Processed.csv            # Cleaned and normalized time-series data
│   ├── House_[ID]_Graphs.pt                # PyTorch graph datasets
│   ├── House_[ID]_GTAE.pth                 # Trained Graph Transformer Autoencoder model
│   ├── House_[ID]_Anomaly_Detection.png    # Reconstruction error visualization
│   ├── Master_Summary.csv                  # Summary statistics for all houses
│   ├── Anomaly_Report.json                 # Consolidated anomaly detection results
│   └── Verification_Report.html            # Data preprocessing verification report
│
├── .venv/                                  # Python virtual environment
│
├── .vscode/                                # VS Code configuration files
│
├── run_pipeline.py                         # Preprocesses raw REFIT data, normalizes power values, generates processed datasets, and creates master summaries
│
├── refit_processor.py                      # Core preprocessing module for cleaning, filtering, feature engineering, and standardizing appliance data
│
├── generate_synthetic_refit.py             # Generates synthetic REFIT-style datasets when raw data is unavailable
│
├── graph_builder.py                        # Builds behavioral graphs using sliding windows and Jaccard similarity between appliance states
│
├── graph_transformer.py                    # Implements the Graph Transformer Autoencoder (GTAE) architecture in PyTorch
│
├── fault_injector.py                       # Simulates appliance failures such as refrigerator drift and motor current spikes for evaluation
│
├── train_eval.py                           # Trains GTAE models, evaluates reconstruction errors, performs anomaly detection, and saves trained models
│
├── predict.py                              # Performs inference on new CSV files using trained GTAE models
│
├── compile_dashboard.py                    # Compiles anomaly reports and metadata into the final interactive HTML dashboard
│
├── dashboard_template.html                 # HTML template used to generate the predictive maintenance dashboard
│
├── report_template.html                    # HTML template for preprocessing verification reports
│
├── server.py                               # Starts a local web server for dashboard visualization and real-time CSV uploads
│
├── test_upload.py                          # Tests the server API by sending CSV files through HTTP POST requests
│
├── Predictive_Maintenance_Dashboard.html   # Final interactive dashboard for visualization and monitoring
│
├── Logo.png                                # Primary project logo
│
├── aegis_logo.png                          # Alternate logo asset
│
└── README.md                               # Project documentation
```


# 🚀 Pipeline Workflow

The project follows a five-stage processing pipeline.

## Phase 1 — Data Preprocessing

Clean, standardize, and normalize raw REFIT appliance power readings.

```bash
.venv\Scripts\python run_pipeline.py
```

This stage:

* Reads REFIT CSV files
* Cleans missing values
* Normalizes appliance power
* Generates processed datasets
* Creates synthetic data if raw data is unavailable

---

## Phase 2 — Graph Construction

Convert appliance behavior into graph representations.

```bash
.venv\Scripts\python graph_builder.py
```

Each sliding window becomes a graph where:

* Nodes → Appliances
* Edges → Behavioral similarity
* Edge weights → Jaccard Similarity

Outputs:

* Graph datasets (.pt)

---

## Phase 3 — Model Training & Evaluation

Train the Graph Transformer Autoencoder and evaluate anomaly detection.

```bash
.venv\Scripts\python train_eval.py
```

This phase:

* Trains GTAE
* Injects synthetic faults
* Computes reconstruction error
* Detects anomalies
* Saves trained models
* Generates visualization plots

---

## Phase 4 — Dashboard Compilation

Generate the interactive maintenance dashboard.

```bash
.venv\Scripts\python compile_dashboard.py
```

Outputs:

* Interactive HTML Dashboard
* Summary reports
* JSON metadata

---

## Phase 5 — Launch Web Server

Start the prediction server.

```bash
.venv\Scripts\python server.py
```

Open:

```
http://localhost:8000
```

The dashboard allows users to:

* Upload new CSV files
* Run predictions
* View detected anomalies
* Explore historical reports

---

# 🧠 Graph Transformer Autoencoder

The proposed model jointly learns:

* Temporal appliance behavior
* Appliance interaction graph

The network optimizes two objectives:

## 1. Feature Reconstruction

Reconstruct node features using Mean Squared Error (MSE).

## 2. Graph Reconstruction

Reconstruct adjacency matrices using Binary Cross Entropy (BCE).

This enables the model to detect:

* Behavioral drift
* Appliance degradation
* Structural relationship changes

---

# 📐 Mathematical Formulation

## Feature Normalization

[
P_{normalized}=\frac{P}{P_{max}}
]

---

## Behavioral Graph Construction

Edge weights are computed using **Jaccard Similarity**.

[
A_{i,j}
=======

\frac{|S_i\cap S_j|}
{|S_i\cup S_j|}
]

Expanded form:

[
A_{i,j}
=======

\frac{\sum_t(S_{i,t}S_{j,t})}
{\sum_tS_{i,t}
+\sum_tS_{j,t}
-\sum_t(S_{i,t}S_{j,t})}
]

Self-loops are retained:

[
A_{i,i}=1
]

---

## Multi-Task Loss

[
L_{total}=L_X+0.2L_A
]

Feature reconstruction loss:

[
L_X=
\frac{1}{BNWF}
\sum
(X-\hat X)^2
]

Graph reconstruction loss:

[
L_A=
-\frac{1}{BNN}
\sum
\left[
A\log(\hat A)
+
(1-A)\log(1-\hat A)
\right]
]

where

* **B** = Batch size
* **N** = Number of appliances
* **W** = Window length
* **F** = Feature dimension

---

# 🔍 Fault Injection Strategy

Synthetic degradation is injected into testing data to validate anomaly detection.

| Appliance        | Injected Fault | Simulated Failure                      |
| ---------------- | -------------- | -------------------------------------- |
| Fridge / Freezer | Cycle Drift    | Thermostat failure / Door seal leakage |
| Washing Machine  | Current Spikes | Bearing wear / Motor degradation       |
| Dryer            | Current Spikes | Heating element deterioration          |
| Dishwasher       | Current Spikes | Pump or motor wear                     |
| Other Appliances | None           | Baseline comparison                    |

---

# 📊 Anomaly Detection Metric

The anomaly score is computed using the reconstruction error drift ratio.

[
Drift\ Ratio=
\frac{MSE_{faulty}}
{MSE_{normal}}
]

Higher values indicate abnormal appliance behavior.

---

# 📈 Performance Evaluation

## Scenario A

Positive class:

```
ALERT (FAULT)
```

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | 57.14%  |
| Precision | 100.00% |
| Recall    | 1.22%   |
| F1 Score  | 2.41%   |

Characteristics:

* Extremely conservative
* No false alarms
* Detects only severe failures

---

## Scenario B

Positive class:

```
ALERT (FAULT) + WARNING
```

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | 100.00% |
| Precision | 100.00% |
| Recall    | 100.00% |
| F1 Score  | 100.00% |

Characteristics:

* Detects early degradation
* Suitable for predictive maintenance
* Captures subtle behavioral drift

---

# 📊 Generated Outputs

The pipeline automatically produces:

* Processed datasets
* PyTorch graph datasets
* Trained GTAE models
* Reconstruction error plots
* Verification reports
* JSON anomaly reports
* Interactive HTML dashboard

---

# ⚡ Real-Time Prediction

Run inference on a new CSV file:

```bash
.venv\Scripts\python predict.py --input-file 1_raw_data/House_1.csv
```

The prediction module:

* Loads trained GTAE models
* Processes uploaded data
* Computes reconstruction error
* Classifies anomalies
* Generates prediction reports

---

# 🛠️ Technology Stack

* Python
* PyTorch
* NumPy
* Pandas
* NetworkX
* HTML
* CSS
* JavaScript
* JSON

---

# 🎯 Applications

* Smart Home Monitoring
* Predictive Maintenance
* Energy Analytics
* Appliance Health Monitoring
* Industrial IoT
* Edge AI
* Smart Buildings

---

# 📚 Dataset

**REFIT Smart Home Dataset**

The REFIT dataset contains high-resolution household electricity consumption data collected from multiple residential homes in the United Kingdom, making it suitable for appliance-level energy analysis and predictive maintenance research.

---

# 👨‍💻 Author

**Internship NIML Project**

**Predictive Maintenance & Anomaly Detection using Graph Transformer Autoencoder**

Developed as an internship project focused on intelligent appliance health monitoring using graph-based deep learning.
