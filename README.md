# 🧠 Face Cluster AI

A robust Computer Vision pipeline for automatically identifying, grouping, and organizing images of the same person using **InsightFace (ArcFace)** embeddings and **DBSCAN** clustering.

This project was developed as part of the **Computer Vision Engineer Assessment**.

---

# 📌 Project Overview

The system automatically detects faces from an unorganized image dataset, extracts high-dimensional facial embeddings, clusters images belonging to the same individual, assigns confidence scores, and generates organized outputs along with reports and visualizations.

---

# ✨ Features

- ✅ Automatic Face Detection
- ✅ Face Embedding Generation using InsightFace (ArcFace)
- ✅ Face Clustering using DBSCAN
- ✅ Confidence Score Calculation
- ✅ Automatic Folder Organization
- ✅ CSV Report Generation
- ✅ PCA-based Cluster Visualization
- ✅ Modular & Clean Architecture
- ✅ Configurable Project Structure

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Programming Language |
| InsightFace | Face Detection & Recognition |
| ArcFace | Face Embeddings |
| OpenCV | Image Processing |
| NumPy | Numerical Computing |
| Scikit-learn | DBSCAN & PCA |
| Matplotlib | Visualization |
| PyYAML | Configuration |
| Rich | Console Output |
| tqdm | Progress Bar |

---

# 📂 Project Structure

```text
face-cluster-ai/
│
├── dataset/
│   └── person_identification/
│
├── output/
│   ├── Person_1/
│   ├── Person_2/
│   └── Person_3/
│
├── report/
│   ├── report.csv
│   └── clusters.png
│
├── src/
│   ├── services/
│   ├── models/
│   ├── utils/
│   └── pipeline/
│
├── tests/
│
├── config.yaml
├── requirements.txt
├── README.md
└── main.py
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/priyanshucodes73/face-cluster-ai.git

cd face-cluster-ai
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📁 Dataset

Place your images inside

```text
dataset/person_identification/
```

Example

```text
dataset/
└── person_identification/
    ├── img1.jpg
    ├── img2.jpg
    ├── img3.jpg
```

---

# ▶️ Usage

Run

```bash
python main.py
```

---

# 📊 Output

After execution

```
Detected Faces
↓

Generated Face Embeddings
↓

Clustered Similar Faces
↓

Calculated Confidence Scores
↓

Generated CSV Report

↓

Created Cluster Visualization

↓

Organized Images into Output Folder
```

---

# 📈 Sample Console Output

```text
Total Images          : 6
Total Clusters        : 3
Average Confidence    : 90.98%
Processing Time       : 11.32 sec

Generated Files

✓ Output Folder : output/
✓ CSV Report    : report/report.csv
✓ Cluster Graph : report/clusters.png
```

---

# 📄 Generated Reports

## CSV Report

```
report/report.csv
```

Contains

- Image Name
- Cluster ID
- Confidence Score

---

## Visualization

```
report/clusters.png
```

Displays clustered face embeddings after PCA dimensionality reduction.

---

# 🎯 Results

The system successfully

- Detects faces from images
- Generates 512-dimensional ArcFace embeddings
- Groups images of the same individual
- Assigns confidence scores
- Creates organized folders
- Generates reports automatically

---

# 🚀 Future Improvements

- Multi-face image support
- Real-time webcam clustering
- FAISS-based large-scale search
- GPU acceleration
- Streamlit Web Interface
- REST API using FastAPI
- Docker Deployment
- Batch Processing
- Face Search by Query Image

---

# 📷 Sample Output

```
output/

├── Person_1
│   ├── person_01_0.jpg
│   └── person_01_1.jpg
│
├── Person_2
│   ├── person_02_0.jpg
│   └── person_02_1.jpg
│
└── Person_3
    ├── person_03_0.jpg
    └── person_03_1.jpg
```

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Mitthu Kumar Bhagat**

GitHub

https://github.com/priyanshucodes73

---

## ⭐ If you found this project useful, consider giving it a star!