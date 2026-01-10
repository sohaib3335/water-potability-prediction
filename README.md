# Water Potability Prediction Using Machine Learning

A comprehensive Machine Learning project that predicts whether water is safe for human consumption based on physicochemical parameters. This project was developed as part of MSc Computing Independent Project.

## Live Demo

**[View the Streamlit App](https://water-potability-prediction-ip-assignment.streamlit.app/)**

## Project Overview

This project implements and compares four machine learning classification algorithms:
- **Logistic Regression** - Linear classifier with balanced class weights
- **Random Forest** - Ensemble of decision trees
- **XGBoost** - Gradient boosting with regularization
- **LightGBM** - Histogram-based gradient boosting

## Dataset

The Water Potability dataset contains **3,276 water samples** with 9 physicochemical quality indicators:

| Feature | Description | Unit |
|---------|-------------|------|
| pH | Acidity/alkalinity measure | 0-14 scale |
| Hardness | Capacity to precipitate soap | mg/L |
| Solids | Total dissolved solids (TDS) | ppm |
| Chloramines | Chloramine concentration | ppm |
| Sulfate | Dissolved sulfate amount | mg/L |
| Conductivity | Electrical conductivity | uS/cm |
| Organic Carbon | Organic carbon content | ppm |
| Trihalomethanes | THM concentration | ug/L |
| Turbidity | Light-emitting properties | NTU |

## Model Performance Results

| Algorithm | Accuracy | Precision | Recall | F1-Score | AUC |
|-----------|----------|-----------|--------|----------|-----|
| Logistic Regression | 52.59% | 0.42 | 0.53 | 0.47 | 0.548 |
| **Random Forest** | **65.85%** | **0.63** | 0.30 | 0.41 | 0.641 |
| XGBoost | 64.18% | 0.56 | 0.40 | 0.46 | 0.626 |
| LightGBM | 65.40% | 0.60 | 0.34 | 0.43 | **0.651** |

**Best Model:** Random Forest with 65.85% accuracy

## Installation

```bash
# Clone the repository
git clone https://github.com/sohaib3335/water-potability-prediction.git
cd water-potability-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

## Project Structure

```
water-potability-prediction/
├── streamlit_app.py                    # Streamlit web application
├── Water_Potability_ML_Project.ipynb   # Jupyter notebook with full analysis
├── Water_Potability_Report_Final.docx  # Final report with mathematical formulations
├── water_potability.csv                # Dataset
├── requirements.txt                    # Python dependencies
├── figures/                            # Generated charts and visualizations
│   ├── 01_missing_values.png
│   ├── 02_target_distribution.png
│   ├── 03_correlation_heatmap.png
│   ├── ...
└── README.md                           # Project documentation
```

## Features

- **Exploratory Data Analysis**: Comprehensive analysis with visualizations
- **Multiple ML Models**: Compare 4 different algorithms
- **Interactive Web App**: Real-time predictions via Streamlit
- **Mathematical Formulations**: Detailed documentation of algorithms
- **Model Evaluation**: ROC curves, PR curves, confusion matrices

## Technologies Used

- Python 3.12
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, LightGBM
- Streamlit (Web Application)
- Jupyter Notebook

## Author

**Sohaib Farooq**  
Email: sohaib.farooq@bigacademy.com  
MSc Computing - Independent Project

## License

This project is for educational purposes as part of MSc Computing coursework.
