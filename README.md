# Water Potability Prediction System 💧

A Machine Learning web application that predicts whether water is safe for human consumption based on physicochemical parameters.

## 🎯 Project Overview

This project implements and compares four machine learning algorithms for water potability classification:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## 📊 Dataset

The Water Potability dataset contains 3,276 water samples with 9 quality indicators:
- pH, Hardness, Solids, Chloramines, Sulfate
- Conductivity, Organic Carbon, Trihalomethanes, Turbidity

## 🚀 Live Demo

[View the Streamlit App](https://your-app-name.streamlit.app)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/water-potability-prediction.git
cd water-potability-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📈 Results

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Logistic Regression | 52.59% | 0.42 | 0.53 | 0.47 |
| Random Forest | **65.85%** | **0.63** | 0.30 | 0.41 |
| XGBoost | 63.72% | 0.55 | 0.36 | 0.44 |
| LightGBM | 64.63% | 0.58 | 0.36 | 0.44 |

## 📁 Project Structure

```
├── app.py                      # Streamlit web application
├── Water_Potability_ML_Project.ipynb  # Jupyter notebook with analysis
├── water_potability.csv        # Dataset
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 👤 Author

**MSc Computing - Independent Project**

## 📄 License

This project is for educational purposes.
