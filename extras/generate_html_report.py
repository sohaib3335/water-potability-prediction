"""
Generate HTML Report with MathJax for proper LaTeX rendering
Author: Sohaib Farooq
Email: sohaib.farooq@bigacademy.com
"""

import os
import base64

def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return None

def create_img_tag(image_path, caption, fig_num):
    """Create HTML img tag with base64 embedded image"""
    b64 = image_to_base64(image_path)
    if b64:
        return f'''
        <figure>
            <img src="data:image/png;base64,{b64}" alt="{caption}" style="max-width: 100%; height: auto;">
            <figcaption><em>Figure {fig_num}: {caption}</em></figcaption>
        </figure>
        '''
    return f'<p><em>[Image not found: {image_path}]</em></p>'

html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Water Potability Prediction Using Machine Learning</title>
    
    <!-- MathJax for LaTeX rendering -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <style>
        body {
            font-family: 'Times New Roman', Times, serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
            line-height: 1.6;
            color: #333;
        }
        
        h1 {
            text-align: center;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }
        
        h2 {
            color: #2c3e50;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 8px;
            margin-top: 40px;
        }
        
        h3 {
            color: #34495e;
            margin-top: 30px;
        }
        
        h4 {
            color: #7f8c8d;
            margin-top: 20px;
        }
        
        .author-info {
            text-align: center;
            margin-bottom: 30px;
            color: #555;
        }
        
        .abstract {
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
        }
        
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        figure {
            text-align: center;
            margin: 30px 0;
            page-break-inside: avoid;
        }
        
        figcaption {
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        .equation-block {
            margin: 20px 0;
            overflow-x: auto;
        }
        
        ul, ol {
            margin: 15px 0;
        }
        
        li {
            margin: 8px 0;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .toc {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .toc ul {
            list-style-type: none;
            padding-left: 20px;
        }
        
        .toc > ul {
            padding-left: 0;
        }
        
        hr {
            border: none;
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }
        
        .findings {
            background-color: #e8f6f3;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        @media print {
            body {
                padding: 20px;
            }
            h2 {
                page-break-before: always;
            }
            h2:first-of-type {
                page-break-before: avoid;
            }
            figure {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>

<h1>Water Potability Prediction Using Machine Learning</h1>

<div class="author-info">
    <p><strong>Author:</strong> Sohaib Farooq<br>
    <strong>Email:</strong> sohaib.farooq@bigacademy.com<br>
    <strong>Date:</strong> January 2026</p>
</div>

<hr>

<div class="abstract">
<h2 style="margin-top: 0; border: none;">Abstract</h2>
<p>This report presents a comprehensive machine learning approach to predict water potability based on physicochemical properties. We implement and compare four classification algorithms: Logistic Regression, Random Forest, XGBoost, and LightGBM. The study includes detailed exploratory data analysis, feature engineering, model training, evaluation, and deployment of a web application for real-time predictions.</p>
</div>

<hr>

<div class="toc">
<h2 style="margin-top: 0; border: none;">Table of Contents</h2>
<ul>
    <li>1. <a href="#intro">Introduction</a></li>
    <li>2. <a href="#dataset">Dataset Description</a></li>
    <li>3. <a href="#eda">Exploratory Data Analysis</a></li>
    <li>4. <a href="#preprocessing">Data Preprocessing</a></li>
    <li>5. <a href="#algorithms">Machine Learning Algorithms</a>
        <ul>
            <li>5.1 <a href="#lr">Logistic Regression</a></li>
            <li>5.2 <a href="#rf">Random Forest</a></li>
            <li>5.3 <a href="#xgb">XGBoost</a></li>
            <li>5.4 <a href="#lgbm">LightGBM</a></li>
        </ul>
    </li>
    <li>6. <a href="#metrics">Evaluation Metrics</a></li>
    <li>7. <a href="#results">Results and Analysis</a></li>
    <li>8. <a href="#webapp">Web Application Deployment</a></li>
    <li>9. <a href="#conclusions">Conclusions</a></li>
    <li>10. <a href="#references">References</a></li>
</ul>
</div>

<hr>

<h2 id="intro">1. Introduction</h2>

<p>Access to clean drinking water is essential for human health. The World Health Organization (WHO) estimates that contaminated water causes approximately 485,000 diarrheal deaths annually. This project aims to develop a machine learning model capable of predicting whether water is safe for human consumption based on measurable physicochemical parameters.</p>

<p>The objective is to build a classification system that can accurately determine water potability, enabling water quality monitoring systems to make rapid assessments without extensive laboratory testing.</p>

<hr>

<h2 id="dataset">2. Dataset Description</h2>

<p>The dataset contains water quality metrics for 3,276 water samples with the following features:</p>

<table>
    <tr>
        <th>Feature</th>
        <th>Description</th>
        <th>Unit</th>
    </tr>
    <tr><td><strong>pH</strong></td><td>Measure of acidity/alkalinity</td><td>0-14 scale</td></tr>
    <tr><td><strong>Hardness</strong></td><td>Capacity of water to precipitate soap</td><td>mg/L</td></tr>
    <tr><td><strong>Solids</strong></td><td>Total dissolved solids (TDS)</td><td>ppm</td></tr>
    <tr><td><strong>Chloramines</strong></td><td>Amount of chloramines</td><td>ppm</td></tr>
    <tr><td><strong>Sulfate</strong></td><td>Amount of sulfate dissolved</td><td>mg/L</td></tr>
    <tr><td><strong>Conductivity</strong></td><td>Electrical conductivity</td><td>&mu;S/cm</td></tr>
    <tr><td><strong>Organic Carbon</strong></td><td>Amount of organic carbon</td><td>ppm</td></tr>
    <tr><td><strong>Trihalomethanes</strong></td><td>Amount of trihalomethanes</td><td>&mu;g/L</td></tr>
    <tr><td><strong>Turbidity</strong></td><td>Measure of light-emitting properties</td><td>NTU</td></tr>
    <tr><td><strong>Potability</strong></td><td>Target variable (0: Not Potable, 1: Potable)</td><td>Binary</td></tr>
</table>

<h3>Missing Values Analysis</h3>
<p>The dataset contains missing values that require handling before model training:</p>

''' + create_img_tag('figures/01_missing_values.png', 'Distribution of missing values across features', 1) + '''

<hr>

<h2 id="eda">3. Exploratory Data Analysis</h2>

<h3>3.1 Target Variable Distribution</h3>
<p>The dataset exhibits class imbalance, with approximately 61% non-potable samples and 39% potable samples:</p>

''' + create_img_tag('figures/02_target_distribution.png', 'Distribution of water potability classes', 2) + '''

<h3>3.2 Feature Correlation Analysis</h3>
<p>Understanding feature relationships is crucial for feature engineering and model interpretation:</p>

''' + create_img_tag('figures/streamlit_03_correlation_heatmap.png', 'Correlation matrix heatmap showing relationships between features', 3) + '''

<p>Key observations:</p>
<ul>
    <li>Most features show weak correlation with each other</li>
    <li>No strong multicollinearity issues detected</li>
    <li>The target variable has weak correlations with all features, indicating classification complexity</li>
</ul>

<h3>3.3 Feature Distributions</h3>

''' + create_img_tag('figures/04_feature_distributions.png', 'Histograms showing feature distributions by potability class', 4) + '''

<h3>3.4 Box Plot Analysis</h3>

''' + create_img_tag('figures/05_box_plots.png', 'Box plots comparing feature distributions between potable and non-potable water', 5) + '''

<hr>

<h2 id="preprocessing">4. Data Preprocessing</h2>

<h3>4.1 Missing Value Imputation</h3>
<p>Missing values were imputed using the median strategy, which is robust to outliers:</p>

<div class="equation-block">
\\[\\tilde{x}_j = \\text{median}(x_{1j}, x_{2j}, \\ldots, x_{nj})\\]
</div>

<p>where \\(\\tilde{x}_j\\) is the median value for feature \\(j\\).</p>

<h3>4.2 Feature Scaling</h3>
<p>Standard scaling was applied to normalize features:</p>

<div class="equation-block">
\\[z = \\frac{x - \\mu}{\\sigma}\\]
</div>

<p>where:</p>
<ul>
    <li>\\(x\\) is the original feature value</li>
    <li>\\(\\mu\\) is the mean of the feature</li>
    <li>\\(\\sigma\\) is the standard deviation</li>
    <li>\\(z\\) is the scaled value</li>
</ul>

<h3>4.3 Train-Test Split</h3>
<p>The dataset was split using stratified sampling:</p>
<ul>
    <li><strong>Training set:</strong> 80% (2,620 samples)</li>
    <li><strong>Test set:</strong> 20% (656 samples)</li>
</ul>

<hr>

<h2 id="algorithms">5. Machine Learning Algorithms</h2>

<h3 id="lr">5.1 Logistic Regression</h3>

<p>Logistic Regression is a linear classification algorithm that models the probability of binary outcomes using the logistic (sigmoid) function.</p>

<h4>Mathematical Formulation</h4>
<p>The hypothesis function:</p>

<div class="equation-block">
\\[h_\\theta(x) = \\sigma(\\theta^T x) = \\frac{1}{1 + e^{-\\theta^T x}}\\]
</div>

<p>where:</p>
<ul>
    <li>\\(\\sigma\\) is the sigmoid function</li>
    <li>\\(\\theta\\) represents the model parameters</li>
    <li>\\(x\\) is the input feature vector</li>
</ul>

<h4>Cost Function (Cross-Entropy Loss)</h4>

<div class="equation-block">
\\[J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y^{(i)} \\log(h_\\theta(x^{(i)})) + (1-y^{(i)}) \\log(1-h_\\theta(x^{(i)})) \\right]\\]
</div>

<p>where:</p>
<ul>
    <li>\\(m\\) is the number of training examples</li>
    <li>\\(y^{(i)}\\) is the actual label for sample \\(i\\)</li>
    <li>\\(h_\\theta(x^{(i)})\\) is the predicted probability</li>
</ul>

<h4>Gradient Descent Update</h4>

<div class="equation-block">
\\[\\theta_j := \\theta_j - \\alpha \\frac{\\partial J(\\theta)}{\\partial \\theta_j}\\]
</div>

<p>where \\(\\alpha\\) is the learning rate.</p>

<h4>Class Weight Balancing</h4>
<p>To address class imbalance, we apply balanced class weights:</p>

<div class="equation-block">
\\[w_c = \\frac{n}{k \\cdot n_c}\\]
</div>

<p>where:</p>
<ul>
    <li>\\(n\\) is the total number of samples</li>
    <li>\\(k\\) is the number of classes</li>
    <li>\\(n_c\\) is the number of samples in class \\(c\\)</li>
</ul>

<hr>

<h3 id="rf">5.2 Random Forest</h3>

<p>Random Forest is an ensemble learning method that constructs multiple decision trees and aggregates their predictions.</p>

<h4>Algorithm</h4>
<p>For \\(b = 1\\) to \\(B\\) (number of trees):</p>
<ol>
    <li>Draw a bootstrap sample \\(Z^*\\) of size \\(n\\) from training data</li>
    <li>Grow a decision tree \\(T_b\\) using recursive partitioning:
        <ul>
            <li>At each node, select \\(m \\approx \\sqrt{p}\\) features randomly</li>
            <li>Find the best split among selected features using Gini impurity</li>
            <li>Split the node into two child nodes</li>
        </ul>
    </li>
</ol>

<h4>Gini Impurity</h4>

<div class="equation-block">
\\[G = \\sum_{c=1}^{C} p_c(1-p_c) = 1 - \\sum_{c=1}^{C} p_c^2\\]
</div>

<p>where \\(p_c\\) is the proportion of samples belonging to class \\(c\\) at the node.</p>

<h4>Information Gain</h4>
<p>The best split maximizes information gain:</p>

<div class="equation-block">
\\[IG(D_p, f) = G(D_p) - \\sum_{j \\in \\{left, right\\}} \\frac{n_j}{n_p} G(D_j)\\]
</div>

<p>where:</p>
<ul>
    <li>\\(D_p\\) is the parent dataset</li>
    <li>\\(n_p\\) is the number of samples at parent node</li>
    <li>\\(D_j\\) is the child dataset after split</li>
</ul>

<h4>Final Prediction (Majority Voting)</h4>

<div class="equation-block">
\\[\\hat{y} = \\text{mode}\\{T_1(x), T_2(x), \\ldots, T_B(x)\\}\\]
</div>

<h4>Feature Importance</h4>

''' + create_img_tag('figures/06_feature_importance.png', 'Feature importance scores from Random Forest model', 6) + '''

<hr>

<h3 id="xgb">5.3 XGBoost</h3>

<p>XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting algorithm with regularization.</p>

<h4>Objective Function</h4>

<div class="equation-block">
\\[\\mathcal{L}(\\phi) = \\sum_{i=1}^{n} l(y_i, \\hat{y}_i) + \\sum_{k=1}^{K} \\Omega(f_k)\\]
</div>

<p>where:</p>
<ul>
    <li>\\(l\\) is the loss function (logistic loss for classification)</li>
    <li>\\(\\Omega\\) is the regularization term</li>
    <li>\\(K\\) is the number of trees</li>
</ul>

<h4>Regularization Term</h4>

<div class="equation-block">
\\[\\Omega(f) = \\gamma T + \\frac{1}{2}\\lambda \\sum_{j=1}^{T} w_j^2\\]
</div>

<p>where:</p>
<ul>
    <li>\\(T\\) is the number of leaves in the tree</li>
    <li>\\(w_j\\) is the weight of leaf \\(j\\)</li>
    <li>\\(\\gamma\\) and \\(\\lambda\\) are regularization parameters</li>
</ul>

<h4>Second-Order Taylor Expansion</h4>
<p>For each iteration \\(t\\):</p>

<div class="equation-block">
\\[\\mathcal{L}^{(t)} \\approx \\sum_{i=1}^{n} \\left[ g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i) \\right] + \\Omega(f_t)\\]
</div>

<p>where:</p>
<ul>
    <li>\\(g_i = \\frac{\\partial l(y_i, \\hat{y}^{(t-1)})}{\\partial \\hat{y}^{(t-1)}}\\) (gradient)</li>
    <li>\\(h_i = \\frac{\\partial^2 l(y_i, \\hat{y}^{(t-1)})}{\\partial (\\hat{y}^{(t-1)})^2}\\) (Hessian)</li>
</ul>

<h4>Optimal Leaf Weight</h4>

<div class="equation-block">
\\[w_j^* = -\\frac{\\sum_{i \\in I_j} g_i}{\\sum_{i \\in I_j} h_i + \\lambda}\\]
</div>

<h4>Split Gain</h4>

<div class="equation-block">
\\[\\text{Gain} = \\frac{1}{2} \\left[ \\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L + G_R)^2}{H_L + H_R + \\lambda} \\right] - \\gamma\\]
</div>

<hr>

<h3 id="lgbm">5.4 LightGBM</h3>

<p>LightGBM is a gradient boosting framework that uses histogram-based algorithms for efficient training.</p>

<h4>Gradient-based One-Side Sampling (GOSS)</h4>
<p>GOSS keeps samples with large gradients and randomly samples from small gradient instances:</p>
<ol>
    <li>Sort training instances by absolute gradient \\(|g_i|\\)</li>
    <li>Select top \\(a \\times 100\\%\\) instances with largest gradients</li>
    <li>Randomly sample \\(b \\times 100\\%\\) from remaining instances</li>
    <li>Amplify sampled small gradient data by factor \\(\\frac{1-a}{b}\\)</li>
</ol>

<h4>Exclusive Feature Bundling (EFB)</h4>
<p>EFB bundles mutually exclusive features to reduce dimensionality:</p>

<div class="equation-block">
\\[\\text{Conflict}(f_i, f_j) = \\sum_{x \\in D} \\mathbb{1}[f_i(x) \\neq 0 \\land f_j(x) \\neq 0]\\]
</div>

<p>Features are bundled if their conflict count is below a threshold.</p>

<h4>Leaf-wise Tree Growth</h4>
<p>Unlike level-wise growth, LightGBM grows trees leaf-wise:</p>
<ul>
    <li>Choose the leaf with maximum delta loss</li>
    <li>Can lead to deeper trees and better accuracy</li>
    <li>Uses <code>max_depth</code> to prevent overfitting</li>
</ul>

<hr>

<h2 id="metrics">6. Evaluation Metrics</h2>

<h3>6.1 Confusion Matrix</h3>
<p>The confusion matrix provides a complete picture of classification performance:</p>

<div class="equation-block">
\\[\\text{Confusion Matrix} = \\begin{bmatrix} TN & FP \\\\ FN & TP \\end{bmatrix}\\]
</div>

<p>where:</p>
<ul>
    <li>\\(TN\\) = True Negatives (correctly predicted non-potable)</li>
    <li>\\(FP\\) = False Positives (non-potable predicted as potable)</li>
    <li>\\(FN\\) = False Negatives (potable predicted as non-potable)</li>
    <li>\\(TP\\) = True Positives (correctly predicted potable)</li>
</ul>

<h3>6.2 Accuracy</h3>

<div class="equation-block">
\\[\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}\\]
</div>

<h3>6.3 Precision</h3>

<div class="equation-block">
\\[\\text{Precision} = \\frac{TP}{TP + FP}\\]
</div>

<p>Precision measures the proportion of positive predictions that are actually correct.</p>

<h3>6.4 Recall (Sensitivity)</h3>

<div class="equation-block">
\\[\\text{Recall} = \\frac{TP}{TP + FN}\\]
</div>

<p>Recall measures the proportion of actual positives that are correctly identified.</p>

<h3>6.5 F1-Score</h3>
<p>The harmonic mean of precision and recall:</p>

<div class="equation-block">
\\[F_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}\\]
</div>

<h3>6.6 ROC-AUC</h3>
<p>The Area Under the Receiver Operating Characteristic Curve:</p>

<div class="equation-block">
\\[\\text{AUC} = \\int_0^1 \\text{TPR}(t) \\, d(\\text{FPR}(t))\\]
</div>

<p>where:</p>
<ul>
    <li>\\(\\text{TPR} = \\frac{TP}{TP + FN}\\) (True Positive Rate)</li>
    <li>\\(\\text{FPR} = \\frac{FP}{FP + TN}\\) (False Positive Rate)</li>
</ul>

<h3>6.7 Average Precision</h3>

<div class="equation-block">
\\[\\text{AP} = \\sum_n (R_n - R_{n-1}) P_n\\]
</div>

<p>where \\(P_n\\) and \\(R_n\\) are precision and recall at threshold \\(n\\).</p>

<hr>

<h2 id="results">7. Results and Analysis</h2>

<h3>7.1 Model Performance Comparison</h3>

<table>
    <tr>
        <th>Model</th>
        <th>Accuracy</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1-Score</th>
        <th>AUC</th>
    </tr>
    <tr><td>Logistic Regression</td><td>0.5259</td><td>0.4159</td><td>0.5312</td><td>0.4666</td><td>0.5475</td></tr>
    <tr><td>Random Forest</td><td>0.6585</td><td>0.6311</td><td>0.3008</td><td>0.4074</td><td>0.6407</td></tr>
    <tr><td>XGBoost</td><td>0.6418</td><td>0.5574</td><td>0.3984</td><td>0.4647</td><td>0.6256</td></tr>
    <tr><td>LightGBM</td><td>0.6540</td><td>0.6000</td><td>0.3398</td><td>0.4339</td><td>0.6512</td></tr>
</table>

''' + create_img_tag('figures/07_model_comparison.png', 'Visual comparison of model performance metrics', 7) + '''

<h3>7.2 Confusion Matrices</h3>

''' + create_img_tag('figures/08_confusion_matrices.png', 'Confusion matrices for all four models', 8) + '''

<h3>7.3 ROC Curves</h3>

''' + create_img_tag('figures/09_roc_curves.png', 'ROC curves comparing model discrimination ability', 9) + '''

<h3>7.4 Precision-Recall Curves</h3>

''' + create_img_tag('figures/10_precision_recall_curves.png', 'Precision-Recall curves for model comparison', 10) + '''

<h3>7.5 Key Findings</h3>

<div class="findings">
<ol>
    <li><strong>Best Overall Accuracy:</strong> Random Forest (65.85%)</li>
    <li><strong>Best AUC Score:</strong> LightGBM (0.6512)</li>
    <li><strong>Best Recall:</strong> Logistic Regression (53.12%) - important for minimizing false negatives</li>
    <li><strong>Best Precision:</strong> Random Forest (63.11%)</li>
</ol>
<p>The class imbalance in the dataset (61% non-potable vs 39% potable) affects model performance. Using balanced class weights in Logistic Regression improved its recall significantly.</p>
</div>

<hr>

<h2 id="webapp">8. Web Application Deployment</h2>

<p>A Streamlit web application was developed and deployed for real-time water potability predictions.</p>

<p><strong>Live Application:</strong> <a href="https://water-potability-prediction-ip-assignment.streamlit.app/">https://water-potability-prediction-ip-assignment.streamlit.app/</a></p>

<h3>8.1 Application Home Page</h3>

''' + create_img_tag('figures/streamlit_01_home_page.png', 'Home page of the Water Potability Prediction application', 11) + '''

<h3>8.2 Data Exploration Features</h3>
<p>The application provides interactive data exploration capabilities:</p>

''' + create_img_tag('figures/streamlit_02_data_distributions.png', 'Interactive data distribution visualizations', 12) + '''

<h3>8.3 Model Training Interface</h3>

''' + create_img_tag('figures/streamlit_04_model_training.png', 'Model training and evaluation interface', 13) + '''

<h3>8.4 Model Comparison Dashboard</h3>

''' + create_img_tag('figures/streamlit_05_model_comparison.png', 'Dashboard comparing all trained models', 14) + '''

<h3>8.5 Prediction Interface</h3>
<p>Users can input water quality parameters to get predictions:</p>

''' + create_img_tag('figures/streamlit_06_prediction_page.png', 'Prediction input interface', 15) + '''

<h3>8.6 Prediction Results</h3>

''' + create_img_tag('figures/streamlit_07_prediction_result.png', 'Sample prediction result showing water potability assessment', 16) + '''

<hr>

<h2 id="conclusions">9. Conclusions</h2>

<h3>9.1 Summary</h3>
<p>This project successfully developed and deployed a machine learning system for water potability prediction. Key achievements include:</p>
<ol>
    <li><strong>Comprehensive EDA:</strong> Identified data patterns, missing values, and class imbalance issues</li>
    <li><strong>Multiple Model Comparison:</strong> Evaluated four different classification algorithms</li>
    <li><strong>Mathematical Foundation:</strong> Provided detailed mathematical formulations for each algorithm</li>
    <li><strong>Web Deployment:</strong> Created an accessible web application for real-time predictions</li>
</ol>

<h3>9.2 Recommendations</h3>
<ol>
    <li><strong>Collect More Data:</strong> Additional samples, especially potable water samples, would help address class imbalance</li>
    <li><strong>Feature Engineering:</strong> Domain-specific features combining multiple water quality parameters</li>
    <li><strong>Ensemble Methods:</strong> Combining multiple models could improve prediction accuracy</li>
    <li><strong>Threshold Optimization:</strong> Adjusting classification thresholds based on cost-sensitivity analysis</li>
</ol>

<h3>9.3 Future Work</h3>
<ul>
    <li>Implement deep learning approaches (Neural Networks)</li>
    <li>Add time-series analysis for water quality monitoring</li>
    <li>Integrate IoT sensors for real-time data collection</li>
    <li>Develop a mobile application for field testing</li>
</ul>

<hr>

<h2 id="references">10. References</h2>

<ol>
    <li>Breiman, L. (2001). Random Forests. <em>Machine Learning</em>, 45(1), 5-32.</li>
    <li>Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <em>KDD '16</em>.</li>
    <li>Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. <em>NIPS</em>.</li>
    <li>World Health Organization. (2017). Guidelines for Drinking-water Quality.</li>
    <li>Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). <em>Applied Logistic Regression</em>.</li>
</ol>

<hr>

<p><strong>GitHub Repository:</strong> <a href="https://github.com/sohaib3335/water-potability-prediction">https://github.com/sohaib3335/water-potability-prediction</a></p>

<hr>

<p style="text-align: center;"><em>Report generated as part of MSc Computing Independent Project</em><br>
<em>Author: Sohaib Farooq | Email: sohaib.farooq@bigacademy.com</em></p>

</body>
</html>
'''

# Write the HTML file
with open('Water_Potability_ML_Report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("=" * 60)
print("HTML Report Generated Successfully!")
print("=" * 60)
print()
print("File: Water_Potability_ML_Report.html")
print()
print("To create a PDF with properly rendered mathematics:")
print("1. Open Water_Potability_ML_Report.html in your web browser")
print("2. Wait for MathJax to render all equations (a few seconds)")
print("3. Press Ctrl+P (or Cmd+P on Mac) to print")
print("4. Select 'Save as PDF' as the destination")
print("5. Click Save")
print()
print("=" * 60)
