# 🏠 House Price Prediction App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://house-price-prediction-celebal-project.streamlit.app/)

## 🚀 Live Demo

**Try the app now:**  
🔗 [https://house-price-prediction-celebal-project.streamlit.app/](https://house-price-prediction-celebal-project.streamlit.app/)

---

## 📋 Overview

An intelligent house price prediction application built with **Streamlit** and **Machine Learning**. This app predicts house prices based on key property features using advanced regression models trained on real estate data.

---

### 🎯 Key Features

- 🧠 **Interactive Price Prediction**  
- 📊 **5 Key Features**: Living area, bedrooms, quality, year built, garage area  
- 🤖 **Multiple ML Models**: Random Forest, Gradient Boosting, Linear Regression  
- 📉 **Data Visualization**: Correlation matrices, feature importance, price distributions  
- 📈 **Model Performance Metrics**: R², MAE, RMSE  
- 📱 **Responsive Design**: Desktop + Mobile

---

### 🏡 Prediction Features

| Feature               | Description                              | Impact     |
|----------------------|------------------------------------------|------------|
| **Living Area**      | Above ground living space (sq ft)        | 🔥 High    |
| **Number of Bedrooms** | Bedrooms above ground level              | 🔥 High    |
| **Overall Quality**  | Material and finish quality (1-10)       | 🔥 High    |
| **Year Built**       | Original construction date               | 🔶 Medium  |
| **Garage Area**      | Garage size (sq ft)                      | 🔶 Medium  |

---

## 🛠️ Running the App Locally

### Prerequisites

- Python 3.7+

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd house-price-prediction
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies 
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```
🔗 Open http://localhost:8501 in your browser.

## 📦 Dependencies
``` bash
streamlit
pandas
numpy
scikit-learn
plotly
```

## 🎮 How to Use the App

### 1. 🔮 Prediction Tab
- Adjust sliders for house features
- Click “Predict Price”
- View predicted price with confidence

### 2. 📈 Model Performance
- View model comparison (R², MAE, RMSE)
- Visualize predictions and feature importance

### 3. 📊 Data Analysis
- Explore dataset stats
- Feature correlation heatmaps
- Price distributions

### 4. ℹ️ About
- Model and feature info
- Dataset background

