import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the house prices dataset"""
    try:
        # Try to load from uploaded file or use sample data
        data = pd.read_csv('train.csv')
    except:
        # Create sample data if file not found
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'GrLivArea': np.random.normal(1500, 500, n_samples),
            'OverallQual': np.random.randint(1, 11, n_samples),
            'YearBuilt': np.random.randint(1900, 2020, n_samples),
            'GarageArea': np.random.normal(500, 200, n_samples),
            'BedroomAbvGr': np.random.randint(1, 6, n_samples),  # Number of bedrooms above ground
            'Neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural', 'Upscale'], n_samples),
            'SalePrice': np.random.normal(200000, 80000, n_samples)
        })
        
        # Make GrLivArea and GarageArea positive
        data['GrLivArea'] = np.abs(data['GrLivArea'])
        data['GarageArea'] = np.abs(data['GarageArea'])
        data['SalePrice'] = np.abs(data['SalePrice'])
    
    return data

@st.cache_data
def prepare_features(data):
    """Prepare and encode features for modeling"""
    # Select top 5 most important features including bedrooms
    if 'GrLivArea' in data.columns:
        features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'BedroomAbvGr', 'GarageArea']
        
        # If BedroomAbvGr not available, try other bedroom columns
        if 'BedroomAbvGr' not in data.columns:
            bedroom_cols = [col for col in data.columns if 'bedroom' in col.lower() or 'bed' in col.lower()]
            if bedroom_cols:
                features[3] = bedroom_cols[0]  # Replace BedroomAbvGr with available bedroom column
            else:
                # Add neighborhood instead if no bedroom data
                if 'Neighborhood' in data.columns:
                    features[3] = 'Neighborhood'
                else:
                    categorical_cols = data.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        features[3] = categorical_cols[0]
        
        # Remove any features not in the dataset
        features = [f for f in features if f in data.columns]
        
        # Add neighborhood if we have space and it's available
        if len(features) < 5 and 'Neighborhood' in data.columns and 'Neighborhood' not in features:
            features.append('Neighborhood')
            
    else:
        # Use the first 5 numeric columns if standard columns not found
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in numeric_cols:
            numeric_cols.remove('SalePrice')
        features = numeric_cols[:4]
        
        # Add one categorical feature if available
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            features.append(categorical_cols[0])
    
    # Ensure we don't exceed 5 features
    features = features[:5]
    
    # Prepare the dataset
    df_model = data[features + ['SalePrice']].copy()
    
    # Handle missing values
    for col in df_model.columns:
        if df_model[col].dtype == 'object':
            df_model[col].fillna('Unknown', inplace=True)
        else:
            df_model[col].fillna(df_model[col].median(), inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in features:
        if df_model[col].dtype == 'object':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            label_encoders[col] = le
    
    return df_model, features, label_encoders

@st.cache_data
def train_models(X, y):
    """Train multiple models and return the best one"""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    model_scores = {}
    trained_models = {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        model_scores[name] = {
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse,
            'Model': model
        }
        trained_models[name] = model
    
    # Find the best model based on R²
    best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['R²'])
    best_model = model_scores[best_model_name]['Model']
    
    return trained_models, model_scores, best_model_name, best_model, X_test, y_test

def create_visualizations(data, features, y_test, y_pred, feature_importance=None):
    """Create visualizations for the app"""
    
    # 1. Feature Correlation Heatmap - only use numeric columns
    numeric_features = []
    for feature in features:
        if data[feature].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(feature)
    
    if len(numeric_features) > 0:
        corr_data = data[numeric_features + ['SalePrice']].select_dtypes(include=[np.number])
        fig_corr = px.imshow(
            corr_data.corr(),
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix (Numeric Features Only)"
        )
    else:
        # Create empty correlation plot if no numeric features
        fig_corr = go.Figure()
        fig_corr.add_annotation(
            text="No numeric features available for correlation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig_corr.update_layout(title="Feature Correlation Matrix")
    
    # 2. Actual vs Predicted scatter plot
    fig_scatter = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Price', 'y': 'Predicted Price'},
        title='Actual vs Predicted Prices'
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        )
    )
    
    # 3. Feature Importance (if available)
    fig_importance = None
    if feature_importance is not None:
        fig_importance = px.bar(
            x=feature_importance,
            y=features,
            orientation='h',
            title='Feature Importance',
            labels={'x': 'Importance', 'y': 'Features'}
        )
    
    # 4. Price Distribution
    fig_dist = px.histogram(
        data, x='SalePrice',
        title='Distribution of House Prices',
        nbins=50
    )
    
    return fig_corr, fig_scatter, fig_importance, fig_dist

def main():
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Model Configuration")
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_and_prepare_data()
        df_model, features, label_encoders = prepare_features(data)
    
    st.sidebar.success(f"Dataset loaded: {len(data)} houses")
    st.sidebar.info(f"Features used: {', '.join(features)}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📈 Model Performance", "📊 Data Analysis", "ℹ️ About"])
    
    # Train models
    X = df_model[features]
    y = df_model['SalePrice']
    
    with st.spinner("Training models..."):
        trained_models, model_scores, best_model_name, best_model, X_test, y_test = train_models(X, y)
        y_pred = best_model.predict(X_test)
    
    with tab1:
        st.header("🔮 Make a Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter House Details")
            
            # Create input fields based on features
            user_inputs = {}
            
            for feature in features:
                if feature in label_encoders:
                    # Categorical feature
                    options = list(label_encoders[feature].classes_)
                    user_inputs[feature] = st.selectbox(f"{feature}", options)
                else:
                    # Numerical feature
                    min_val = float(df_model[feature].min())
                    max_val = float(df_model[feature].max())
                    mean_val = float(df_model[feature].mean())
                    
                    if feature == 'GrLivArea':
                        user_inputs[feature] = st.slider(
                            "Living Area (sq ft)", 
                            min_value=int(min_val), 
                            max_value=int(max_val), 
                            value=int(mean_val)
                        )
                    elif feature == 'OverallQual':
                        user_inputs[feature] = st.slider(
                            "Overall Quality (1-10)", 
                            min_value=int(min_val), 
                            max_value=int(max_val), 
                            value=int(mean_val)
                        )
                    elif feature == 'YearBuilt':
                        user_inputs[feature] = st.slider(
                            "Year Built", 
                            min_value=int(min_val), 
                            max_value=int(max_val), 
                            value=int(mean_val)
                        )
                    elif feature == 'BedroomAbvGr' or 'bedroom' in feature.lower():
                        user_inputs[feature] = st.slider(
                            "Number of Bedrooms", 
                            min_value=int(max(1, min_val)), 
                            max_value=int(max_val), 
                            value=int(mean_val)
                        )
                    elif feature == 'GarageArea':
                        user_inputs[feature] = st.slider(
                            "Garage Area (sq ft)", 
                            min_value=int(min_val), 
                            max_value=int(max_val), 
                            value=int(mean_val)
                        )
                    else:
                        user_inputs[feature] = st.number_input(
                            f"{feature}", 
                            min_value=min_val, 
                            max_value=max_val, 
                            value=mean_val
                        )
        
        with col2:
            st.subheader("Prediction")
            
            if st.button("🔍 Predict Price", type="primary"):
                # Prepare input for prediction
                input_data = []
                for feature in features:
                    if feature in label_encoders:
                        encoded_value = label_encoders[feature].transform([user_inputs[feature]])[0]
                        input_data.append(encoded_value)
                    else:
                        input_data.append(user_inputs[feature])
                
                # Make prediction
                prediction = best_model.predict([input_data])[0]
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Price</h2>
                    <h1>${prediction:,.0f}</h1>
                    <p>Using {best_model_name}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence interval (rough estimate)
                std_error = np.std(y_test - best_model.predict(X_test))
                lower_bound = prediction - 1.96 * std_error
                upper_bound = prediction + 1.96 * std_error
                
                st.info(f"95% Confidence Interval: ${lower_bound:,.0f} - ${upper_bound:,.0f}")
    
    with tab2:
        st.header("📈 Model Performance")
        
        # Model comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Comparison")
            comparison_df = pd.DataFrame(model_scores).T
            comparison_df = comparison_df.drop('Model', axis=1)
            st.dataframe(comparison_df.style.highlight_max(axis=0))
        
        with col2:
            st.subheader("Best Model")
            st.success(f"**{best_model_name}** performs best!")
            best_scores = model_scores[best_model_name]
            
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("R² Score", f"{best_scores['R²']:.3f}")
            with col2_2:
                st.metric("MAE", f"${best_scores['MAE']:,.0f}")
            with col2_3:
                st.metric("RMSE", f"${best_scores['RMSE']:,.0f}")
        
        # Visualizations
        st.subheader("Model Visualizations")
        
        # Get feature importance for tree-based models
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        
        fig_corr, fig_scatter, fig_importance, fig_dist = create_visualizations(
            data, features, y_test, y_pred, feature_importance
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_plot")
        with col2:
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True, key="importance_plot")
            else:
                st.plotly_chart(fig_dist, use_container_width=True, key="dist_plot_alt")
    
    with tab3:
        st.header("📊 Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.dataframe(data.describe())
        
        with col2:
            st.subheader("Feature Correlations")
            # Only show numeric features for correlation
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'SalePrice' in numeric_cols and len(numeric_cols) > 1:
                fig_corr = px.imshow(
                    data[numeric_cols].corr(),
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig_corr, use_container_width=True, key="correlation_plot")
            else:
                st.info("Correlation matrix requires numeric features")
        
        st.subheader("Price Distribution")
        fig_dist = px.histogram(data, x='SalePrice', title='Distribution of House Prices', nbins=50)
        st.plotly_chart(fig_dist, use_container_width=True, key="price_distribution")
        
        # Feature analysis
        st.subheader("Feature Analysis")
        selected_feature = st.selectbox("Select feature to analyze:", features)
        
        if selected_feature in label_encoders or data[selected_feature].dtype == 'object':
            # Categorical feature - use box plot
            fig_box = px.box(data, x=selected_feature, y='SalePrice', 
                           title=f'Price by {selected_feature}')
            st.plotly_chart(fig_box, use_container_width=True, key=f"box_plot_{selected_feature}")
        else:
            # Numeric feature - use scatter plot
            fig_scatter_feat = px.scatter(data, x=selected_feature, y='SalePrice',
                                        title=f'Price vs {selected_feature}')
            st.plotly_chart(fig_scatter_feat, use_container_width=True, key=f"scatter_{selected_feature}")
    
    with tab4:
        st.header("ℹ️ About This App")
        
        st.markdown("""
        ### 🏠 House Price Prediction Model
        
        This application predicts house prices based on key property features using machine learning.
        
        **Features Used:**
        - **Living Area (GrLivArea)**: Above ground living area in square feet
        - **Overall Quality**: Overall material and finish quality (1-10 scale)
        - **Year Built**: Original construction date
        - **Number of Bedrooms**: Bedrooms above ground level
        - **Garage Area**: Size of garage in square feet
        - **Neighborhood**: Location category (when available)
        
        **Models Implemented:**
        - **Random Forest**: Ensemble of decision trees
        - **Gradient Boosting**: Sequential ensemble method
        - **Linear Regression**: Traditional linear approach
        
        **Model Selection:**
        The best performing model is automatically selected based on R² score.
        
        **Accuracy Metrics:**
        - **R² Score**: Proportion of variance explained by the model
        - **MAE**: Mean Absolute Error in dollars
        - **RMSE**: Root Mean Square Error in dollars
        
        
        ### 📊 Data Source
        Based on the Kaggle House Prices: Advanced Regression Techniques competition dataset.
        """)
        
        # Technical details
        with st.expander("Technical Details"):
            st.code(f"""
            Dataset Shape: {data.shape}
            Features Used: {features}
            Best Model: {best_model_name}
            Model Accuracy: {model_scores[best_model_name]['R²']:.3f} R²
            """)

if __name__ == "__main__":
    main()