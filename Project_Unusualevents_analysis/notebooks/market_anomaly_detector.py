import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import sys
from pathlib import Path

# Set up path to allow import from src/unusualevents
BASE_DIR = Path('Project_Unusualevents_analysis/src/unusualevents/detectorclass.py').resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR)) # Insert at front

# Now import
from unusualevents.detectorclass import MarketAnomalyDetector, create_visualizations
warnings.filterwarnings('ignore')
def main():
    """
    Main Streamlit application function
    """
    st.set_page_config(page_title="Market Anomaly Detection", layout="wide")
    
    st.title("🔍 Advanced Market Anomaly Detection System")
    st.markdown("""
    This application implements state-of-the-art statistical and machine learning methods 
    for detecting unusual events in financial time series data.
    """)

    
    # Sidebar for parameters
    st.sidebar.header("Configuration")
    
    # Stock selection
    ticker = st.sidebar.text_input("Stock Ticker", value="SPY", help="Enter stock symbol (e.g., AAPL, MSFT, SPY)")
    
    # Date range
    start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp.now() - pd.Timedelta(days=365*2))
    end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.now())
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    stat_threshold = st.sidebar.slider("Statistical Z-Score Threshold", 1.0, 5.0, 3.0)
    vol_threshold = st.sidebar.slider("Volatility Threshold", 1.0, 4.0, 2.0)
    contamination = st.sidebar.slider("Expected Anomaly Rate (%)", 1, 20, 5) / 100
    
    # Run Analysis Button
    if st.sidebar.button("Run Analysis", type="primary"):
        try:
            # Validate inputs
            if not ticker:
                st.error("Please enter a stock ticker")
                return
            
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return
            
            # Download data
            with st.spinner("📊 Downloading market data..."):
                try:
                    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                except Exception as e:
                    st.error(f"Error downloading data: {str(e)}")
                    return
                
            if stock_data.empty:
                st.error(f"No data found for ticker '{ticker}' in the specified date range.")
                st.info("Please check the ticker symbol and date range.")
                return
            
            # Data quality check
            if len(stock_data) < 50:
                st.warning(f"Limited data available: only {len(stock_data)} observations. Results may be less reliable.")
            
            st.info(f"Downloaded {len(stock_data)} data points for {ticker}")
            
            # Initialize detector
            with st.spinner("🔧 Initializing anomaly detection..."):
                try:
                    detector = MarketAnomalyDetector(stock_data)
                except Exception as e:
                    st.error(f"Error initializing detector: {str(e)}")
                    return
            
            # Run detection methods
            with st.spinner("🔍 Running anomaly detection algorithms..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Detection Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Statistical outliers
                status_text.text("Running statistical outlier detection...")
                detector.detect_statistical_outliers(threshold=stat_threshold)
                progress_bar.progress(20)
                
                # Volatility clustering
                status_text.text("Detecting volatility clusters...")
                detector.detect_volatility_clusters(threshold=vol_threshold)
                progress_bar.progress(40)
                
                # Jump detection
                status_text.text("Running jump diffusion analysis...")
                detector.detect_jump_diffusion()
                progress_bar.progress(60)
                
                # Multivariate outliers
                status_text.text("Performing multivariate anomaly detection...")
                detector.detect_multivariate_outliers(contamination=contamination)
                progress_bar.progress(80)
                
                # Regime changes
                status_text.text("Detecting regime changes...")
                detector.detect_regime_changes()
                progress_bar.progress(100)
                
                status_text.text("✅ Analysis complete!")
            
            # Display results
            st.success("🎉 Anomaly detection completed successfully!")
            
            # Summary statistics
            st.subheader("📊 Detection Summary")
            summary = detector.get_anomaly_summary()
            
            if summary:
                # Display metrics
                cols = st.columns(len(summary))
                for i, (method, stats) in enumerate(summary.items()):
                    with cols[i]:
                        st.metric(
                            f"{method.title()} Anomalies",
                            f"{stats['total_anomalies']}",
                            f"{stats['anomaly_rate']:.2f}%"
                        )
                
                # Detailed statistics table
                summary_df = pd.DataFrame(summary).T
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No anomalies detected with current parameters.")
            
            # Visualizations
            st.subheader("📈 Interactive Visualizations")
            fig = create_visualizations(detector, stock_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data preview
            st.subheader("📋 Data Preview")
            with st.expander("View Raw Data"):
                st.dataframe(stock_data.tail(20))
            
            # Export functionality
            st.subheader("💾 Export Results")
            if st.button("Generate Export Data"):
                try:
                    # Combine all anomalies
                    export_data = stock_data.copy()
                    for method in detector.anomalies:
                        if 'anomalies' in detector.anomalies[method]:
                            anomalies = detector.anomalies[method]['anomalies']
                            export_data[f'{method}_anomaly'] = False
                            if hasattr(anomalies, 'index'):
                                export_data.loc[anomalies.index[anomalies], f'{method}_anomaly'] = True
                    
                    csv = export_data.to_csv()
                    st.download_button(
                        label="📥 Download Full Dataset with Anomaly Flags",
                        data=csv,
                        file_name=f"{ticker}_anomaly_analysis.csv",
                        mime="text/csv"
                    )
                    st.success("Export data generated successfully!")
                except Exception as e:
                    st.error(f"Error generating export data: {str(e)}")
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.error("Please check your inputs and try again.")
            
            # Debug information
            with st.expander("Debug Information"):
                st.write("Error details:", str(e))
                st.write("Ticker:", ticker)
                st.write("Date range:", start_date, "to", end_date)
    
    # Educational content
    with st.expander("📚 Methodology & Theory"):
        st.markdown("""
        ### Statistical Methods Implemented:
        
        **1. Statistical Process Control (Z-Score)**
        - Assumes local normality of returns
        - Detects points beyond statistical control limits
        - Effective for identifying outliers in return distributions
        
        **2. Volatility Clustering Detection**
        - Based on ARCH/GARCH effects in financial data
        - Identifies periods of unusual volatility persistence
        - Critical for risk management applications
        
        **3. Jump-Diffusion Detection**
        - Uses Bipower Variation methodology (Barndorff-Nielsen & Shephard, 2004)
        - Distinguishes continuous price movements from discrete jumps
        - Essential for options pricing and risk assessment
        
        **4. Isolation Forest (Multivariate)**
        - Ensemble method for high-dimensional anomaly detection
        - Isolates anomalies by randomly partitioning feature space
        - Effective when multiple indicators signal unusual behavior
        
        **5. Regime Change Detection**
        - Monitors changes in multiple statistical moments simultaneously
        - Identifies structural breaks in market behavior
        - Important for adaptive trading strategies
        
        ### Applications in Finance:
        - **Risk Management**: Early warning system for market stress
        - **Algorithmic Trading**: Signal generation for mean-reversion strategies  
        - **Portfolio Management**: Dynamic hedging and position sizing
        - **Regulatory Compliance**: Market surveillance and manipulation detection
        """)

if __name__ == "__main__":
    main()