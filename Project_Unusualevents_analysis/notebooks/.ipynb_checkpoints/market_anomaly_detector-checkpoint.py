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
warnings.filterwarnings('ignore')
def main():
    st.set_page_config(page_title="Market Anomaly Detection", layout="wide")
    
    st.title("🔍 Advanced Market Anomaly Detection System")
    st.markdown("""
    This application implements state-of-the-art statistical and machine learning methods 
    for detecting unusual events in financial time series data.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Configuration")
    
    # Stock selection
    ticker = st.sidebar.text_input("Stock Ticker", value="SPY", help="Enter stock symbol (e.g., AAPL, MSFT)")
    
    # Date range
    start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp.now() - pd.Timedelta(days=365*10))
    end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.now())
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    stat_threshold = st.sidebar.slider("Statistical Z-Score Threshold", 1.0, 5.0, 3.0)
    vol_threshold = st.sidebar.slider("Volatility Threshold", 1.0, 4.0, 2.0)
    contamination = st.sidebar.slider("Expected Anomaly Rate (%)", 1, 20, 5) / 100
    
    if st.sidebar.button("Run Analysis"):
        try:
            # Download data
            with st.spinner("Downloading market data..."):
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                
            if stock_data.empty:
                st.error("No data found for the specified ticker and date range.")
                return
            
            # Initialize detector
            with st.spinner("Initializing anomaly detection..."):
                detector = MarketAnomalyDetector(stock_data)
            
            # Run detection methods
            with st.spinner("Running anomaly detection algorithms..."):
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
                
                status_text.text("Analysis complete!")
            
            # Display results
            st.success("✅ Anomaly detection completed successfully!")
            
            # Summary statistics
            st.subheader("📊 Detection Summary")
            summary = detector.get_anomaly_summary()
            
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
            summary_df = summary_df.round(2)
            st.dataframe(summary_df, use_container_width=True)
            
            # Visualizations
            st.subheader("📈 Interactive Visualizations")
            fig = create_visualizations(detector, stock_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly details
            st.subheader("🔍 Anomaly Details")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Statistical", "Volatility", "Jumps", "Multivariate", "Regime"
            ])
            
            with tab1:
                if 'statistical' in detector.anomalies:
                    anomalies = detector.anomalies['statistical']['anomalies']
                    if anomalies.sum() > 0:
                        anomaly_data = stock_data.loc[anomalies.index[anomalies]]
                        st.dataframe(anomaly_data[['Close', 'Returns', 'Volatility']])
                    else:
                        st.info("No statistical anomalies detected.")
            
            with tab2:
                if 'volatility' in detector.anomalies:
                    anomalies = detector.anomalies['volatility']['anomalies']
                    if anomalies.sum() > 0:
                        anomaly_data = stock_data.loc[anomalies.index[anomalies]]
                        st.dataframe(anomaly_data[['Close', 'Returns', 'Volatility']])
                    else:
                        st.info("No volatility anomalies detected.")
            
            with tab3:
                if 'jumps' in detector.anomalies:
                    anomalies = detector.anomalies['jumps']['anomalies']
                    if anomalies.sum() > 0:
                        anomaly_dates = anomalies.index[anomalies]
                        jump_data = stock_data.loc[anomaly_dates]
                        jump_components = detector.anomalies['jumps']['jump_component'].loc[anomaly_dates]
                        jump_data['Jump_Component'] = jump_components
                        st.dataframe(jump_data[['Close', 'Returns', 'Jump_Component']])
                    else:
                        st.info("No jump anomalies detected.")
            
            with tab4:
                if 'multivariate' in detector.anomalies:
                    anomalies = detector.anomalies['multivariate']['anomalies']
                    if anomalies.sum() > 0:
                        anomaly_data = stock_data.loc[anomalies.index[anomalies]]
                        scores = detector.anomalies['multivariate']['scores'].loc[anomalies.index[anomalies]]
                        anomaly_data['Anomaly_Score'] = scores
                        st.dataframe(anomaly_data[['Close', 'Returns', 'Volatility', 'Anomaly_Score']])
                    else:
                        st.info("No multivariate anomalies detected.")
            
            with tab5:
                if 'regime' in detector.anomalies:
                    anomalies = detector.anomalies['regime']['anomalies']
                    if anomalies.sum() > 0:
                        anomaly_data = stock_data.loc[anomalies.index[anomalies]]
                        st.dataframe(anomaly_data[['Close', 'Returns', 'Volatility']])
                    else:
                        st.info("No regime change anomalies detected.")
            
            # Download results
            st.subheader("💾 Export Results")
            if st.button("Generate Export Data"):
                # Combine all anomalies
                export_data = stock_data.copy()
                for method in detector.anomalies:
                    if 'anomalies' in detector.anomalies[method]:
                        anomalies = detector.anomalies[method]['anomalies']
                        export_data[f'{method}_anomaly'] = False
                        export_data.loc[anomalies.index[anomalies], f'{method}_anomaly'] = True
                
                csv = export_data.to_csv()
                st.download_button(
                    label="Download Full Dataset with Anomaly Flags",
                    data=csv,
                    file_name=f"{ticker}_anomaly_analysis.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your inputs and try again.")
    
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