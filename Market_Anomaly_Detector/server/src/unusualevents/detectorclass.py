# market_anomaly_detector.py
# Complete fixed version - Save this as a single .py file

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance with error handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.error("yfinance is not installed. Please install it using: pip install yfinance")

# ====================================================================
# CLASS DEFINITION - This must come BEFORE it's used
# ====================================================================

class MarketAnomalyDetector:
    """
    Advanced Market Anomaly Detection System
    
    This class implements multiple statistical and machine learning methods
    for detecting unusual events in financial time series data.
    
    Methods include:
    1. Statistical Process Control (SPC) using control charts
    2. Z-Score based outlier detection with rolling statistics
    3. Isolation Forest for multivariate anomaly detection
    4. Volatility clustering detection using GARCH-like approach
    5. Jump detection using Bipower Variation
    """
    
    def __init__(self, data):
        """
        Initialize the detector with price data
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and datetime index
        """
        if data is None or data.empty:
            raise ValueError("Input data is empty or None")
            
        self.data = data.copy()
        
        # Ensure we have the required columns
        if 'Close' not in self.data.columns:
            # Try different column names
            if 'close' in self.data.columns:
                self.data['Close'] = self.data['close']
            elif 'Adj Close' in self.data.columns:
                self.data['Close'] = self.data['Adj Close']
            else:
                raise ValueError("No 'Close' price column found in data")
        
        self.returns = None
        self.features = None
        self.anomalies = {}
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Prepare financial features for analysis
        
        Theory: Financial returns are more stationary than prices and exhibit
        properties like volatility clustering, fat tails, and mean reversion.
        """
        try:
            # Calculate returns
            self.data['Returns'] = self.data['Close'].pct_change()
            
            # Log returns (more theoretically sound for continuous compounding)
            self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
            
            # Realized volatility (rolling standard deviation)
            self.data['Volatility'] = self.data['Returns'].rolling(window=20, min_periods=1).std()
            
            # Volume-weighted returns (if volume available)
            #if 'Volume' in self.data.columns and not self.data['Volume'].isna().all():
                # Add small constant to avoid log(0)
                #volume_safe = self.data['Volume'].replace(0, 1)
                #self.data['Volume_Weighted_Returns'] = self.data['Returns'] * np.log(volume_safe)
            # Handle volume
            if 'Volume' in self.data.columns:
                st.write("Step: Volume column exists")
                if isinstance(self.data['Volume'], pd.Series):
                    if not self.data['Volume'].isna().all():
                        st.write("Step: Calculating volume-weighted returns")
                        volume_safe = self.data['Volume'].replace(0, 1)
                        self.data['Volume_Weighted_Returns'] = self.data['Returns'] * np.log(volume_safe)
                else:
                    st.warning("Volume column is not a valid Series")

            st.write("Step: Higher moments")
            
            # Higher moments for fat-tail analysis
            self.data['Returns_Squared'] = self.data['Returns'] ** 2
            self.data['Returns_Cubed'] = self.data['Returns'] ** 3
            
            # Rolling skewness and kurtosis (20-day window)
            self.data['Rolling_Skewness'] = self.data['Returns'].rolling(window=20, min_periods=5).skew()
            self.data['Rolling_Kurtosis'] = self.data['Returns'].rolling(window=20, min_periods=5).kurt()
            
            # Price momentum indicators
            self.data['RSI'] = self._calculate_rsi(self.data['Close'], window=14)
            
            # Drop NaN values but keep at least some data
            initial_length = len(self.data)
            self.data = self.data.dropna()
            
            if len(self.data) < 30:  # Minimum required for analysis
                raise ValueError(f"Insufficient data after cleaning. Only {len(self.data)} valid observations remaining from {initial_length}")
            
            self.returns = self.data['Returns'].values
            
        except Exception as e:
            raise ValueError(f"Error in data preparation: {str(e)}")
    
    def _calculate_rsi(self, prices, window=14):
        """
        Calculate Relative Strength Index
        
        Theory: RSI measures momentum and can indicate overbought/oversold conditions
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            
            # Avoid division by zero
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series(50, index=prices.index)  # Return neutral RSI if calculation fails
    
    def detect_statistical_outliers(self, window=60, threshold=3):
        """
        Statistical Process Control using rolling Z-scores
        
        Theory: Assumes returns follow a normal distribution locally.
        Points beyond threshold standard deviations are flagged as anomalies.
        """
        try:
            # Adjust window size if data is too small
            window = min(window, len(self.data) // 2)
            window = max(window, 10)  # Minimum window size
            
            # Rolling mean and standard deviation
            rolling_mean = self.data['Returns'].rolling(window=window, min_periods=window//2).mean()
            rolling_std = self.data['Returns'].rolling(window=window, min_periods=window//2).std()
            
            # Calculate Z-scores, handle division by zero
            z_scores = (self.data['Returns'] - rolling_mean) / (rolling_std + 1e-10)
            
            # Detect anomalies
            anomalies = np.abs(z_scores) > threshold
            
            self.anomalies['statistical'] = {
                'anomalies': anomalies,
                'z_scores': z_scores,
                'threshold': threshold
            }
            
            return anomalies
            
        except Exception as e:
            st.warning(f"Error in statistical outlier detection: {str(e)}")
            return pd.Series(False, index=self.data.index)
    
    def detect_volatility_clusters(self, window=20, threshold=2):
        """
        Volatility Clustering Detection
        
        Theory: Financial markets exhibit volatility clustering - periods of high
        volatility tend to be followed by high volatility (ARCH/GARCH effects).
        """
        try:
            # Adjust window size
            window = min(window, len(self.data) // 3)
            window = max(window, 5)
            
            # Calculate rolling volatility statistics
            vol_mean = self.data['Volatility'].rolling(window=window, min_periods=window//2).mean()
            vol_std = self.data['Volatility'].rolling(window=window, min_periods=window//2).std()
            
            # Standardize current volatility
            vol_z_score = (self.data['Volatility'] - vol_mean) / (vol_std + 1e-10)
            
            # Detect volatility spikes
            vol_anomalies = vol_z_score > threshold
            
            self.anomalies['volatility'] = {
                'anomalies': vol_anomalies,
                'z_scores': vol_z_score,
                'threshold': threshold
            }
            
            return vol_anomalies
            
        except Exception as e:
            st.warning(f"Error in volatility cluster detection: {str(e)}")
            return pd.Series(False, index=self.data.index)
    
    def detect_jump_diffusion(self, window=20, significance_level=0.01):
        """
        Jump Detection using Bipower Variation
        
        Theory: Distinguishes between continuous price movements (diffusion)
        and discontinuous jumps. Based on Barndorff-Nielsen & Shephard (2004).
        """
        try:
            returns = self.data['Returns'].values
            window = min(window, len(returns) // 4)
            window = max(window, 5)
            
            # Calculate bipower variation
            abs_returns = np.abs(returns)
            bipower_var = np.zeros(len(returns))
            
            for i in range(window, len(returns)):
                # Bipower variation estimator
                abs_ret_window = abs_returns[i-window+1:i+1]
                if len(abs_ret_window) >= 2:
                    bipower_var[i] = np.pi/2 * np.sum(abs_ret_window[:-1] * abs_ret_window[1:]) / window
            
            # Realized variance
            realized_var = np.array([np.sum(returns[max(0, i-window+1):i+1]**2) 
                                    for i in range(len(returns))])
            
            # Jump component
            jump_component = realized_var - bipower_var
            jump_component = np.maximum(jump_component, 0)  # Ensure non-negative
            
            # Statistical test for jumps
            test_stat = jump_component / np.sqrt(bipower_var + 1e-8)
            critical_value = stats.norm.ppf(1 - significance_level)
            
            jump_anomalies = test_stat > critical_value
            
            self.anomalies['jumps'] = {
                'anomalies': pd.Series(jump_anomalies, index=self.data.index),
                'test_statistic': pd.Series(test_stat, index=self.data.index),
                'jump_component': pd.Series(jump_component, index=self.data.index)
            }
            
            return pd.Series(jump_anomalies, index=self.data.index)
            
        except Exception as e:
            st.warning(f"Error in jump detection: {str(e)}")
            return pd.Series(False, index=self.data.index)
    
    def detect_multivariate_outliers(self, contamination=0.05):
        """
        Multivariate Anomaly Detection using Isolation Forest
        
        Theory: Isolation Forest works by randomly selecting features and split values,
        isolating anomalies which require fewer splits.
        """
        try:
            # Select features for multivariate analysis
            feature_columns = ['Returns', 'Volatility', 'Rolling_Skewness', 
                              'Rolling_Kurtosis', 'RSI']
            
            # Add volume-based features if available
            if 'Volume_Weighted_Returns' in self.data.columns:
                feature_columns.append('Volume_Weighted_Returns')
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in self.data.columns]
            features = self.data[available_features].dropna()
            
            if len(features) < 10:
                st.warning("Insufficient data for multivariate analysis")
                return pd.Series(False, index=self.data.index)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            anomaly_labels = iso_forest.fit_predict(features_scaled)
            anomalies = anomaly_labels == -1
            
            # Calculate anomaly scores
            anomaly_scores = iso_forest.decision_function(features_scaled)
            
            # Create full-length series
            full_anomalies = pd.Series(False, index=self.data.index)
            full_scores = pd.Series(0.0, index=self.data.index)
            
            full_anomalies.loc[features.index] = anomalies
            full_scores.loc[features.index] = anomaly_scores
            
            self.anomalies['multivariate'] = {
                'anomalies': full_anomalies,
                'scores': full_scores,
                'features_used': available_features
            }
            
            return full_anomalies
            
        except Exception as e:
            st.warning(f"Error in multivariate outlier detection: {str(e)}")
            return pd.Series(False, index=self.data.index)
    
    def detect_regime_changes(self, window=60, threshold=2.5):
        """
        Regime Change Detection using Rolling Statistics
        
        Theory: Financial markets experience regime changes - periods with
        different statistical properties.
        """
        try:
            window = min(window, len(self.data) // 3)
            window = max(window, 10)
            
            # Calculate rolling statistics
            rolling_mean = self.data['Returns'].rolling(window=window, min_periods=window//2).mean()
            rolling_std = self.data['Returns'].rolling(window=window, min_periods=window//2).std()
            rolling_skew = self.data['Returns'].rolling(window=window, min_periods=window//2).skew()
            rolling_kurt = self.data['Returns'].rolling(window=window, min_periods=window//2).kurt()
            
            # Detect significant changes in statistics
            mean_change = np.abs(rolling_mean.diff()) > threshold * (rolling_std + 1e-10)
            vol_change = np.abs(rolling_std.pct_change()) > threshold * 0.1
            skew_change = np.abs(rolling_skew.diff()) > threshold * 0.5
            kurt_change = np.abs(rolling_kurt.diff()) > threshold * 1.0
            
            # Fill NaN values with False
            mean_change = mean_change.fillna(False)
            vol_change = vol_change.fillna(False)
            skew_change = skew_change.fillna(False)
            kurt_change = kurt_change.fillna(False)
            
            # Regime change when multiple statistics change
            regime_changes = (mean_change.astype(int) + vol_change.astype(int) + 
                             skew_change.astype(int) + kurt_change.astype(int)) >= 2
            
            self.anomalies['regime'] = {
                'anomalies': regime_changes,
                'mean_change': mean_change,
                'vol_change': vol_change,
                'skew_change': skew_change,
                'kurt_change': kurt_change
            }
            
            return regime_changes
            
        except Exception as e:
            st.warning(f"Error in regime change detection: {str(e)}")
            return pd.Series(False, index=self.data.index)
    
    def get_anomaly_summary(self):
        """
        Generate comprehensive summary of all detected anomalies
        """
        summary = {}
        
        for method, results in self.anomalies.items():
            if 'anomalies' in results:
                anomalies = results['anomalies']
                try:
                    total_anomalies = int(anomalies.sum()) if hasattr(anomalies, 'sum') else sum(anomalies)
                    anomaly_rate = (total_anomalies / len(anomalies) * 100) if len(anomalies) > 0 else 0
                    
                    summary[method] = {
                        'total_anomalies': total_anomalies,
                        'anomaly_rate': round(anomaly_rate, 2),
                        'first_anomaly': anomalies.index[anomalies].min() if hasattr(anomalies, 'index') and total_anomalies > 0 else None,
                        'last_anomaly': anomalies.index[anomalies].max() if hasattr(anomalies, 'index') and total_anomalies > 0 else None
                    }
                except Exception as e:
                    st.warning(f"Error summarizing {method}: {str(e)}")
        
        return summary

# ====================================================================
# VISUALIZATION FUNCTIONS
# ====================================================================

def create_visualizations(detector, data):
    """
    Create comprehensive visualizations for anomaly detection results
    """
    try:
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Price with Anomalies', 'Returns with Statistical Outliers', 
                        'Volatility Clustering', 'Jump Detection'],
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}]]
        )
        
        # Price chart with all anomalies
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add anomaly markers for each method
        colors = {'statistical': 'red', 'volatility': 'orange', 'jumps': 'purple', 
                'multivariate': 'green', 'regime': 'brown'}
        
        for method, color in colors.items():
            if method in detector.anomalies:
                anomalies = detector.anomalies[method]['anomalies']
                if hasattr(anomalies, 'index'):
                    anomaly_dates = anomalies.index[anomalies]
                    if len(anomaly_dates) > 0:
                        try:
                            anomaly_prices = data.loc[anomaly_dates, 'Close']
                            fig.add_trace(
                                go.Scatter(x=anomaly_dates, y=anomaly_prices, 
                                        mode='markers', name=f'{method.title()} Anomalies',
                                        marker=dict(color=color, size=8, symbol='diamond')),
                                row=1, col=1
                            )
                        except Exception as e:
                            st.warning(f"Could not plot {method} anomalies: {str(e)}")
        print("Line 439 work:")
        # Returns with statistical outliers
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Returns'], name='Returns', 
                    line=dict(color='darkblue')),
            row=2, col=1
        )
        print("Line 446 204")
        # Add statistical outlier markers
        if 'statistical' in detector.anomalies:
            stat_anomalies = detector.anomalies['statistical']['anomalies']
            if hasattr(stat_anomalies, 'index'):
                anomaly_indices = stat_anomalies.index[stat_anomalies]
                if len(anomaly_indices) > 0:
                    try:
                        anomaly_returns = data.loc[anomaly_indices, 'Returns']
                        fig.add_trace(
                            go.Scatter(x=anomaly_indices, y=anomaly_returns,
                                    mode='markers', name='Statistical Outliers',
                                    marker=dict(color='red', size=6)),
                            row=2, col=1
                        )
                    except:
                        pass
        print("Line 446 204")
        # Volatility clustering
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Volatility'], name='Volatility',
                    line=dict(color='green')),
            row=3, col=1
        )
        
        # Jump detection
        if 'jumps' in detector.anomalies:
            try:
                jump_component = detector.anomalies['jumps']['jump_component']
                fig.add_trace(
                    go.Scatter(x=jump_component.index, y=jump_component, 
                            name='Jump Component', line=dict(color='purple')),
                    row=4, col=1
                )
            except:
                pass
        
        fig.update_layout(height=1200, title='Comprehensive Market Anomaly Detection')
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        return go.Figure()