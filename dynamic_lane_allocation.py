import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.cluster import KMeans
import folium
from folium.plugins import HeatMap
np.random.seed(42)
tf.random.set_seed(42)
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df
def preprocess_data(df):
    processed_df = df.copy()
    
    if ':' in processed_df['initiated_time'][0] and len(processed_df['initiated_time'][0]) <= 5:
        processed_df['initiated_time'] = pd.to_datetime('2023-01-01 ' + processed_df['initiated_time'], format='%Y-%m-%d %H:%M', errors='coerce')
    else:
        processed_df['initiated_time'] = pd.to_datetime(processed_df['initiated_time'], errors='coerce')
    
    processed_df['hour'] = processed_df['initiated_time'].dt.hour
    processed_df['minute'] = processed_df['initiated_time'].dt.minute
    processed_df['time_of_day'] = pd.cut(processed_df['hour'], bins=[0, 6, 12, 18, 24], 
                                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    processed_df['inn_rr_time_sec'] = pd.to_numeric(processed_df['inn_rr_time_sec'], errors='coerce')
    processed_df['txn_amount'] = pd.to_numeric(processed_df['txn_amount'], errors='coerce')
    
    le_vehicle = LabelEncoder()
    processed_df['vehicle_class_encoded'] = le_vehicle.fit_transform(processed_df['vehicle_class_code'])
    
    le_merchant = LabelEncoder()
    processed_df['merchant_encoded'] = le_merchant.fit_transform(processed_df['merchant_name'])
    
    le_direction = LabelEncoder()
    processed_df['direction_encoded'] = le_direction.fit_transform(processed_df['direction'])
    
    processed_df['is_commercial'] = processed_df['vehicle_comvehicle'].map({'T': 1, 'F': 0, True: 1, False: 0})
    
    try:
        geo_df = processed_df['geocode'].str.split(',', expand=True)
        processed_df['latitude'] = pd.to_numeric(geo_df[0], errors='coerce')
        processed_df['longitude'] = pd.to_numeric(geo_df[1], errors='coerce')
    except Exception as e:
        print(f"Error parsing geocode: {e}")
    
    critical_columns = ['initiated_time', 'inn_rr_time_sec']
    processed_df = processed_df.dropna(subset=critical_columns)
    
    return processed_df
def perform_eda(df):
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    hourly_count = df.groupby('hour').size()
    hourly_count.plot(kind='bar')
    plt.title('Transaction Volume by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Transactions')
    plt.savefig('plots/transactions_by_hour.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['inn_rr_time_sec'], bins=50, kde=True)
    plt.title('Distribution of Processing Times')
    plt.xlabel('Processing Time (seconds)')
    plt.ylabel('Frequency')
    plt.xlim(0, df['inn_rr_time_sec'].quantile(0.95))
    plt.savefig('plots/processing_time_distribution.png')
    plt.close()
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='vehicle_class_code', y='inn_rr_time_sec', data=df)
    plt.title('Processing Time by Vehicle Type')
    plt.xlabel('Vehicle Class')
    plt.ylabel('Processing Time (seconds)')
    plt.xticks(rotation=90)
    plt.ylim(0, df['inn_rr_time_sec'].quantile(0.95))
    plt.savefig('plots/processing_time_by_vehicle.png')
    plt.close()
    
    plt.figure(figsize=(14, 7))
    plaza_processing = df.groupby('merchant_name')['inn_rr_time_sec'].mean().sort_values().reset_index()
    sns.barplot(x='merchant_name', y='inn_rr_time_sec', data=plaza_processing)
    plt.title('Average Processing Time by Toll Plaza')
    plt.xlabel('Toll Plaza')
    plt.ylabel('Avg Processing Time (seconds)')
    plt.xticks(rotation=90)
    plt.savefig('plots/avg_processing_by_plaza.png')
    plt.close()
    
    plt.figure(figsize=(14, 7))
    lane_volume = df['lane'].value_counts().sort_index()
    lane_volume.plot(kind='bar')
    plt.title('Lane Utilization')
    plt.xlabel('Lane')
    plt.ylabel('Number of Transactions')
    plt.savefig('plots/lane_utilization.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    peak_hours = df.groupby('hour').size()
    peak_threshold = peak_hours.quantile(0.75)
    is_peak = peak_hours > peak_threshold
    peak_df = pd.DataFrame({'hour': peak_hours.index, 'count': peak_hours.values, 'is_peak': is_peak.values})
    
    colors = ['orange' if x else 'skyblue' for x in peak_df['is_peak']]
    plt.bar(peak_df['hour'], peak_df['count'], color=colors)
    plt.title('Peak Hours Identification')
    plt.xlabel('Hour of Day')
    plt.ylabel('Transaction Count')
    plt.savefig('plots/peak_hours.png')
    plt.close()
    
    plt.figure(figsize=(14, 10))
    hourly_vehicle_dist = df.groupby(['hour', 'vehicle_class_code']).size().unstack(fill_value=0)
    hourly_vehicle_dist = hourly_vehicle_dist.div(hourly_vehicle_dist.sum(axis=1), axis=0)
    sns.heatmap(hourly_vehicle_dist, cmap='YlGnBu', annot=False)
    plt.title('Vehicle Type Distribution by Hour')
    plt.ylabel('Hour of Day')
    plt.xlabel('Vehicle Class')
    plt.savefig('plots/vehicle_distribution_by_hour.png')
    plt.close()
    
    return peak_df[peak_df['is_peak']]['hour'].tolist()
def engineer_features(df, peak_hours):
        # Time since last transaction at the same lane
    df_sorted = df.sort_values(['merchant_name', 'lane', 'initiated_time'])
    df_sorted['prev_time'] = df_sorted.groupby(['merchant_name', 'lane'])['initiated_time'].shift(1)
    df_sorted['time_since_last_txn'] = (df_sorted['initiated_time'] - df_sorted['prev_time']).dt.total_seconds()
    df_sorted['time_since_last_txn'] = df_sorted['time_since_last_txn'].fillna(60)  # Default to 60 seconds if first transaction
    
    # Capture recent processing time trends
    df_sorted['recent_avg_time'] = df_sorted.groupby(['merchant_name', 'lane'])['inn_rr_time_sec'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Normalize lane efficiency within
    # each plaza
    df_sorted['lane_efficiency'] = df_sorted.groupby(['merchant_name', 'lane'])['inn_rr_time_sec'].transform(
        lambda x: (x.mean() - x) / x.std()
    )
    
    # One-hot encode time of day
    one_hot = pd.get_dummies(df_sorted['time_of_day'], prefix='time_of_day')
    df_sorted = df_sorted.join(one_hot)
    
    # Binary feature for peak hours
    df_sorted['is_peak_hour'] = df_sorted['hour'].apply(lambda x: 1 if x in peak_hours else 0)
    
    # Interaction features
    df_sorted['vehicle_lane_interaction'] = df_sorted['vehicle_class_encoded'] * df_sorted['lane']
    df_sorted['time_amount_interaction'] = df_sorted['hour'] * df_sorted['txn_amount']
    
    # Lagged features
    df_sorted['lag_1_processing_time'] = df_sorted.groupby(['merchant_name', 'lane'])['inn_rr_time_sec'].shift(1)
    df_sorted['lag_2_processing_time'] = df_sorted.groupby(['merchant_name', 'lane'])['inn_rr_time_sec'].shift(2)
    df_sorted = df_sorted.copy()
    df_sorted.fillna(df_sorted.mean(numeric_only=True), inplace=True)
    
    return df_sorted
def train_lstm_model(df):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    X = df['hour'].values.reshape(-1, 1, 1)
    y = df['inn_rr_time_sec'].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2)
    model.save('lstm_model_lane_allocation.h5')
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"LSTM Model MSE: {mse}")
    
    return model
def predict_processing_time(df, model):
    df['predicted_processing_time'] = model.predict(df['hour'].values.reshape(-1, 1, 1))
    return df
def optimize_lane_allocation(df):
    optimized_df = df.copy()
    optimized_df['optimized_processing_time'] = optimized_df['predicted_processing_time'].copy()
    optimized_df['optimized_lane'] = optimized_df['lane'].copy()
    
    for merchant in optimized_df['merchant_name'].unique():
        merchant_data = optimized_df[optimized_df['merchant_name'] == merchant].copy()
        for hour in merchant_data['hour'].unique():
            hour_data = merchant_data[merchant_data['hour'] == hour].copy()
            
            if len(hour_data) > 1:
                lanes = hour_data['lane'].unique()
                avg_times = hour_data.groupby('lane')['predicted_processing_time'].mean()
                
                if len(lanes) > 1:
                    fastest_lane = avg_times.idxmin()
                    slowest_lane = avg_times.idxmax()
                    
                    slow_lane_indices = hour_data[hour_data['lane'] == slowest_lane].index
                    optimized_df.loc[slow_lane_indices, 'optimized_lane'] = fastest_lane
                    optimized_df.loc[slow_lane_indices, 'optimized_processing_time'] = hour_data[hour_data['lane'] == fastest_lane]['predicted_processing_time'].mean()
    
    return optimized_df
def analyze_optimization_results(original_df, optimized_df):
    original_mean_time = original_df['inn_rr_time_sec'].mean()
    predicted_mean_time = original_df['predicted_processing_time'].mean()
    optimized_mean_time = optimized_df['optimized_processing_time'].mean()
    total_improvement = original_mean_time - optimized_mean_time
    
    hourly_comparison = original_df.groupby('hour')['inn_rr_time_sec'].mean().reset_index()
    hourly_comparison['predicted'] = original_df.groupby('hour')['predicted_processing_time'].mean().values
    hourly_comparison['optimized'] = optimized_df.groupby('hour')['optimized_processing_time'].mean().values
    
    lane_change_dist = optimized_df[optimized_df['lane'] != optimized_df['optimized_lane']].groupby('merchant_name')['lane'].count()
    
    vehicle_impact = optimized_df.groupby('vehicle_class_code').agg({
        'inn_rr_time_sec': 'mean',
        'optimized_processing_time': 'mean'
    })
    
    os.makedirs('plots', exist_ok=True)
    
    # Plotting hourly comparison
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_comparison['hour'], hourly_comparison['inn_rr_time_sec'], label='Original')
    plt.plot(hourly_comparison['hour'], hourly_comparison['predicted'], label='Predicted')
    plt.plot(hourly_comparison['hour'], hourly_comparison['optimized'], label='Optimized')
    plt.title('Hourly Processing Time Comparison')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Processing Time (seconds)')
    plt.legend()
    plt.savefig('plots/hourly_comparison.png')
    plt.close()
    
    # Plotting lane change distribution
    plt.figure(figsize=(12, 6))
    lane_change_dist.plot(kind='bar')
    plt.title('Lane Change Distribution by Toll Plaza')
    plt.xlabel('Toll Plaza')
    plt.ylabel('Number of Lane Changes')
    plt.savefig('plots/lane_change_distribution.png')
    plt.close()
    
    # Plotting vehicle impact
    plt.figure(figsize=(14, 7))
    vehicle_impact.plot(kind='bar')
    plt.title('Impact of Optimization on Processing Time by Vehicle Type')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Average Processing Time (seconds)')
    plt.xticks(rotation=45)
    plt.savefig('plots/vehicle_impact.png')
    plt.close()
    
    # Vehicle type improvement plot
    plt.figure(figsize=(14, 7))
    vehicle_impact['improvement'] = vehicle_impact['inn_rr_time_sec'] - vehicle_impact['optimized_processing_time']
    vehicle_impact['improvement'].plot(kind='bar', color='green')
    plt.title('Processing Time Improvement by Vehicle Type')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Improvement (seconds)')
    plt.xticks(rotation=45)
    plt.savefig('plots/vehicle_type_improvement.png')
    plt.close()
    
    # Heatmap of processing time improvements
    plaza_hour_improvement = optimized_df.groupby(['merchant_name', 'hour']).agg({
        'inn_rr_time_sec': 'mean',
        'optimized_processing_time': 'mean'
    }).reset_index()
    plaza_hour_improvement['improvement_pct'] = ((plaza_hour_improvement['inn_rr_time_sec'] - plaza_hour_improvement['optimized_processing_time']) / 
                                                plaza_hour_improvement['inn_rr_time_sec'] * 100)
    plt.figure(figsize=(16, 10))
    pivot_table = plaza_hour_improvement.pivot(index='merchant_name', columns='hour', values='improvement_pct')
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=0.5)
    plt.title('Processing Time Improvement by Plaza and Hour (%)', fontsize=15)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Toll Plaza', fontsize=12)
    plt.savefig('plots/plaza_hour_improvement_heatmap', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'original_mean_time': original_mean_time,
        'predicted_mean_time': predicted_mean_time,
        'optimized_mean_time': optimized_mean_time,
        'total_improvement': total_improvement,
        'hourly_comparison': hourly_comparison,
        'lane_change_distribution': lane_change_dist,
        'vehicle_impact': vehicle_impact,
        'plaza_hour_improvement': plaza_hour_improvement
    }
def perform_clustering(df):
    geo_data = df[['latitude', 'longitude']].dropna()
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(geo_data)
    
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude'])
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='longitude', y='latitude', hue='cluster', data=df, palette='viridis')
    plt.scatter(cluster_centers['longitude'], cluster_centers['latitude'], marker='X', s=200, color='red', label='Cluster Centers')
    plt.title('Transaction Clusters')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig('plots/transaction_clusters.png')
    plt.close()
    
    return df
def visualize_heatmap(df):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)
    heat_data = [[row['latitude'], row['longitude']] for index, row in df.iterrows() if pd.notna(row['latitude']) and pd.notna(row['longitude'])]
    HeatMap(heat_data).add_to(m)
    m.save('plots/heatmap_transactions.html')
def main(file_path):
    df = load_data(file_path)
    enhanced_df = preprocess_data(df)
    peak_hours = perform_eda(enhanced_df)
    enhanced_df = engineer_features(enhanced_df, peak_hours)
    
    lstm_model = train_lstm_model(enhanced_df)
    enhanced_df = predict_processing_time(enhanced_df, lstm_model)
    optimized_df = optimize_lane_allocation(enhanced_df)
    analysis_results = analyze_optimization_results(enhanced_df, optimized_df)
    
    clustered_df = perform_clustering(enhanced_df)
    visualize_heatmap(clustered_df)
    
    print("Analysis complete. Plots saved in 'plots' directory.")
    return analysis_results
if __name__ == "__main__":
    file_path = "/content/cleaned_toll_data.csv"
    results = main(file_path)
    print(results)