import pandas as pd
import numpy as np
import glob
from datetime import timezone, timedelta
import os

KST = timezone(timedelta(hours=9))

def process_cgm_data(file_pattern='data/raw/PastaConnect_Data_*.csv'):
    all_files = glob.glob(file_pattern)
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, encoding='utf-8')
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)

    column_mapping = {
        '측정차수': 'MeasurementNumber',
        '측정 및 기록 시간': 'DateTime',
        '혈당 값 (mg/dL)': 'GlucoseValue',
        '기록 구분': 'RecordType'
    }
    df.rename(columns=column_mapping, inplace=True)

    df = df[df['RecordType'] == '혈당']
    df['MeasurementNumber'] = df['MeasurementNumber'].astype(int)

    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.round('T')
    df['DateTime'] = df['DateTime'].dt.tz_localize(KST)

    df.sort_values(['MeasurementNumber', 'DateTime'], inplace=True)

    def process_group(group):
        start_time = group['DateTime'].min()
        end_time = group['DateTime'].max()
        full_date_range = pd.date_range(start=start_time, end=end_time, freq='T', tz=KST)
        
        interpolated = pd.DataFrame({'DateTime': full_date_range})
        interpolated = pd.merge(interpolated, group[['DateTime', 'GlucoseValue', 'MeasurementNumber']], 
                                on='DateTime', how='left')
        
        interpolated.set_index('DateTime', inplace=True)
        interpolated['IsInterpolated'] = interpolated['GlucoseValue'].isnull()
        interpolated['GlucoseValue'] = interpolated['GlucoseValue'].interpolate(method='time')
        interpolated['MeasurementNumber'] = interpolated['MeasurementNumber'].fillna(method='ffill').astype(int)
        
        return interpolated.reset_index()

    grouped = df.groupby('MeasurementNumber')
    interpolated_groups = [process_group(group) for _, group in grouped]
    interpolated_df = pd.concat(interpolated_groups, ignore_index=True)

    interpolated_df = interpolated_df[['MeasurementNumber', 'DateTime', 'GlucoseValue', 'IsInterpolated']]

    output_file = os.path.join('data', 'processed', 'processed_glucose_data.csv')
    interpolated_df.to_csv(output_file, index=False)
    print(f"Saved processed glucose data to {output_file}")

    return interpolated_df

def calculate_statistics(df):
    return {
        'Mean Glucose': df['GlucoseValue'].mean(),
        'Median Glucose': df['GlucoseValue'].median(),
        'Min Glucose': df['GlucoseValue'].min(),
        'Max Glucose': df['GlucoseValue'].max(),
        'Standard Deviation': df['GlucoseValue'].std(),
        'Percentage of Interpolated Values': df['IsInterpolated'].mean() * 100
    }

if __name__ == "__main__":
    processed_data = process_cgm_data()
    print("Glucose data processing complete.")
    print(f"Processed data shape: {processed_data.shape}")
    print("\nFirst few rows of processed data:")
    print(processed_data.head())
    
    statistics = calculate_statistics(processed_data)
    print("\nStatistics:")
    for stat, value in statistics.items():
        print(f"{stat}: {value:.2f}")