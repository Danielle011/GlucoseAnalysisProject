import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import timezone, timedelta

KST = timezone(timedelta(hours=9))

def ensure_datetime(df, columns):
    for col in columns:
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], utc=True)
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize('UTC')
            df[col] = df[col].dt.tz_convert(KST)
    return df

def categorize_carbs(carbs):
    if carbs <= 30:
        return 'low'
    elif carbs <= 75:
        return 'medium'
    else:
        return 'high'

def calculate_interval_steps(meal_time, interval_start, interval_end, step_data):
    """Calculate steps for a specific time interval using weighted calculation"""
    relevant_steps = step_data[
        (step_data['end_time'] > interval_start) & 
        (step_data['start_time'] < interval_end)
    ].copy()
    
    if relevant_steps.empty:
        return 0.0
    
    total_steps = 0
    for _, row in relevant_steps.iterrows():
        # Calculate overlap duration in minutes
        overlap_start = max(interval_start, row['start_time'])
        overlap_end = min(interval_end, row['end_time'])
        overlap_duration = (overlap_end - overlap_start).total_seconds() / 60  # in minutes
        
        # Calculate weight (proportion of 10-minute interval that overlaps)
        interval_duration = (row['end_time'] - row['start_time']).total_seconds() / 60  # should be 10
        weight = overlap_duration / interval_duration
        
        # Add weighted steps
        total_steps += row['steps'] * weight
    
    return total_steps

def is_active(step_data):
    """Determine if activity level is active based on step count criteria:
    - total steps >= 600 OR
    - any 10-minute interval has >= 200 steps
    """
    if step_data.empty:
        return False
    total_steps = step_data['steps'].sum()
    max_interval_steps = step_data['steps'].max()
    return not (total_steps < 600 and max_interval_steps <= 200)

def extract_glucose_features(meal_data, glucose_data, step_data, window_hours=2):
    features = []
    
    for i, meal in meal_data.iterrows():
        meal_time = meal['meal_time']
        
        # Determine window end time
        if i < len(meal_data) - 1:
            next_meal_time = meal_data.iloc[i+1]['meal_time']
            window_end = min(meal_time + pd.Timedelta(hours=window_hours), next_meal_time)
        else:
            window_end = meal_time + pd.Timedelta(hours=window_hours)
        
        # Get glucose data for the window
        window_data = glucose_data[
            (glucose_data['DateTime'] >= meal_time) & 
            (glucose_data['DateTime'] <= window_end)
        ]
        
        if len(window_data) < 2:
            continue
        
        glucose_values = window_data['GlucoseValue']
        time_values = (window_data['DateTime'] - meal_time).dt.total_seconds() / 60
        
        # Calculate basic glucose metrics
        baseline = glucose_values.iloc[0]
        max_glucose = glucose_values.max()
        
        # Calculate AUC metrics
        values_above_baseline = np.maximum(glucose_values - baseline, 0)
        values_below_baseline = np.minimum(glucose_values - baseline, 0)
        values_above_140 = np.maximum(glucose_values - 140, 0)
        
        auc_above_baseline = np.trapz(values_above_baseline, time_values)
        auc_below_baseline = np.trapz(values_below_baseline, time_values)
        auc_above_140 = np.trapz(values_above_140, time_values)

        # Add here:
        net_auc = auc_above_baseline + auc_below_baseline
                
        # Calculate time to peak
        max_index = glucose_values.idxmax()
        time_to_peak = time_values.loc[max_index] if len(glucose_values) > 0 else np.nan
        
        # Calculate glucose variability
        glucose_variability = glucose_values.std() if len(glucose_values) > 1 else 0
        
        # Calculate steps for each 30-minute interval
        steps_intervals = []
        for j in range(4):  # 0-30, 30-60, 60-90, 90-120 minutes
            interval_start = meal_time + pd.Timedelta(minutes=30*j)
            interval_end = meal_time + pd.Timedelta(minutes=30*(j+1))
            
            # If interval extends beyond window_end, set to NaN
            if interval_start >= window_end:
                steps_intervals.append(np.nan)
            else:
                interval_end = min(interval_end, window_end)
                steps = calculate_interval_steps(meal_time, interval_start, interval_end, step_data)
                steps_intervals.append(steps)
                
        # Get step data for activity classification
        window_step_data = step_data[
            (step_data['start_time'] >= meal_time) & 
            (step_data['start_time'] < window_end)
        ]
        
        feature = {
            'meal_time': meal_time,
            'food_name': meal['food_name'],
            'baseline_glucose': baseline,
            'max_glucose': max_glucose,
            'glucose_range': max_glucose - baseline,
            'auc_above_baseline': auc_above_baseline,
            'auc_below_baseline': auc_below_baseline,
            'auc_above_140': auc_above_140,
            'net_auc': net_auc,
            'time_to_peak': time_to_peak,
            'glucose_variability': glucose_variability,
            'window_duration': (window_end - meal_time).total_seconds() / 60,  # in minutes
            'carb_category': categorize_carbs(meal['carbohydrates']),
            'is_active': is_active(window_step_data),
            'steps_0_30': steps_intervals[0],
            'steps_30_60': steps_intervals[1],
            'steps_60_90': steps_intervals[2],
            'steps_90_120': steps_intervals[3],
            'measurement_number': meal.get('measurement_number', None),
            'calories': meal.get('calories', None),
            'carbohydrates': meal.get('carbohydrates', None),
            'sugars': meal.get('sugars', None),
            'protein': meal.get('protein', None),
            'fat': meal.get('fat', None),
            'saturated_fat': meal.get('saturated_fat', None),
            'transfat': meal.get('transfat', None),
            'cholestrerol': meal.get('cholestrerol', None),
            'sodium': meal.get('sodium', None),
            'meal_type': meal.get('meal_type', None)
        }
        
        features.append(feature)
    
    return pd.DataFrame(features)

def extract_features(meal_data, glucose_data, workout_data, step_data):
    # Ensure datetime columns are properly formatted
    meal_data = ensure_datetime(meal_data, ['meal_time'])
    glucose_data = ensure_datetime(glucose_data, ['DateTime'])
    step_data = ensure_datetime(step_data, ['start_time', 'end_time'])
    
    # Note: workout_data is kept as parameter for future use but not currently used
    return extract_glucose_features(meal_data, glucose_data, step_data)

def save_features(features, output_path):
    # Convert timezone-aware datetime to string before saving
    for col in features.select_dtypes(include=['datetime64']).columns:
        features[col] = features[col].dt.strftime('%Y-%m-%d %H:%M:%S%z')
    
    features.to_csv(output_path, index=False)
    print(f"Extracted features saved to {output_path}")

if __name__ == "__main__":
    print("This is the feature extractor module. Run main.py to process data and extract features.")