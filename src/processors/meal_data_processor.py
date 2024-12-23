import pandas as pd
import numpy as np
import glob
import os
from datetime import timedelta, timezone

KST = timezone(timedelta(hours=9))

def process_csv_file(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Filter rows where 기록 구분 is '식사'
    df = df[df['기록 구분'] == '식사']
    
    # Select measurement number, time, and all columns starting with [식사]
    meal_columns = [col for col in df.columns if col.startswith('[식사]')]
    columns_to_keep = ['측정차수', '측정 및 기록 시간'] + meal_columns
    df = df[columns_to_keep]
    
    # Create column name mapping
    column_name_mapping = {
        '측정차수': 'measurement_number',
        '측정 및 기록 시간': 'meal_time',
        '[식사] 음식명': 'food_name',
        '[식사] 칼로리 (kcal)': 'calories',
        '[식사] 탄수화물 (g)': 'carbohydrates',
        '[식사] 당류 (g)': 'sugars',
        '[식사] 단백질 (g)': 'protein',
        '[식사] 지방 (g)': 'fat',
        '[식사] 포화지방 (g)': 'saturated_fat',
        '[식사] 트랜스지방 (g)': 'trans_fat',
        '[식사] 콜레스테롤 (mg)': 'cholesterol',
        '[식사] 나트륨 (mg)': 'sodium'
    }
    df.rename(columns=column_name_mapping, inplace=True)
    
    # Convert meal_time to timezone-aware datetime
    df['meal_time'] = pd.to_datetime(df['meal_time']).dt.tz_localize(KST)
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['calories', 'carbohydrates', 'sugars', 'protein', 'fat',
                      'saturated_fat', 'trans_fat', 'cholesterol', 'sodium']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    return df

def merge_csv_files():
    # Get all CSV files matching the pattern
    csv_files = glob.glob('data/raw/PastaConnect_Data_*.csv')
    
    # Sort the files based on the measurement order
    csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Process and merge all CSV files
    merged_df = pd.concat([process_csv_file(file) for file in csv_files], ignore_index=True)
    
    return merged_df

def combine_continuous_meals(df, time_threshold=40):
    """
    Combine meals that occur within the specified time threshold.
    """
    df = df.sort_values('meal_time')
    df['time_diff'] = df['meal_time'].diff().dt.total_seconds() / 60
    df['new_meal'] = (df['time_diff'] > time_threshold) | (df['time_diff'].isna())
    df['meal_group'] = df['new_meal'].cumsum()
    
    combined_meals = df.groupby('meal_group').agg({
        'measurement_number': 'first',  # Keep the measurement number from the first record
        'meal_time': 'first',
        'food_name': lambda x: ' / '.join(x),
        'calories': 'sum',
        'carbohydrates': 'sum',
        'sugars': 'sum',
        'protein': 'sum',
        'fat': 'sum',
        'saturated_fat': 'sum',
        'trans_fat': 'sum',
        'cholesterol': 'sum',
        'sodium': 'sum'
    }).reset_index(drop=True)
    
    return combined_meals

def categorize_meals(df):
    """
    Categorize meals based on time, calorie content, and Korean food names.
    """
    def get_meal_type(row):
        hour = row['meal_time'].hour
        calories = row['calories']
        food_name = row['food_name']
        
        main_meal_keywords = [
            '샐러드', '쌀라다', '샐러드', '수프', '국',  # salad, soup
            '비빔밥', '볶음밥', '덮밥',  # rice dishes
            '김밥', '샌드위치',  # kimbap, sandwich
            '스테이크', '구이', '찌개',  # steak, grilled dishes, stew
            '파스타', '면', '국수',  # pasta, noodles
            '채식', '식사',  # vegetarian meal, meal
            '닭',
        ]
        
        is_potential_main_meal = any(keyword in food_name for keyword in main_meal_keywords)
        
        if 5 <= hour < 11:
            return 'Breakfast' if calories > 500 or is_potential_main_meal else 'Snack'
        elif 11 <= hour < 17:
            return 'Lunch' if calories > 500 or is_potential_main_meal else 'Snack'
        elif 17 <= hour < 24:
            return 'Dinner' if calories > 500 or is_potential_main_meal else 'Snack'
        else:  # Between midnight and 5 AM
            return 'Snack'
    
    df['meal_type'] = df.apply(get_meal_type, axis=1)
    return df

def process_meal_data():
    """
    Main function to process meal data:
    1. Merge all CSV files
    2. Combine continuous meals
    3. Categorize meals
    4. Save processed data
    """
    print("Merging meal data files...")
    merged_data = merge_csv_files()
    
    print("Combining continuous meals...")
    combined_df = combine_continuous_meals(merged_data)
    
    print("Categorizing meals...")
    categorized_df = categorize_meals(combined_df)
    
    # Save processed data
    output_file = os.path.join('data', 'processed', 'processed_meal_data.csv')
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    categorized_df.to_csv(output_file, index=False)
    print(f"Saved processed meal data to {output_file}")
    
    return categorized_df

if __name__ == "__main__":
    processed_data = process_meal_data()
    print("\nFirst few rows of processed data:")
    print(processed_data.head())
    
    print("\nMeal type distribution:")
    print(processed_data['meal_type'].value_counts())