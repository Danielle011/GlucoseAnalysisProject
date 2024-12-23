import os
import pandas as pd
from processors.apple_health_processor import parse_apple_health_data, extract_time_periods, save_dataframes_to_csv
from processors.meal_data_processor import process_meal_data
from processors.glucose_data_processor import process_cgm_data
from analyzers.feature_extractor import extract_features, save_features

def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process glucose data first to get time periods
    print("Processing glucose data...")
    glucose_data = process_cgm_data()
    glucose_data_file = os.path.join(current_dir, "..", "data", "processed", "processed_glucose_data.csv")

    # Extract time periods from processed glucose data
    time_periods = extract_time_periods(glucose_data_file)

    # Process Apple Health data
    apple_health_file = os.path.join(current_dir, "..", "data", "raw", "export.xml")
    if os.path.exists(apple_health_file):
        print("Processing Apple Health data...")
        heart_rate_df, aggregated_data, sleep_df, workout_df = parse_apple_health_data(apple_health_file, time_periods)
        print("Apple Health data processing complete.")
        
        # Save Apple Health data to CSV
        save_dataframes_to_csv(heart_rate_df, aggregated_data, sleep_df, workout_df)
        
        # You can add additional analysis or reporting here if needed
        print(f"Heart Rate data shape: {heart_rate_df.shape}")
        print(f"Aggregated Step and Distance data shape: {aggregated_data.shape}")
        print(f"Sleep data shape: {sleep_df.shape}")
        print(f"Workout data shape: {workout_df.shape}")
    else:
        print(f"Apple Health data file not found at: {apple_health_file}")
        aggregated_data = pd.DataFrame()  # Create an empty DataFrame if Apple Health data is not available
        workout_df = pd.DataFrame()  # Create an empty DataFrame if Apple Health data is not available

    # Process meal data
    print("Processing meal data...")
    meal_data = process_meal_data()

    print("All data processed successfully!")

    # Add feature extraction step
    print("Extracting features...")
    processed_dir = os.path.join(current_dir, "..", "data", "processed")
    output_path = os.path.join(processed_dir, 'glucose_features_with_activity.csv')

    extracted_features = extract_features(meal_data, glucose_data, workout_df, aggregated_data)
    save_features(extracted_features, output_path)

    print("Feature extraction complete!")

if __name__ == "__main__":
    main()