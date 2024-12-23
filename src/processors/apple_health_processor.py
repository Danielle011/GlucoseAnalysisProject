import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import os

def parse_date(date_string):
    """Convert date string to timezone-aware datetime"""
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S %z")

def is_in_time_period(date, time_periods):
    """Check if date is within specified time periods"""
    return any(start <= date <= end for start, end in time_periods)

def parse_heart_rate(record):
    """Parse heart rate records"""
    return {
        'type': record.get('type'),
        'value': float(record.get('value')),
        'unit': record.get('unit'),
        'start_date': parse_date(record.get('startDate')),
        'end_date': parse_date(record.get('endDate')),
        'device': record.get('device'),
        'source_name': record.get('sourceName'),
        'source_version': record.get('sourceVersion'),
        'creation_date': parse_date(record.get('creationDate')),
        'motion_context': next((entry.get('value') for entry in record.findall('MetadataEntry') 
                              if entry.get('key') == 'HKMetadataKeyHeartRateMotionContext'), None)
    }

def parse_activity_record(record):
    """Generic parser for activity records (steps, distance, flights)"""
    try:
        return {
            'start_date': parse_date(record.get('startDate')),
            'end_date': parse_date(record.get('endDate')),
            'value': float(record.get('value')),
            'source': record.get('sourceName'),
            'device': record.get('device')
        }
    except (ValueError, TypeError):
        return None

def deduplicate_health_records(records):
    """
    Deduplicate health records prioritizing Apple Watch data.
    When Watch and iPhone data overlap in time, use Watch data only.
    When no overlap, keep records from both devices.
    """
    if not records:
        return []

    def has_time_overlap(rec1, rec2):
        """Check if two records overlap in time"""
        return (rec1['start_date'] <= rec2['end_date'] and 
                rec2['start_date'] <= rec1['end_date'])

    sorted_records = sorted(records, key=lambda x: x['start_date'])
    watch_records = [r for r in sorted_records if "Watch" in r['source']]
    iphone_records = [r for r in sorted_records if "Watch" not in r['source']]
    
    final_records = watch_records.copy()
    for iphone_record in iphone_records:
        has_overlap = any(has_time_overlap(iphone_record, watch_record) 
                         for watch_record in watch_records)
        if not has_overlap:
            final_records.append(iphone_record)
    
    return sorted(final_records, key=lambda x: x['start_date'])

def parse_sleep(record):
    """Parse sleep analysis records"""
    return {
        'type': record.get('type'),
        'value': record.get('value'),
        'start_date': parse_date(record.get('startDate')),
        'end_date': parse_date(record.get('endDate')),
        'source_name': record.get('sourceName'),
        'source_version': record.get('sourceVersion'),
        'creation_date': parse_date(record.get('creationDate'))
    }

def parse_workout(workout):
    """Parse workout records"""
    workout_data = {
        'type': workout.get('workoutActivityType'),
        'start_time': parse_date(workout.get('startDate')),
        'end_time': parse_date(workout.get('endDate')),
        'duration': float(workout.get('duration')),
        'duration_unit': workout.get('durationUnit'),
        'total_distance': None,
        'total_energy_burned': None,
        'avg_heart_rate': None,
        'min_heart_rate': None,
        'max_heart_rate': None,
        'avg_mets': None,
        'temperature': None,
        'humidity': None,
        'elevation_ascended': None,
        'num_segments': 0,
    }

    for entry in workout.findall('MetadataEntry'):
        key, value = entry.get('key'), entry.get('value')
        if key == 'HKWeatherTemperature':
            workout_data['temperature'] = float(value.split()[0])
        elif key == 'HKWeatherHumidity':
            workout_data['humidity'] = float(value.split()[0]) / 100
        elif key == 'HKAverageMETs':
            workout_data['avg_mets'] = float(value.split()[0])
        elif key == 'HKElevationAscended':
            workout_data['elevation_ascended'] = float(value.split()[0])

    workout_data['num_segments'] = len(workout.findall('WorkoutEvent'))

    for stat in workout.findall('WorkoutStatistics'):
        stat_type = stat.get('type')
        if stat_type == 'HKQuantityTypeIdentifierDistanceWalkingRunning':
            workout_data['total_distance'] = float(stat.get('sum'))
        elif stat_type == 'HKQuantityTypeIdentifierActiveEnergyBurned':
            workout_data['total_energy_burned'] = float(stat.get('sum'))
        elif stat_type == 'HKQuantityTypeIdentifierHeartRate':
            workout_data['avg_heart_rate'] = float(stat.get('average'))
            workout_data['min_heart_rate'] = float(stat.get('minimum'))
            workout_data['max_heart_rate'] = float(stat.get('maximum'))

    return workout_data

def process_sleep_data(sleep_data):
    """
    Process sleep records based on creation_date (date only).
    If both iPhone and Watch data exist for the same date, use only Watch data.
    """
    if not sleep_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(sleep_data)
    
    # Add date-only column for creation_date grouping
    df['creation_date_day'] = df['creation_date'].dt.date
    
    def process_sleep_group(group):
        """Process a group of sleep records with same creation date"""
        # Check if this date has any Watch data
        has_watch_data = group['source_name'].str.contains('Watch', na=False).any()
        
        if has_watch_data:
            # Use only Watch sleep states
            watch_sleep_states = group[
                (group['source_name'].str.contains('Watch', na=False)) &
                (group['value'].isin([
                    'HKCategoryValueSleepAnalysisAsleepCore',
                    'HKCategoryValueSleepAnalysisAsleepDeep',
                    'HKCategoryValueSleepAnalysisAsleepREM'
                ]))
            ]
            
            if watch_sleep_states.empty:
                return None
                
            start_date = watch_sleep_states['start_date'].min()
            end_date = watch_sleep_states['end_date'].max()
            
            # Calculate duration for each sleep state
            core_sleep = pd.Timedelta(0)
            deep_sleep = pd.Timedelta(0)
            rem_sleep = pd.Timedelta(0)
            
            for _, row in watch_sleep_states.iterrows():
                duration = row['end_date'] - row['start_date']
                if row['value'] == 'HKCategoryValueSleepAnalysisAsleepCore':
                    core_sleep += duration
                elif row['value'] == 'HKCategoryValueSleepAnalysisAsleepDeep':
                    deep_sleep += duration
                elif row['value'] == 'HKCategoryValueSleepAnalysisAsleepREM':
                    rem_sleep += duration
            
            total_sleep = core_sleep + deep_sleep + rem_sleep
            
        else:
            # Use iPhone InBed data
            iphone_data = group[
                (group['source_name'].str.contains('iPhone', na=False)) &
                (group['value'] == 'HKCategoryValueSleepAnalysisInBed')
            ]
            
            if iphone_data.empty:
                return None
                
            start_date = iphone_data['start_date'].min()
            end_date = iphone_data['end_date'].max()
            
            # For iPhone data, calculate total duration as core sleep
            total_sleep = core_sleep = pd.Timedelta(0)
            for _, row in iphone_data.iterrows():
                duration = row['end_date'] - row['start_date']
                total_sleep += duration
            
            core_sleep = total_sleep
            deep_sleep = pd.Timedelta(0)
            rem_sleep = pd.Timedelta(0)
        
        return pd.Series({
            'start_date': start_date,
            'end_date': end_date,
            'total_sleep_time': total_sleep,
            'core_sleep_time': core_sleep,
            'deep_sleep_time': deep_sleep,
            'rem_sleep_time': rem_sleep
        })
    
    # Process each creation_date_day group
    results = []
    for _, group in df.groupby('creation_date_day'):
        result = process_sleep_group(group)
        if result is not None:
            results.append(result)
    
    if not results:
        return pd.DataFrame()
    
    # Create final DataFrame and sort by start_date
    result_df = pd.DataFrame(results).sort_values('start_date')
    
    # Ensure all timestamps are in KST
    for col in ['start_date', 'end_date']:
        if result_df[col].dt.tz is None:
            result_df[col] = result_df[col].dt.tz_localize('Asia/Seoul')
        else:
            result_df[col] = result_df[col].dt.tz_convert('Asia/Seoul')
    
    return result_df

def aggregate_data(df, interval='10min'):
    """Aggregate data into specified time intervals"""
    if df.empty:
        # Create a dummy DataFrame with proper structure but no data
        return pd.DataFrame({
            'start_time': pd.Series(dtype='datetime64[ns, UTC]'),
            'end_time': pd.Series(dtype='datetime64[ns, UTC]'),
            'value': pd.Series(dtype='float64')
        })

    distributed_records = []
    for _, row in df.iterrows():
        start = row['start_date']
        end = row['end_date']
        value = row['value']
        
        # Ensure timezone-aware datetime
        intervals = pd.date_range(
            start=start.floor(interval),
            end=end.ceil(interval),
            freq=interval,
            tz=start.tzinfo
        )
        
        for interval_start, interval_end in zip(intervals, intervals[1:]):
            overlap_start = max(start, interval_start)
            overlap_end = min(end, interval_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            total_duration = (end - start).total_seconds()
            
            if overlap_duration > 0:
                # interval_start and interval_end are already timezone-aware
                distributed_records.append({
                    'start_time': interval_start,
                    'end_time': interval_end,
                    'value': value * (overlap_duration / total_duration)
                })
    
    result_df = pd.DataFrame(distributed_records)
    if not result_df.empty:
        result_df['value'] = result_df['value'].astype(float)
        aggregated = result_df.groupby(['start_time', 'end_time'])['value'].sum().reset_index()
        return aggregated
    return result_df

def parse_apple_health_data(xml_file, time_periods):
    """Main function to parse Apple Health data"""
    print("Reading XML file...")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    heart_rate_data = []
    sleep_data = []
    step_data = []
    distance_data = []
    flights_climbed_data = []
    workout_data = []
    
    print("Processing records...")
    for record in root.findall('.//Record'):
        start_date = parse_date(record.get('startDate'))
        if is_in_time_period(start_date, time_periods):
            record_type = record.get('type')
            
            if record_type == 'HKQuantityTypeIdentifierHeartRate':
                heart_rate_data.append(parse_heart_rate(record))
            elif record_type == 'HKCategoryTypeIdentifierSleepAnalysis':
                sleep_data.append(parse_sleep(record))
            elif record_type == 'HKQuantityTypeIdentifierStepCount':
                parsed_record = parse_activity_record(record)
                if parsed_record:
                    step_data.append(parsed_record)
            elif record_type == 'HKQuantityTypeIdentifierDistanceWalkingRunning':
                parsed_record = parse_activity_record(record)
                if parsed_record:
                    distance_data.append(parsed_record)
            elif record_type == 'HKQuantityTypeIdentifierFlightsClimbed':
                parsed_record = parse_activity_record(record)
                if parsed_record:
                    flights_climbed_data.append(parsed_record)

    print("Processing workout data...")
    for workout in root.findall('.//Workout'):
        start_date = parse_date(workout.get('startDate'))
        if is_in_time_period(start_date, time_periods):
            workout_data.append(parse_workout(workout))

    print("Deduplicating and aggregating activity data...")
    # Deduplicate and aggregate each activity metric
    deduplicated_steps = deduplicate_health_records(step_data)
    deduplicated_distance = deduplicate_health_records(distance_data)
    deduplicated_flights = deduplicate_health_records(flights_climbed_data)

    # Convert to DataFrames
    heart_rate_df = pd.DataFrame(heart_rate_data)
    steps_df = pd.DataFrame(deduplicated_steps)
    distance_df = pd.DataFrame(deduplicated_distance)
    flights_df = pd.DataFrame(deduplicated_flights)
    workout_df = pd.DataFrame(workout_data)

    # Aggregate activity data
    print("Creating final DataFrames...")
    if not steps_df.empty:
        aggregated_steps = aggregate_data(steps_df)
        aggregated_steps['value'] = aggregated_steps['value'].round().astype(int)
        aggregated_steps = aggregated_steps.rename(columns={'value': 'steps'})
    else:
        aggregated_steps = pd.DataFrame(columns=['start_time', 'end_time', 'steps'])

    if not distance_df.empty:
        aggregated_distance = aggregate_data(distance_df)
        aggregated_distance['value'] = aggregated_distance['value'].round(3)
        aggregated_distance = aggregated_distance.rename(columns={'value': 'distance'})
    else:
        aggregated_distance = pd.DataFrame(columns=['start_time', 'end_time', 'distance'])

    if not flights_df.empty:
        aggregated_flights = aggregate_data(flights_df)
        aggregated_flights['value'] = aggregated_flights['value'].round(3)
        aggregated_flights = aggregated_flights.rename(columns={'value': 'flights'})
    else:
        aggregated_flights = pd.DataFrame(columns=['start_time', 'end_time', 'flights'])

    # Process sleep data
    sleep_df = process_sleep_data(sleep_data)

    # Merge all activity data
    print("Merging activity data...")
    
    # Ensure datetime types before merge
    for df in [aggregated_steps, aggregated_distance, aggregated_flights]:
        if not df.empty:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
    
    # First merge steps and distance
    aggregated_data = pd.merge(
        aggregated_steps,
        aggregated_distance[['start_time', 'end_time', 'distance']],
        on=['start_time', 'end_time'],
        how='outer'
    )
    
    # Then merge flights
    aggregated_data = pd.merge(
        aggregated_data,
        aggregated_flights[['start_time', 'end_time', 'flights']],
        on=['start_time', 'end_time'],
        how='outer'
    )
    
    # Fill NaN values with 0 for numeric columns only
    numeric_columns = ['steps', 'distance', 'flights']
    aggregated_data[numeric_columns] = aggregated_data[numeric_columns].fillna(0)
    
    # Ensure datetime types after merge
    aggregated_data['start_time'] = pd.to_datetime(aggregated_data['start_time'])
    aggregated_data['end_time'] = pd.to_datetime(aggregated_data['end_time'])
    
    # Sort by start_time
    aggregated_data = aggregated_data.sort_values('start_time')

    return heart_rate_df, aggregated_data, sleep_df, workout_df


def save_dataframes_to_csv(heart_rate_df, aggregated_data, sleep_df, workout_df):
    """Save processed data to CSV files"""
    output_dir = os.path.join('data', 'processed')
    print(f"Attempting to save files to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    def format_date(dt):
        if pd.isna(dt):
            return dt
        return dt.strftime('%Y-%m-%d %H:%M:%S%z')
    
    for df, filename in [
        (heart_rate_df, 'heart_rate_data.csv'),
        (aggregated_data, 'aggregated_activity_data.csv'),
        (sleep_df, 'sleep_data.csv'),
        (workout_df, 'workout_data.csv')
    ]:
        if not df.empty:
            file_path = os.path.join(output_dir, filename)
            print(f"Saving {filename} to: {file_path}")
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].apply(format_date)
            df.to_csv(file_path, index=False)
            print(f"Saved {filename}")


def extract_time_periods(glucose_data_file):
    """Extract time periods from glucose data"""
    df = pd.read_csv(glucose_data_file)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Define KST timezone
    kst = timezone(timedelta(hours=9))
    
    # Check if datetime is timezone-naive
    if df['DateTime'].dt.tz is None:
        df['DateTime'] = df['DateTime'].dt.tz_localize(kst)
    else:
        # If already tz-aware, convert to KST
        df['DateTime'] = df['DateTime'].dt.tz_convert(kst)
    
    time_periods = []
    for measurement_number, group in df.groupby('MeasurementNumber'):
        start_time = group['DateTime'].min()
        end_time = group['DateTime'].max()
        time_periods.append((start_time, end_time))
    
    return time_periods        