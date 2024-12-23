# Glucose Analysis Project

A comprehensive data analysis pipeline for processing and analyzing continuous glucose monitoring (CGM) data along with Apple Health and meal data.

## Features

- Process and analyze Continuous Glucose Monitoring (CGM) data
- Integration with Apple Health data (steps, heart rate, sleep, workouts)
- Meal data analysis with automatic meal type categorization
- Feature extraction for glucose response analysis
- Time-series data interpolation and processing
- Activity and glucose correlation analysis

## Project Structure

```
glucose_analysis_project/
│
├── src/
│   ├── analyzers/
│   │   ├── __init__.py
│   │   └── feature_extractor.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── apple_health_processor.py
│   │   ├── glucose_data_processor.py
│   │   └── meal_data_processor.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── utils.py
│   └── main.py
│
├── data/
│   ├── raw/
│   │   ├── export.xml
│   │   └── PastaConnect_Data_*.csv
│   └── processed/
│
├── venv/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd glucose-analysis-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements

You need to prepare two types of data files and place them in the `data/raw/` directory:

1. **Apple Health Data**
   - Export your Apple Health data from the Health app on your iPhone
   - Settings -> Health -> Your Account -> Export All Health Data
   - Extract the exported zip file
   - Copy the `export.xml` file to `data/raw/` directory

2. **CGM and Meal Data from PASTA Health**
   - Log in to [PASTA Health Connect](https://connect.pastahealth.com)
   - Download '전체 데이터' (Full Data) for each measurement period
   - Rename the downloaded files to `PastaConnect_Data_1.csv`, `PastaConnect_Data_2.csv`, etc.
     - The number should match the measurement number (측정차수) in the data
   - Place all renamed files in the `data/raw/` directory

Example directory structure:
```
data/
└── raw/
    ├── export.xml
    ├── PastaConnect_Data_1.csv
    ├── PastaConnect_Data_2.csv
    └── PastaConnect_Data_3.csv
```

Note: The PASTA Health data files contain both glucose measurements and meal records, which will be automatically processed by the pipeline.

## Usage

1. Prepare your data files in the `data/raw/` directory

2. Run the main analysis script:
   ```bash
   python src/main.py
   ```

3. Check the processed results in `data/processed/` directory:
   - Processed glucose data
   - Extracted features
   - Activity correlations
   - Meal analysis results

## Output Files

The pipeline generates several processed data files:
- `processed_glucose_data.csv`: Cleaned and interpolated glucose readings
- `processed_meal_data.csv`: Categorized and processed meal data
- `glucose_features_with_activity.csv`: Extracted features combining glucose, meal, and activity data
- Additional Apple Health processed data files (if available)

## Configuration

- Timezone is set to KST (Korea Standard Time)
- Glucose readings are interpolated to 1-minute intervals
- Meal combinations are processed with a 40-minute threshold

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.