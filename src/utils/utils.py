import pandas as pd
from pathlib import Path
from typing import Union, List

def ensure_directory(directory: Union[str, Path]) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
    directory (str or Path): The directory path to check/create.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def safe_read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Safely read a CSV file, handling potential errors.
    
    Args:
    file_path (str or Path): The path to the CSV file.
    **kwargs: Additional keyword arguments to pass to pd.read_csv().

    Returns:
    pd.DataFrame: The data from the CSV file.
    """
    try:
        return pd.read_csv(file_path, **kwargs)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: Empty file - {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return pd.DataFrame()

def safe_to_datetime(series: pd.Series, errors: str = 'coerce') -> pd.Series:
    """
    Safely convert a series to datetime, handling errors.
    
    Args:
    series (pd.Series): The series to convert.
    errors (str): How to handle errors. Default is 'coerce'.

    Returns:
    pd.Series: The converted datetime series.
    """
    return pd.to_datetime(series, errors=errors)

def get_file_paths(directory: Union[str, Path], pattern: str) -> List[Path]:
    """
    Get a list of file paths in a directory matching a pattern.
    
    Args:
    directory (str or Path): The directory to search.
    pattern (str): The glob pattern to match files.

    Returns:
    List[Path]: A list of matching file paths.
    """
    return list(Path(directory).glob(pattern))

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame has the required columns.
    
    Args:
    df (pd.DataFrame): The DataFrame to validate.
    required_columns (List[str]): List of required column names.

    Returns:
    bool: True if valid, False otherwise.
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return False
    return True