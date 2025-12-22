"""
Data import utilities for reading experimental data files.
"""

import numpy as np
import re


def load_txt_file(filepath, delimiter=None, comments='#', skip_header=0):
    """
    Load data from TXT file with automatic header detection.
    
    Parameters
    ----------
    filepath : str
        Path to TXT file
    delimiter : str or None, optional
        Delimiter between columns. If None, auto-detect (space, tab, comma)
    comments : str, optional
        Character indicating comment lines, default '#'
    skip_header : int, optional
        Number of header lines to skip, default 0
    
    Returns
    -------
    x : ndarray
        X-axis data (first column)
    y : ndarray
        Y-axis data (second column)
    
    Raises
    ------
    ValueError
        If file cannot be parsed or doesn't have at least 2 columns
    """
    # First, try to auto-detect header lines
    auto_skip = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            # Skip empty lines and comment lines
            if not line or line.startswith(comments):
                auto_skip += 1
                continue
            
            # Try to parse first data line
            try:
                # Split line
                if delimiter:
                    parts = line.split(delimiter)
                else:
                    # Try whitespace split
                    parts = line.split()
                
                # Try to convert first two elements to float
                float(parts[0])
                float(parts[1])
                # If successful, we found the data start
                break
            except (ValueError, IndexError):
                # This line doesn't contain numeric data - treat as header
                auto_skip += 1
                continue
    
    # Use auto-detected skip value if no manual skip specified
    if skip_header == 0:
        skip_header = auto_skip
    
    try:
        # Try to load with numpy
        data = np.loadtxt(filepath, delimiter=delimiter, comments=comments, 
                         skiprows=skip_header, ndmin=2)
        
        if data.shape[1] < 2:
            raise ValueError(f"File must have at least 2 columns, found {data.shape[1]}")
        
        x = data[:, 0]
        y = data[:, 1]
        
        # Validate data
        if not np.all(np.isfinite(x)):
            raise ValueError("X data contains NaN or Inf values")
        if not np.all(np.isfinite(y)):
            raise ValueError("Y data contains NaN or Inf values")
        
        return x, y
        
    except Exception as e:
        raise ValueError(f"Error loading file '{filepath}': {e}")


def auto_detect_delimiter(filepath, max_lines=10):
    """
    Automatically detect delimiter in text file.
    
    Parameters
    ----------
    filepath : str
        Path to file
    max_lines : int, optional
        Number of lines to check, default 10
    
    Returns
    -------
    str or None
        Detected delimiter (space, tab, comma, or None)
    """
    with open(filepath, 'r') as f:
        lines = []
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
    
    if not lines:
        return None
    
    # Check for common delimiters
    delimiters = {
        ',': 'comma',
        '\t': 'tab',
        ' ': 'space',
    }
    
    for delim, name in delimiters.items():
        # Count delimiter occurrences in each line
        counts = [line.count(delim) for line in lines]
        # If consistent and more than one column
        if len(set(counts)) == 1 and counts[0] > 0:
            return delim
    
    return None


def load_data_file(filepath):
    """
    Load data file with automatic format detection.
    
    Parameters
    ----------
    filepath : str
        Path to data file
    
    Returns
    -------
    x : ndarray
        X-axis data
    y : ndarray
        Y-axis data
    """
    # Try automatic delimiter detection
    delimiter = auto_detect_delimiter(filepath)
    
    try:
        x, y = load_txt_file(filepath, delimiter=delimiter)
        return x, y
    except Exception as e:
        # Try without delimiter (whitespace-separated)
        try:
            x, y = load_txt_file(filepath, delimiter=None)
            return x, y
        except:
            raise ValueError(f"Could not load data file: {e}")


def validate_data(x, y):
    """
    Validate experimental data.
    
    Parameters
    ----------
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    
    Returns
    -------
    bool
        True if data is valid
    
    Raises
    ------
    ValueError
        If data validation fails
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError(f"X and Y must have same length: {len(x)} vs {len(y)}")
    
    if len(x) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(x)}")
    
    if not np.all(np.isfinite(x)):
        raise ValueError("X data contains NaN or Inf")
    
    if not np.all(np.isfinite(y)):
        raise ValueError("Y data contains NaN or Inf")
    
    return True
