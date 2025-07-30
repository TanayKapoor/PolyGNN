#!/usr/bin/env python3
"""
Data Preview Analysis - Display first few rows of files in data/raw/

This script provides a quick overview of the data files in the raw data directory,
showing the structure and first few rows of each file.
"""

import os
import pandas as pd
import pickle
import json
from pathlib import Path


def preview_csv_file(filepath, filename, n_rows=5):
    """Preview CSV file with first n rows"""
    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print(f"{'='*60}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            if encoding != 'utf-8':
                print(f"Note: File read with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
            continue
    
    if df is None:
        print("Error: Could not read CSV file with any supported encoding")
        return
        
    try:
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst {} rows:".format(n_rows))
        print(df.head(n_rows).to_string())
        
        # Show data types
        print(f"\nData Types:")
        print(df.dtypes.to_string())
        
        # Show basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\nBasic Statistics (numeric columns):")
            print(df[numeric_cols].describe().to_string())
        
        # Show missing values summary
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\nMissing Values:")
            print(missing_data[missing_data > 0].to_string())
            
    except Exception as e:
        print(f"Error processing CSV file: {e}")


def preview_pickle_file(filepath, filename):
    """Preview pickle file contents"""
    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print(f"{'='*60}")
    
    try:
        # Pandas compatibility hack for older pickle files  
        # Older pandas pickles choke on new versions due to module renames
        import sys
        if 'pandas.core.indexes.numeric' not in sys.modules:
            sys.modules['pandas.core.indexes.numeric'] = pd.core.indexes.base  # Fakes the old path
            
        data = pd.read_pickle(filepath)
        
        print(f"Type: {type(data)}")
        
        if hasattr(data, 'shape'):
            print(f"Shape: {data.shape}")
        
        if hasattr(data, 'head'):
            print("\nFirst 5 rows:")
            try:
                print(data.head().to_string())
            except Exception as e:
                print(f"Error displaying head: {e}")
                print("Raw data preview:")
                print(str(data)[:500])
        elif hasattr(data, 'columns') and hasattr(data, 'index'):
            # Looks like a DataFrame but head() failed
            print(f"Columns: {list(data.columns) if hasattr(data.columns, '__iter__') else 'Cannot display columns'}")
            print(f"Index: {len(data.index) if hasattr(data.index, '__len__') else 'Cannot display index length'}")
            print("Raw data preview:")
            print(str(data)[:500])
        elif isinstance(data, (list, tuple)):
            print(f"Length: {len(data)}")
            print("First few items:")
            for i, item in enumerate(data[:5]):
                print(f"  [{i}]: {str(item)[:100]}...")
        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            print("First few key-value pairs:")
            for i, (k, v) in enumerate(list(data.items())[:5]):
                print(f"  {k}: {str(v)[:100]}...")
        else:
            print(f"Content preview: {str(data)[:500]}...")
            
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        print("This might be due to version compatibility issues with pandas or other dependencies.")


def preview_json_file(filepath, filename, n_items=5):
    """Preview JSON file contents"""
    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            print("First few key-value pairs:")
            for i, (k, v) in enumerate(list(data.items())[:n_items]):
                print(f"  {k}: {str(v)[:100]}...")
        elif isinstance(data, list):
            print(f"Length: {len(data)}")
            print("First few items:")
            for i, item in enumerate(data[:n_items]):
                print(f"  [{i}]: {str(item)[:100]}...")
        else:
            print(f"Content preview: {str(data)[:500]}...")
            
    except Exception as e:
        print(f"Error reading JSON file: {e}")


def preview_text_file(filepath, filename, n_lines=10):
    """Preview text file contents"""
    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print(f"{'='*60}")
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            if encoding != 'utf-8':
                print(f"Note: File read with {encoding} encoding")
            
            print(f"Total lines: {len(lines)}")
            print(f"\nFirst {min(n_lines, len(lines))} lines:")
            for i, line in enumerate(lines[:n_lines]):
                print(f"  {i+1}: {line.rstrip()}")
            
            break
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading text file with {encoding}: {e}")
            continue
    else:
        print("Error: Could not read text file with any supported encoding")


def main():
    """Main function to preview all data files"""
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    
    print("POLYMER GNN DATA PREVIEW")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return
    
    # Get all files in the directory
    files = list(data_dir.glob('*'))
    files = [f for f in files if f.is_file()]
    
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {f.name}")
    
    # Process each file
    for filepath in sorted(files):
        filename = filepath.name
        
        # Skip hidden files and .gitkeep
        if filename.startswith('.'):
            continue
            
        if filename.endswith('.csv'):
            preview_csv_file(filepath, filename)
        elif filename.endswith(('.pkl', '.pickle')):
            preview_pickle_file(filepath, filename)
        elif filename.endswith(('.json', '.jsonl')):
            preview_json_file(filepath, filename)
        elif filename.endswith(('.txt', '.text')):
            preview_text_file(filepath, filename)
        else:
            print(f"\n{'='*60}")
            print(f"FILE: {filename}")
            print(f"{'='*60}")
            print(f"File type not supported for preview: {filepath.suffix}")
            print(f"File size: {filepath.stat().st_size} bytes")
    
    print(f"\n{'='*60}")
    print("PREVIEW COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()