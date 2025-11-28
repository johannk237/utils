#!/usr/bin/env python3
"""
PDF File Lister
Scans a directory for PDF files and exports their names to a text file.
"""

import glob
import os
from pathlib import Path
from typing import List


def list_pdf_files(directory: str = ".", output_file: str = "pdf_list.txt", delimiter: str = ";") -> List[str]:
    """
    List all PDF files in a directory and save to a text file.
    
    Args:
        directory: Path to the directory to scan (default: current directory)
        output_file: Name of the output text file (default: pdf_list.txt)
        delimiter: Character to separate filenames (default: semicolon)
    
    Returns:
        List of PDF filenames found in the directory
    
    Raises:
        OSError: If the directory doesn't exist or isn't accessible
    """
    # Validate directory
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        raise OSError(f"Directory does not exist: {directory}")
    if not dir_path.is_dir():
        raise OSError(f"Path is not a directory: {directory}")
    
    # Find all PDF files
    pdf_pattern = str(dir_path / "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    # Extract filenames only
    pdf_filenames = [os.path.basename(f) for f in pdf_files]
    
    # Sort for consistent output
    pdf_filenames.sort()
    
    # Create delimited string
    file_list_string = delimiter.join(pdf_filenames)
    
    # Write to output file
    output_path = dir_path / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(file_list_string)
    
    print(f"Found {len(pdf_filenames)} PDF file(s)")
    print(f"List saved to: {output_path}")
    
    return pdf_filenames


def main() -> None:
    """Main entry point for the script."""
    try:
        pdf_list = list_pdf_files()
        if pdf_list:
            print(f"\nPDF files: {', '.join(pdf_list)}")
        else:
            print("No PDF files found in the current directory.")
    except OSError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()