# Utils

A curated collection of production-ready Python utilities for PDF processing, file handling, and automation tasks.

## 📑 Table of Contents

- [Overview](#overview)
- [Available Utilities](#available-utilities)
- [Installation](#installation)
- [License](#license)

## Overview

This repository contains a collection of specialized Python scripts designed to streamline common automation tasks, with a focus on PDF manipulation and file processing. Each utility is self-contained and includes its own dependencies.

## Available Utilities

### 🔗 Merge PDF Files

**Location:** `merge_pdf_files/`

An industrial-strength PDF processing pipeline that repairs, compresses, and merges PDF files with intelligent size management.

**Features:**
- ✅ Automatic PDF repair using `pikepdf`
- 🗜️ Multi-stage compression with `qpdf` and optional Ghostscript fallback
- 📦 Smart chunking: splits output into multiple files with configurable size limits (default: 200MB)
- ⚡ Parallel processing for fast batch operations
- 🛡️ Robust error handling and detailed logging
- 🎯 Preserves original file order during merging

**Usage:**
```bash
cd merge_pdf_files
pip install -r requirements.txt

# Basic usage (processes PDFs in current directory)
python script.py

# Advanced usage with custom settings
python script.py --workers 6 --max-size 200 --keep-temp --ghostscript
```

**Options:**
- `--folder, -d`: Directory containing PDF files (default: current directory)
- `--workers, -w`: Number of parallel workers (default: CPU count - 1)
- `--max-size`: Maximum output file size in MB (default: 200)
- `--keep-temp`: Keep temporary directory for debugging
- `--final-prefix`: Prefix for merged output files (default: "MERGED")
- `--ghostscript`: Enable Ghostscript fallback for image-heavy PDFs

**Requirements:**
- Python 3.6+
- `pikepdf`
- `qpdf` (CLI tool)
- `ghostscript` (optional, for fallback compression)

---

### 📋 List PDF Files in Folder

**Location:** `list_pdf_file_in_a_folder/`

A simple utility that scans a directory for PDF files and exports their names to a semicolon-delimited text file.

**Features:**
- 📝 Generates a semicolon-separated list of PDF filenames
- 💾 Outputs to `pdf_list.txt` in the current directory
- 🔍 Scans current working directory automatically

**Usage:**
```bash
cd list_pdf_file_in_a_folder
pip install -r requirements.txt
python script.py
```

**Output:**
Creates a `pdf_list.txt` file containing:
```
file1.pdf;file2.pdf;file3.pdf
```

## Installation

Each utility is self-contained with its own `requirements.txt` file. Navigate to the specific utility directory and install dependencies:

```bash
cd <utility_name>
pip install -r requirements.txt
```

For global installation of all utilities:

```bash
# Install dependencies for all utilities
for dir in */; do
    if [ -f "$dir/requirements.txt" ]; then
        pip install -r "$dir/requirements.txt"
    fi
done
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author:** Johann Kengne  
**Year:** 2025
