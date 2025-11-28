import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    """
    Converts a CSV file to a JSON file (list of dictionaries).

    Assumes the first row of the CSV contains the headers.

    Args:
        csv_file_path (str): The path to the CSV file.
        json_file_path (str): The path to save the JSON file.
    """
    data = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8', newline='') as csvfile:
             # Use DictReader to read rows as dictionaries
            csv_reader = csv.DictReader(csvfile)
            # Convert each row (OrderedDict) to a standard dict and add to the list
            for row in csv_reader:
                data.append(dict(row)) # Changed from data.extend(row)

        with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
            # Dump the list of dictionaries to JSON
            json.dump(data, jsonfile, ensure_ascii=False, indent=4)
        print(f"Successfully converted '{csv_file_path}' to '{json_file_path}'")

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage with the active file path
# Make sure this CSV file exists in the same directory or provide the full path
csv_file_path = "global_cancer_patients_2015_2024.csv"
json_file_path = "global_cancer_patients_2015_2024.json"

csv_to_json(csv_file_path, json_file_path)
