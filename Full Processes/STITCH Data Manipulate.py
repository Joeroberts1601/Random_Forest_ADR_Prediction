import time
import csv
import subprocess

# List of Python files to execute
python_files = ["STITCH_Identifiers/Filter to get all matched drugs IDs.py", "STITCH_Identifiers/Get Drug Protein interaction matrix.py"]

# CSV file to store results
output_csv = "Timings/STITCH Manipulation Execution Times.csv"

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Script Name", "Execution Time (seconds)"])

    # Loop through each Python file
    for py_file in python_files:
        start_time = time.time()  # Start timing
        try:
            # Run the Python file
            subprocess.run(["python", py_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {py_file}: {e}")
        
        end_time = time.time()  # End timing

        # Calculate execution time
        execution_time = end_time - start_time

        # Write to CSV
        writer.writerow([py_file, execution_time])
        print(f"{py_file} executed in {execution_time:.2f} seconds")
