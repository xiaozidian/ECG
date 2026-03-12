import os
import sys
from glob import glob

# Add parent directory to sys.path to import clinical_processor
# import clinical_processor
import clinical_processor
from generate_clinical_report import generate_report

def main():
    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    real_data_root = os.path.join(base_dir, "real_data")
    report_root = os.path.join(base_dir, "real_report")
    
    if not os.path.exists(report_root):
        os.makedirs(report_root)
        
    # Find all .DATA files
    # Pattern: real_data/*/data/*.DATA
    data_files = glob(os.path.join(real_data_root, '*', 'data', '*.DATA'))
    
    print(f"Found {len(data_files)} records to process.")
    
    for file_path in data_files:
        record_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n{'='*50}")
        print(f"Start processing: {record_name}")
        
        # Create output directory: real_report/{record_name}/
        output_dir = os.path.join(report_root, record_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. Process data and generate predictions.csv
        try:
            csv_path = clinical_processor.process_clinical_record(file_path, output_dir=output_dir)
            if not csv_path:
                print(f"Skipping report generation for {record_name} due to processing error.")
                continue
        except Exception as e:
            print(f"Error processing {record_name}: {e}")
            continue
            
        # 2. Generate Report
        try:
            # Mock patient info (can be improved by reading XML if available)
            p_info = {'name': 'Unknown', 'sex': 'Unknown', 'age': 'Unknown'}
            
            generate_report(
                record_name, 
                input_dir=output_dir, 
                output_dir=output_dir, 
                patient_info=p_info
            )
        except Exception as e:
            print(f"Error generating report for {record_name}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\n{'='*50}")
    print("Batch processing completed.")

if __name__ == "__main__":
    main()
