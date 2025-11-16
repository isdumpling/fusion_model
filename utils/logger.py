import os
import sys
import csv
from datetime import datetime

class logger:
    """
    Simple logger class for training logs
    """
    def __init__(self, args):
        self.args = args
        self.log_file = None
        self.csv_file = None
        self.csv_writer = None
        self.csv_initialized = False
        
        # Create output directory if it doesn't exist
        if hasattr(args, 'out') and args.out:
            os.makedirs(args.out, exist_ok=True)
            log_path = os.path.join(args.out, 'training_log.txt')
            self.log_file = open(log_path, 'w')
            print(f"Logging to: {log_path}")
            
            # Create CSV log file
            csv_path = os.path.join(args.out, 'training_metrics.csv')
            self.csv_file = open(csv_path, 'w', newline='')
            print(f"CSV logging to: {csv_path}")
    
    def __call__(self, message, level=1):
        """
        Log a message with specified indentation level
        
        Args:
            message: Message to log
            level: Indentation level (1, 2, 3, etc.)
        """
        indent = "  " * (level - 1)
        formatted_message = f"{indent}{message}"
        
        # Print to console
        print(formatted_message)
        
        # Write to file
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"[{timestamp}] {formatted_message}\n")
            self.log_file.flush()
    
    def log_metrics_csv(self, metrics_dict):
        """
        Log metrics to CSV file
        
        Args:
            metrics_dict: Dictionary containing metrics to log
        """
        if not self.csv_file:
            return
        
        # Initialize CSV writer with header on first call
        if not self.csv_initialized:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=metrics_dict.keys())
            self.csv_writer.writeheader()
            self.csv_initialized = True
        
        # Write metrics row
        self.csv_writer.writerow(metrics_dict)
        self.csv_file.flush()
    
    def __del__(self):
        """Close log files when object is destroyed"""
        if self.log_file:
            self.log_file.close()
        if self.csv_file:
            self.csv_file.close()