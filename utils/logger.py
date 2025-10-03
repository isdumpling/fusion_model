import os
import sys
from datetime import datetime

class logger:
    """
    Simple logger class for training logs
    """
    def __init__(self, args):
        self.args = args
        self.log_file = None
        
        # Create output directory if it doesn't exist
        if hasattr(args, 'out') and args.out:
            os.makedirs(args.out, exist_ok=True)
            log_path = os.path.join(args.out, 'training_log.txt')
            self.log_file = open(log_path, 'w')
            print(f"Logging to: {log_path}")
    
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
    
    def __del__(self):
        """Close log file when object is destroyed"""
        if self.log_file:
            self.log_file.close()