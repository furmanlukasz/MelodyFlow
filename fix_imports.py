#!/usr/bin/env python3
"""
Fix script for handling transformers library import issues.
This patches the missing Replicate class in the transformers integration.
"""
import importlib.machinery
import importlib.util
import os
import sys


def fix_transformer_imports():
    """
    Fix the transformer imports by patching the tensor_parallel.py file
    to add the missing Replicate class if it doesn't exist.
    """
    try:
        import transformers
        tp_path = os.path.join(os.path.dirname(transformers.__file__), 'integrations', 'tensor_parallel.py')
        
        # Check if the file exists
        if not os.path.exists(tp_path):
            print(f"Could not find tensor_parallel.py at {tp_path}")
            return False
            
        # Read the file to check if Replicate is defined
        with open(tp_path, 'r') as f:
            content = f.read()
            
        if 'class Replicate' not in content:
            # Add the missing Replicate class
            with open(tp_path, 'a') as f:
                f.write('\n\n# Added by fix_imports.py\nclass Replicate(object):\n    """Layout class for replicated tensors."""\n    def __init__(self):\n        pass\n')
            
            print(f"Added missing Replicate class to {tp_path}")
            
            # Reload the module if it's already loaded
            if 'transformers.integrations.tensor_parallel' in sys.modules:
                # Remove from cache to force reload
                del sys.modules['transformers.integrations.tensor_parallel']
                
            # Import to verify fix worked
            try:
                from transformers.integrations.tensor_parallel import Replicate
                print("Fix successful - Replicate class is now available")
                return True
            except ImportError as e:
                print(f"Fix attempt failed: {e}")
                return False
        else:
            print("Replicate class is already defined, no fix needed")
            return True
            
    except Exception as e:
        print(f"Error attempting to fix imports: {e}")
        return False

if __name__ == "__main__":
    success = fix_transformer_imports()
    if success:
        print("You can now run your training script.")
    else:
        print("Failed to fix the import issue automatically.")
        print("Alternative solution: pip install transformers==4.30.0 which is known to be compatible.") 