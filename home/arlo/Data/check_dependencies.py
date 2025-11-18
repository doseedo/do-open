#!/usr/bin/env python3
"""
Check for missing dependencies in genfrominterface.py
Run this before deploying to catch missing scripts/modules
"""
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_dependencies():
    missing = []
    
    # Read genfrominterface.py
    with open(os.path.join(SCRIPT_DIR, 'genfrominterface.py'), 'r') as f:
        content = f.read()
    
    # Find local module imports
    local_modules = ['dataloader', 'output_paths', 'conditioning_encoder', 'trainer_performer', 
                     'trainer_performerCN2', 'trainer_performer_backup']
    
    for module in local_modules:
        module_file = os.path.join(SCRIPT_DIR, f"{module}.py")
        if not os.path.exists(module_file):
            missing.append(f"Module: {module}.py")
    
    # Find Python script calls
    script_patterns = [
        r'python3?\s+([a-z_][a-z0-9_]*\.py)',
        r'"python",\s*"([a-z_][a-z0-9_]*\.py)"',
    ]
    
    scripts = set()
    for pattern in script_patterns:
        scripts.update(re.findall(pattern, content))
    
    for script in scripts:
        script_file = os.path.join(SCRIPT_DIR, script)
        if not os.path.exists(script_file):
            missing.append(f"Script: {script}")
    
    if missing:
        print("❌ Missing dependencies:")
        for item in missing:
            print(f"   {item}")
        return False
    else:
        print("✅ All dependencies found!")
        return True

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)
