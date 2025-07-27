#!/usr/bin/env python3
"""
Background HPO runner for PolyGNN - runs in chunks to avoid timeouts
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

def run_hpo_chunk(n_trials: int = 10):
    """Run a small chunk of HPO trials"""
    print(f"🚀 Starting HPO chunk with {n_trials} trials...")
    
    # Run the HPO with reduced trials
    cmd = ["python", "run_hpo_simple.py", "--n_trials", str(n_trials)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print(f"✅ HPO chunk completed successfully!")
            print("📊 Results summary:")
            # Extract results from output
            lines = result.stdout.split('\n')
            for line in lines[-20:]:  # Last 20 lines for summary
                if any(keyword in line for keyword in ['Best', 'R²', 'Target', 'Time']):
                    print(f"   {line}")
            return True
        else:
            print(f"❌ HPO chunk failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ HPO chunk timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ HPO chunk failed with exception: {e}")
        return False

def main():
    """Run multiple HPO chunks"""
    total_trials = 50
    chunk_size = 10
    completed_trials = 0
    
    print("🎯 Starting Background HPO Optimization")
    print(f"📊 Target: {total_trials} total trials in chunks of {chunk_size}")
    print("="*60)
    
    while completed_trials < total_trials:
        remaining = total_trials - completed_trials
        current_chunk = min(chunk_size, remaining)
        
        print(f"\n🔄 Running chunk {completed_trials//chunk_size + 1}")
        print(f"📈 Progress: {completed_trials}/{total_trials} trials completed")
        
        success = run_hpo_chunk(current_chunk)
        
        if success:
            completed_trials += current_chunk
            print(f"✅ Chunk completed. Total progress: {completed_trials}/{total_trials}")
        else:
            print(f"❌ Chunk failed. Continuing with next chunk...")
            completed_trials += current_chunk  # Skip failed chunk
        
        # Brief pause between chunks
        if completed_trials < total_trials:
            print("⏸️  Pausing 30 seconds before next chunk...")
            time.sleep(30)
    
    print("\n" + "="*60)
    print("🎉 Background HPO Optimization Complete!")
    print(f"📊 Completed {completed_trials} trials total")
    print("="*60)

if __name__ == "__main__":
    main()