#!/usr/bin/env python3
"""
Test script to verify MLflow connection and data availability.
Run this script to check if the web interface can connect to MLflow.
"""

import os
import sys
import requests
import mlflow
import mlflow.tracking
from datetime import datetime

def test_mlflow_connection():
    """Test MLflow connection and data availability."""
    print("🔍 Testing MLflow Connection for QEPO Web Interface")
    print("=" * 60)
    
    # Test 1: Check MLflow server availability
    print("1. Testing MLflow server availability...")
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("   ✅ MLflow server is running on localhost:5000")
        else:
            print(f"   ❌ MLflow server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Cannot connect to MLflow server: {e}")
        print("   💡 Start MLflow server with: python start_mlflow.py")
        return False
    
    # Test 2: Check MLflow Python client
    print("\n2. Testing MLflow Python client...")
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.tracking.MlflowClient()
        print("   ✅ MLflow Python client connected successfully")
    except Exception as e:
        print(f"   ❌ MLflow Python client error: {e}")
        return False
    
    # Test 3: Check for experiments
    print("\n3. Checking for experiments...")
    try:
        experiments = mlflow.search_experiments()
        print(f"   ✅ Found {len(experiments)} experiments")
        
        if len(experiments) == 0:
            print("   ⚠️  No experiments found. Create some runs to see data in the web interface.")
            return True
        
        # Test 4: Check for runs
        print("\n4. Checking for runs...")
        total_runs = 0
        for exp in experiments:
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=10
            )
            total_runs += len(runs)
            print(f"   📊 Experiment '{exp.name}': {len(runs)} runs")
        
        if total_runs == 0:
            print("   ⚠️  No runs found in any experiment.")
            print("   💡 Run some quantum portfolio commands to create data:")
            print("      cd ../quantum-portfolio")
            print("      python -m qepo.cli data ingest")
            print("      python -m qepo.cli optimize")
            return True
        else:
            print(f"   ✅ Found {total_runs} total runs across all experiments")
        
        # Test 5: Check run data structure
        print("\n5. Testing run data structure...")
        try:
            # Get the first run
            first_run = None
            for exp in experiments:
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1
                )
                if len(runs) > 0:
                    first_run = runs.iloc[0]
                    break
            
            if first_run is not None:
                run_id = first_run['run_id']
                run_info = mlflow.get_run(run_id)
                
                print(f"   📋 Sample run: {run_id[:8]}...")
                print(f"   📅 Start time: {first_run['start_time']}")
                print(f"   📊 Status: {first_run['status']}")
                
                # Check metrics
                if hasattr(run_info, 'data') and run_info.data.metrics:
                    print(f"   📈 Metrics: {len(run_info.data.metrics)} found")
                    for key, value in list(run_info.data.metrics.items())[:3]:
                        print(f"      - {key}: {value}")
                else:
                    print("   ⚠️  No metrics found in this run")
                
                # Check parameters
                if hasattr(run_info, 'data') and run_info.data.params:
                    print(f"   ⚙️  Parameters: {len(run_info.data.params)} found")
                    for key, value in list(run_info.data.params.items())[:3]:
                        print(f"      - {key}: {value}")
                else:
                    print("   ⚠️  No parameters found in this run")
                
                print("   ✅ Run data structure looks good")
            else:
                print("   ⚠️  No runs found to test data structure")
        
        except Exception as e:
            print(f"   ❌ Error testing run data: {e}")
            return False
        
        print("\n🎉 All tests passed! The web interface should show real data.")
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking experiments: {e}")
        return False

def main():
    """Main test function."""
    success = test_mlflow_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ MLflow connection test completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Start the web interface backend: python server.py")
        print("   2. Start the frontend: npm run dev")
        print("   3. Go to Reporting & Analytics tab to see your runs")
    else:
        print("❌ MLflow connection test failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure MLflow server is running: python start_mlflow.py")
        print("   2. Check if port 5000 is available")
        print("   3. Verify MLflow installation: pip install mlflow>=2.0.0")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
