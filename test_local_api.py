#!/usr/bin/env python3
"""
Local API testing script for ZUS Pension Simulator
Tests the full API functionality including Supabase integration
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_model_metrics():
    """Test the model metrics endpoint"""
    print("🔍 Testing model metrics endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/model/metrics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model metrics retrieved: {len(data)} keys")
            return True
        else:
            print(f"❌ Model metrics failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model metrics failed: {e}")
        return False

def test_forecast_endpoint():
    """Test the forecast endpoint with sample data"""
    print("🔍 Testing forecast endpoint...")
    
    # Sample forecast data
    test_payload = {
        "age": 40,
        "sex": "M",
        "gross_salary_now": 10000,
        "start_year": 2025,
        "include_sickleave": True,
        "desired_pension": 5000,
        "postal_code": "00-001"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/forecast",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Forecast request successful!")
            print(f"   Retirement year: {data['result']['retirement_year']}")
            print(f"   Forecasted pension: {data['result']['pension_first_year_nominal']} PLN")
            print(f"   Supabase saved: {data.get('supabase_saved', 'Not specified')}")
            return True
        else:
            print(f"❌ Forecast request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Forecast request failed: {e}")
        return False

def test_forecast_report_endpoint():
    """Test the forecast report endpoint"""
    print("🔍 Testing forecast report endpoint...")
    
    test_payload = {
        "age": 35,
        "sex": "F",
        "gross_salary_now": 8000,
        "start_year": 2020,
        "include_sickleave": False,
        "desired_pension": 4000
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/forecast/report",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Forecast report request successful!")
            print(f"   Content-Type: {response.headers.get('Content-Type')}")
            print(f"   File size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Forecast report request failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Forecast report request failed: {e}")
        return False

def test_multiple_forecasts():
    """Test multiple forecast requests to verify Supabase integration"""
    print("🔍 Testing multiple forecasts for Supabase integration...")
    
    test_cases = [
        {
            "age": 30,
            "sex": "M",
            "gross_salary_now": 6000,
            "start_year": 2020,
            "include_sickleave": True,
            "postal_code": "30-001"
        },
        {
            "age": 45,
            "sex": "F", 
            "gross_salary_now": 12000,
            "start_year": 2010,
            "include_sickleave": False,
            "desired_pension": 6000,
            "postal_code": "50-001"
        },
        {
            "age": 55,
            "sex": "M",
            "gross_salary_now": 15000,
            "start_year": 1995,
            "include_sickleave": True,
            "accumulated_now": 200000,
            "postal_code": "80-001"
        }
    ]
    
    success_count = 0
    for i, payload in enumerate(test_cases, 1):
        print(f"   Test case {i}/3: Age {payload['age']}, Sex {payload['sex']}")
        try:
            response = requests.post(
                f"{API_BASE_URL}/forecast",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                supabase_saved = data.get('supabase_saved', False)
                print(f"   ✅ Success - Supabase saved: {supabase_saved}")
                success_count += 1
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"📊 Multiple forecasts test: {success_count}/{len(test_cases)} successful")
    return success_count == len(test_cases)

def check_supabase_data():
    """Check if data was actually saved to Supabase"""
    print("🔍 Checking Supabase data...")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Supabase credentials not configured")
        return False
    
    try:
        from supabase import create_client, Client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get recent records
        result = supabase.table('forecast').select('*').order('created_at', desc=True).limit(5).execute()
        
        if result.data:
            print(f"✅ Found {len(result.data)} recent records in Supabase")
            for record in result.data[:3]:  # Show first 3 records
                print(f"   - Age: {record['wiek']}, Sex: {record['plec']}, Pension: {record['prognozowana_emerytura']}")
            return True
        else:
            print("⚠️  No records found in Supabase")
            return False
            
    except Exception as e:
        print(f"❌ Error checking Supabase data: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Starting ZUS Pension Simulator Local API Tests")
    print("=" * 60)
    
    # Check if API is running
    print("🔍 Checking if API is running...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running. Please start it with: python3 app.py")
            return
    except requests.exceptions.RequestException:
        print("❌ API is not running. Please start it with: python3 app.py")
        return
    
    print("✅ API is running, starting tests...")
    print("")
    
    # Run tests
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Model Metrics", test_model_metrics),
        ("Forecast Endpoint", test_forecast_endpoint),
        ("Forecast Report", test_forecast_report_endpoint),
        ("Multiple Forecasts", test_multiple_forecasts),
        ("Supabase Data Check", check_supabase_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API and Supabase integration are working correctly.")
    else:
        print("💥 Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
