#!/usr/bin/env python3
"""
Test script for Supabase integration
Run this to verify that the Supabase connection and table operations work correctly
"""

import os
import json
from supabase import create_client, Client

def test_supabase_connection():
    """Test Supabase connection and table operations"""
    
    # Get credentials from environment
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ SUPABASE_URL and SUPABASE_KEY environment variables not set")
        return False
    
    try:
        # Initialize client
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Supabase client created successfully")
        
        # Test connection by reading from the forecast table
        result = supabase.table('forecast').select('*').limit(1).execute()
        print("✅ Successfully connected to Supabase and accessed 'forecast' table")
        
        # Test inserting a sample record
        test_data = {
            'data_symulacji': '01.01.2025, 12:00',
            'wiek': 30,
            'plec': 'M',
            'wynagrodzenie': 5000.00,
            'kod_pocztowy': '00-001',
            'pozadana_emerytura': 3000.00,
            'zgromadzone_srodki': 100000.00,
            'l4_wliczone': 'Nie',
            'prognozowana_emerytura': 2500.00,
            'realna_emerytura': 2200.00
        }
        
        insert_result = supabase.table('forecast').insert(test_data).execute()
        print("✅ Successfully inserted test record")
        print(f"   Inserted record ID: {insert_result.data[0]['id'] if insert_result.data else 'Unknown'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Supabase connection: {e}")
        return False

if __name__ == "__main__":
    print("Testing Supabase integration...")
    print("=" * 50)
    
    success = test_supabase_connection()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! Supabase integration is working correctly.")
    else:
        print("💥 Tests failed. Please check your Supabase configuration.")
        print("\nTo fix this:")
        print("1. Set SUPABASE_URL and SUPABASE_KEY environment variables")
        print("2. Run the SQL schema from supabase_schema.sql in your Supabase project")
        print("3. Ensure your Supabase project is active and accessible")
        
