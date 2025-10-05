#!/usr/bin/env python3
"""
Quick test to verify Supabase connection and setup
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_supabase_setup():
    print("🔍 Testing Supabase setup...")
    
    # Check environment variables
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    print(f"SUPABASE_URL: {url}")
    print(f"SUPABASE_KEY: {key[:20]}..." if key else "Not set")
    
    if not url or not key:
        print("❌ Missing Supabase credentials")
        return False
    
    try:
        from supabase import create_client, Client
        supabase: Client = create_client(url, key)
        
        # Try to access the database
        result = supabase.table('forecast').select('*').limit(1).execute()
        print("✅ Successfully connected to Supabase!")
        print(f"   Found {len(result.data)} records in forecast table")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        print("\n🔧 To fix this:")
        print("1. Go to your Supabase project dashboard")
        print("2. Check if the project is active")
        print("3. Get the correct URL and API key from Settings > API")
        print("4. Run the SQL schema from supabase_schema.sql")
        return False

if __name__ == "__main__":
    test_supabase_setup()
