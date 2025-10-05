#!/bin/bash

# Setup script for local development and testing
echo "🚀 Setting up ZUS Pension Simulator for local development..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Set up environment variables for local testing
echo "🔑 Setting up environment variables..."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Local development environment variables
PORT=8000
ZUS_CSV_PATH=data/processed/zus_forecast_all.csv
BASE_AVG_WAGE_TODAY=8000

EOF
    echo "✅ Created .env file with Supabase credentials"
else
    echo "✅ .env file already exists"
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

echo "🧪 Testing Supabase connection..."
python3 test_supabase.py

echo ""
echo "🎉 Setup complete! You can now:"
echo "1. Run the app locally: python3 app.py"
echo "2. Test the API: python3 test_local_api.py"
echo "3. Check Supabase data in your dashboard"
