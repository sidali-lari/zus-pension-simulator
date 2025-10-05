#!/bin/bash

echo "🔧 Fixing production 502 error..."

# Check if we're logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "❌ Not logged in to Fly.io. Please run: flyctl auth login"
    exit 1
fi

echo "✅ Logged in to Fly.io"

# Deploy the fixed version
echo "🚀 Deploying fixed version..."
flyctl deploy --app zus-pension-simulator

echo "✅ Deployment complete!"
echo "🔍 Testing the fix..."

# Wait a moment for the app to start
sleep 10

# Test the health endpoint
echo "Testing health endpoint..."
curl -s https://zus-pension-simulator.fly.dev/health

echo ""
echo "🎉 Production should now be working!"
echo "📊 Check your app at: https://zus-pension-simulator.fly.dev"
