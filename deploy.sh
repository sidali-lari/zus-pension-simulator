#!/bin/bash

# ZUS Pension Simulator Deployment Script for Fly.io

echo "🚀 Deploying ZUS Pension Simulator to Fly.io..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first:"
    echo "   curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "🔐 Please log in to Fly.io first:"
    echo "   flyctl auth login"
    exit 1
fi

# Deploy the application
echo "📦 Building and deploying..."
flyctl deploy

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at: https://zus-pension-simulator.fly.dev"
