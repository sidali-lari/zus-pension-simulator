# ZUS Pension Simulator - Deployment Guide

## 🚀 Fly.io Deployment

### Prerequisites
1. Install Fly.io CLI: `curl -L https://fly.io/install.sh | sh`
2. Login to Fly.io: `flyctl auth login`
3. Ensure you have a Fly.io account

### Quick Deployment
```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy to Fly.io
./deploy.sh
```

### Manual Deployment Steps

1. **Initialize Fly.io app** (if not already done):
   ```bash
   flyctl launch
   ```

2. **Deploy the application**:
   ```bash
   flyctl deploy
   ```

3. **Check deployment status**:
   ```bash
   flyctl status
   ```

4. **View logs**:
   ```bash
   flyctl logs
   ```

### Configuration Files Created

- `fly.toml` - Fly.io configuration
- `Dockerfile` - Docker configuration with OpenMP support
- `.dockerignore` - Docker build optimization
- `deploy.sh` - Automated deployment script

### Environment Variables

The app uses these environment variables:
- `PORT` - Server port (default: 8080)
- `ZUS_CSV_PATH` - Path to CSV data file
- `BASE_AVG_WAGE_TODAY` - Base average wage in PLN

### Production Features

- ✅ OpenMP support for XGBoost
- ✅ Optimized Docker build
- ✅ Health check endpoint
- ✅ CORS enabled
- ✅ Auto-scaling configuration

### API Endpoints

- `GET /health` - Health check
- `POST /forecast` - Pension forecast
- `POST /forecast/report` - Excel report download
- `GET /model/metrics` - Model performance metrics
- `GET /admin/usage-report` - Usage analytics

### Troubleshooting

1. **OpenMP Issues**: The Dockerfile includes `libomp-dev` installation
2. **Port Conflicts**: App uses port 8080 in production
3. **Memory**: Configured for 1GB RAM
4. **CPU**: Shared CPU configuration

### Local Testing

```bash
# Run locally on port 8001
PORT=8001 python3 app.py

# Test endpoints
curl http://localhost:8001/health
curl -X POST http://localhost:8001/forecast -H "Content-Type: application/json" -d '{"age": 40, "sex": "M", "gross_salary_now": 10000, "start_year": 2010}'
```
