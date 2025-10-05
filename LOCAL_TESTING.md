# Local Testing Guide for ZUS Pension Simulator

This guide will help you test the ZUS Pension Simulator with Supabase integration locally.

## Prerequisites

- Python 3.8 or higher
- Your Supabase project set up (see `SUPABASE_SETUP.md`)
- The database table created (run `supabase_schema.sql` in your Supabase project)

## Quick Start

### 1. Automated Setup

Run the setup script to prepare your local environment:

```bash
./setup_local_env.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up environment variables
- Test Supabase connection

### 2. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Supabase credentials
```

### 3. Start the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Start the Flask app
python3 app.py
```

The API will be available at `http://localhost:8000`

### 4. Run Tests

In a new terminal window:

```bash
# Activate virtual environment
source venv/bin/activate

# Run comprehensive API tests
python3 test_local_api.py

# Or test just Supabase connection
python3 test_supabase.py
```

## Testing Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Model Metrics
```bash
curl http://localhost:8000/model/metrics
```

### Forecast (with Supabase saving)
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "age": 40,
    "sex": "M",
    "gross_salary_now": 10000,
    "start_year": 2025,
    "include_sickleave": true,
    "desired_pension": 5000,
    "postal_code": "00-001"
  }'
```

### Forecast Report (Excel download)
```bash
curl -X POST http://localhost:8000/forecast/report \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "sex": "F",
    "gross_salary_now": 8000,
    "start_year": 2020
  }' \
  --output report.xlsx
```

## What to Expect

### Successful API Response
```json
{
  "inputs": {
    "age": 40,
    "sex": "M",
    "gross_salary_now": 10000,
    "start_year": 2025,
    "include_sickleave": true,
    "desired_pension": 5000,
    "postal_code": "00-001"
  },
  "result": {
    "retirement_year": 2050,
    "pension_first_year_nominal": 4500.00,
    "pension_first_year_real_today": 3800.00,
    "replacement_rate_percent": 45.0,
    // ... more results
  },
  "admin_usage_saved_to": "zus_usage_report.xlsx",
  "supabase_saved": true
}
```

### Supabase Data
Check your Supabase dashboard to see the saved records in the `forecast` table with columns:
- `data_symulacji` (simulation date)
- `wiek` (age)
- `plec` (sex)
- `wynagrodzenie` (salary)
- `kod_pocztowy` (postal code)
- `pozadana_emerytura` (desired pension)
- `zgromadzone_srodki` (accumulated funds)
- `l4_wliczone` (sick leave included)
- `prognozowana_emerytura` (forecasted pension)
- `realna_emerytura` (real pension)

## Troubleshooting

### API Not Starting
- Check if port 8000 is available
- Ensure all dependencies are installed
- Check for missing data files

### Supabase Connection Issues
- Verify your `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- Check if your Supabase project is active
- Ensure the `forecast` table exists in your database

### Test Failures
- Make sure the API is running before running tests
- Check your internet connection for Supabase access
- Verify your Supabase credentials are correct

## Development Tips

1. **Hot Reload**: The Flask app will reload automatically when you change code
2. **Logs**: Check the console output for detailed error messages
3. **Database**: Use your Supabase dashboard to inspect saved data
4. **Testing**: Run tests frequently to catch issues early

## Next Steps

Once local testing is successful:
1. Deploy to Fly.io with your Supabase credentials
2. Test the production API
3. Integrate with your frontend application
