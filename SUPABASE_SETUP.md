# Supabase Setup for ZUS Pension Simulator

This document explains how to set up Supabase integration for the ZUS Pension Simulator to save forecast data to a database.

## Prerequisites

1. A Supabase account and project
2. Access to your Supabase project's URL and API key

## Setup Steps

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Create a new project
3. Note down your project URL and API key from the project settings

### 2. Create the Database Table

Run the SQL script in `supabase_schema.sql` in your Supabase SQL editor:

```sql
-- Copy and paste the contents of supabase_schema.sql
```

This will create the `forecast` table with the following columns:
- `id` (auto-incrementing primary key)
- `data_symulacji` (simulation date/time)
- `wiek` (age)
- `plec` (sex: M/F)
- `wynagrodzenie` (salary)
- `kod_pocztowy` (postal code)
- `pozadana_emerytura` (desired pension)
- `zgromadzone_srodki` (accumulated funds)
- `l4_wliczone` (sick leave included: Tak/Nie)
- `prognozowana_emerytura` (forecasted pension)
- `realna_emerytura` (real pension)
- `created_at` (timestamp)
- `updated_at` (timestamp)

### 3. Configure Environment Variables

Update your `fly.toml` file with your Supabase credentials:

```toml
[env]
  PORT = "8080"
  ZUS_CSV_PATH = "data/processed/zus_forecast_all.csv"
  BASE_AVG_WAGE_TODAY = "8000"
  SUPABASE_URL = "your-supabase-url"
  SUPABASE_KEY = "your-supabase-anon-key"
```

### 4. Deploy the Application

Deploy your application with the updated configuration:

```bash
flyctl deploy
```

## How It Works

When a user makes a POST request to `/forecast`, the application will:

1. Process the pension simulation as usual
2. Save the forecast data to the local Excel file (existing functionality)
3. **NEW**: Save the forecast data to the Supabase `forecast` table
4. Return the response with an additional `supabase_saved` field indicating whether the database save was successful

## API Response

The `/forecast` endpoint now returns:

```json
{
  "inputs": { ... },
  "result": { ... },
  "admin_usage_saved_to": "zus_usage_report.xlsx",
  "supabase_saved": true
}
```

## Database Schema

The forecast data is saved with Polish column names matching your frontend requirements:

- `data_symulacji`: "10.04.2025, 14:30" format
- `wiek`: Age as integer
- `plec`: "M" or "F"
- `wynagrodzenie`: Monthly salary
- `kod_pocztowy`: Postal code or "—"
- `pozadana_emerytura`: Desired pension amount
- `zgromadzone_srodki`: Accumulated funds
- `l4_wliczone`: "Tak" or "Nie"
- `prognozowana_emerytura`: Forecasted pension
- `realna_emerytura`: Real (inflation-adjusted) pension

## Troubleshooting

- If `supabase_saved` is `false`, check your environment variables
- Check the application logs for Supabase connection errors
- Ensure your Supabase project has the correct table structure
- Verify that your API key has the necessary permissions
