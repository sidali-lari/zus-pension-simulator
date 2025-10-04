# ZUS Pension Simulator (Backend)

Flask API that forecasts pension amounts using macro assumptions and an XGBoost-based replacement-rate path to 2080. Includes PDF→CSV extraction script for reproducibility.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional envs
export ZUS_CSV_PATH=./data/processed/zus_forecast_all.csv
export BASE_AVG_WAGE_TODAY=8000
export PORT=8000

python app.py
# → http://127.0.0.1:8000
