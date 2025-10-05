-- Create the 'forecast' table in Supabase
-- This table will store pension forecast data from the ZUS simulator

CREATE TABLE IF NOT EXISTS forecast (
    id SERIAL PRIMARY KEY,
    data_symulacji TEXT NOT NULL,
    wiek INTEGER NOT NULL,
    plec TEXT NOT NULL CHECK (plec IN ('M', 'F')),
    wynagrodzenie DECIMAL(10,2) NOT NULL,
    kod_pocztowy TEXT,
    pozadana_emerytura DECIMAL(10,2),
    zgromadzone_srodki DECIMAL(12,2),
    l4_wliczone TEXT NOT NULL CHECK (l4_wliczone IN ('Tak', 'Nie')),
    prognozowana_emerytura DECIMAL(10,2),
    realna_emerytura DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index on the simulation date for faster queries
CREATE INDEX IF NOT EXISTS idx_forecast_data_symulacji ON forecast(data_symulacji);

-- Create an index on age for filtering
CREATE INDEX IF NOT EXISTS idx_forecast_wiek ON forecast(wiek);

-- Create an index on sex for filtering
CREATE INDEX IF NOT EXISTS idx_forecast_plec ON forecast(plec);

-- Add a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_forecast_updated_at 
    BEFORE UPDATE ON forecast 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions (adjust as needed for your Supabase setup)
-- ALTER TABLE forecast ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Enable read access for all users" ON forecast FOR SELECT USING (true);
-- CREATE POLICY "Enable insert for all users" ON forecast FOR INSERT WITH CHECK (true);
