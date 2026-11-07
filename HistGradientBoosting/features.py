"""Feature engineering for SSU Data Dash 2026.
Only uses columns present at prediction time; no sold_price, date_sold,
external AVM, or days_on_market.
"""
import numpy as np
import pandas as pd

CATEGORICAL = ['house_type_name', 'house_style', 'municipality', 'community']
NUMERIC = ['area_sqft', 'bedrooms', 'washrooms', 'parking_total', 'garage',
           'lot_front', 'lot_depth', 'lot_area', 'latitude', 'longitude',
           'year_listed', 'month_listed', 'months_since_start',
           'lat_lon_interaction', 'remarks_length', 'remarks_renovated',
           'remarks_luxury', 'remarks_needs_work', 'remarks_basement']

def square_feet(series):
    parts = series.astype('string').str.split('-', n=1, expand=True)
    lo = pd.to_numeric(parts[0].str.strip(), errors='coerce')
    hi = pd.to_numeric(parts[1].str.strip(), errors='coerce') if 1 in parts.columns else lo
    return (lo + hi.fillna(lo)) / 2

def make_features(homes, list_aware=False):
    x = pd.DataFrame(index=homes.index)
    for col in CATEGORICAL:
        x[col] = homes[col].fillna('Unknown').astype(str) if col in homes else 'Unknown'
    x['area_sqft'] = square_feet(homes['house_area'])
    for col in ['bedrooms', 'washrooms', 'parking_total', 'garage', 'lot_front',
                'lot_depth', 'latitude', 'longitude']:
        x[col] = pd.to_numeric(homes[col], errors='coerce') if col in homes else np.nan
    x['lot_area'] = x['lot_front'] * x['lot_depth']
    dates = pd.to_datetime(homes['date_listed'], errors='coerce')
    x['year_listed'] = dates.dt.year.astype(float)
    x['month_listed'] = dates.dt.month.astype(float)
    x['months_since_start'] = (dates - pd.Timestamp('2021-01-01')).dt.days / 30.44
    x['lat_lon_interaction'] = x['latitude'] * x['longitude']
    remarks = homes['remarks'].fillna('').astype(str).str.lower() if 'remarks' in homes else pd.Series('', index=homes.index)
    x['remarks_length'] = remarks.str.len().astype(float)
    for key, pattern in {'renovated':'renovat|updated|upgraded',
                         'luxury':'luxur|premium|high.end',
                         'needs_work':'fixer|needs work|as.is',
                         'basement':'basement|finished lower'}.items():
        x['remarks_' + key] = remarks.str.contains(pattern, regex=True).astype(float)
    if list_aware:
        x['list_price'] = pd.to_numeric(homes['list_price'], errors='coerce')
        x['log_list_price'] = np.log(x['list_price'].clip(lower=1))
    return x

def off_market_features(homes):
    return make_features(homes, False)

def list_aware_features(homes):
    return make_features(homes, True)
