# ============================================
# SMART RIDE AI - Taxi Demand Forecasting
# + Demand Category
# + Surge Pricing Recommendation
# + Fleet Allocation
# ============================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Load Dataset
df = pd.read_csv(
    r"C:\Portfolio-ML\Datasets\Dubai_Smart_Mobility_1M_Cleaned.csv"
)

# Extract unique zones before processing
VALID_ZONES = list(df['Pickup_Zone'].dropna().unique())

# --------------------------------------------
# Feature Engineering & Aggregation
# --------------------------------------------
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Define columns we want to keep for an operational forecast
agg_cols = {
    'Taxi_Demand': 'sum',
    'Weather': 'first',
    'Event_Type': 'first',
    'Day_Type': 'first'
}

# Group by time and origin zone
df = df.groupby(['Date', 'Hour', 'Pickup_Zone']).agg(agg_cols).reset_index()

df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday

df.drop(['Date'], axis=1, inplace=True)

# Create a template row for interactive predictions
df_template = df.mode().iloc[[0]].copy()

# One-hot Encoding
categorical_cols = [
    'Pickup_Zone',
    'Weather',
    'Event_Type',
    'Day_Type'
]

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

# --------------------------------------------
# Model Training
# --------------------------------------------
X = df.drop('Taxi_Demand', axis=1)
Y = df['Taxi_Demand']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
m = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
m.fit(X_train,Y_train)

# --------------------------------------------
# Predictions
# --------------------------------------------
predictions = m.predict(X_test)

# --------------------------------------------
# Demand Category
# --------------------------------------------
def demand_category(x):
    if x < 800:
        return "LOW"
    elif x < 1500:
        return "MEDIUM"
    elif x < 2500:
        return "HIGH"
    else:
        return "PEAK"

# --------------------------------------------
# Surge Pricing Recommendation
# --------------------------------------------
def surge_multiplier(x):
    if x < 800:
        return 1.0
    elif x < 1500:
        return 1.2
    elif x < 2500:
        return 1.5
    else:
        return 2.0

# --------------------------------------------
# Fleet Allocation
# --------------------------------------------
def allocate_drivers(demand):
    sedan = int(demand * 0.5)
    suv = int(demand * 0.3)
    luxury = int(demand * 0.2)
    return sedan,suv,luxury

print("MODEL PERFORMANCE")
print("-"*60)
print("R2 Score :",round(r2_score(Y_test,predictions),4))
print("Train Score               :",round(m.score(X_train,Y_train),4))
print("Test Score                :",round(m.score(X_test,Y_test),4))
print("="*60)

# --------------------------------------------
# Interactive Prediction
# --------------------------------------------
print("\n" + "="*60)
print("           INTERACTIVE PREDICTION")
print("="*60)

import difflib

def get_valid_zone(prompt_text):
    while True:
        val = input(prompt_text).strip()
        if not val:
            return None
        for vz in VALID_ZONES:
            if val.lower() == vz.lower():
                return vz
        matches = difflib.get_close_matches(val, VALID_ZONES, n=1, cutoff=0.5)
        if matches:
            confirm = input(f"   Did you mean '{matches[0]}'? (y/n, default y): ").strip().lower()
            if confirm in ['y', 'yes', '']:
                return matches[0]
        print(f"   ❌ Invalid zone. Examples: {', '.join(VALID_ZONES[:5])}...")

def get_valid_date(prompt_text):
    while True:
        val = input(prompt_text).strip()
        if not val:
            return None, None
        try:
            dt = pd.to_datetime(val, dayfirst=True)
            if dt.year < 2020 or dt.year > 2030:
                print("   ❌ Please enter a realistic year (e.g., 2023 or 2024).")
                continue
            return dt, val
        except Exception:
            print("   ❌ Invalid date format. Please use DD/MM/YYYY.")

try:
    print("\nEnter details (Leave blank to use defaults from the dataset):")
    
    # Print available zones
    zones_str = ", ".join(sorted(VALID_ZONES))
    print(f"\nAvailable Zones: \n{zones_str}\n")
    
    pickup = get_valid_zone("Pickup Zone (from list above): ")
    hour = input("Hour of day (0-23, optional): ").strip()
    dt, date_input = get_valid_date("Date (DD/MM/YYYY): ")
    
    user_input_df = df_template.copy()
    
    if pickup:
        user_input_df['Pickup_Zone'] = pickup
    if hour:
        user_input_df['Hour'] = int(hour)
    if dt is not None:
        user_input_df['Day'] = dt.day
        user_input_df['WeekOfYear'] = dt.isocalendar().week
        if 'Month' in user_input_df.columns:
            user_input_df['Month'] = dt.month
        if 'Weekday' in user_input_df.columns:
            user_input_df['Weekday'] = dt.weekday()
        if 'Day_Type' in user_input_df.columns:
            user_input_df['Day_Type'] = 'Weekend' if dt.weekday() >= 5 else 'Weekday'

    user_input_encoded = pd.get_dummies(user_input_df, columns=categorical_cols)
    
    if 'Taxi_Demand' in user_input_encoded.columns:
        user_input_encoded = user_input_encoded.drop('Taxi_Demand', axis=1)
        
    user_input_encoded = user_input_encoded.reindex(columns=X_train.columns, fill_value=0).astype(float)
    
    user_pred = m.predict(user_input_encoded)[0]
    
    user_pred_int = int(round(user_pred))
    s, su, l = allocate_drivers(user_pred_int)
    
    route_pickup = user_input_df['Pickup_Zone'].iloc[0]
    
    if date_input:
        disp_date = date_input
    else:
        # Reconstruct default date string if not provided
        d = int(user_input_df['Day'].iloc[0])
        m = int(user_input_df['Month'].iloc[0])
        # year isn't in df_template explicitly, so we just say "Dataset Default"
        disp_date = f"{d:02d}/{m:02d}/(Default Year)"
        
    actual_hour = int(user_input_df['Hour'].iloc[0])
    disp_time = f"{actual_hour:02d}:00"
    
    print("\n" + "="*60)
    print(" 🚕  ZONE DEMAND FORECAST DASHBOARD  🚕")
    print("="*60)
    print(f"📍 Zone  : {route_pickup}")
    print(f"📅 Date  : {disp_date}")
    print(f"⏰ Time  : {disp_time}")
    print("-" * 60)
    print(f"📈 Total Outbound Demand : {user_pred_int} rides ({demand_category(user_pred_int)})")
    print(f"⚡ Recommended Surge     : {surge_multiplier(user_pred_int)}x")
    print("-" * 60)
    print("🚗 Outbound Fleet Allocation Required:")
    print(f"   • Standard Sedan : {s} vehicles")
    print(f"   • Premium SUV    : {su} vehicles")
    print(f"   • Luxury VIP     : {l} vehicles")
    print("=" * 60)
    print("\n")

except Exception as e:
    print(f"Error processing input: {e}")
