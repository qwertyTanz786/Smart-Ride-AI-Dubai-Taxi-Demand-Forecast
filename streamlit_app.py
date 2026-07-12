import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

st.set_page_config(page_title="Smart Ride AI Dashboard", page_icon="🚕", layout="wide")

st.markdown("""
<style>
/* Main App Fade-In */
@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
div.block-container {
    animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Premium Metric Cards */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease !important;
    will-change: transform;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-8px) !important;
    box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2) !important;
    background: rgba(255, 255, 255, 0.08) !important;
    z-index: 999;
}

/* For Light Mode Compatibility */
@media (prefers-color-scheme: light) {
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03) !important;
    }
    div[data-testid="stMetric"]:hover {
        background: #f8fafc !important;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1) !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================
# Core Logic & Model Caching
# ============================================
@st.cache_resource(show_spinner="Training ML Model & Processing Data...")
def load_and_train_model():
    df = pd.read_csv(r"C:\Portfolio-ML\Datasets\Dubai_Smart_Mobility_1M_Cleaned.csv")
    
    real_coords = {
        'Al Barsha': [25.110, 55.200],
        'Al Nahda': [25.286, 55.362],
        'Arabian Ranches': [25.050, 55.269],
        'Bur Dubai': [25.261, 55.296],
        'Business Bay': [25.185, 55.266],
        'DIFC': [25.213, 55.281],
        'Deira': [25.269, 55.318],
        'Downtown Dubai': [25.197, 55.274],
        'Dubai Airport': [25.253, 55.365],
        'Dubai Hills': [25.112, 55.253],
        'Dubai Marina': [25.080, 55.140],
        'Expo City': [24.965, 55.149],
        'JLT': [25.069, 55.140],
        'Jumeirah': [25.216, 55.233],
        'Karama': [25.244, 55.303],
        'Mirdif': [25.219, 55.418],
        'Motor City': [25.046, 55.242],
        'Palm Jumeirah': [25.112, 55.139],
        'Satwa': [25.228, 55.275],
        'Silicon Oasis': [25.125, 55.385]
    }
    
    zone_coords = pd.DataFrame([
        {'Pickup_Zone': zone, 'Latitude': coords[0], 'Longitude': coords[1]} 
        for zone, coords in real_coords.items()
    ])
    
    # Feature Engineering & Aggregation
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    agg_cols = {
        'Taxi_Demand': 'sum',
        'Weather': 'first',
        'Event_Type': 'first',
        'Day_Type': 'first'
    }
    df = df.groupby(['Date', 'Hour', 'Pickup_Zone']).agg(agg_cols).reset_index()

    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df.drop(['Date'], axis=1, inplace=True)

    df_template = df.mode().iloc[[0]].copy()
    
    categorical_cols = ['Pickup_Zone', 'Weather', 'Event_Type', 'Day_Type']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop('Taxi_Demand', axis=1)
    Y = df['Taxi_Demand']

    m = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    m.fit(X, Y)
    
    return m, X.columns, df_template, categorical_cols, zone_coords

model, model_cols, template_df, cat_cols, zone_map_df = load_and_train_model()

# ============================================
# Helper Functions
# ============================================
def surge_multiplier(x):
    if x < 800: return 1.0
    elif x < 1500: return 1.2
    elif x < 2500: return 1.5
    else: return 2.0

def allocate_drivers(demand):
    return int(demand * 0.5), int(demand * 0.3), int(demand * 0.2)

def predict_demand(pickup, dt, hour):
    user_input_df = template_df.copy()
    
    user_input_df['Pickup_Zone'] = pickup
    user_input_df['Hour'] = int(hour)
    user_input_df['Day'] = dt.day
    user_input_df['WeekOfYear'] = dt.isocalendar().week
    if 'Month' in user_input_df.columns:
        user_input_df['Month'] = dt.month
    if 'Weekday' in user_input_df.columns:
        user_input_df['Weekday'] = dt.weekday()
    if 'Day_Type' in user_input_df.columns:
        user_input_df['Day_Type'] = 'Weekend' if dt.weekday() >= 5 else 'Weekday'
        
    user_input_encoded = pd.get_dummies(user_input_df, columns=cat_cols)
    if 'Taxi_Demand' in user_input_encoded.columns:
        user_input_encoded = user_input_encoded.drop('Taxi_Demand', axis=1)
        
    user_input_encoded = user_input_encoded.reindex(columns=model_cols, fill_value=0).astype(float)
    
    pred = model.predict(user_input_encoded)[0]
    return int(round(pred))

valid_zones = sorted(zone_map_df['Pickup_Zone'].tolist())

# Synchronize Map and Sidebar Selections
if 'sidebar_zone' not in st.session_state:
    st.session_state.sidebar_zone = valid_zones[0]
if 'prev_map_selection' not in st.session_state:
    st.session_state.prev_map_selection = None

if 'map_selection' in st.session_state:
    sel = st.session_state.map_selection
    if sel != st.session_state.prev_map_selection:
        st.session_state.prev_map_selection = sel
        try:
            # Extract objects from the selection event
            selection_data = sel.get("selection", {}) if isinstance(sel, dict) else getattr(sel, "selection", {})
            objects = selection_data.get("objects", {}) if isinstance(selection_data, dict) else getattr(selection_data, "objects", {})
            
            # objects can be a list or a dict mapping layer IDs to lists depending on PyDeck/Streamlit version
            if isinstance(objects, dict):
                for layer_id, layer_objs in objects.items():
                    if layer_objs and isinstance(layer_objs, list):
                        new_zone = layer_objs[0].get("Pickup_Zone")
                        if new_zone:
                            st.session_state.sidebar_zone = new_zone
                            break
            elif isinstance(objects, list) and objects:
                new_zone = objects[0].get("Pickup_Zone")
                if new_zone:
                    st.session_state.sidebar_zone = new_zone
        except Exception:
            pass

# ============================================
# Sidebar Configuration
# ============================================
st.sidebar.title("🚕 Smart Ride Config")
st.sidebar.markdown("Configure operational parameters to forecast demand.")

selected_date = st.sidebar.date_input("Target Date")
selected_hour = st.sidebar.slider("Time of Day (Hour)", 0, 23, 8)

selected_zone = st.sidebar.selectbox(
    "Pickup Zone", 
    valid_zones,
    key="sidebar_zone"
)

# ============================================
# Main Dashboard UI
# ============================================
st.title(f"Operational Forecast: {selected_zone}")
st.markdown(f"**Date:** {selected_date.strftime('%d %B %Y')} | **Time:** {selected_hour:02d}:00")

# Run Prediction for exact hour
current_demand = predict_demand(selected_zone, selected_date, selected_hour)
surge = surge_multiplier(current_demand)
sedan, suv, luxury = allocate_drivers(current_demand)

# 1. Top Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Demand", f"{current_demand} rides")
c2.metric("Optimal Surge", f"{surge}x")
c3.metric("Sedan / SUV Split", f"{sedan} / {suv}")
c4.metric("Luxury Fleet Required", f"{luxury}")

st.markdown("---")

# 2. Map & 24H Forecast Chart
col_map, col_chart = st.columns([1, 1.2])

with col_map:
    st.subheader("Zone Location")
    # Color logic: Red for selected zone, blue for others
    zone_map_df['color'] = zone_map_df['Pickup_Zone'].apply(lambda z: [255, 0, 0, 200] if z == selected_zone else [0, 100, 255, 100])
    zone_map_df['radius'] = zone_map_df['Pickup_Zone'].apply(lambda z: 2000 if z == selected_zone else 1000)
    
    selected_lat = zone_map_df[zone_map_df['Pickup_Zone'] == selected_zone]['Latitude'].values[0]
    selected_lon = zone_map_df[zone_map_df['Pickup_Zone'] == selected_zone]['Longitude'].values[0]

    view_state = pdk.ViewState(
        latitude=selected_lat, 
        longitude=selected_lon, 
        zoom=10, 
        pitch=45,
        transition_duration=1200,
        transition_easing="ease-out-cubic"
    )
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        id="zones_layer",
        data=zone_map_df,
        get_position='[Longitude, Latitude]',
        get_color='color',
        get_radius='radius',
        pickable=True,
        auto_highlight=True
    )
    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Pickup_Zone}"}),
        on_select="rerun",
        selection_mode="single-object",
        key="map_selection"
    )

with col_chart:
    st.subheader("Forecast Insights")
    c_pie, c_trend = st.columns(2)
    
    with c_pie:
        # 1. Fleet Mix Pie Chart
        fleet_data = pd.DataFrame({
            "Vehicle Type": ["Standard Sedan", "Premium SUV", "Luxury VIP"],
            "Count": [sedan, suv, luxury]
        })
        fig_pie = px.pie(fleet_data, values="Count", names="Vehicle Type", hole=0.4,
                         color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'])
        fig_pie.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=350, 
                              legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_trend:
        # 2. Next 7 Days Forecast (Something New)
        from datetime import timedelta
        days = []
        preds = []
        for i in range(7):
            next_day = selected_date + timedelta(days=i)
            days.append(next_day.strftime('%b %d'))
            preds.append(predict_demand(selected_zone, next_day, selected_hour))
            
        trend_data = pd.DataFrame({"Date": days, "Forecast": preds})
        fig_trend = px.line(trend_data, x="Date", y="Forecast", markers=True, line_shape="spline",
                            color_discrete_sequence=['#ef4444'])
        fig_trend.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=350,
                                xaxis_title="Upcoming Week", yaxis_title="Rides")
        st.plotly_chart(fig_trend, use_container_width=True)


