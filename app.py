import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from datetime import datetime

st.set_page_config(layout="wide", page_title="Credit Fraud Explorer", initial_sidebar_state="expanded")

@st.cache_data(show_spinner=False)
def load_data(path: str):
    df = pd.read_csv(path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    

    expected = {'trans_date_trans_time','cc_num','merchant','category','amt',
                'first','last','gender','street','city','state','zip','lat',
                'long','city_pop','job','dob','trans_num','unix_time','merch_lat',
                'merch_long','is_fraud','trans_dt'}
    missing = expected - set(df.columns)
    if missing:
        st.warning(f"Missing expected columns: {missing}")
    

    if 'trans_date_trans_time' in df.columns:
        df['trans_dt'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    
   
    df['amt'] = pd.to_numeric(df['amt'], errors='coerce').fillna(0)
    df['is_fraud'] = df['is_fraud'].astype(int)

    for c in ['lat','long','merch_lat','merch_long']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


path_1 = "../fraud_part1.csv"
path_2 = "../fraud_part2.csv"
path_3 = "../fraud_part2.csv"

df1 = load_data(path_1)
df2 = load_data(path_2)
df3 = load_data(path_3)

df = pd.concat([df1, df2, df3], ignore_index=True)

if df.shape[0] == 0:
    st.stop()


st.sidebar.title("Filters")
min_date = df['trans_dt'].min()
max_date = df['trans_dt'].max()
date_range = st.sidebar.date_input("Transaction date range", value=(min_date.date(), max_date.date()))
start_date, end_date = date_range if isinstance(date_range, tuple) else (date_range, date_range)

amt_min = float(df['amt'].min())
amt_max = float(df['amt'].quantile(0.995))
amt_range = st.sidebar.slider("Amount range (USD)", min_value=0.0, max_value=float(df['amt'].max()), value=(amt_min, amt_max), step=1.0)

states = sorted(df['state'].dropna().unique())
selected_states = st.sidebar.multiselect("States (leave blank = all)", options=states, default=[])

merchants = sorted(df['merchant'].dropna().unique())
selected_merchants = st.sidebar.multiselect("Merchants (sample)", options=merchants[:200], default=[])

fraud_mode = st.sidebar.selectbox("Show", options=["Both", "Fraud only", "Non-fraud only"])

agg_by = st.sidebar.selectbox("Aggregate time by", options=["D (day)", "W (week)", "M (month)"], index=0)
refresh = st.sidebar.button("Refresh / Apply filters")


filtered = df.copy()
filtered = filtered[filtered['trans_dt'].notna()] 

filtered = filtered[(filtered['trans_dt'].dt.date >= start_date) & (filtered['trans_dt'].dt.date <= end_date)]
filtered = filtered[(filtered['amt'] >= amt_range[0]) & (filtered['amt'] <= amt_range[1])]

if selected_states:
    filtered = filtered[filtered['state'].isin(selected_states)]
if selected_merchants:
    filtered = filtered[filtered['merchant'].isin(selected_merchants)]

if fraud_mode == "Fraud only":
    filtered = filtered[filtered['is_fraud'] == 1]
elif fraud_mode == "Non-fraud only":
    filtered = filtered[filtered['is_fraud'] == 0]


st.title("Credit Fraud Explorer — Interactive Dashboard")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_tx = len(filtered)
fraud_tx = int(filtered['is_fraud'].sum())
fraud_rate = 100 * fraud_tx / total_tx if total_tx > 0 else 0
avg_amt = filtered['amt'].mean() if total_tx > 0 else 0
unique_merchants = filtered['merchant'].nunique()

kpi1.metric("Transactions (filtered)", f"{total_tx:,}")
kpi2.metric("Fraudulent tx", f"{fraud_tx:,}", f"{fraud_rate:.2f}%")
kpi3.metric("Avg. Amount (USD)", f"${avg_amt:,.2f}")
kpi4.metric("Unique Merchants", f"{unique_merchants:,}")


left_col, right_col = st.columns((2,1))

with left_col:
    st.subheader("Transactions over time")
    ts = filtered.copy()
    freq_map = {"D (day)":"D", "W (week)":"W", "M (month)":"M"}
    agg_freq = freq_map.get(agg_by, "D")
    ts['period'] = ts['trans_dt'].dt.to_period(agg_freq).dt.to_timestamp()
    ts_agg = ts.groupby('period').agg(total=('trans_dt','count'), frauds=('is_fraud','sum'), avg_amt=('amt','mean')).reset_index()
    ts_agg['fraud_rate'] = 100 * ts_agg['frauds'] / ts_agg['total']
    fig_ts = px.line(ts_agg, x='period', y=['total','frauds'], labels={'value':'count','period':'date'}, title="Transactions and Fraud Counts")
    fig_ts.update_traces(mode='lines+markers')
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Amount distribution by Fraud label")
    fig_amt = px.violin(filtered, x='is_fraud', y='amt', box=True, points='outliers', labels={'is_fraud':'is_fraud','amt':'Amount (USD)'}, title='Transaction Amount Distribution (Fraud vs Non-Fraud)')
    st.plotly_chart(fig_amt, use_container_width=True)

    st.subheader("Fraud rate by U.S. state")
    fraud_by_state = filtered.groupby('state').agg(total=('trans_dt','count'), frauds=('is_fraud','sum')).reset_index()
    fraud_by_state['fraud_rate'] = fraud_by_state['frauds']/fraud_by_state['total']
    fraud_by_state = fraud_by_state[fraud_by_state['state'].notna()]
    if not fraud_by_state.empty:
        fig_state = px.choropleth(fraud_by_state, locations='state', locationmode='USA-states', color='fraud_rate', color_continuous_scale='Reds', scope='usa', hover_data=['total','frauds'], labels={'fraud_rate':'Fraud rate'})
        fig_state.update_layout(margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        st.info("No state-level data available.")

with right_col:
    st.subheader("Merchant / Category breakdown")
    sb = filtered.groupby(['merchant','category']).size().reset_index(name='count')
    if not sb.empty:
        top_sb = sb.sort_values('count', ascending=False).head(200)
        fig_sb = px.sunburst(top_sb, path=['category','merchant'], values='count', maxdepth=3, title="Top merchants by category")
        st.plotly_chart(fig_sb, use_container_width=True)
    else:
        st.info("No merchant/category data to show.")

    st.subheader("Data snapshot")
    st.dataframe(filtered.head(200))
    csv = filtered.to_csv(index=False)
    st.download_button(label="Download filtered CSV", data=csv, file_name="filtered_fraud_data.csv", mime="text/csv")

st.subheader("Geospatial Overview — Customer & Merchant locations (PyDeck)")
map_options = st.multiselect("Map layers to show", options=["Customer heatmap","Merchant heatmap","Customer scatter","Merchant scatter"], default=["Customer heatmap","Customer scatter"])

center_lat = filtered['lat'].median() if 'lat' in filtered.columns else 37.0902
center_lon = filtered['long'].median() if 'long' in filtered.columns else -95.7129
initial_view_state = pdk.ViewState(latitude=float(center_lat), longitude=float(center_lon), zoom=4, pitch=40)

layers = []
if "Customer heatmap" in map_options and 'lat' in filtered.columns and 'long' in filtered.columns:
    heat_data = filtered[['lat','long','is_fraud']].dropna()
    if not heat_data.empty:
        layers.append(pdk.Layer("HeatmapLayer", data=heat_data, get_position=['long','lat'], aggregation='SUM', get_weight='is_fraud + 1e-6'))

if "Merchant heatmap" in map_options and 'merch_lat' in filtered.columns and 'merch_long' in filtered.columns:
    mheat = filtered[['merch_lat','merch_long','is_fraud']].dropna()
    if not mheat.empty:
        layers.append(pdk.Layer("HeatmapLayer", data=mheat, get_position=['merch_long','merch_lat'], aggregation='SUM', get_weight='is_fraud + 1e-6'))

if "Customer scatter" in map_options and 'lat' in filtered.columns and 'long' in filtered.columns:
    scatter_c = filtered[['lat','long','is_fraud','amt','city']].dropna()
    if not scatter_c.empty:
        layers.append(pdk.Layer("ScatterplotLayer", data=scatter_c, get_position=['long','lat'], get_fill_color='[255*is_fraud,140,30]', get_radius=200, pickable=True, auto_highlight=True))

if "Merchant scatter" in map_options and 'merch_lat' in filtered.columns and 'merch_long' in filtered.columns:
    scatter_m = filtered[['merch_lat','merch_long','is_fraud','amt','merchant']].dropna()
    if not scatter_m.empty:
        layers.append(pdk.Layer("ScatterplotLayer", data=scatter_m, get_position=['merch_long','merch_lat'], get_fill_color='[30,144,255*(1-is_fraud)]', get_radius=250, pickable=True, auto_highlight=True))

if layers:
    r = pdk.Deck(layers=layers, initial_view_state=initial_view_state, tooltip={"text":"{city}\nAmount: {amt}\nFraud: {is_fraud}"})
    st.pydeck_chart(r)
else:
    st.info("No geospatial layers could be created (missing lat/long columns).")

st.subheader("Temporal patterns")
if 'trans_dt' in filtered.columns:
    filtered['hour'] = filtered['trans_dt'].dt.hour
    filtered['weekday'] = filtered['trans_dt'].dt.day_name()
    col1, col2 = st.columns(2)
    with col1:
        fig_h = px.histogram(filtered, x='hour', color='is_fraud', nbins=24, barmode='group', title='Transactions by Hour')
        st.plotly_chart(fig_h, use_container_width=True)
    with col2:
        wd_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        fig_w = px.histogram(filtered, x='weekday', category_orders={'weekday':wd_order}, color='is_fraud', title='Transactions by Weekday')
        st.plotly_chart(fig_w, use_container_width=True)
else:
    st.info("No transaction datetime available.")


st.markdown("-------")
st.markdown(
    """
    **Notes & tips**
    - Use sidebar filters to focus on time ranges, amounts, merchants, or states.
    - PyDeck map layers are interactive: zoom, rotate, and click points.
    """
)
