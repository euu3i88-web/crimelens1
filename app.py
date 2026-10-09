import json
from pathlib import Path
from math import radians, sin, cos, asin, sqrt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CrimeLens | Area Safety Intelligence", page_icon="🧭", layout="wide")

DATA_PATH = Path(__file__).parent / "data.csv"
MODEL_DIR = Path(__file__).parent / "models"

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at top right,#172c42 0,#0b1220 45%,#080d16 100%); color:#f1f5f9}
[data-testid="stSidebar"] {background:#101b2a}
h1,h2,h3 {color:#f8fafc}
div.stButton > button {background:#36d399;color:#07111b;border:0;border-radius:10px;font-weight:700}
div[data-testid="stMetric"] {background:#132338;padding:16px;border-radius:14px;border:1px solid #263c54}
.small-note {color:#b8c7d9;font-size:0.92rem}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Latitude","Longitude","Timestamp","Crime_Category"])
    return df

def haversine(lat1, lon1, lat2, lon2):
    r=6371.0
    p1,p2=radians(float(lat1)),radians(float(lat2))
    dlat=radians(float(lat2)-float(lat1)); dlon=radians(float(lon2)-float(lon1))
    a=sin(dlat/2)**2+cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2*r*asin(sqrt(a))

def area_lookup(df, query):
    q=query.strip().lower()
    if not q: return None, "Enter a place, district, or grid cell."
    for col in ["Spatial_Grid_ID","District","Location_Type"]:
        matches=df[df[col].astype(str).str.lower().str.contains(q, regex=False, na=False)]
        if not matches.empty:
            return matches, f"Matched dataset field: {col}"
    # Common Kerala place names can be searched against district values. If absent, user can provide coordinates.
    return None, "No name match in the dataset. Use coordinates/live location to estimate from the nearest recorded area."

def nearest_rows(df, lat, lon, n=100):
    tmp=df.copy()
    tmp["_distance_km"]=np.sqrt(((tmp["Latitude"]-lat)*111.0)**2+((tmp["Longitude"]-lon)*111.0*np.cos(np.radians(lat)))**2)
    return tmp.nsmallest(n, "_distance_km")

def risk_summary(rows, distance_km=None):
    if rows is None or rows.empty: return None
    # Descriptive proxy based on recorded incidents, not a calibrated probability of future crime.
    recent=rows[rows["Timestamp"] >= (rows["Timestamp"].max()-pd.Timedelta(days=365))]
    if recent.empty: recent=rows
    counts=recent["Crime_Category"].value_counts()
    top=counts.index[0] if len(counts) else "Unknown"
    mean_sev=float(recent["Offense_Severity_Index"].mean()) if "Offense_Severity_Index" in recent else 0
    per_grid=recent["Spatial_Grid_ID"].nunique()
    # Relative index among dataset locations; label explicitly as historical indicator.
    count=len(recent)
    score=min(100, max(0, 18 + 9*mean_sev + 2.2*np.log1p(count)))
    return {"score":round(score), "count":count, "severity":mean_sev, "top":top, "counts":counts, "recent":recent}

df=load_data()
st.title("🧭 CrimeLens")
st.caption("Area crime-pattern explorer • Kerala • Historical-data-based decision support")
st.warning("Research prototype, not an official police system. The score is a historical  crime Dataset For emergencies, contact local emergency services.")

with st.sidebar:
    st.header("Find an area")
    mode=st.radio("Search method",["Place / district / grid","Coordinates","Live location"])
    place=""
    lat=lon=None
    if mode=="Place / district / grid":
        place=st.text_input("Place, district, grid cell, or location type", placeholder="e.g. Thrissur or grid cell")
        st.caption("If no name matches, enter coordinates or use live location.")
    elif mode=="Coordinates":
        c1,c2=st.columns(2)
        lat=c1.number_input("Latitude", value=10.5276, format="%.6f")
        lon=c2.number_input("Longitude", value=76.2144, format="%.6f")
    else:
        st.write("Allow browser location access when prompted.")
        try:
            from streamlit_geolocation import streamlit_geolocation
            loc=streamlit_geolocation()
            if loc and loc.get("latitude") is not None:
                lat=float(loc["latitude"]); lon=float(loc["longitude"])
                st.success(f"Location received: {lat:.5f}, {lon:.5f}")
            else:
                st.info("Waiting for location permission. If it does not appear, use Coordinates.")
        except Exception:
            st.info("Location widget unavailable. Use Coordinates or install streamlit-geolocation.")

    analyze=st.button("Analyze area", use_container_width=True)

if "result" not in st.session_state: st.session_state.result=None
if analyze:
    rows=None; description=""
    if mode=="Place / district / grid":
        rows,description=area_lookup(df,place)
        if rows is not None:
            center_lat=float(rows["Latitude"].median()); center_lon=float(rows["Longitude"].median())
            rows=nearest_rows(df,center_lat,center_lon, n=max(100,len(rows)))
            # retain rows associated with matched named area where possible, plus local context
            description += f" • Approx. center: {center_lat:.4f}, {center_lon:.4f}"
        else:
            st.session_state.result={"error":description}
    else:
        if lat is not None and lon is not None:
            rows=nearest_rows(df,lat,lon, n=150)
            nearest_km=float(rows["_distance_km"].min())
            description=f"Nearest recorded dataset points • closest record approx. {nearest_km:.2f} km away"
        else:
            st.session_state.result={"error":"Location not available. Allow permission or choose Coordinates."}
    if rows is not None and not rows.empty:
        st.session_state.result={"rows":rows,"description":description,"lat":lat,"lon":lon,"mode":mode}


def model_forecast(rows):
    """Optional trained-model inference; gracefully falls back until artifacts are uploaded."""
    meta_path=MODEL_DIR/"metadata.json"
    cnn_path=MODEL_DIR/"crime_cnn.keras"
    lstm_path=MODEL_DIR/"crime_lstm.keras"
    if not (meta_path.exists() and cnn_path.exists() and lstm_path.exists()):
        return None
    try:
        import tensorflow as tf
        meta=json.loads(meta_path.read_text())
        sample=rows.sort_values("_distance_km").iloc[0] if "_distance_km" in rows else rows.iloc[0]
        now=pd.Timestamp.now()
        hour=int(now.hour); month=int(now.month); weekday=int(now.dayofweek)
        dm=meta.get("district_mapping",{}); lm=meta.get("location_mapping",{}); gm=meta.get("grid_mapping",{})
        feat={
          "Latitude":float(sample["Latitude"]), "Longitude":float(sample["Longitude"]),
          "Hour_sin":float(np.sin(2*np.pi*hour/24)), "Hour_cos":float(np.cos(2*np.pi*hour/24)),
          "Month_sin":float(np.sin(2*np.pi*month/12)), "Month_cos":float(np.cos(2*np.pi*month/12)),
          "DayOfWeek":weekday, "District_code":dm.get(str(sample.get("District","")),0),
          "Location_code":lm.get(str(sample.get("Location_Type","")),0),
          "Grid_code":gm.get(str(sample.get("Spatial_Grid_ID","")),0)
        }
        x=np.array([[feat[k] for k in meta["features"]]],dtype=float)
        x=(x-np.array(meta["scaler_mean"])) / np.array(meta["scaler_scale"])
        cnn=tf.keras.models.load_model(cnn_path,compile=False)
        probs=cnn.predict(x[...,None],verbose=0)[0]
        classes=meta["crime_classes"]
        top=classes[int(np.argmax(probs))]
        confidence=float(np.max(probs))
        # LSTM sequence comes from the training notebook's hourly grid-count export.
        forecast=None
        hourly_path=MODEL_DIR/"hourly_counts.csv"
        if hourly_path.exists():
            h=pd.read_csv(hourly_path)
            grid=str(sample.get("Spatial_Grid_ID",""))
            g=h[h["Spatial_Grid_Cell"].astype(str)==grid].sort_values("Timestamp")
            if len(g)>=int(meta.get("sequence_length",24)):
                seq=g["count"].tail(int(meta["sequence_length"])).to_numpy(dtype=float)
                seq=(seq-float(meta["count_mean"]))/float(meta["count_std"])
                lstm=tf.keras.models.load_model(lstm_path,compile=False)
                forecast=max(0.0,float(lstm.predict(seq.reshape(1,-1,1),verbose=0)[0][0])*float(meta["count_std"])+float(meta["count_mean"]))
        return {"category":top,"confidence":confidence,"next_hour_count":forecast}
    except Exception as e:
        return {"error":str(e)}

res=st.session_state.result
if res:
    if "error" in res:
        st.error(res["error"])
        st.info("Try a district or grid-cell name present in the dataset, or search by coordinates/live location.")
    else:
        rows=res["rows"]; summ=risk_summary(rows)
        st.subheader("Area assessment")
        st.caption(res["description"])
        a,b,c,d=st.columns(4)
        a.metric("Historical indicator", f'{summ["score"]}/100')
        b.metric("Recorded incidents in sample", f'{summ["count"]:,}')
        c.metric("Average severity index", f'{summ["severity"]:.2f}/5')
        d.metric("Most frequent category", summ["top"])
        st.progress(summ["score"]/100, text="Relative historical indicator (not a crime probability)")
        forecast=model_forecast(rows)
        if forecast and not forecast.get("error"):
            st.markdown("### Deep-learning model estimates")
            m1,m2=st.columns(2)
            m1.metric("CNN likely recorded category", forecast["category"])
            m1.caption(f"Model softmax score: {forecast['confidence']:.1%}; not a calibrated probability.")
            if forecast["next_hour_count"] is not None:
                m2.metric("LSTM next-hour incident-count estimate", f"{forecast['next_hour_count']:.1f}")
            else:
                m2.info("LSTM estimate unavailable for this grid: not enough sequence history.")
        else:
            st.info("Train the CNN/LSTM in CrimeLens_Train_CNN_LSTM.ipynb and upload the generated files into the repository's models/ folder to enable deep-learning estimates.")
        st.markdown("### Recorded crime-category mix")
        chart=summ["counts"].head(10)
        st.bar_chart(chart)
        st.markdown("### Nearby recorded incidents")
        show=rows.sort_values("_distance_km").head(12).copy()
        if "_distance_km" in show:
            show["Approx. distance (km)"]=show["_distance_km"].round(2)
        cols=[c for c in ["Timestamp","District","Spatial_Grid_ID","Location_Type","Crime_Category","Offense_Severity_Index","Approx. distance (km)"] if c in show.columns]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)
        st.caption("The closest area is selected from the uploaded dataset's coordinates. A nearby estimate may be less reliable when the closest records are far away.")
else:
    st.markdown("## Know the pattern before you travel")
    x,y,z=st.columns(3)
    x.markdown("### 🔎 Search an area\nFind a district, location type, or spatial grid recorded in the dataset.")
    y.markdown("### 📍 Use location\nUse coordinates or browser location to inspect the nearest recorded area.")
    z.markdown("### 📊 Understand patterns\nExplore recorded categories and severity indicators transparently.")
    st.markdown("### Dataset coverage")
    st.write(f"{len(df):,} rows • {df['District'].nunique()} districts • {df['Spatial_Grid_Cell'].nunique()} spatial grid id")
    st.caption("The CNN and LSTM are trained in the provided notebook. This web prototype currently presents historical-neighbourhood summaries; model artifacts can be added for a future model-backed forecast. It does not claim to predict actual individual crimes.")
