import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt



RAW_DATA_DIR = "data/raw_stations/"
LOCATION_FILE = "data/PCP_location.txt"
TARGET_STATION = "8625_2250"
NUM_NEIGHBORS = 5


def read_station_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    start_date = lines[0].strip()
    rainfall_values = [float(x.strip()) for x in lines[1:] if x.strip() != '']
    date_range = pd.date_range(start=pd.to_datetime(start_date, format="%Y%m%d"), periods=len(rainfall_values))
    return pd.Series(rainfall_values, index=date_range)

def load_all_stations(raw_dir, max_files=None):
    data = {}
    station_files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    
    if max_files is not None:
        station_files = station_files[:max_files]
    
    print(f"🗃️ Found {len(station_files)} station files. Reading...\n")

    for i, fname in enumerate(station_files):
        station_id = fname.replace(".txt", "")
        filepath = os.path.join(raw_dir, fname)
        
        try:
            s = read_station_file(filepath)
            data[station_id] = s
        except Exception as e:
            print(f"❌ Error reading {fname}: {e}")
        
        if (i + 1) % 25 == 0 or i == len(station_files) - 1:
            print(f"✅ Processed {i+1}/{len(station_files)} files")

    return pd.DataFrame(data)

def load_station_locations(filepath):
    df = pd.read_csv(filepath)
    df["NAME"] = df["NAME"].astype(str)

    
    df.rename(columns={"LAT": "LONG", "LONG": "LAT"}, inplace=True)

    return df.set_index("NAME")[["LAT", "LONG"]]  




def is_valid_coord(lat, lon):
    return -90 <= lat <= 90 and -180 <= lon <= 180



def get_nearest_stations(target_id, location_df, rainfall_df, k=5):
    if target_id not in location_df.index:
        raise ValueError(f"{target_id} not in location file")

    if "LAT" not in location_df.columns or "LONG" not in location_df.columns:
        raise ValueError("Expected columns 'LAT' and 'LONG' not found in location_df")

    target_lat = location_df.loc[target_id]["LAT"]
    target_long = location_df.loc[target_id]["LONG"]
    print(f"\n🧭 Target Station Coordinates for {target_id}: LAT={target_lat}, LONG={target_long}")
    
    target_coord = (target_lat, target_long)

    
    valid_stations = []
    for name, row in location_df.iterrows():
        lat, lon = row["LAT"], row["LONG"]
        if is_valid_coord(lat, lon):
            valid_stations.append(name)
        else:
            print(f"❌ Invalid coordinates for station {name}: LAT={lat}, LONG={lon}")
    
    location_df = location_df.loc[valid_stations]
    
    
    distances = location_df.apply(
        lambda row: geodesic(target_coord, (row["LAT"], row["LONG"])).km, axis=1
    )

    nearest = distances.nsmallest(k + 1)  
    all_neighbors = [station for station in nearest.index if station != target_id]

    
    valid_neighbors = [station for station in all_neighbors if station in rainfall_df.columns]
    excluded = [s for s in all_neighbors if s not in rainfall_df.columns]

    if excluded:
        print(f"⚠️ Excluding {len(excluded)} neighbor(s) not in rainfall data: {excluded}")
    
    if not valid_neighbors:
        print(f"❌ No valid neighbors found for station {target_id} in rainfall DataFrame.")
    
    return valid_neighbors[:k]




def prepare_dataset(df, target_station, neighbors):
   
    print("Available columns in DataFrame:", df.columns)

   
    valid_neighbors = [neighbor for neighbor in neighbors if neighbor in df.columns]
    
    
    print("Valid neighbors:", valid_neighbors)

    
    if not valid_neighbors:
        raise ValueError(f"No valid neighbors found in the DataFrame. Available columns: {df.columns}")
    
   
    if target_station not in df.columns:
        raise ValueError(f"Target station '{target_station}' is not found in the DataFrame.")
    
   
    valid_columns = [target_station] + valid_neighbors
    
    
    print("Columns used for dropna:", valid_columns)
    
    
    df = df.dropna(subset=valid_columns) 
    
   
    print(f"Rows remaining after dropping missing values: {len(df)}")

   
    X = df[valid_neighbors]
    y = df[target_station]

    
    print("First few rows of the dataset (X and y):")
    print(X.head())
    print(y.head())

    return X, y



def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X_test)

    importances = model.feature_importances_
    features = X.columns

    plt.figure(figsize=(10, 5))
    plt.barh(features, importances)
    plt.xlabel('Feature Importance')
    plt.title('Station-wise Importance in Rainfall Prediction')
    plt.tight_layout()
    plt.show()

    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

    return model

def main():
    print("📦 Loading data...")
    all_data = load_all_stations(RAW_DATA_DIR)
    station_locations = load_station_locations(LOCATION_FILE)

    print("📍 Finding nearest stations...")
    neighbors = get_nearest_stations(TARGET_STATION, station_locations, all_data, NUM_NEIGHBORS)

    print(f"Selected neighbors: {neighbors}")

    print("🧹 Preparing dataset...")
    X, y = prepare_dataset(all_data, TARGET_STATION, neighbors) 

    print("📈 Training model...")
    model = train_and_evaluate(X, y)

   
    example_input = X.iloc[0]
    predicted = model.predict([example_input])[0]
    print(f"\nExample prediction:")
    print(f"Date: {example_input.name.date()}, Actual: {y.iloc[0]}, Predicted: {predicted:.2f}")

if __name__ == "__main__":
    main()
