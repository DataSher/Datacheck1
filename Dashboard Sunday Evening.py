import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
from iapws import IAPWS97
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam



# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title(" :bar_chart: Turbine Health Index Monitoring Dashboard- AM")

# Define the directory to save files
save_directory = "saved_data"

# Ensure the save directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Initialize uploaded_data as a dictionary to store data by station and unit
uploaded_data = {}

# Check if files for each station/unit combination exist
stations_units = [("BBGS", "Unit 1"), ("BBGS", "Unit 2"), ("BBGS", "Unit 3"),
                  ("SGS", "Unit 1"), ("SGS", "Unit 2"),
                  ("DIL", "Unit 1"), ("DIL", "Unit 2"),
                  ("HEL", "Unit 1"), ("HEL", "Unit 2")]

# Load existing data into uploaded_data
for station, unit in stations_units:
    file_path = os.path.join(save_directory, f"{station}_{unit}_data.csv")
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path, encoding="ISO-8859-1")
        df_existing['Date'] = pd.to_datetime(df_existing['Date'])
        uploaded_data[f"{station}_{unit}"] = df_existing

# Selection of Station and Unit
col1, col2 = st.columns(2)

with col1:
    station = st.selectbox("Select Station", ["BBGS", "SGS", "DIL", "HEL"])

with col2:
    if station == "BBGS":
        unit = st.selectbox("Select Unit", ["Unit 1", "Unit 2", "Unit 3"])
    else:
        unit = st.selectbox("Select Unit", ["Unit 1", "Unit 2"])

# File uploader
fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "txt", "xlsx", "xls"])

if fl is not None:
    filename = fl.name

    # Load the uploaded file into a DataFrame
    try:
        if filename.endswith(".csv"):
            df_new = pd.read_csv(fl, encoding="ISO-8859-1")
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df_new = pd.read_excel(fl)
        elif filename.endswith(".txt"):
            df_new = pd.read_csv(fl, delimiter="\t", encoding="ISO-8859-1")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Ensure 'Date' column is in datetime format
    if 'Date' in df_new.columns:
        df_new['Date'] = pd.to_datetime(df_new['Date'], errors='coerce')
        if df_new['Date'].isnull().all():
            st.error("The 'Date' column contains no valid dates.")
            st.stop()
    else:
        st.error("The uploaded file must contain a 'Date' column.")
        st.stop()  # Stop further execution if there's an error

    # Add the selected Station and Unit as columns to the dataframe
    df_new['Station'] = station
    df_new['Unit'] = unit

    # Generate a filename based on the station and unit
    save_filename = f"{station}_{unit}_data.csv"
    save_filepath = os.path.join(save_directory, save_filename)

    # If the file exists, load the existing data and append new data
    if os.path.exists(save_filepath):
        df_existing = pd.read_csv(save_filepath, encoding="ISO-8859-1")
        df_existing['Date'] = pd.to_datetime(df_existing['Date'])

        # Combine existing and new data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['Date'], keep='last')
        df_combined.to_csv(save_filepath, index=False, encoding="ISO-8859-1")
        st.success(f"Data successfully appended to {save_filename}.")
        uploaded_data[f"{station}_{unit}"] = df_combined
    else:
        df_new.to_csv(save_filepath, index=False, encoding="ISO-8859-1")
        st.success(f"Data successfully saved as {save_filename}.")
        uploaded_data[f"{station}_{unit}"] = df_new

# Analysis section
col3, col4 = st.columns(2)
with col3:
    selected_station = st.selectbox("Select Station for Analysis", ["BBGS", "SGS", "DIL", "HEL"])
with col4:
    selected_unit = st.selectbox("Select Unit for Analysis", ["Unit 1", "Unit 2", "Unit 3"] if selected_station == "BBGS" else ["Unit 1", "Unit 2"])

selected_key = f"{selected_station}_{selected_unit}"  # Consistent key format

# Check if any data has been uploaded or loaded
if uploaded_data:
    # Ensure selected key exists in uploaded_data
    if selected_key in uploaded_data:
        df_filtered = uploaded_data[selected_key]
        
        # Date filter inputs
        start_date = df_filtered['Date'].min().date()
        end_date = df_filtered['Date'].max().date()

        col5, col6 = st.columns(2)
        with col5:
            selected_start_date = st.date_input("Start Date", start_date)
        with col6:
            selected_end_date = st.date_input("End Date", end_date)

        # Filter the DataFrame based on selected dates
        df_filtered = df_filtered[(df_filtered['Date'] >= pd.to_datetime(selected_start_date)) & 
                                  (df_filtered['Date'] <= pd.to_datetime(selected_end_date))]

        if df_filtered.empty:
            st.warning("No data available for the selected date range.")
        else:
            # Proceed with calculations for bearing health index

            bearing_data = {
                'BBGS Unit 1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                'BBGS Unit 2': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                'BBGS Unit 3': [1, 2, 3, 4, 5, 6, 7],
                'SGS Unit 1': [1, 2, 3, 4],
                'SGS Unit 2': [1, 2, 3, 4],
                'DIL Unit 1': [1, 2, 3, 4, 5, 6, 7],
                'DIL Unit 2': [1, 2, 3, 4, 5, 6, 7],
                'HEL Unit 1': [1, 2, 3, 4, 5, 6, 7],
                'HEL Unit 2': [1, 2, 3, 4, 5, 6, 7]
            }

            selected_bearing_key = f"{selected_station} {selected_unit}"
            bearings = bearing_data.get(selected_bearing_key, [])
            
            # Create a dataframe for the health index table
bearing_health_index = []

for bearing in bearings:
    drain_temp_column = f"Drain {bearing}"
    metal_temp_column = f"METAL TEMP {bearing}"
    
    # Check if both columns are in the filtered DataFrame
    if drain_temp_column in df_filtered.columns and metal_temp_column in df_filtered.columns:
        drain_temp_value = df_filtered[drain_temp_column].mean()
        metal_temp_value = df_filtered[metal_temp_column].mean()

        # Calculate health indices
        drain_health_index = max(0, min(10, 0.0196 * drain_temp_value**2 - 2.0701 * drain_temp_value + 54.671))
        metal_health_index = max(0, min(10, 0.0066 * metal_temp_value**2 - 0.8211 * metal_temp_value + 26.614))
        
        # Initialize vibration indexes
        vibration_health_index_x = 0
        vibration_health_index_y = 0
        pedestal_health_index = 0

        # Check if it is Bearing 9
        if bearing == 9:
            weighted_health_index = 100 / ((3 * metal_health_index + 1 * drain_health_index) / 4)
            weighted_health_index = max(1, min(100.0, weighted_health_index))
        else:
            # Calculate vibration health indexes based on bearing range
           # Calculate vibration health indexes based on bearing range
                if bearing <= 4:
                    # For bearings 1 to 4
                    shaft_vib_x_column = f"SHAFT VIB X {bearing}"
                    shaft_vib_y_column = f"SHAFT VIB Y {bearing}"
                    housing_vib_column = f"HOUSING VIB {bearing}"

                    if shaft_vib_x_column in df_filtered.columns:
                        shaft_vib_x_value = df_filtered[shaft_vib_x_column].mean()
                        vibration_health_index_x = 0.0001 * shaft_vib_x_value**2 + 0.0478 * shaft_vib_x_value
                        vibration_health_index_x = max(0, min(10, vibration_health_index_x))

                    if shaft_vib_y_column in df_filtered.columns:
                        shaft_vib_y_value = df_filtered[shaft_vib_y_column].mean()
                        vibration_health_index_y = 0.0001 * shaft_vib_y_value**2 + 0.0478 * shaft_vib_y_value
                        vibration_health_index_y = max(0, min(10, vibration_health_index_y))

                    if housing_vib_column in df_filtered.columns:
                        housing_vib_value = df_filtered[housing_vib_column].mean()
                        pedestal_health_index = 0.0426 * housing_vib_value**2 + 0.7936 * housing_vib_value + 0.1713
                        pedestal_health_index = max(0, min(10, pedestal_health_index))

                else:
                    # For bearings 5 to 8
                    shaft_vib_x_column = f"SHAFT VIB X {bearing}"
                    shaft_vib_y_column = f"SHAFT VIB Y {bearing}"
                    housing_vib_column = f"HOUSING VIB {bearing}"

                    if shaft_vib_x_column in df_filtered.columns:
                        shaft_vib_x_value = df_filtered[shaft_vib_x_column].mean()
                        vibration_health_index_x = 0.0003 * shaft_vib_x_value**2 + 0.0056 * shaft_vib_x_value - 0.15
                        vibration_health_index_x = max(0, min(10, vibration_health_index_x))

                    if shaft_vib_y_column in df_filtered.columns:
                        shaft_vib_y_value = df_filtered[shaft_vib_y_column].mean()
                        vibration_health_index_y = 0.0003 * shaft_vib_y_value**2 + 0.0056 * shaft_vib_y_value - 0.15
                        vibration_health_index_y = max(0, min(10, vibration_health_index_y))

                    if housing_vib_column in df_filtered.columns:
                        housing_vib_value = df_filtered[housing_vib_column].mean()
                        pedestal_health_index = 0.0624 * housing_vib_value**2 + 0.4793 * housing_vib_value
                        pedestal_health_index = max(0, min(10, pedestal_health_index))

                        
            # Overall health index calculation here...
                weighted_health_index = 100 / ((3 * metal_health_index + 4 * vibration_health_index_x + 
                                             4 * vibration_health_index_y + 2 * pedestal_health_index + 2 * drain_health_index) / 15)

        # Append to the health index list
        bearing_health_index.append({
            'Bearing': bearing,
            'Drain Temp Index': drain_health_index,
            'Metal Temp Index': metal_health_index,
            'Vibration Index X': vibration_health_index_x,
            'Vibration Index Y': vibration_health_index_y,
            'Pedestal Index': pedestal_health_index,
            'Overall Health Index': weighted_health_index
        })

# Creating the health index DataFrame
health_index_df = pd.DataFrame(bearing_health_index)

# Check if the DataFrame is empty before trying to calculate the overall health index
if not health_index_df.empty:
    # Calculate and display overall health index
    overall_health_index = health_index_df['Overall Health Index'].mean()
    simplified_health_index_df = health_index_df[['Bearing', 'Overall Health Index']].copy()
    simplified_health_index_df.columns = ['Bearing No', 'Health Index']

    overall_row = pd.DataFrame(data={'Bearing No': ['Overall Bearing Health Index'], 
                                      'Health Index': [overall_health_index]})
    simplified_health_index_df = pd.concat([overall_row, simplified_health_index_df], ignore_index=True)
    col7,col8,col9 = st.columns(3)

    with col7:

        # Displaying the results
        st.subheader("Turbine Health Index")
        st.write(simplified_health_index_df.style.background_gradient(cmap="Blues_r"))
        # Download option
        csv = simplified_health_index_df.to_csv(index=False).encode('utf-8')
        #st.download_button("Download Health Index CSV", csv, "health_index.csv", "text/csv")
else:
    st.warning("No valid bearings to calculate the health index.")


 # Link to show detailed breakdown


if st.button("Show Detailed Breakdown"):
        st.subheader("Detailed Breakdown of Bearing Health Index")
        
        # Display detailed health index values with actual values before conversion
        detailed_health_index_df = pd.DataFrame({
            'Bearing': [bearing['Bearing'] for bearing in bearing_health_index],
            'Drain Temp Index': [bearing['Drain Temp Index'] for bearing in bearing_health_index],
            'Metal Temp Index': [bearing['Metal Temp Index'] for bearing in bearing_health_index],
            'Vibration Index X': [bearing['Vibration Index X'] for bearing in bearing_health_index],
            'Vibration Index Y': [bearing['Vibration Index Y'] for bearing in bearing_health_index],
            'Pedestal Index': [bearing['Pedestal Index'] for bearing in bearing_health_index],
            'Overall Health Index': [bearing['Overall Health Index'] for bearing in bearing_health_index]
        })
        # Create a DataFrame for the actual values from the uploaded data
        # Assuming 'df' is your main DataFrame containing actual values
    # Replace 'df' with the name of your existing DataFrame

    # Create a list to hold actual values
        actual_values = []

        for bearing in range(1, 10):  # Assuming 1-9 for your bearings
        # Retrieve average values directly from your DataFrame
            drain_temp_avg = df_filtered[f'Drain {bearing}'].mean() if f'Drain {bearing}' in df_filtered.columns else None
            metal_temp_avg = df_filtered[f'METAL TEMP {bearing}'].mean() if f'METAL TEMP {bearing}' in df_filtered.columns else None
            vibration_x_avg = df_filtered[f'SHAFT VIB X {bearing}'].mean() if f'SHAFT VIB X {bearing}' in df_filtered.columns else None
            vibration_y_avg = df_filtered[f'SHAFT VIB Y {bearing}'].mean() if f'SHAFT VIB Y {bearing}' in df_filtered.columns else None
            vibration_ped_avg = df_filtered[f'HOUSING VIB {bearing}'].mean() if f'HOUSING VIB {bearing}' in df_filtered.columns else None

        # Append actual values
            actual_values.append({
            'Bearing': bearing,
            'Drain Oil Avg': drain_temp_avg,
            'Metal Temp Avg': metal_temp_avg,
            'Vibration X Avg': vibration_x_avg,
            'Vibration Y Avg': vibration_y_avg,
            'Vibration Ped Avg': vibration_ped_avg
        })

    # Create a DataFrame for the actual values
            actual_values_df = pd.DataFrame(actual_values)

    # Create a DataFrame for the health index values (as before)
            detailed_health_index_df = pd.DataFrame({
        'Bearing': [bearing['Bearing'] for bearing in bearing_health_index],
        'Overall Health Index': [bearing['Overall Health Index'] for bearing in bearing_health_index]
    })

    # Merge with actual values DataFrame
            detailed_health_index_df = detailed_health_index_df.merge(actual_values_df, on='Bearing', how='left')

    # Display the table with reversed color gradient
        st.write(detailed_health_index_df.style.background_gradient(cmap="Blues_r"))
        
# Cylinder Module part: First Calculate the Efficiency from uploaded data
# After calculation of efficiency, need to compare with Design

# Initialize the XSteam object with MKS unit system
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

# Function to calculate efficiency
def calculate_efficiency(inlet_pressure, inlet_temp, outlet_pressure, outlet_temp):
    # Convert pressures from kg/cm² gauge to bar absolute
    inlet_pressure_bar_abs = (inlet_pressure + 1.01325) * 1.0  # Convert from kg/cm² gauge to bar absolute
    outlet_pressure_bar_abs = (outlet_pressure + 1.01325) * 1.0  # Convert from kg/cm² gauge to bar absolute

    # Get inlet properties
    inlet_h = steamTable.h_pt(inlet_pressure_bar_abs, inlet_temp)  # Enthalpy at inlet pressure and temperature
    inlet_s = steamTable.s_pt(inlet_pressure_bar_abs, inlet_temp)  # Entropy at inlet pressure and temperature

    # Get outlet properties
    outlet_h = steamTable.h_pt(outlet_pressure_bar_abs, outlet_temp)  # Enthalpy at outlet pressure and temperature

    # Actual enthalpy difference
    actual_enthalpy_drop = inlet_h - outlet_h

    # Isentropic efficiency calculation (isentropic process)
    isentropic_outlet_h = steamTable.h_ps(outlet_pressure_bar_abs, inlet_s)  # Isentropic process: constant entropy
    isentropic_enthalpy_drop = inlet_h - isentropic_outlet_h

    # Check for valid isentropic enthalpy drop to avoid division by zero
    if isentropic_enthalpy_drop == 0:
        raise ValueError("Isentropic enthalpy drop is zero. Efficiency cannot be calculated.")

    # Actual efficiency (in decimal form)
    efficiency = actual_enthalpy_drop / isentropic_enthalpy_drop
    return efficiency * 100  # Convert to percentage (0-100)

# Function to compute efficiencies for each cylinder section
def compute_section_efficiencies(df):
    efficiencies = {}

    # HP section (inlet: Main Steam, outlet: CRH)
    efficiencies['HP Cylinder'] = calculate_efficiency(df['Main Steam Pressure'], df['Main Steam Temperature'], 
                                                       df['CRH Pressure'], df['CRH Temperature'])

    # IP section (inlet: HRH, outlet: LP4)
    efficiencies['IP Cylinder'] = calculate_efficiency(df['HRH Pressure'], df['HRH Temperature'], 
                                                       df['LP4 Pressure'], df['LP4 Temperature'])

    # LP section (inlet: LP4, outlet: LP3)
    efficiencies['LP Cylinder till 3rd extraction'] = calculate_efficiency(df['LP4 Pressure'], df['LP4 Temperature'], 
                                                                           df['LP3 Pressure'], df['LP3 Temperature'])

    return efficiencies

# Calculate actual efficiencies for each row
efficiency_results = []
for index, row in df_filtered.iterrows():
    efficiencies = compute_section_efficiencies(row)
    efficiencies['Timestamp'] = row['Date']  # Add the timestamp for each row
    efficiencies['Load'] = row['MW']  # Assuming 'Load' is the column name in your DataFrame
    efficiency_results.append(efficiencies)

# Create DataFrame with date and load as first two columns
efficiency_df = pd.DataFrame(efficiency_results)

# Reorder columns to have 'Timestamp' and 'Load' at the front
efficiency_df = efficiency_df[['Timestamp', 'Load', 'HP Cylinder', 'IP Cylinder', 'LP Cylinder till 3rd extraction']]

# Format efficiency values to 2 decimal places and convert to percentage strings
efficiency_df['HP Cylinder'] = efficiency_df['HP Cylinder'].apply(lambda x: f"{x:.2f}%")
efficiency_df['IP Cylinder'] = efficiency_df['IP Cylinder'].apply(lambda x: f"{x:.2f}%")
efficiency_df['LP Cylinder till 3rd extraction'] = efficiency_df['LP Cylinder till 3rd extraction'].apply(lambda x: f"{x:.2f}%")

# Show only the last three rows
efficiency_df_last_three = efficiency_df.head(3)



# Load design data from CSV
design_data_path = 'Design Data.csv'
design_df = pd.read_csv(design_data_path)

# Calculate design efficiencies using the same logic
design_efficiency_results = []
for index, row in design_df.iterrows():
    efficiencies = compute_section_efficiencies(row)
    efficiencies['Load'] = row['MW']  # Assuming 'Load' is the column name in design data
    design_efficiency_results.append(efficiencies)

design_efficiency_df = pd.DataFrame(design_efficiency_results)

# Plot efficiency vs load curve for all three cylinders
def plot_efficiency_vs_load():
    plt.figure(figsize=(10, 6))

    # Actual efficiency vs Load
    plt.plot(efficiency_df['Load'], efficiency_df['HP Cylinder'].apply(lambda x: float(x.strip('%'))), label='HP Cylinder Actual', marker='o')
    plt.plot(efficiency_df['Load'], efficiency_df['IP Cylinder'].apply(lambda x: float(x.strip('%'))), label='IP Cylinder Actual', marker='o')
    plt.plot(efficiency_df['Load'], efficiency_df['LP Cylinder till 3rd extraction'].apply(lambda x: float(x.strip('%'))), label='LP Cylinder Actual', marker='o')

    # Design efficiency vs Load
    plt.plot(design_efficiency_df['Load'], design_efficiency_df['HP Cylinder'], label='HP Cylinder Design', linestyle='--', marker='x')
    plt.plot(design_efficiency_df['Load'], design_efficiency_df['IP Cylinder'], label='IP Cylinder Design', linestyle='--', marker='x')
    plt.plot(design_efficiency_df['Load'], design_efficiency_df['LP Cylinder till 3rd extraction'], label='LP Cylinder Design', linestyle='--', marker='x')

    plt.xlabel('Load (MW)')
    plt.ylabel('Efficiency (%)')
    plt.title('Efficiency vs Load Curve for HP, IP, and LP Cylinders')
    plt.legend()
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)

# Button to plot efficiency vs load curve
with col8:
    # Display turbine efficiencies (only last three rows)
    st.subheader("Actual Turbine Efficiencies")
    st.dataframe(efficiency_df_last_three)

    # Download option for efficiencies
    csv_efficiency = efficiency_df_last_three.to_csv(index=False).encode('utf-8')
    #st.download_button("Download Efficiency Data", csv_efficiency, "efficiency_data.csv", "text/csv")
    if st.button('Show Efficiency vs Load Curve'):
        plot_efficiency_vs_load()


