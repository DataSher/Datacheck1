import streamlit as st
import pandas as pd
import os

# Define the directory to save files
save_directory = "saved_data"

# Ensure the save directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Initialize uploaded_data as a dictionary to store data by station and unit
uploaded_data = {}

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
    if filename.endswith(".csv"):
        df_new = pd.read_csv(fl, encoding="ISO-8859-1")
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df_new = pd.read_excel(fl)
    elif filename.endswith(".txt"):
        df_new = pd.read_csv(fl, delimiter="\t", encoding="ISO-8859-1")

    # Ensure 'Date' column is in datetime format
    if 'Date' in df_new.columns:
        df_new['Date'] = pd.to_datetime(df_new['Date'])
    else:
        st.error("The uploaded file must contain a 'Date' column.")
        st.stop()  # Stop further execution if there's an error

    # Add the selected Station and Unit as columns to the dataframe
    df_new['Station'] = station
    df_new['Unit'] = unit

    # Generate a filename based on the station and unit
    save_filename = f"{station}_{unit}_data.csv"
    save_filepath = os.path.join(save_directory, save_filename)

    if os.path.exists(save_filepath):
        # If the file exists, load the existing data
        df_existing = pd.read_csv(save_filepath, encoding="ISO-8859-1")
        df_existing['Date'] = pd.to_datetime(df_existing['Date'])
        
        # Find new rows that are not in the existing file
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['Date'], keep='last')

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
                    weighted_health_index = max(1, min(100, weighted_health_index))
                else:
                    # Calculate vibration health indexes based on bearing range
                    # [The same code you had for vibrations should be placed here]

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
        
        # Calculate and display overall health index
        overall_health_index = health_index_df['Overall Health Index'].mean()
        simplified_health_index_df = health_index_df[['Bearing', 'Overall Health Index']].copy()
        simplified_health_index_df.columns = ['Bearing No', 'Health Index']

        overall_row = pd.DataFrame(data={'Bearing No': ['Overall Bearing Health Index'], 
                                          'Health Index': [overall_health_index]})
        simplified_health_index_df = pd.concat([overall_row, simplified_health_index_df], ignore_index=True)

        # Displaying the results
        st.subheader("Turbine Health Index")
        st.dataframe(simplified_health_index_df)

        # Download option
        csv = simplified_health_index_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Health Index CSV", csv, "health_index.csv", "text/csv")
else:
    st.warning("No data available for the selected station and unit. Please upload data.")
