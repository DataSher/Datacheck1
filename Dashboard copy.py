import streamlit as st
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title(" :bar_chart: Turbine Health Index Monitoring Dashboard")
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
    #st.write(f"File uploaded: {filename}")
    
    # Load the file into a DataFrame
    if filename.endswith(".csv"):
        df = pd.read_csv(fl, encoding="ISO-8859-1")
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(fl)
    elif filename.endswith(".txt"):
        df = pd.read_csv(fl, delimiter="\t", encoding="ISO-8859-1")

    # Ensure 'Date' column is in datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("The uploaded file must contain a 'Date' column.")
    
    # Add the selected Station and Unit as columns to the dataframe
    df['Station'] = station
    df['Unit'] = unit
    
    # Filter DataFrame for selected station and unit only
    df_filtered = df[(df['Station'] == station) & (df['Unit'] == unit)]
    
    # Display filtered data for the selected station and unit
    #st.write(f"Showing data for {station} - {unit}")
 

    # Date filter inputs
    start_date = df_filtered['Date'].min().date()  # Minimum date in the filtered DataFrame
    end_date = df_filtered['Date'].max().date()    # Maximum date in the filtered DataFrame

    col3, col4 = st.columns(2)
    with col3:
        selected_start_date = st.date_input("Start Date", start_date)
    with col4:
        selected_end_date = st.date_input("End Date", end_date)

    # Filter the DataFrame based on selected dates
    df_filtered = df_filtered[(df_filtered['Date'] >= pd.to_datetime(selected_start_date)) & 
                              (df_filtered['Date'] <= pd.to_datetime(selected_end_date))]

    # Display the filtered data after applying date filter
    #st.write(f"Filtered data from {selected_start_date} to {selected_end_date}:")


    # Create a dictionary for bearings based on the selected station and unit
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

    # Filter bearing data for the selected station and unit
    selected_bearing_key = f"{station} {unit}"
    if selected_bearing_key in bearing_data:
        bearings = bearing_data[selected_bearing_key]
    else:
        st.error(f"No bearing data available for {station} - {unit}")
        bearings = []

    # Creating a dataframe for the health index table
    bearing_health_index = []

    # Calculate the health index for each bearing
    for bearing in bearings:
        # Use appropriate column format from the dataset for bearing temperatures
        drain_temp_column = f"Drain {bearing}"
        metal_temp_column = f"METAL TEMP {bearing}"

        if drain_temp_column in df_filtered.columns and metal_temp_column in df_filtered.columns:
            # Mean values of both temperatures
            drain_temp_value = df_filtered[drain_temp_column].mean()
            metal_temp_value = df_filtered[metal_temp_column].mean()

            # Calculate drain health index
            drain_health_index = 0.0196 * drain_temp_value**2 - 2.0701 * drain_temp_value + 54.671
            drain_health_index = max(0, min(10, drain_health_index))  # Limit between 0 and 10
            
            # Calculate metal temperature health index
            metal_health_index = 0.0066 * metal_temp_value**2 - 0.8211 * metal_temp_value + 26.614
            metal_health_index = max(0, min(10, metal_health_index))  # Limit between 0 and 10
            
            # Initialize vibration indexes
            vibration_health_index_x = 0
            vibration_health_index_y = 0
            pedestal_health_index = 0

            # Check if it is Bearing 9
            if bearing == 9:
                # Only consider drain and metal temp for Bearing 9
                weighted_health_index = 100/((3 * metal_health_index + 1 * drain_health_index) / 4)
                weighted_health_index = max(1,min(100,weighted_health_index))  # Weight 3 for metal temp, 1 for drain temp
            else:
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

                # Calculate the overall health index for bearings 1-8
                weighted_health_index = 100/((3 * metal_health_index + 4 * vibration_health_index_x + 
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

        # Create a DataFrame for the health index results
        health_index_df = pd.DataFrame(bearing_health_index)

    # Calculate overall bearing health index as the average of the health index
    overall_health_index = health_index_df['Overall Health Index'].mean()

    # Create a new DataFrame to display only Bearing No and Health Index
    simplified_health_index_df = health_index_df[['Bearing', 'Overall Health Index']].copy()
    simplified_health_index_df.columns = ['Bearing No', 'Health Index']

    # Insert the overall health index at the top
    overall_row = pd.DataFrame(data={'Bearing No': ['Overall Bearing Health Index'], 
                                    'Health Index': [overall_health_index]})
    simplified_health_index_df = pd.concat([overall_row, simplified_health_index_df], ignore_index=True)

    # Display the simplified health index DataFrame
    st.subheader("Turbine Bearing Health Index")
    st.write(simplified_health_index_df.style.background_gradient(cmap="Blues_r"))

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
            drain_temp_avg = df[f'Drain {bearing}'].mean() if f'Drain {bearing}' in df.columns else None
            metal_temp_avg = df[f'METAL TEMP {bearing}'].mean() if f'METAL TEMP {bearing}' in df.columns else None
            vibration_x_avg = df[f'SHAFT VIB X {bearing}'].mean() if f'SHAFT VIB X {bearing}' in df.columns else None
            vibration_y_avg = df[f'SHAFT VIB Y {bearing}'].mean() if f'SHAFT VIB Y {bearing}' in df.columns else None
            vibration_ped_avg = df[f'HOUSING VIB {bearing}'].mean() if f'HOUSING VIB {bearing}' in df.columns else None

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
        st.write(detailed_health_index_df.style.background_gradient(cmap="Blues"))
        

