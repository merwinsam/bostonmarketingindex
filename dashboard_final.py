#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import os
from math import radians, sin, cos, sqrt, atan2

# Streamlit page config
st.set_page_config(page_title="Boston Retail and Restaurant Interactive Dashboard", layout="wide")

# Haversine function to calculate distance between two points (in km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Load and clean Yelp data (restaurants and retailers)
@st.cache_data
def load_yelp_data():
    file_path = "yelp_boston.csv"
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['name', 'latitude', 'longitude'])
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
        df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce').fillna(0)
        df['neighborhood'] = df['neighborhood'].fillna('Unknown').str.strip()
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df['cuisines'] = 'Unknown'
        df['price_range'] = pd.NA
        df['categories'] = df.get('categories', 'Unknown').fillna('Unknown').astype(str).str.strip()
        return df
    except FileNotFoundError:
        st.error("Error: yelp_boston.csv not found")
        return pd.DataFrame()

# Load and clean data_merged.csv (restaurants only)
@st.cache_data
def load_merged_data():
    file_path = "data_merged.csv"
    try:
        df = pd.read_csv(file_path)
        df = df[df['error'] == False]
        df = df[df['latitude'] != 0]
        df['name'] = df['name_restaurant'].str.strip()
        df['review_count'] = pd.to_numeric(df['amount_ratings'], errors='coerce').fillna(0)
        review_cols = [
            'amount_ratings_excellent', 'amount_ratings_vgood', 
            'amount_ratings_average', 'amount_ratings_poor', 
            'amount_ratings_terrible'
        ]
        for col in review_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['rating'] = (
            5 * df['amount_ratings_excellent'] +
            4 * df['amount_ratings_vgood'] +
            3 * df['amount_ratings_average'] +
            2 * df['amount_ratings_poor'] +
            1 * df['amount_ratings_terrible']
        ) / df['amount_ratings'].replace(0, 1)
        df['rating'] = df['rating'].fillna(0).round(2)
        df['cuisines'] = df[['category_1', 'category_2', 'category_3']].apply(
            lambda x: ', '.join([str(c) for c in x if pd.notnull(c) and c != '0']), axis=1
        ).replace('', 'Unknown')
        df = df[['name', 'rating', 'review_count', 'latitude', 'longitude', 'cuisines', 'price_range']]
        df['neighborhood'] = 'Unknown'
        df['categories'] = 'Restaurants'
        df = df[(df['name'].notnull()) & (df['name'] != '') &
                (df['rating'] > 0) & (df['review_count'] > 0) &
                (df['price_range'].notnull()) &
                (df['cuisines'].notnull()) & (df['cuisines'] != 'Unknown')]
        return df
    except FileNotFoundError:
        st.error("Error: data_merged.csv not found.")
        return pd.DataFrame()

# Merge datasets, assign neighborhoods, and handle retail
@st.cache_data
def merge_datasets():
    yelp_df = load_yelp_data()
    merged_df = load_merged_data()
    
    if yelp_df.empty and merged_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    yelp_restaurants = yelp_df[yelp_df['categories'].str.contains('Restaurants', case=False, na=False)].copy()
    yelp_retail = yelp_df[yelp_df['categories'].str.contains('Grocery|Delis|Convenience Stores', case=False, na=False)].copy()
    
    yelp_restaurants = yelp_restaurants[
        (yelp_restaurants['name'].notnull()) & (yelp_restaurants['name'] != '') &
        (yelp_restaurants['rating'] > 0) & (yelp_restaurants['review_count'] > 0)
    ]
    
    if not merged_df.empty and not yelp_restaurants.empty:
        merged_df = merged_df[~merged_df['name'].str.lower().isin(yelp_restaurants['name'].str.lower())]
    
    combined_restaurants = pd.concat([
        yelp_restaurants[['name', 'rating', 'review_count', 'neighborhood', 'latitude', 'longitude', 'cuisines', 'price_range', 'categories']],
        merged_df[['name', 'rating', 'review_count', 'neighborhood', 'latitude', 'longitude', 'cuisines', 'price_range', 'categories']]
    ], ignore_index=True)
    
    yelp_known = yelp_df[(yelp_df['neighborhood'] != 'Unknown') & (yelp_df['latitude'] != 0) & (yelp_df['longitude'] != 0)]
    if not yelp_known.empty and not combined_restaurants.empty:
        for idx in combined_restaurants.index:
            if combined_restaurants.loc[idx, 'neighborhood'] == 'Unknown':
                lat, lon = combined_restaurants.loc[idx, 'latitude'], combined_restaurants.loc[idx, 'longitude']
                distances = yelp_known.apply(
                    lambda row: haversine(lat, lon, row['latitude'], row['longitude']),
                    axis=1
                )
                if not distances.empty:
                    nearest_idx = distances.idxmin()
                    combined_restaurants.loc[idx, 'neighborhood'] = yelp_known.loc[nearest_idx, 'neighborhood']
    
    combined_retail = yelp_retail[['name', 'rating', 'review_count', 'neighborhood', 'latitude', 'longitude', 'cuisines', 'price_range', 'categories']].copy()
    if not yelp_known.empty and not combined_retail.empty:
        for idx in combined_retail.index:
            if combined_retail.loc[idx, 'neighborhood'] == 'Unknown':
                lat, lon = combined_retail.loc[idx, 'latitude'], combined_retail.loc[idx, 'longitude']
                distances = yelp_known.apply(
                    lambda row: haversine(lat, lon, row['latitude'], row['longitude']),
                    axis=1
                )
                if not distances.empty:
                    nearest_idx = distances.idxmin()
                    combined_retail.loc[idx, 'neighborhood'] = yelp_known.loc[nearest_idx, 'neighborhood']
    
    return combined_restaurants, combined_retail, merged_df

# Main dashboard
st.title("Boston Retail and Restaurant Interactive Dashboard")

# Load and merge data
combined_restaurants, combined_retail, merged_df = merge_datasets()

# Compute Product Price Index (for restaurants)
@st.cache_data
def compute_price_index(df):
    if df.empty:
        return 50.0
    price_map = {'$': 1, '$$ - $$$': 2, '$$$$': 3}
    df['price_numeric'] = df['price_range'].map(price_map).fillna(2)
    mean_price = df['price_numeric'].mean()
    price_index = (mean_price - 1) / 2 * 100
    return round(price_index, 2)

# Hardcoded Census data
census_data = {
    'median_household_income': 94755,
    'restaurant_establishments': len(combined_restaurants) if not combined_restaurants.empty else 0,
    'avg_employees_per_establishment': '20 - 99',
    'broadband_penetration': 543499.35  # 639,411 * 0.85
}

if combined_restaurants.empty and combined_retail.empty:
    st.error("No valid data loaded. Please check CSV files or ensure restaurants have complete name, rating, review count, price range, and cuisines, or retailers have available data.")
    st.stop()

# Filters
col1, col2, col3, col4 = st.columns(4)
with col1:
    business_type = st.selectbox("Business Type", ["Restaurants", "Retail (Grocery, Deli, Convenience)"])
with col2:
    if business_type == "Restaurants":
        names = ['All'] + sorted(combined_restaurants['name'].unique().tolist())
    else:
        names = ['All'] + sorted(combined_retail['name'].unique().tolist())
    selected_name = st.selectbox("Name", names)
with col3:
    if business_type == "Restaurants":
        neighborhoods = ['All'] + sorted(combined_restaurants['neighborhood'].unique().tolist())
    else:
        neighborhoods = ['All'] + sorted(combined_retail['neighborhood'].unique().tolist())
    selected_neighborhood = st.selectbox("Neighborhood", neighborhoods)
with col4:
    view = st.selectbox("View", ["Businesses (Micro)", "Neighborhoods (Macro)"])
    map_view = 'micro' if view == "Businesses (Micro)" else 'macro'

# Select data based on business type
if business_type == "Restaurants":
    combined_df = combined_restaurants
else:
    combined_df = combined_retail

# Linked filtering logic
filtered_df = combined_df.copy()
if selected_name != 'All':
    filtered_df = filtered_df[filtered_df['name'] == selected_name]
    if not filtered_df.empty:
        # Auto-select the neighborhood of the selected business
        selected_neighborhood = filtered_df['neighborhood'].iloc[0]
        st.session_state['selected_neighborhood'] = selected_neighborhood
elif selected_neighborhood != 'All':
    filtered_df = filtered_df[filtered_df['neighborhood'] == selected_neighborhood]
    # Update name dropdown to show only businesses in the selected neighborhood
    names = ['All'] + sorted(filtered_df['name'].unique().tolist())
    if selected_name not in names:
        selected_name = 'All'
        st.session_state['selected_name'] = 'All'

if map_view == 'micro' and selected_neighborhood != 'All':
    filtered_df = filtered_df[filtered_df['neighborhood'] == selected_neighborhood]
elif map_view == 'macro' and selected_neighborhood != 'All':
    filtered_df = filtered_df[filtered_df['neighborhood'] == selected_neighborhood]

# Neighborhood aggregates
neighborhood_agg = combined_df.groupby('neighborhood').agg({
    'rating': 'mean',
    'review_count': 'sum',
    'latitude': 'mean',
    'longitude': 'mean',
    'name': 'count'
}).reset_index()

if 'cuisines' in combined_df.columns:
    cuisine_agg = combined_df.groupby('neighborhood')['cuisines'].apply(
        lambda x: ', '.join(sorted(set(x.dropna())))
    ).reset_index()
    neighborhood_agg = neighborhood_agg.merge(cuisine_agg, on='neighborhood', how='left')
else:
    neighborhood_agg['cuisines'] = 'Unknown'

neighborhood_agg['rating'] = neighborhood_agg['rating'].round(2)
neighborhood_agg = neighborhood_agg.rename(columns={'name': 'business_count'})

if map_view == 'macro' and selected_neighborhood != 'All':
    neighborhood_agg = neighborhood_agg[neighborhood_agg['neighborhood'] == selected_neighborhood]

# Interesting fact
if not neighborhood_agg.empty and 'rating' in neighborhood_agg.columns and not neighborhood_agg['rating'].isna().all():
    highest_rated = neighborhood_agg.loc[neighborhood_agg['rating'].idxmax()]
    st.markdown(f"**Interesting Fact**: {highest_rated['neighborhood']} has the highest average rating ({highest_rated['rating']} stars).")
else:
    st.markdown("**Interesting Fact**: Limited neighborhood data available.")

# Plotly map
if map_view == 'micro':
    df_to_plot = filtered_df
    if business_type == "Restaurants":
        custom_data = ['name', 'neighborhood', 'rating', 'review_count', 'cuisines', 'price_range']
        hover_template = (
            "<b>Name of the Establishment</b>: %{customdata[0]}<br>"
            "<b>Location</b>: %{customdata[1]}<br>"
            "<b>Average rating by customers</b>: %{customdata[2]}<br>"
            "<b>No. of reviews by customers</b>: %{customdata[3]}<br>"
            "<b>Cuisines Served</b>: %{customdata[4]}<br>"
            "<b>Cost of Food</b>: %{customdata[5]}<extra></extra>"
        )
        title = "Restaurant Locations (Rating, Review Count, Cuisines, Price Range)"
    else:
        custom_data = ['name', 'neighborhood', 'rating', 'review_count']
        hover_template = (
            "<b>Name of the Establishment</b>: %{customdata[0]}<br>"
            "<b>Location</b>: %{customdata[1]}<br>"
            "<b>Average rating by customers</b>: %{customdata[2]}<br>"
            "<b>No. of reviews by customers</b>: %{customdata[3]}<extra></extra>"
        )
        title = "Retail Locations (Rating, Review Count)"
else:
    df_to_plot = neighborhood_agg
    custom_data = ['neighborhood', 'rating', 'review_count', 'business_count']
    hover_template = (
        "<b>Location</b>: %{customdata[0]}<br>"
        "<b>Average rating by customers</b>: %{customdata[1]}<br>"
        "<b>No. of reviews by customers</b>: %{customdata[2]}<br>"
        "<b>Business Count</b>: %{customdata[3]}<extra></extra>"
    )
    title = "Neighborhood Aggregates (Avg Rating, Total Reviews, Business Count)"

if not df_to_plot.empty and all(col in df_to_plot.columns for col in ['latitude', 'longitude', 'rating', 'review_count']):
    fig = px.scatter_mapbox(
        df_to_plot,
        lat='latitude',
        lon='longitude',
        size='review_count',
        color='rating',
        color_continuous_scale='Viridis',
        size_max=30,
        zoom=11,
        center={'lat': 42.3601, 'lon': -71.0589},
        mapbox_style='open-street-map',
        title=title,
        custom_data=custom_data
    )
    fig.update_traces(hovertemplate=hover_template)
    fig.update_layout(margin={'l': 0, 'r': 0, 't': 50, 'b': 0}, height=500)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Insufficient data to display on the map.")

# Indexes table
st.subheader("Marketing Indexes")
if business_type == "Restaurants":
    price_index = compute_price_index(merged_df if selected_neighborhood == 'All' else merged_df[merged_df['neighborhood'] == selected_neighborhood])
    if map_view == 'macro' and selected_neighborhood != 'All':
        # Macro view: Use neighborhood-specific metrics
        neighborhood_data = neighborhood_agg[neighborhood_agg['neighborhood'] == selected_neighborhood]
        avg_rating = neighborhood_data['rating'].iloc[0] if not neighborhood_data.empty else "N/A"
        avg_reviews = neighborhood_data['review_count'].iloc[0] / neighborhood_data['business_count'].iloc[0] if not neighborhood_data.empty and neighborhood_data['business_count'].iloc[0] > 0 else "N/A"
        business_count = neighborhood_data['business_count'].iloc[0] if not neighborhood_data.empty else 0
        indexes = [
            {
                "Index": "Market Need",
                "Metric": "Median Household Income",
                "Value": f"${census_data['median_household_income']:,}",
                "Calculation": "This is a fixed number ($94,755) from the 2023 American Community Survey, showing the middle income for households in Boston.",
                "Data Source": "https://data.census.gov/table/ACSDT5Y2023.B19013?q=B19013&g=040XX00US08_160XX00US2507000"
            },
            {
                "Index": "Market Size",
                "Metric": "Restaurant Establishments",
                "Value": f"{business_count}",
                "Calculation": f"Open yelp_boston.csv and data_merged.csv. Filter yelp_boston.csv for rows where 'categories' includes 'Restaurants'. Combine with all rows from data_merged.csv. Keep only rows where 'neighborhood' is '{selected_neighborhood}'. Count the unique 'name' entries.",
                "Data Source": "https://shorturl.at/1Es5F, https://shorturl.at/pYUfv"
            },
            {
                "Index": "Market Competition",
                "Metric": "Restaurants in Neighborhood",
                "Value": f"{business_count}",
                "Calculation": f"Same as Market Size: In yelp_boston.csv, take rows where 'categories' includes 'Restaurants'. Add all rows from data_merged.csv. Filter for 'neighborhood' = '{selected_neighborhood}'. Count unique 'name' entries.",
                "Data Source": "https://shorturl.at/1Es5F, https://shorturl.at/pYUfv"
            },
            {
                "Index": "Consumer Satisfaction",
                "Metric": "Average Rating",
                "Value": f"{avg_rating:.2f}" if isinstance(avg_rating, (int, float)) else avg_rating,
                "Calculation": f"In yelp_boston.csv, take rows where 'categories' includes 'Restaurants'. Add all rows from data_merged.csv. Filter for 'neighborhood' = '{selected_neighborhood}'. Take the 'rating' column, add all values, and divide by the number of restaurants. Round to 2 decimals.",
                "Data Source": "https://shorturl.at/1Es5F, https://shorturl.at/pYUfv"
            },
            {
                "Index": "Market Social Media",
                "Metric": "Average Review Count",
                "Value": f"{avg_reviews:.0f}" if isinstance(avg_reviews, (int, float)) else avg_reviews,
                "Calculation": f"In yelp_boston.csv, take rows where 'categories' includes 'Restaurants'. Add all rows from data_merged.csv. Filter for 'neighborhood' = '{selected_neighborhood}'. Sum the 'review_count' column. Divide by the number of unique 'name' entries. Round to the nearest whole number.",
                "Data Source": "https://shorturl.at/1Es5F, https://shorturl.at/pYUfv"
            },
            {
                "Index": "Supply Chain",
                "Metric": "Avg Employees per Establishment",
                "Value": f"{census_data['avg_employees_per_establishment']}",
                "Calculation": "This is a fixed range (20–99 employees) from the 2022 Economic Census, showing the typical number of workers per restaurant in Boston.",
                "Data Source": "https://data.census.gov/table/ECNBASIC2022.EC2272BASIC?q=EC2272BASIC&g=160XX00US2507000"
            },
            {
                "Index": "Market Technology",
                "Metric": "Internet Access",
                "Value": f"{census_data['broadband_penetration']:,}",
                "Calculation": "Take Boston’s population (639,411) from World Population Review. Multiply by 0.85 (85% of people with home internet from StateScoop survey). So, 639,411 × 0.85 = 543,499.35.",
                "Data Source": "https://worldpopulationreview.com/us-cities/massachusetts/boston, https://statescoop.com/boston-digital-equity-survey-internet-affordability/"
            },
            {
                "Index": "Product Price",
                "Metric": "Average Price Level",
                "Value": f"{price_index}",
                "Calculation": f"Open data_merged.csv. Filter for 'neighborhood' = '{selected_neighborhood}'. Take the 'price_range' column. Change '$' to 1, '$$ - $$$' to 2, '$$$$' to 3. If blank, use 2. Add all these numbers and divide by the number of rows. Subtract 1, divide by 2, multiply by 100, and round to 2 decimals.",
                "Data Source": "https://shorturl.at/pYUfv"
            }
        ]
    else:
        # Micro view or All neighborhoods
        indexes = [
            {
                "Index": "Market Need",
                "Metric": "Median Household Income",
                "Value": f"${census_data['median_household_income']:,}",
                "Calculation": "This is a fixed number ($94,755) from the 2023 American Community Survey, showing the middle income for households in Boston.",
                "Data Source": "https://data.census.gov/table/ACSDT5Y2023.B19013?q=B19013&g=040XX00US08_160XX00US2507000"
            },
            {
                "Index": "Market Size",
                "Metric": "Restaurant Establishments",
                "Value": f"{len(combined_restaurants):,}",
                "Calculation": "Open yelp_boston.csv and data_merged.csv. Filter yelp_boston.csv for rows where 'categories' includes 'Restaurants'. Combine with all rows from data_merged.csv. Count the unique 'name' entries.",
                "Data Source": "https://shorturl.at/1Es5F, https://shorturl.at/pYUfv"
            },
            {
                "Index": "Market Competition",
                "Metric": "Avg Restaurants per Neighborhood",
                "Value": f"{neighborhood_agg['business_count'].mean():.1f}" if not neighborhood_agg.empty else "0",
                "Calculation": "In yelp_boston.csv, take rows where 'categories' includes 'Restaurants'. Add all rows from data_merged.csv. Group by 'neighborhood'. Count unique 'name' entries per neighborhood. Add these counts and divide by the number of neighborhoods. Round to 1 decimal.",
                "Data Source": "https://shorturl.at/1Es5F, https://shorturl.at/pYUfv"
            },
            {
                "Index": "Consumer Satisfaction",
                "Metric": "Average Rating",
                "Value": f"{combined_df['rating'].mean():.2f}" if not combined_df.empty and 'rating' in combined_df.columns else "N/A",
                "Calculation": "In yelp_boston.csv, take rows where 'categories' includes 'Restaurants'. Add all rows from data_merged.csv. Take the 'rating' column, add all values, and divide by the number of restaurants. Round to 2 decimals.",
                "Data Source": "https://shorturl.at/1Es5F, https://shorturl.at/pYUfv"
            },
            {
                "Index": "Market Social Media",
                "Metric": "Average Review Count",
                "Value": f"{combined_df['review_count'].mean():.0f}" if not combined_df.empty and 'review_count' in combined_df.columns else "N/A",
                "Calculation": "In yelp_boston.csv, take rows where 'categories' includes 'Restaurants'. Add all rows from data_merged.csv. Take the 'review_count' column, add all values, and divide by the number of restaurants. Round to the nearest whole number.",
                "Data Source": "https://shorturl.at/1Es5F"
            },
            {
                "Index": "Supply Chain",
                "Metric": "Avg Employees per Establishment",
                "Value": f"{census_data['avg_employees_per_establishment']}",
                "Calculation": "This is a fixed range (20–99 employees) from the 2022 Economic Census, showing the typical number of workers per restaurant in Boston.",
                "Data Source": "https://data.census.gov/table/ECNBASIC2022.EC2272BASIC?q=EC2272BASIC&g=160XX00US2507000"
            },
            {
                "Index": "Market Technology",
                "Metric": "Internet Access",
                "Value": f"{census_data['broadband_penetration']:,}",
                "Calculation": "Take Boston’s population (639,411) from World Population Review. Multiply by 0.85 (85% of people with home internet from StateScoop survey). So, 639,411 × 0.85 = 543,499.35.",
                "Data Source": "https://worldpopulationreview.com/us-cities/massachusetts/boston, https://statescoop.com/boston-digital-equity-survey-internet-affordability"
            },
            {
                "Index": "Product Price",
                "Metric": "Average Price Level",
                "Value": f"{price_index}",
                "Calculation": "Open data_merged.csv. Take the 'price_range' column. Change '$' to 1, '$$ - $$$' to 2, '$$$$' to 3. If blank, use 2. Add all these numbers and divide by the number of rows. Subtract 1, divide by 2, multiply by 100, and round to 2 decimals.",
                "Data Source": "https://shorturl.at/pYUfv"
            }
        ]
else:
    if map_view == 'macro' and selected_neighborhood != 'All':
        # Macro view: Use neighborhood-specific metrics
        neighborhood_data = neighborhood_agg[neighborhood_agg['neighborhood'] == selected_neighborhood]
        avg_rating = neighborhood_data['rating'].iloc[0] if not neighborhood_data.empty else "N/A"
        avg_reviews = neighborhood_data['review_count'].iloc[0] / neighborhood_data['business_count'].iloc[0] if not neighborhood_data.empty and neighborhood_data['business_count'].iloc[0] > 0 else "N/A"
        business_count = neighborhood_data['business_count'].iloc[0] if not neighborhood_data.empty else 0
        indexes = [
            {
                "Index": "Market Need",
                "Metric": "Median Household Income",
                "Value": f"${census_data['median_household_income']:,}",
                "Calculation": "This is a fixed number ($94,755) from the 2023 American Community Survey, showing the middle income for households in Boston.",
                "Data Source": "https://data.census.gov/table/ACSDT5Y2023.B19013?q=B19013&g=040XX00US08_160XX00US2507000"
            },
            {
                "Index": "Market Size",
                "Metric": "Retail Establishments",
                "Value": f"{business_count}",
                "Calculation": f"Open yelp_boston.csv. Filter for rows where 'categories' includes 'Grocery', 'Delis', or 'Convenience Stores'. Keep only rows where 'neighborhood' is '{selected_neighborhood}'. Count the unique 'name' entries.",
                "Data Source": "https://shorturl.at/1Es5F"
            },
            {
                "Index": "Market Competition",
                "Metric": "Retailers in Neighborhood",
                "Value": f"{business_count}",
                "Calculation": f"Same as Market Size: In yelp_boston.csv, take rows where 'categories' includes 'Grocery', 'Delis', or 'Convenience Stores'. Filter for 'neighborhood' = '{selected_neighborhood}'. Count unique 'name' entries.",
                "Data Source": "https://shorturl.at/1Es5F"
            },
            {
                "Index": "Consumer Satisfaction",
                "Metric": "Average Rating",
                "Value": f"{avg_rating:.2f}" if isinstance(avg_rating, (int, float)) else avg_rating,
                "Calculation": f"In yelp_boston.csv, take rows where 'categories' includes 'Grocery', 'Delis', or 'Convenience Stores'. Filter for 'neighborhood' = '{selected_neighborhood}'. Take the 'rating' column, add all values, and divide by the number of retailers. Round to 2 decimals.",
                "Data Source": "https://shorturl.at/1Es5F"
            },
            {
                "Index": "Market Technology",
                "Metric": "Internet Access",
                "Value": f"{census_data['broadband_penetration']:,}",
                "Calculation": "Take Boston’s population (639,411) from World Population Review. Multiply by 0.85 (85% of people with home internet from StateScoop survey). So, 639,411 × 0.85 = 543,499.35.",
                "Data Source": "https://worldpopulationreview.com/us-cities/massachusetts/boston, https://statescoop.com/boston-digital-equity-survey-internet-affordability/"
            }
        ]
    else:
        # Micro view or All neighborhoods
        indexes = [
            {
                "Index": "Market Need",
                "Metric": "Median Household Income",
                "Value": f"${census_data['median_household_income']:,}",
                "Calculation": "This is a fixed number ($94,755) from the 2023 American Community Survey, showing the middle income for households in Boston.",
                "Data Source": "https://data.census.gov/table/ACSDT5Y2023.B19013?q=B19013&g=040XX00US08_160XX00US2507000"
            },
            {
                "Index": "Market Size",
                "Metric": "Retail Establishments",
                "Value": f"{len(combined_retail):,}",
                "Calculation": "Open yelp_boston.csv. Filter for rows where 'categories' includes 'Grocery', 'Delis', or 'Convenience Stores'. Count the unique 'name' entries.",
                "Data Source": "https://shorturl.at/1Es5F"
            },
            {
                "Index": "Market Competition",
                "Metric": "Avg Retailers per Neighborhood",
                "Value": f"{neighborhood_agg['business_count'].mean():.1f}" if not neighborhood_agg.empty else "0",
                "Calculation": "In yelp_boston.csv, take rows where 'categories' includes 'Grocery', 'Delis', or 'Convenience Stores'. Group by 'neighborhood'. Count unique 'name' entries per neighborhood. Add these counts and divide by the number of neighborhoods. Round to 1 decimal.",
                "Data Source": "https://shorturl.at/1Es5F"
            },
            {
                "Index": "Consumer Satisfaction",
                "Metric": "Average Rating",
                "Value": f"{combined_df['rating'].mean():.2f}" if not combined_df.empty and 'rating' in combined_df.columns and not combined_df['rating'].empty else "N/A",
                "Calculation": "In yelp_boston.csv, take rows where 'categories' includes 'Grocery', 'Delis', or 'Convenience Stores'. Take the 'rating' column, add all values, and divide by the number of retailers. Round to 2 decimals.",
                "Data Source": "https://shorturl.at/1Es5F"
            },
            {
                "Index": "Market Technology",
                "Metric": "Internet Access",
                "Value": f"{census_data['broadband_penetration']:,}",
                "Calculation": "Take Boston’s population (639,411) from World Population Review. Multiply by 0.85 (85% of people with home internet from StateScoop survey). So, 639,411 × 0.85 = 543,499.35.",
                "Data Source": "https://worldpopulationreview.com/us-cities/massachusetts/boston, https://statescoop.com/boston-digital-equity-survey-internet-affordability/"
            }
        ]

# Display the indexes table with HTML support for hyperlinks
st.table(indexes)

# Summary
st.markdown("""
This dashboard visualizes marketing indexes for Boston's retail and restaurant sectors using Yelp and Census data. 
Restaurants are included only if they have complete name, rating, review count, price range, and cuisines data, with neighborhoods assigned based on proximity to known locations from yelp_boston.csv. 
Retailers (grocery, deli, convenience stores) are shown with available data from yelp_boston.csv. The table details each index's calculation and source, with Census data hyperlinked to https://data.census.gov/ in the Data Source column. 
In Macro view, indexes reflect neighborhood-specific metrics when a neighborhood is selected. 
Use the filters to explore businesses or neighborhoods on the map (ratings: color, review counts: size, cuisines/price range: hover for restaurants). 
Census data is hardcoded from ACS 2023 (B19013, B28002) and ECNBASIC 2022 (EC2272BASIC); update values from https://data.census.gov/ for accuracy.
""")

