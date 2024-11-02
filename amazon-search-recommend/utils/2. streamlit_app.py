import streamlit as st
import pandas as pd
import json

# Set the title of the app
st.title("Amazon-Like Product Display")

# Upload CSV file for meta data
meta_file = 'data/Digital_Music_Meta.csv'
import ast

# Check if the file is uploaded
if meta_file is not None:
    # Load the data into a DataFrame
    meta_df = pd.read_csv(meta_file)

    # Display each product in a formatted way
    for index, row in meta_df[:10].iterrows():
        st.subheader(row['title'])  # Product title
        
        # Try to decode the images JSON
        images = []
        if pd.notna(row['images']):
            print(f"Row {index} images data: {row['images']}")  # Print the raw data
            try:
                images = ast.literal_eval(row['images'])  # Attempt to decode the JSON
            except json.JSONDecodeError as e:
                st.warning(f"Error decoding JSON for images in row {index}: {e}. Images data may be malformed.")

        # Display the first image (main image)
        if images:
            main_image_url = images[0]['large'] if 'large' in images[0] else images[0]['thumb']
            st.image(main_image_url, width=150)  # Product image
        st.write(f"**Description:** {', '.join(row['description']) if isinstance(row['description'], list) else row['description']}")  # Product description
        
        # Ratings and pricing information
        average_rating = row['average_rating']  # Directly access the average rating
        rating_number = row['rating_number']  # Directly access the rating count

        if pd.notna(average_rating):
            st.write(f"**Rating:** {average_rating:.1f} ({int(rating_number)} ratings)")
        else:
            st.write("**Rating:** No ratings available")
        
        st.write(f"**Price:** ${row['price']:.2f}" if pd.notna(row['price']) else "Price not available")  # Display price
        st.write(f"**Details:** {row['details'] if pd.notna(row['details']) else 'No details available'}")  # Additional product details
        st.write(f"**Category:** {row['main_category']}")  # Product main category
        
        # Display additional metadata if available
        if 'bought_together' in row and pd.notna(row['bought_together']):
            st.write(f"**Frequently Bought Together:** {row['bought_together']}")  # Products bought together

        st.markdown("---")  # Separator line for clarity
