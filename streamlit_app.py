import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import re
import io

# Function to normalize text
def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(text.split())        # Remove extra whitespace
    return text

# Streamlit app
def main():
    st.title("Product Matching App")

    # File upload
    uploaded_file1 = st.file_uploader("Choose the first Excel file", type="xlsx")
    uploaded_file2 = st.file_uploader("Choose the second Excel file", type="xlsx")

    if uploaded_file1 and uploaded_file2:
        df1 = pd.read_excel(uploaded_file1)
        df2 = pd.read_excel(uploaded_file2)

        st.write("DF1 Columns:", df1.columns.tolist())
        st.write("DF2 Columns:", df2.columns.tolist())

        # Preprocess product names
        df1['product_name_normalized'] = df1['product_name'].apply(normalize)
        df2['product_name_normalized'] = df2['product_name'].apply(normalize)

        # Compute embeddings for product names
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings1 = model.encode(df1['product_name_normalized'].tolist(), convert_to_tensor=False)
        embeddings2 = model.encode(df2['product_name_normalized'].tolist(), convert_to_tensor=False)

        # Fit NearestNeighbors model on embeddings from df2
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine').fit(embeddings2)

        # Find nearest neighbors in df2 for entries in df1
        distances, indices = nbrs.kneighbors(embeddings1)

        # Set a similarity threshold
        threshold = st.slider("Set similarity threshold", 0.0, 1.0, 0.2)

        matches = []
        unmatched_indices = []

        for idx, (distance_array, index_array) in enumerate(zip(distances, indices)):
            index = index_array[0]
            distance = distance_array[0]
            similarity = 1 - distance  # Cosine similarity
            if distance <= threshold:
                matches.append({
                    'product_name_1': df1.iloc[idx]['product_name'],
                    'product_price_1': df1.iloc[idx]['product_price'],
                    'product_name_2': df2.iloc[index]['product_name'],
                    'product_price_2': df2.iloc[index]['product_price'],
                    'similarity': similarity
                })
            else:
                unmatched_indices.append(idx)

        # Compile matched and unmatched products
        matched_df = pd.DataFrame(matches)
        unmatched_df = df1.iloc[unmatched_indices]

        st.write("Matched Products:")
        st.dataframe(matched_df)

        st.write("Unmatched Products:")
        st.dataframe(unmatched_df)

        # Ensure product prices are numeric
        matched_df['product_price_1'] = pd.to_numeric(matched_df['product_price_1'], errors='coerce')
        matched_df['product_price_2'] = pd.to_numeric(matched_df['product_price_2'], errors='coerce')

        # Price comparison
        matched_df['price_difference'] = matched_df['product_price_1'] - matched_df['product_price_2']

        # Convert DataFrames to Excel files in memory
        matched_buffer = io.BytesIO()
        unmatched_buffer = io.BytesIO()

        # Write DataFrames to the in-memory buffers
        with pd.ExcelWriter(matched_buffer, engine='xlsxwriter') as writer:
            matched_df.to_excel(writer, index=False)
        with pd.ExcelWriter(unmatched_buffer, engine='xlsxwriter') as writer:
            unmatched_df.to_excel(writer, index=False)

        # Download buttons
        st.download_button(
            label="Download Matched Products",
            data=matched_buffer.getvalue(),
            file_name='matched_products.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        st.download_button(
            label="Download Unmatched Products",
            data=unmatched_buffer.getvalue(),
            file_name='unmatched_products.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

if __name__ == "__main__":
    main()
