import streamlit as st
import numpy as np

# Absolute package layer imports
from data_loader import initialize_engine
from recommender import apply_scoring_rule, apply_ranking_rule

st.set_page_config(page_title="Music Vector Engine", layout="centered")

st.title("🎵 Vector-Based Music Recommender")

# Initialize and pull shared data structures from cache
with st.spinner("Caching and normalizing 1.2M track data matrix..."):
    df, feature_matrix, feature_cols = initialize_engine()

st.write(f"Engine online. Currently indexing **{len(df):,}** clean tracks in RAM.")

# Search UI inputs
search_song = st.text_input("Enter a Song Name:", placeholder="e.g., Blinding Lights")
search_artist = st.text_input("Enter Artist Name (Optional):", placeholder="e.g., The Weeknd")

if st.button("Generate Recommendations", type="primary"):
    if search_song:
        # String comparison mask building
        query_mask = df['name'].str.lower() == search_song.lower()
        if search_artist:
            query_mask = query_mask & (df['artists'].str.lower().str.contains(search_artist.lower()))
            
        song_lookup = df[query_mask]
        
        if not song_lookup.empty:
            matched_song = song_lookup.iloc[0]
            st.success(f"Target track found: **{matched_song['name']}** by *{matched_song['artists']}*")
            
            # Extract target vector coordinates
            seed_vector = matched_song[feature_cols].to_numpy(dtype=np.float32)
            
            # Execute calculation pipelines
            with st.spinner("Processing linear algebra matching matrix..."):
                scores = apply_scoring_rule(seed_vector, feature_matrix)
                recommendations = apply_ranking_rule(scores, df, top_n=5)
            
            # Deduplicate matching tracks (remove the seed song from its own recommendations list)
            recommendations = recommendations[
                ~((recommendations['name'].str.lower() == matched_song['name'].lower()) & 
                  (recommendations['artists'].str.lower() == matched_song['artists'].lower()))
            ].head(5)
            
            # Render recommendations feed 
            st.subheader("Your Custom Algorithmic Matches:")
            for idx, row in recommendations.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{row['name']}**")
                        st.markdown(f"*{row['artists']}*")
                    with col2:
                        st.metric(label="Match", value=f"{row['match_score']}%")
        else:
            st.error("Song not found in the local database catalog. Please verify spelling.")
    else:
        st.warning("Please enter at least a track name to initialize recommendations processing.")