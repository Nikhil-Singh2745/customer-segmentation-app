import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import 'os' for creating robust file paths

# --- 0. FILE PATH SETUP ---

# Function to build a path relative to the script
# This makes it work in Streamlit Community Cloud
def get_path(file_name):
    """Constructs an absolute path to the file."""
    # os.path.dirname(__file__) gets the directory of the current script
    return os.path.join(os.path.dirname(__file__), file_name)

# --- 1. SETUP AND LOADING ---

# Set page configuration for a wider layout and a clear title
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# Load the trained K-Means model
try:
    with open(get_path('models/kmeans_model.pkl'), 'rb') as model_file:
        kmeans = pickle.load(model_file)
except FileNotFoundError:
    st.error("Error: The 'kmeans_model.pkl' file was not found. Please check the 'models/' folder.")
    st.stop() # Stop the app if the model can't be loaded

# Load the saved StandardScaler
try:
    with open(get_path('models/scaler.pkl'), 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Error: The 'scaler.pkl' file was not found. Please check the 'models/' folder.")
    st.stop()

# Load the customer dataset
try:
    # We assume the file was renamed to 'mall_customers.csv' as recommended
    customer_data = pd.read_csv(get_path('data/mall_customers.csv'))
    # Extract the features for clustering
    X = customer_data.iloc[:, [3, 4]].values
except FileNotFoundError:
    st.error("Error: The 'mall_customers.csv' file was not found. Please check the 'data/' folder.")
    st.stop()

# --- PRE-COMPUTE CLUSTERS FOR EXISTING DATA (Needed for Insights & EDA) ---
# Scale the existing data
X_scaled = scaler.transform(X)
# Predict clusters for ALL existing data and add to the dataframe
customer_data['Cluster'] = kmeans.predict(X_scaled)

# --- 2. PERSONA DEFINITIONS ---

# Define the cluster personas, colors, and strategies
# (You can adjust these based on your notebook analysis)
cluster_personas = {
    0: {
        "persona": "Standard",
        "color": "#1f77b4", # Muted Blue
        "strategy": "Offer standard loyalty programs and seasonal promotions. Focus on consistent value."
    },
    1: {
        "persona": "Target Customer",
        "color": "#2ca02c", # Green
        "strategy": "Engage with high-value offers, exclusive previews, and premium membership benefits. This is your key segment."
    },
    2: {
        "persona": "Careful Spender",
        "color": "#ff7f0e", # Orange
        "strategy": "Attract with 'save money' messaging, bundle deals, and clearance events. Emphasize value and utility."
    },
    3: {
        "persona": "Careless Spender",
        "color": "#d62728", # Red
        "strategy": "Use impulse-buy tactics, point-of-sale promotions, and 'new arrivals' alerts. Focus on trends and excitement."
    },
    4: {
        "persona": "Low-Income / Thrifty",
        "color": "#9467bd", # Purple
        "strategy": "Target with coupons, discount codes, and 'value-pack' offers. Build loyalty through consistent savings."
    }
}


# --- 3. USER INTERFACE ---

st.title('🛍️ Customer Segmentation Dashboard')

# Create two tabs: one for the tool, one for data exploration
tab1, tab2 = st.tabs(["Clustering Tool", "Data Exploration"])

# --- TAB 1: CLUSTERING TOOL ---
with tab1:
    st.write("""
    This tool uses a K-Means model to segment customers. 
    Use the sliders to input a new customer's data and see which segment they belong to.
    """)
    
    # Layout with two columns: 1 for input, 2 for the main plot
    col1, col2 = st.columns((1, 2))

    with col1:
        st.header("👤 New Customer Input")

        # Sliders for user input
        annual_income = st.slider(
            'Annual Income (k$)',
            min_value=int(customer_data['Annual Income (k$)'].min()),
            max_value=int(customer_data['Annual Income (k$)'].max()),
            value=50,
            step=1
        )

        spending_score = st.slider(
            'Spending Score (1-100)',
            min_value=1,
            max_value=100,
            value=50,
            step=1
        )

        # --- 4. MODEL PREDICTION ---
        
        # Create the new customer's data array
        new_customer_data = [[annual_income, spending_score]]
        
        # **NEW**: Scale the new customer's data
        new_customer_data_scaled = scaler.transform(new_customer_data)
        
        # Predict the cluster
        predicted_cluster = kmeans.predict(new_customer_data_scaled)[0]
        
        # Get the persona results
        result = cluster_personas.get(predicted_cluster, {"persona": "Unknown", "color": "gray", "strategy": "No strategy defined."})

        # --- 5. DISPLAY PREDICTION RESULTS ---
        
        st.header("📈 Prediction Result")
        st.subheader(f"The customer belongs to Cluster {predicted_cluster}")
        
        st.markdown(f"""
        **Customer Persona:** <span style='color:{result['color']}; font-size: 24px; font-weight: bold;'>
        {result['persona']}
        </span>
        """, unsafe_allow_html=True)
        
        st.write("**Recommended Marketing Strategy:**")
        st.info(result['strategy'])

        # --- 6. **NEW**: DEEPER CLUSTER INSIGHTS ---
        
        st.header(f"Insights for '{result['persona']}' (Cluster {predicted_cluster})")

        # Filter the dataframe for only customers in the predicted cluster
        cluster_df = customer_data[customer_data['Cluster'] == predicted_cluster]

        # Create two columns for these new insights
        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            # Gender distribution for this cluster
            st.write("**Gender Breakdown**")
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            cluster_df['Gender'].value_counts().plot.pie(
                autopct='%1.1f%%', 
                ax=ax_pie, 
                colors=['#66b3ff', '#ff9999'] # Light blue / light red
            )
            ax_pie.set_ylabel('') # Hide the 'Gender' label
            st.pyplot(fig_pie)
            
        with insight_col2:
            # Age distribution for this cluster
            st.write("**Age Distribution**")
            fig_hist, ax_hist = plt.subplots(figsize=(5, 5))
            sns.histplot(cluster_df['Age'], kde=True, ax=ax_hist, bins=15)
            ax_hist.set_xlabel("Age")
            ax_hist.set_ylabel("Count")
            st.pyplot(fig_hist)


    with col2:
        st.header("📊 Customer Segments Visualization")
        st.write("The plot shows all customers, colored by their segment. The new customer is marked with a black star (★).")

        # --- 7. VISUALIZATION ---

        fig, ax = plt.subplots(figsize=(12, 8))

        # Get the colors for each cluster from our persona mapping
        cluster_colors = [cluster_personas[i]['color'] for i in range(len(cluster_personas))]

        # Scatter plot of the existing clusters
        # We plot the ORIGINAL 'X' data, but color it by the 'Cluster' column
        sns.scatterplot(
            data=customer_data,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Cluster',
            palette=cluster_colors,
            s=60,
            ax=ax,
            legend='full'
        )
        
        # Plot the new customer's data point
        ax.scatter(annual_income, spending_score, s=250, c='black', marker='*', label='New Customer')

        # **NEW**: Plot the cluster centroids (must be inverse-transformed)
        original_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(
            original_centroids[:, 0], 
            original_centroids[:, 1], 
            s=200, 
            c='cyan', 
            marker='P', 
            label='Centroids', 
            edgecolors='black'
        )

        ax.set_title('Customer Segments')
        ax.set_xlabel('Annual Income (k$)')
        ax.set_ylabel('Spending Score (1-100)')
        
        # Create custom legend handles
        handles, labels = ax.get_legend_handles_labels()
        # Create persona labels for the legend
        persona_labels = [f"Cluster {i}: {cluster_personas[i]['persona']}" for i in range(len(cluster_personas))]
        
        # Update legend labels
        new_labels = []
        for label in labels:
            if label.isdigit():
                new_labels.append(persona_labels[int(label)])
            else:
                new_labels.append(label)
        
        # Re-create the legend
        ax.legend(handles, new_labels, title='Segments', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        
        st.pyplot(fig)


# --- TAB 2: DATA EXPLORATION (EDA) ---
with tab2:
    st.header("Exploratory Data Analysis")
    st.write("Understanding the customer demographics before clustering.")
    
    # Show a snippet of the data
    st.subheader("Raw Data")
    st.dataframe(customer_data.head())
    
    # Show summary statistics
    st.subheader("Summary Statistics")
    # We drop CustomerID (non-numeric) and Cluster (categorical) from describe()
    st.dataframe(customer_data.drop(columns=["CustomerID", "Cluster"]).describe())
    
    # Add some plots
    st.subheader("Data Distributions")
    
    # Create columns for plots
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        # Gender Distribution
        st.write("**Gender Distribution**")
        fig_gender, ax_gender = plt.subplots()
        sns.countplot(data=customer_data, x='Gender', ax=ax_gender, palette=['#66b3ff', '#ff9999'])
        st.pyplot(fig_gender)
        
        # Age Distribution
        st.write("**Age Distribution**")
        fig_age, ax_age = plt.subplots()
        sns.histplot(customer_data['Age'], bins=20, kde=True, ax=ax_age)
        st.pyplot(fig_age)
        
    with plot_col2:
        # Annual Income Distribution
        st.write("**Annual Income (k$) Distribution**")
        fig_income, ax_income = plt.subplots()
        sns.histplot(customer_data['Annual Income (k$)'], bins=20, kde=True, ax=ax_income, color='green')
        st.pyplot(fig_income)
        
        # Spending Score Distribution
        st.write("**Spending Score (1-100) Distribution**")
        fig_score, ax_score = plt.subplots()
        sns.histplot(customer_data['Spending Score (1-100)'], bins=20, kde=True, ax=ax_score, color='red')
        st.pyplot(fig_score)
        
    # Pairplot (Very impressive for interviews)
    st.subheader("Correlation Pairplot")
    st.write("Shows relationships between all features, colored by Gender.")
    
    # Use st.spinner to show a loading message as this can be slow
    with st.spinner('Generating pairplot... This may take a moment.'):
        # We drop CustomerID (not useful) and Cluster (categorical, use hue instead)
        fig_pairplot = sns.pairplot(
            customer_data.drop('CustomerID', axis=1), 
            hue='Gender', 
            palette=['#66b3ff', '#ff9999']
        )
        st.pyplot(fig_pairplot)
