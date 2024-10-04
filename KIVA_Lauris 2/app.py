import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import time 

# Set page config
st.set_page_config(page_title="KIVA Loans Dashboard", layout="centered")

@st.cache_data
def load_data(file_paths):
    try:
        return {name: pd.read_csv(path) for name, path in file_paths.items()}
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Define file paths
file_paths = {
    'Kiva Loans': 'kiva_loans.csv',
    'Kiva MPI Region Locations': 'kiva_mpi_region_locations.csv',
    'Loan Theme IDs': 'loan_theme_ids.csv',
    'Loan Themes by Region': 'loan_themes_by_region.csv'
}

# Load datasets
dataframes = load_data(file_paths)

# Load model and scaler
def load_model_scaler(model_path, scaler_path):
    try:
        model = joblib.load(model_path)  # Load the XGBoost model trained to predict lender count
        scaler = joblib.load(scaler_path)  # Load the scaler used for scaling the features
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

model_path = 'best_xgb_model.pkl'  # Model that predicts lender count
scaler_path = 'scaler.pkl'  # Scaler for feature scaling
model, scaler = load_model_scaler(model_path, scaler_path)

# Lists of sectors and countries (unchanged as these are still input features)
sectors = ['Arts', 'Clothing', 'Construction', 'Education', 'Entertainment', 'Food', 
           'Health', 'Housing', 'Manufacturing', 'Personal Use', 'Retail', 
           'Services', 'Transportation', 'Wholesale']

countries = ['Armenia', 'Azerbaijan', 'Belize', 'Benin', 'Bolivia', 'Brazil', 'Burkina Faso', 
             'Burundi', 'Cambodia', 'Cameroon', 'China', 'Colombia', 'Costa Rica', 
             'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Georgia', 
             'Ghana', 'Guatemala', 'Haiti', 'Honduras', 'India', 'Indonesia', 'Israel', 
             'Jordan', 'Kenya', 'Kyrgyzstan', 'Lao People\'s Democratic Republic', 'Lebanon', 
             'Lesotho', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mexico', 'Moldova', 
             'Mongolia', 'Mozambique', 'Myanmar (Burma)', 'Nepal', 'Nicaragua', 'Nigeria', 
             'Pakistan', 'Palestine', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Rwanda', 
             'Samoa', 'Senegal', 'Sierra Leone', 'Solomon Islands', 'Somalia', 'South Africa', 
             'South Sudan', 'Suriname', 'Tajikistan', 'Tanzania', 'Thailand', 'The Democratic Republic of the Congo', 
             'Timor-Leste', 'Togo', 'Turkey', 'Uganda', 'Ukraine', 'United States', 'Vietnam', 
             'Yemen', 'Zambia', 'Zimbabwe']


# Set the app title and sidebar header
st.title("Kiva Loans Dashboard 📊")
st.sidebar.header("Explore the Kiva Data")

# Sidebar options list for easier management
sidebar_options = [
    'Introduction', 'Borrower Details', 'Kiva Loan Themes', 
    'Monthly Loan Analysis', 'Average Kiva Customer', 'Prediction Tool'
]

# Sidebar selectbox for navigation
option = st.sidebar.selectbox(
    'What would you like to explore?',
    sidebar_options
)


# Welcome Page
if option == 'Introduction':
    st.markdown("""
        ## Welcome to the Kiva Loans Dashboard 
        Kiva is a global platform aimed at empowering underserved communities through financial support. This dashboard lets you dive deep into key insights from Kiva's loan data—loan distribution, repayment patterns, popular loan themes, and more.
        
        Use the sidebar to explore detailed insights into borrower demographics, monthly loan trends, and Kiva’s target customers.
    """)

    # Expander for the objectives section
    with st.expander("🎯  **Dashboard Objectives**"):
        st.markdown("""
        This dashboard is designed to explore Kiva's loan data and answer these key questions:

        1. **Borrower Details**:
           - Where are Kiva's customers located globally?
           - What is the gender distribution among borrowers in different countries?
           - What are the repayment patterns across regions?

        2. **Loan Themes**:
           - What are the most common loan themes?
           - Which activities are most funded?

        3. **Monthly Loan Analysis**:
           - How are loans disbursed month by month?
           - Analyze trends in loan amounts by selecting a specific date range.

        4. **Target Customers**:
           - Who are Kiva’s typical borrowers?

        5. **Lender Count Prediction Tool**:
           - Estimate how many lenders are likely to contribute to a loan based on borrower details and loan characteristics.
           - Use this tool to optimize lender engagement and forecast loan performance.
        """)

# Borrower Details Page
elif option == 'Borrower Details':
    st.title('Borrower Details')

    # Load necessary datasets
    df_loans = dataframes['Kiva Loans']
    df_themes_by_region = dataframes['Loan Themes by Region']

    st.write("In this section, you'll find key insights into borrower demographics and loan distribution patterns across different countries. The map below shows the geographical distribution of loans, and the charts further explore gender distribution and repayment patterns in selected countries.")

    # Map Section
    st.subheader("Map View")
    
    # Display the map 
    if {'lat', 'lon'}.issubset(df_themes_by_region.columns):
        st.map(df_themes_by_region[['lat', 'lon']].dropna())

    st.write("🌍 **Insight**: This map shows the geographical distribution of loans, highlighting regional loan concentrations worldwide.")

    # Sidebar filters for country selection
    st.sidebar.header('Filters for Loan Distribution')

    # Filter gender data for 'male' and 'female'
    df_loans_cleaned = df_loans[df_loans['borrower_genders'].isin(['male', 'female'])]
    all_countries = df_loans_cleaned['country'].unique().tolist()

    # Sidebar multiselect with up to 10 countries allowed
    selected_countries = st.sidebar.multiselect(
        'Select countries (up to 10):',
        options=all_countries,
        default=all_countries[:5],
        help="Select up to 10 countries."
    )

    # Enforce a limit of 10 countries
    if len(selected_countries) > 10:
        st.sidebar.error("⚠️ You can select a maximum of 10 countries.")
        selected_countries = selected_countries[:10]

    # Filter loans by selected countries
    df_filtered = df_loans_cleaned[df_loans_cleaned['country'].isin(selected_countries)]

    # Gender Distribution Section
    st.subheader('Gender Distribution in Selected Countries')
    gender_by_country = df_filtered.groupby(['country', 'borrower_genders'])['lender_count'].sum().reset_index()

    gender_chart = alt.Chart(gender_by_country).mark_bar().encode(
        x=alt.X('country:N', title='Country', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('lender_count:Q', title='Number of Loans'),
        color='borrower_genders:N',
        tooltip=['country', 'lender_count', 'borrower_genders']
    ).properties(width=700, height=400)

    st.altair_chart(gender_chart, use_container_width=True)
    st.write("👥 **Insight**: This chart shows loan distribution by gender across selected countries. You can filter the countries using the sidebar.")

    # Repayment Patterns Section
    st.subheader('Repayment Patterns Across Selected Countries')
    repayment_by_country = df_filtered.groupby(['country', 'repayment_interval'])['lender_count'].sum().reset_index()

    repayment_chart = alt.Chart(repayment_by_country).mark_bar().encode(
        x=alt.X('country:N', title='Country', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('lender_count:Q', title='Number of Loans'),
        color='repayment_interval:N',
        tooltip=['country', 'repayment_interval', 'lender_count']
    ).properties(width=700, height=400)

    st.altair_chart(repayment_chart, use_container_width=True)
    st.write("💰 **Insight**: This chart visualizes repayment patterns across selected countries. Adjust the sidebar filters to explore country-specific repayment behaviors.")

# Themes Page
elif option == "Kiva Loan Themes":
    st.title("Kiva Loan Themes")

    st.markdown("This section displays the most frequent loan themes and activities. Use the sidebar to filter the top items and choose the sorting order.")

    # Sidebar filter controls
    st.sidebar.header('Filter Options')
    
    top_n = st.sidebar.selectbox('Number of top items to display:', options=range(3, 11), index=2)
    sort_order = st.sidebar.selectbox('Sort order:', ['Descending', 'Ascending'])
    ascending = sort_order == 'Ascending'

    # Reusable function for bar charts
    def create_bar_chart(df, x_col, y_col, x_label, chart_title):
        df_counts = df.value_counts().reset_index()
        df_counts.columns = [x_col, y_col]
        df_counts = df_counts.sort_values(y_col, ascending=ascending).head(top_n)

        chart = alt.Chart(df_counts).mark_bar().encode(
            x=alt.X(f'{x_col}:N', sort=alt.EncodingSortField(field=y_col, order=sort_order.lower()), 
                    axis=alt.Axis(labelAngle=-45)),  # Rotate x-axis labels
            y=f'{y_col}:Q',
            color=f'{x_col}:N',
            tooltip=[x_col, y_col]
        ).properties(title=chart_title, width=700, height=400)

        return chart

    # Create tabs for loan themes and activities
    tab1, tab2 = st.tabs(["Loan Themes", "Loan Activities"])

    # Loan Themes Tab
    with tab1:
        df_theme_ids = dataframes['Loan Theme IDs']['Loan Theme Type']
        theme_chart = create_bar_chart(df_theme_ids, 'Loan Theme Type', 'Number of Loans', 'Loan Theme Type', 'Top Loan Themes')
        st.altair_chart(theme_chart, use_container_width=True)
        st.write("💡 **Insight**: The chart above shows the most frequent loan themes in the Kiva dataset.")

    # Loan Activities Tab
    with tab2:
        df_loans_activity = dataframes['Kiva Loans']['activity']
        activity_chart = create_bar_chart(df_loans_activity, 'Loan Activity Type', 'Number of Loans', 'Loan Activity Type', 'Top Loan Activities')
        st.altair_chart(activity_chart, use_container_width=True)
        st.write("📊 **Insight**: The chart above shows the most common loan activities. Use the filter to explore more details.")

# Monthly Results Page
elif option == 'Monthly Loan Analysis':
    st.title("Monthly Loan Analysis")

    st.write("This section analyzes the monthly loan disbursements over a selected time period. Use the sidebar to filter the date range and observe trends in loan amounts distributed each month. The chart below updates dynamically as data loads, giving you a clear view of how loan disbursements have evolved over time. Adjust the date range to explore different timeframes and patterns in loan activities.")

    # Data preparation and cleaning
    df_loans = dataframes['Kiva Loans']
    df_loans['date'] = pd.to_datetime(df_loans['date'], errors='coerce')
    df_loans = df_loans.dropna(subset=['date'])  # Remove rows with invalid dates
    df_loans = df_loans[df_loans['loan_amount'] > 0]  # Filter out non-positive loan amounts

    # Sidebar for date range selection
    min_date, max_date = df_loans['date'].min(), df_loans['date'].max()
    start_date, end_date = st.sidebar.date_input("Select date range:", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filter loans by selected date range
    df_filtered = df_loans[(df_loans['date'] >= pd.Timestamp(start_date)) & (df_loans['date'] <= pd.Timestamp(end_date))]

    # Aggregate monthly loan amounts
    df_filtered['year_month'] = df_filtered['date'].dt.to_period('M')
    monthly_loan_amount = df_filtered.groupby('year_month')['loan_amount'].sum()

    # Remove the last month (it had missing info)
    monthly_loan_amount = monthly_loan_amount[monthly_loan_amount.index != pd.Period('2017-07')]

    # Display the chart with animation and progress bar
    st.subheader(f"Monthly Loan Amount Trend ({start_date} to {end_date})")
    
    progress_bar = st.sidebar.progress(0)
    chart = st.line_chart([])

    # Animating the chart
    total_steps = len(monthly_loan_amount)
    for i, (month, amount) in enumerate(monthly_loan_amount.items()):
        # Add data step by step with animation
        chart.add_rows(pd.DataFrame({'Loan Amount': [amount]}, index=[month.to_timestamp()]))
        progress_bar.progress((i + 1) / total_steps)  # Update progress bar
        time.sleep(0.1)  # Slight delay to simulate the animation

    # Clear the progress bar once done
    progress_bar.empty()
    st.button("Re-run")

# Our Average Customer Page
elif option == 'Average Kiva Customer':
    st.title("Kiva Target Customer")

    # Load the loans data
    df_loans = dataframes['Kiva Loans']

    # Clean the gender data for 'male' and 'female'
    df_loans_cleaned = df_loans[df_loans['borrower_genders'].isin(['male', 'female'])]

    # Calculate the average loan amount (in USD)
    average_loan_amount_usd = df_loans['loan_amount'].mean()

    # Most common attributes
    most_common_gender = df_loans_cleaned['borrower_genders'].mode()[0].capitalize()
    most_common_country = df_loans['country'].mode()[0]
    most_common_activity = df_loans['activity'].mode()[0]
    most_common_repayment_plan = df_loans['repayment_interval'].mode()[0].capitalize()

    # Display the information
    st.markdown("### The Average Kiva Borrower")
    
    st.write(f"**Gender**: {most_common_gender}")
    st.write(f"**Country**: {most_common_country}")
    st.write(f"**Average Loan Amount**: ${average_loan_amount_usd:,.2f} USD")
    st.write(f"**Loan Purpose**: {most_common_activity}")
    st.write(f"**Repayment Plan**: {most_common_repayment_plan}")

    # Display image 
    st.image('Kiva customer.webp', caption='Image generated using DALL·E.', use_column_width=True)

elif option == 'Prediction Tool':
    st.title("KIVA Lender Count Prediction Tool")
    st.subheader("Estimate the Number of Lenders")

    # Contextual information for the user
    st.markdown("""
    Enter borrower and loan details below to estimate how many lenders will contribute to the loan. This tool uses historical Kiva data to help predict the level of lender engagement based on various factors.
    """)

    # User inputs with empty initial values
    gender = st.selectbox("Borrower's Gender", ["", "Male", "Female"], help="Select the borrower's gender.")
    sector = st.selectbox("Loan Sector", [""] + sectors, help="Choose the category or industry for the loan (e.g., Agriculture, Retail).")
    country = st.selectbox("Borrower's Country", [""] + countries, help="Select the borrower's country of origin.")
    loan_amount = st.number_input("Loan Amount (USD)", min_value=0, value=0, 
                                  help="Enter the requested loan amount in USD.")
    days_to_fund = st.number_input("Days to Fund", min_value=0, max_value=365, value=0, 
                                   help="Enter how many days it took or is expected to take for the loan to be fully funded. Shorter times suggest higher popularity.")

    # Prediction button
    if st.button("Predict Lender Count 🚀"):
        if gender == "" or sector == "" or country == "":
            st.error("Please fill out all fields before making a prediction.")
        else:
            with st.spinner("Calculating your estimate..."):
                try:
                    # Prepare input features
                    is_male, is_female = (1, 0) if gender == "Male" else (0, 1)
                    sector_encoded = [1 if s == sector else 0 for s in sectors]
                    country_encoded = [1 if c == country else 0 for c in countries]
                    features = np.array([[is_male, is_female, loan_amount, days_to_fund] + sector_encoded + country_encoded])

                    # Scale numeric features
                    features[:, 2:4] = scaler.transform(features[:, 2:4])

                    # Predict lender count
                    prediction = np.maximum(model.predict(features), 0)

                    st.success(f"🎉 Estimated Lender Count: {int(prediction[0])}")

                    # SHAP explanation for model prediction
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(features)
                    feature_names = ['Days to Fund', 'Is Male', 'Is Female', 'Loan Amount'] + \
                                    [f'Sector: {s}' for s in sectors] + [f'Country: {c}' for c in countries]

                    # SHAP force plot explanation
                    st.markdown("### Why This Estimate? SHAP Breakdown")
                    st.markdown("The chart below shows how each factor influenced the estimated lender count.")
                    
                    shap.initjs()
                    plt.figure(figsize=(10, 6))
                    shap.force_plot(explainer.expected_value, shap_values[0], features[0], feature_names=feature_names, matplotlib=True)
                    st.pyplot(plt)

                except Exception as e:
                    st.error(f"An error occurred while predicting: {str(e)}")
