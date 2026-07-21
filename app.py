import streamlit as st
import pandas as pd
import mysql.connector
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "mysql369",
    "database": "real_estate_investment"
}

MLFLOW_TRACKING_URI = (
    "mysql+mysqlconnector://root:mysql369"
    "@localhost:3306/mlflow_tracking"
)

FEATURE_COLS = [
    'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
    'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals', 'Amenity_Count',
    'Transport_Score', 'Security_Score'
] + [
    c + '_enc' for c in [
        'State', 'City', 'Locality', 'Property_Type',
        'Furnished_Status', 'Facing', 'Owner_Type',
        'Availability_Status'
    ]
]

# =========================================================
# LOAD RESOURCES
# =========================================================
@st.cache_resource
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


@st.cache_resource
def load_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    clf_model = mlflow.pyfunc.load_model(
        "models:/GoodInvestmentClassifier@production"
    )
    reg_model = mlflow.pyfunc.load_model(
        "models:/FuturePriceRegressor@production"
    )
    return clf_model, reg_model


@st.cache_resource
def load_native_clf_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(
        "models:/GoodInvestmentClassifier@production"
    )


@st.cache_resource
def load_preprocessors():
    label_encoders = joblib.load("label_encoders.pkl")
    scaler_clf = joblib.load("scaler_clf.pkl")
    scaler_reg = joblib.load("scaler_reg.pkl")
    return label_encoders, scaler_clf, scaler_reg


@st.cache_data
def load_location_data():
    return pd.read_csv("india_housing_prices_cleaned.csv")


clf_model, reg_model = load_models()
label_encoders, scaler_clf, scaler_reg = load_preprocessors()
location_df = load_location_data()

# =========================================================
# NAVIGATION
# =========================================================
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Predict Investment", "🔍 Property Explorer", "📊 Market Insights"]
)

# =========================================================
# PAGE 1: PREDICT INVESTMENT
# =========================================================
if page == "🏠 Predict Investment":

    st.title("Real Estate Investment Advisor")
    st.subheader("Enter Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        # State dropdown
        state = st.selectbox(
            "State",
            sorted(location_df["State"].dropna().unique())
        )

        # City dropdown depends on selected State
        filtered_cities = (
            location_df[location_df["State"] == state]["City"]
            .dropna()
            .unique()
        )

        city = st.selectbox(
            "City",
            sorted(filtered_cities)
        )

        locality = st.selectbox(
            "Locality",
            sorted(label_encoders["Locality"].classes_)
        )

        property_type = st.selectbox(
            "Property Type",
            sorted(label_encoders["Property_Type"].classes_)
        )

        bhk = st.number_input(
            "BHK",
            min_value=1,
            max_value=10,
            value=2
        )

    with col2:
        size_sqft = st.number_input(
            "Size (Sq Ft)",
            min_value=300,
            max_value=10000,
            value=1200
        )

        price_lakhs = st.number_input(
            "Current Price (Lakhs ₹)",
            min_value=5.0,
            max_value=1000.0,
            value=80.0
        )

        year_built = st.number_input(
            "Year Built",
            min_value=1950,
            max_value=2026,
            value=2015
        )

        furnished_status = st.selectbox(
            "Furnished Status",
            sorted(label_encoders["Furnished_Status"].classes_)
        )

        floor_no = st.number_input(
            "Floor No.",
            min_value=0,
            max_value=100,
            value=3
        )

    with col3:
        total_floors = st.number_input(
            "Total Floors",
            min_value=1,
            max_value=100,
            value=10
        )

        nearby_schools = st.slider(
            "Nearby Schools",
            1,
            10,
            5
        )

        nearby_hospitals = st.slider(
            "Nearby Hospitals",
            1,
            10,
            5
        )

        transport_access = st.selectbox(
            "Public Transport Accessibility",
            ["Low", "Medium", "High"]
        )

        facing = st.selectbox(
            "Facing",
            sorted(label_encoders["Facing"].classes_)
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        parking = st.radio(
            "Parking Space",
            ["Yes", "No"],
            horizontal=True
        )

    with col5:
        security = st.radio(
            "Security",
            ["Yes", "No"],
            horizontal=True
        )

    with col6:
        owner_type = st.selectbox(
            "Owner Type",
            sorted(label_encoders["Owner_Type"].classes_)
        )

    availability_status = st.selectbox(
        "Availability Status",
        sorted(label_encoders["Availability_Status"].classes_)
    )

    amenities_selected = st.multiselect(
        "Amenities",
        ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
    )

    predict_btn = st.button(
        "Predict Investment Potential",
        type="primary"
    )

    if predict_btn:

        age_of_property = 2026 - year_built
        amenity_count = max(len(amenities_selected), 1)

        transport_map = {
            "Low": 0.0,
            "Medium": 0.5,
            "High": 1.0
        }

        transport_score = transport_map[transport_access]
        security_score = 1 if security == "Yes" else 0
        price_per_sqft = price_lakhs * 100000 / size_sqft

        input_dict = {
            "BHK": bhk,
            "Size_in_SqFt": size_sqft,
            "Price_in_Lakhs": price_lakhs,
            "Price_per_SqFt": price_per_sqft,
            "Year_Built": year_built,
            "Floor_No": floor_no,
            "Total_Floors": total_floors,
            "Age_of_Property": age_of_property,
            "Nearby_Schools": nearby_schools,
            "Nearby_Hospitals": nearby_hospitals,
            "Amenity_Count": amenity_count,
            "Transport_Score": transport_score,
            "Security_Score": security_score,
            "State_enc": label_encoders["State"].transform([state])[0],
            "City_enc": label_encoders["City"].transform([city])[0],
            "Locality_enc": label_encoders["Locality"].transform([locality])[0],
            "Property_Type_enc": label_encoders["Property_Type"].transform([property_type])[0],
            "Furnished_Status_enc": label_encoders["Furnished_Status"].transform([furnished_status])[0],
            "Facing_enc": label_encoders["Facing"].transform([facing])[0],
            "Owner_Type_enc": label_encoders["Owner_Type"].transform([owner_type])[0],
            "Availability_Status_enc": label_encoders["Availability_Status"].transform([availability_status])[0]
        }

        input_df = pd.DataFrame([input_dict])[FEATURE_COLS]

        input_scaled_clf = scaler_clf.transform(input_df)
        input_scaled_reg = scaler_reg.transform(input_df)

        clf_pred = clf_model.predict(input_scaled_clf)[0]
        reg_pred = reg_model.predict(input_scaled_reg)[0]

        st.divider()

        res1, res2 = st.columns(2)

        with res1:
            if clf_pred == 1:
                st.success("✅ Good Investment")
            else:
                st.error("❌ Not a Recommended Investment")

        with res2:
            st.metric(
                "Estimated Price After 5 Years",
                f"₹{reg_pred:.2f} Lakhs",
                delta=f"+₹{reg_pred - price_lakhs:.2f} Lakhs"
            )

        try:
            native_clf = load_native_clf_model()
            proba = native_clf.predict_proba(input_scaled_clf)[0]

            st.write(
                f"Model confidence: {max(proba) * 100:.1f}%"
            )

            importances = (
                pd.Series(
                    native_clf.feature_importances_,
                    index=FEATURE_COLS
                )
                .sort_values(ascending=False)
                .head(10)
            )

            fig, ax = plt.subplots(figsize=(6, 4))

            sns.barplot(
                x=importances.values,
                y=importances.index,
                hue=importances.index,
                palette="crest",
                legend=False,
                ax=ax
            )

            ax.set_title("Top 10 Feature Importances")
            st.pyplot(fig)

        except Exception:
            st.caption(
                "Confidence score or feature importance unavailable "
                "for this model type."
            )

# =========================================================
# PAGE 2: PROPERTY EXPLORER
# =========================================================
elif page == "🔍 Property Explorer":

    st.title("Property Explorer")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT city FROM properties ORDER BY city"
    )

    city_options = [row[0] for row in cursor.fetchall()]
    cursor.close()

    f1, f2, f3 = st.columns(3)

    with f1:
        filter_city = st.selectbox(
            "City",
            ["All"] + city_options
        )

    with f2:
        price_range = st.slider(
            "Price Range (Lakhs ₹)",
            0,
            500,
            (0, 500)
        )

    with f3:
        filter_bhk = st.selectbox(
            "BHK",
            ["All", 1, 2, 3, 4, 5]
        )

    query = """
        SELECT *
        FROM properties
        WHERE price_lakhs BETWEEN %s AND %s
    """

    params = [
        price_range[0],
        price_range[1]
    ]

    if filter_city != "All":
        query += " AND city = %s"
        params.append(filter_city)

    if filter_bhk != "All":
        query += " AND bhk = %s"
        params.append(filter_bhk)

    query += " LIMIT 200"

    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, tuple(params))

    results = pd.DataFrame(
        cursor.fetchall()
    )

    cursor.close()

    st.write(
        f"Found {len(results)} matching properties"
    )

    st.dataframe(
        results,
        use_container_width=True
    )

# =========================================================
# PAGE 3: MARKET INSIGHTS
# =========================================================
elif page == "📊 Market Insights":

    st.title("Market Insights")

    conn = get_connection()

    df_insights = pd.read_sql(
        """
        SELECT
            city,
            state,
            price_lakhs,
            price_per_sqft,
            bhk,
            good_investment
        FROM properties
        """,
        conn
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Avg Price by City (Top 15)")

        city_avg = (
            df_insights
            .groupby("city")["price_lakhs"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        sns.barplot(
            x=city_avg.values,
            y=city_avg.index,
            hue=city_avg.index,
            palette="crest",
            legend=False,
            ax=ax
        )

        st.pyplot(fig)

    with col2:
        st.subheader("Good Investment Rate by City (Top 15)")

        inv_rate = (
            df_insights
            .groupby("city")["good_investment"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        sns.barplot(
            x=inv_rate.values,
            y=inv_rate.index,
            hue=inv_rate.index,
            palette="flare",
            legend=False,
            ax=ax
        )

        st.pyplot(fig)

    st.subheader(
        "Price Heatmap — State x Property Type"
    )

    df_insights2 = pd.read_sql(
        """
        SELECT
            state,
            property_type,
            price_per_sqft
        FROM properties
        """,
        conn
    )

    pivot = df_insights2.pivot_table(
        index="state",
        columns="property_type",
        values="price_per_sqft",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)