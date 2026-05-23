import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Title
st.title("🏠 House Price Predictor")
st.markdown("Fill the details and click **Predict** 👇")

if "size_val" not in st.session_state:
    st.session_state.size_val = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "size_input" not in st.session_state:
    st.session_state.size_input = ""

if "bedrooms_input" not in st.session_state:
    st.session_state.bedrooms_input = ""

if "age_input" not in st.session_state:
    st.session_state.age_input = ""


# Load data
data = pd.read_csv("data.csv")

# Cleaning
data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].mean())
data['age'] = data['age'].fillna(data['age'].mean())
data = data.dropna(subset=['price'])

# Features & target
X = data[['size', 'bedrooms', 'age']]
y = data['price']

# Cache model
@st.cache_resource
def train_model():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model()

# ---------- TEXT INPUT UI ----------
col1, col2 = st.columns(2)

with col1:
    size = st.text_input("📏 Size (sq ft)", key="size_input")

with col2:
    bedrooms = st.text_input("🛏 Bedrooms", key="bedrooms_input")

age = st.text_input("🏚 Age of House", key="age_input")

# Prediction
if st.button("🚀 Predict Price"):
    try:
        st.session_state.size_val = float(size)
        bedrooms_val = int(bedrooms)
        age_val = int(age)

        input_data = scaler.transform([[st.session_state.size_val, bedrooms_val, age_val]])
        pred = model.predict(input_data)

        st.session_state.prediction = pred[0]

        st.success(f"💰 Predicted Price: ₹ {pred[0]:.2f} Lakhs")

    except ValueError:
        st.error("⚠️ Please enter valid numeric values")

#  RESET BUTTON (separate block)
if st.button("🔄 Reset"):
    st.session_state.clear()
    st.rerun()


# ---------- GRAPH (always visible) ----------

st.markdown(
    """
    <div style='background-color:#f9f9f9;padding:15px;border-radius:10px;border:1px solid #ddd'>
    """,
    unsafe_allow_html=True
)

st.subheader("📊 Data Visualization")

fig, ax = plt.subplots(figsize=(6, 4))  #  control size

# Scatter plot
ax.scatter(X['size'], y)

# Always show dataset
ax.scatter(X['size'], y)

# Show prediction only if exists
if st.session_state.size_val is not None and st.session_state.prediction is not None:
    ax.scatter(
        st.session_state.size_val,
        st.session_state.prediction,
        marker='x',
        s=100
    )

# Labels
ax.set_xlabel("Size (sq ft)", fontsize=10)
ax.set_ylabel("Price (Lakhs)", fontsize=10)
ax.set_title("House Price vs Size", fontsize=12)

# Add grid
ax.grid(True)

# Tight layout (important)
fig.tight_layout()

# Display in Streamlit
st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)