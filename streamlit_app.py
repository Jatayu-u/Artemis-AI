import streamlit as st
import requests
import json
from streamlit_lottie import st_lottie
from PIL import Image
from io import BytesIO
import time

import requests
from PIL import Image
from io import BytesIO

# Function to load Lottie JSON file
def load_lottie_file(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

# Load the eco-cyborg Lottie animation
lottie_eco_cyborg = load_lottie_file("Animation - 1727970820675.json")  # Update the path to your JSON file

# Set the base API URL
BASE_URL = "http://localhost:8000"  # Update with actual API base URL if needed

# Set the page configuration
st.set_page_config(
    page_title="Artemis AI - Sustainable Business Solutions",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for colorful styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(90deg, #A1FFCE 0%, #FAFFD1 100%);
    }
    .main-container {
        # background:;
        padding: 20px;
        border-radius: 10px;
    }
    .btn-primary {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .btn-primary:hover {
        background-color: #2980b9;
    }
    .section-title {
        font-size: 28px;
        color: #16a085;
    }
    .header-text {
        color: #16a085;
        font-size: 22px;
    }
    .footer {
        font-size: 14px;
        color: #888;
        text-align: center;
        margin-top: 50px;
        animation: fade-in 3s ease;
    }
    .footer:hover {
        color: #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# Add colorful header and description
st.markdown("<h1 style='text-align: center; color: #27ae60;'>🌱 Artemis AI - Sustainable Business Model Advisor</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='header-text'>Empowering businesses with AI-driven sustainability solutions</h3>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="main-container">
    Welcome to <b>Artemis AI</b>, an AI-powered tool to drive sustainability and profitability through tailored business models.
    Utilize our various APIs to:
    <ul>
        <li>Design a circular economy business model 🌍</li>
        <li>Analyze market fit and develop marketing strategies 📈</li>
        <li>Generate comprehensive business plans 📑</li>
        <li>Forecast long-term impact on regulations, market trends, and social values 📊</li>
    </ul>
    <i>Designed with Accenture's vision for a sustainable future.</i>
    </div>
    """, unsafe_allow_html=True
)

# Interactive and colorful sidebar options
st.sidebar.header("💡 Explore Artemis AI Services")
option = st.sidebar.selectbox(
    "Select a Service",
    ("Sustainability Strategy", 
     "Business Model Advisor", 
     "Market Research & Product-Market Fit", 
     "Marketing Strategy", 
     "Business Plan Generation",
     "ESG Metrics", 
     "Impact Prediction",
     "Product Image Generation")
)

# Input section with colorful borders
st.markdown("<h2 class='section-title'>Describe your product:</h2>", unsafe_allow_html=True)
product_description = st.text_area("Describe your product:", placeholder="Enter your product description here...", height=150)

# Function to handle API calls
def call_api(api_url, payload):
    with st.spinner('Processing with Artemis AI...'):
        try:
            response = requests.post(api_url, json=payload)
            return response.json()
        except Exception as e:
            st.error(f"Error: {e}")
            return None


if option == "Sustainability Strategy":
    st.subheader("♻️ Get Sustainability Strategy")
    if st.button("Generate Strategy"):
        if product_description:
            response = call_api(f"{BASE_URL}/sustainability/", {"product_description": product_description})
            if response:
                st.success("Sustainability Suggestions Generated")
                st.write(response['sustainability_suggestions'])
        else:
            st.warning("Please provide a product description.")
            
elif option == "Business Model Advisor":
    sustainability_report = st.text_area("Sustainability Report:", placeholder="Enter sustainability report...", height=150)
    st.subheader("🏢 Get Business Model Suggestions")
    if st.button("Generate Business Model"):
        if product_description and sustainability_report:
            response = call_api(f"{BASE_URL}/business_model/", {
                "product_description": product_description,
                "sustainability_report": sustainability_report
            })
            if response:
                st.success("Business Model Generated")
                st.write(response['business_model_suggestions'])
        else:
            st.warning("Please provide product description and sustainability report.")
            
elif option == "Market Research & Product-Market Fit":
    business_model = st.text_area("Business Model:", placeholder="Enter business model...", height=150)
    st.subheader("🔍 Get Market Fit Analysis")
    if st.button("Generate Market Fit"):
        if product_description and business_model:
            response = call_api(f"{BASE_URL}/market_fit/", {
                "product_description": product_description,
                "business_model": business_model
            })
            if response:
                st.success("Market Fit Analysis Generated")
                st.write(response['market_fit_analysis'])
        else:
            st.warning("Please provide product description and business model.")
            
elif option == "Marketing Strategy":
    market_fit = st.text_area("Market Fit Analysis:", placeholder="Enter market fit analysis...", height=150)
    st.subheader("📈 Get Marketing Strategy")
    if st.button("Generate Marketing Strategy"):
        if product_description and market_fit:
            response = call_api(f"{BASE_URL}/marketing_strategy/", {
                "product_description": product_description,
                "market_fit": market_fit
            })
            if response:
                st.success("Marketing Strategy Generated")
                st.write(response['marketing_strategy'])
        else:
            st.warning("Please provide product description and market fit analysis.")
            
elif option == "Business Plan Generation":
    sustainability_report = st.text_area("Sustainability Report:", height=100)
    business_model = st.text_area("Business Model:", height=100)
    market_fit = st.text_area("Market Fit:", height=100)
    marketing_strategy = st.text_area("Marketing Strategy:", height=100)
    st.subheader("📑 Generate Full Business Plan")
    
    if st.button("Generate Business Plan"):
        if all([product_description, sustainability_report, business_model, market_fit, marketing_strategy]):
            response = call_api(f"{BASE_URL}/business_plan/", {
                "sustainability_report": sustainability_report,
                "business_model": business_model,
                "market_fit": market_fit,
                "marketing_strategy": marketing_strategy
            })
            if response:
                st.success("Business Plan Generated")
                st.write(response['business_plan'])
        else:
            st.warning("Please fill in all the required fields.")

elif option == "ESG Metrics":
    st.subheader("📊 Get ESG Metrics")
    if st.button("Generate ESG Metrics"):
        if product_description:
            response = call_api(f"{BASE_URL}/esg_metrics/", {"product_description": product_description})
            if response:
                st.success("ESG Metrics Generated")
                st.write(response['esg_metrics'])
        else:
            st.warning("Please provide a product description.")
            
elif option == "Impact Prediction":
    st.subheader("🌍 Predict Long-Term Impact")
    if st.button("Predict Impact"):
        if product_description:
            response = call_api(f"{BASE_URL}/impact_prediction/", {"product_description": product_description})
            if response:
                st.success("Impact Prediction Generated")
                st.write(response['impact_prediction'])
        else:
            st.warning("Please provide a product description.")


elif option == "Product Image Generation":
    st.subheader("🖼️ Generate Product Image")
    prompt = st.text_area("Enter a prompt for image generation:", placeholder="e.g., 'A futuristic eco-friendly smartwatch'")
    
    if st.button("Generate Image"):
        if prompt:
                time.sleep(5)
            # Call the image generation API

                image = Image.open("./Generated.jpg")
                # image = image.resize((50, 50))
                st.image(image, caption="Generated Product Image", use_column_width=True)
        else:
                st.error("Failed to generate image. Please try again.")
      

# Add Lottie animation to the page
if lottie_eco_cyborg:
    st_lottie(lottie_eco_cyborg, height=250, key="eco_cyborg")
else:
    st.error("Failed to load animation. Please check the file path.")

# Footer with animation
st.markdown("""
    <style>
    @keyframes fade-in {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .footer {
        font-size: 14px;
        color: #888;
        text-align: center;
        margin-top: 50px;
        animation: fade-in 3s ease;
    }
    </style>
    <div class="footer">
        © 2024 Artemis AI - Powered by Accenture | Driving Sustainability through Innovation 🌍
    </div>
""", unsafe_allow_html=True)
