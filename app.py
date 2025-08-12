import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pulmonary Fibrosis Detection",
    page_icon="🩺",
    layout="centered", # Keep this as centered
    initial_sidebar_state="auto"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the pre-trained ViT model and image processor."""
    try:
        model_path = './FibroViT Pretrained Model' 
        model = ViTForImageClassification.from_pretrained(model_path)
        image_processor = ViTImageProcessor.from_pretrained(model_path)
        return model, image_processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Corrected the error message to match the actual folder name
        st.error("Please ensure the 'FibroViT Pretrained Model' folder is in the same directory.")
        return None, None

model, image_processor = load_model()

# ---  UI STYLING ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app background */
    .stApp {
        background-image: linear-gradient( 178.3deg,  rgba(171,245,173,1) 13.8%, rgba(39,161,255,1) 90.9% );
        font-family: 'Inter', sans-serif;
    }

    /* Force the main content block to be wider than the Streamlit default. */
    .main .block-container {
        max-width: 1000px !important;
        padding: 3rem 3rem 4rem 3rem !important;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        margin: 2rem auto;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Headers */
    h1 { 
        text-align: center; 
        color: #2c3e50; 
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 { 
        text-align: center; 
        color: #34495e; 
        font-weight: 600;
    }
    
    .sub-header { 
        text-align: center; 
        color: #6c757d; 
        margin-bottom: 3rem; 
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div > div > div {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%) !important;
        border: 2px dashed rgba(102, 126, 234, 0.6) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    /* Hide the default file list */
    [data-testid="stFileUploaderFileList"] { 
        display: none; 
    }

    /* Style the uploaded image */
    div[data-testid="stImage"] { 
        text-align: center; 
        margin: 2rem 0; 
    }
    
    div[data-testid="stImage"] img { 
        margin: auto; 
        border: 4px solid #ffffff; 
        border-radius: 16px; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        max-width: 100%;
        height: auto;
    }
    
    /* Center the button */
    div[data-testid="stButton"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 280px !important;
        height: 60px !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
        color: #667eea !important;
        border: 2px solid #667eea !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border-color: #667eea !important;
        transform: translateY(-3px) !important;
    }

    /* Results box styling */
    .results-container { 
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e9ecef;
        border-left: 6px solid #667eea;
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .results-container .prediction-label,
    .results-container .confidence-label {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .results-container .prediction-value {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .results-container .confidence-value {
        display: inline-block;
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1) !important;
        margin-top: 1.5rem !important;
        backdrop-filter: blur(10px) !important;
        background-color: rgba(255, 255, 255, 0.8) !important;
    }
    
    /* --- STYLES FOR THE ENHANCED FOOTER --- */
    .footer-container {
        text-align: center;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    .footer-container p {
        margin: 0.5rem 0;
        font-size: 1rem;
        color: #4a5568; /* A nice, readable dark gray */
    }
    .footer-container a {
        color: #39a1ff; /* A bright blue from your background gradient */
        text-decoration: none;
        font-weight: 600;
    }
    .footer-container a:hover {
        text-decoration: underline;
        color: #2c3e50;
    }
    .social-links {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.5rem;
        margin-top: 1.5rem;
    }
    .social-links a {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background-color: #f0f2f6;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        color: #2d3748;
        font-weight: 600;
    }
    .social-links a:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-decoration: none;
        background-color: #e2e8f0;
    }
    .footer-disclaimer {
        margin-top: 2rem;
        font-size: 0.9rem;
        font-style: italic;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("🩺 AI-Powered Pulmonary Fibrosis Detection")
st.markdown('<p class="sub-header">Upload a chest CT scan to predict the likelihood of pulmonary fibrosis using advanced AI technology.</p>', unsafe_allow_html=True)


# --- FILE UPLOADER ---
uploaded_file = st.file_uploader(
    "📁 Drag and drop a CT scan image here", 
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.info(f"📋 **File ready for analysis:** {uploaded_file.name}", icon="✅")
    st.image(image, caption='🔍 Uploaded CT Scan for Analysis', use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button('🔬 ANALYZE IMAGE', key="analyze_btn", type="primary", use_container_width=True):
        if model is not None and image_processor is not None:
            with st.spinner('🔬 Analyzing CT scan... Please wait'):
                try:
                    inputs = image_processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                    
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    predicted_class_idx = probabilities.argmax(-1).item()
                    predicted_class_label = model.config.id2label[predicted_class_idx]
                    confidence = probabilities[0][predicted_class_idx].item() * 100
                    
                    st.markdown(f"""
                    <div class="results-container">
                        <div class="prediction-label">🎯 Prediction Result</div>
                        <div class="prediction-value">{predicted_class_label}</div>
                        <div class="confidence-label">📊 Confidence Level</div>
                        <div class="confidence-value">{confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if predicted_class_label.lower() == "fibrosis":
                        st.error("⚠️ **Model Prediction:** Signs consistent with Pulmonary Fibrosis detected. Please consult with a healthcare professional for proper diagnosis and treatment.", icon="🚨")
                    else:
                        st.success("✅ **Model Prediction:** No significant signs of Pulmonary Fibrosis detected (Normal). Regular monitoring is still recommended.", icon="🎉")

                    st.info("💡 **Note:** This AI analysis is a screening tool and should be used in conjunction with clinical expertise.", icon="ℹ️")

                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {e}", icon="🔧")
                    st.error("Please try uploading the image again or contact support if the issue persists.")
        else:
            st.error("🚫 Model is not loaded. Cannot perform analysis. Please check your model files.", icon="❌")

else:
    # Show helpful information when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; margin: 2rem 0;">
        <h3 style="color: #6c757d; margin-bottom: 1rem;">📤 Ready to Analyze</h3>
        <p style="color: #6c757d; font-size: 1.1rem;">Upload a chest CT scan image to get started with AI-powered pulmonary fibrosis detection.</p>
        <div style="margin-top: 1.5rem;">
            <span style="display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: white; border-radius: 20px; color: #667eea; font-weight: 600; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">PNG</span>
            <span style="display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: white; border-radius: 20px; color: #667eea; font-weight: 600; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">JPG</span>
            <span style="display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: white; border-radius: 20px; color: #667eea; font-weight: 600; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">JPEG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- ENHANCED FOOTER ---
st.markdown("---")

# Step 1: Your links are defined as Python string variables here
GITHUB_URL = "https://github.com/waseemkathia"
LINKEDIN_URL = "https://www.linkedin.com/in/waseemkathia/"
WEBSITE_URL = "https://waseemkathia.github.io/"
PAPER_URL = "https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2023.1282200/full"

# Step 2: The f-string uses the variable names inside the curly braces {}
st.markdown(f"""
<div class="footer-container">
    <h3>Connect & Support</h3>
    <p>
        This application is an open-source implementation of research published in 
        <a href="{PAPER_URL}" target="_blank">Frontiers in Medicine</a>.
    </p>
    <p>
        If you find this tool useful, please consider giving the project a ⭐ on GitHub and connecting with me on LinkedIn.
    </p>
    <div class="social-links">
        <a href="{GITHUB_URL}" target="_blank">⭐ GitHub</a>
        <a href="{LINKEDIN_URL}" target="_blank">🔗 LinkedIn</a>
        <a href="{WEBSITE_URL}" target="_blank">🌐 My Website</a>
    </div>
    <div class="footer-disclaimer">
        <p>
            ⚕️ This tool is for educational and research purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult with qualified healthcare professionals for medical concerns.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)