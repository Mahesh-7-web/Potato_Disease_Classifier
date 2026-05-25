import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Set page configuration with a custom page title and layout
st.set_page_config(
    page_title="Potato Leaf Disease Classifier",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load your trained model
@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model("my_model.keras")

try:
    model = load_disease_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Inject Custom CSS for premium glassmorphism styling
st.markdown("""
<style>
/* Hide Streamlit default styling elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom premium container styling */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
}

/* Gradient Header */
.main-title {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.6rem;
    margin-bottom: 5px;
}

.subtitle {
    color: #9CA3AF;
    font-size: 1.1rem;
    margin-bottom: 25px;
}

/* Colored Alerts for Predictions */
.result-box {
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.healthy-bg {
    background: rgba(16, 185, 129, 0.1);
    border-left: 5px solid #10B981;
}
.early-bg {
    background: rgba(245, 158, 11, 0.1);
    border-left: 5px solid #F59E0B;
}
.late-bg {
    background: rgba(239, 68, 68, 0.1);
    border-left: 5px solid #EF4444;
}

.badge {
    padding: 4px 8px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 0.85rem;
}
.badge-healthy { background-color: #10B981; color: white; }
.badge-early { background-color: #F59E0B; color: black; }
.badge-late { background-color: #EF4444; color: white; }

.section-title {
    font-weight: bold;
    font-size: 1.25rem;
    color: #F3F4F6;
    margin-bottom: 15px;
}

/* Progress bar customization */
.stProgress > div > div > div > div {
    background-color: #10B981;
}
</style>
""", unsafe_allow_html=True)

# Function to preprocess image for prediction
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to the input shape of your model
    image = np.array(image, dtype=np.float32)  # Convert to NumPy array

    # Check if the image has 3 channels (RGB)
    if image.shape[-1] == 3:
        # Normalize pixel values to [0, 1]
        image = np.clip(image, 1, 256)  

        # Convert to TensorFlow tensor and add batch dimension
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        return image
    else:
        raise ValueError("Uploaded image is not in RGB format.")

# Function to make a prediction
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return predictions[0]

# --- Sidebar ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1518977676601-b53f82aba655?q=80&w=400&auto=format&fit=crop", use_column_width=True)
    st.markdown("### 🌾 About the Project")
    st.info(
        "This deep learning classifier analyzes potato leaf images to diagnose crop diseases. "
        "Built with TensorFlow/Keras and deployed as a real-time web application for agriculture tech demos."
    )
    st.markdown("---")
    st.markdown("### 🔍 Diagnoseable Diseases")
    st.markdown("🥬 **Healthy Potato**: Spot-free, fully functional leaves.")
    st.markdown("🍂 **Early Blight**: Caused by *Alternaria solani*, shows target-board spots.")
    st.markdown("🌧️ **Late Blight**: Caused by *Phytophthora infestans*, highly destructive water-soaked lesions.")

# --- Main Layout ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<h1 class='main-title'>🥔 Potato Leaf Health</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Diagnostic Assistant for Agriculture</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📤 Diagnostic Upload</div>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Upload a high-quality JPG image of a potato leaf to begin classification...",
        type=["jpg", "jpeg"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🖼️ Leaf Preview</div>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_image is not None:
        st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Analysis Results</div>", unsafe_allow_html=True)
        
        with st.spinner("Analyzing leaf structure..."):
            predictions = predict(image)
        
        class_names = ['Early Blight', 'Late Blight', 'Healthy']
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100

        # Custom result boxes depending on health state
        if predicted_class == 'Healthy':
            bg_class = "healthy-bg"
            badge_class = "badge-healthy"
            description = "The leaf exhibits normal structure with no visible pathogenetic activity."
            treatment = (
                "**Recommended Action:**\n"
                "* Keep regular watering schedule without drowning the soil.\n"
                "* Monitor crop spacing to avoid microclimate issues."
            )
        elif predicted_class == 'Early Blight':
            bg_class = "early-bg"
            badge_class = "badge-early"
            description = "Detected concentric dark rings on leaf tissue, symptomatic of *Alternaria solani*."
            treatment = (
                "**Treatment Plan:**\n"
                "* **Foliar Removal:** Prune infected lower leaves immediately to stop vertical spread.\n"
                "* **Fungicides:** Apply copper-based fungicides at regular intervals.\n"
                "* **Irrigation:** Use drip irrigation instead of overhead sprayers to keep foliage dry."
            )
        else:
            bg_class = "late-bg"
            badge_class = "badge-late"
            description = "Detected water-soaked leaf spots, characteristic of *Phytophthora infestans*. Highly contagious."
            treatment = (
                "**Urgent Treatment Plan:**\n"
                "* **Quarantine:** Instantly remove and destroy heavily infected plants.\n"
                "* **Chemical Control:** Apply chlorothalonil or protectant fungicides immediately.\n"
                "* **Ventilation:** Prune adjacent non-potato plants to increase airflow."
            )

        st.markdown(f"""
        <div class="result-box {bg_class}">
            <h3 style="margin: 0 0 10px 0;">Diagnosis: <span class="badge {badge_class}">{predicted_class}</span></h3>
            <p style="margin: 0; font-size: 1rem; color: #E5E7EB;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ")
        st.markdown(f"**Confidence Level**: `{confidence:.2f}%`")
        st.progress(int(confidence))
        
        st.markdown("---")
        st.markdown("<div class='section-title'>🛡️ Recommended Treatment & Care</div>", unsafe_allow_html=True)
        st.markdown(treatment)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card' style='text-align: center; padding: 50px 20px;'>", unsafe_allow_html=True)
        st.markdown("<h3>💡 Ready for Diagnosis</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9CA3AF;'>Upload a leaf image on the left panel to trigger the model and display diagnostic treatment advice.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
