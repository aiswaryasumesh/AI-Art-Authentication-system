import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Art Authenticator",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔍 AI Art Authenticator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered detection of AI-generated artwork'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            "resnet50_cifake_3epochs.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader("📤 Upload Artwork")

uploaded_file = st.file_uploader(
    "Choose an image to authenticate",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# IMAGE DISPLAY
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Artwork",
        width="stretch"
    )

    st.write("")

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Artwork",
        use_container_width=True
    ):

        with st.spinner("Analyzing artwork..."):

            image_tensor = transform(image)

            image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(device)

            with torch.no_grad():

                outputs = model(image_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                predicted_class = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

            confidence = (
                probabilities[
                    0,
                    predicted_class
                ].item() * 100
            )

        # ====================================================
        # RESULT
        # ====================================================

        class_names = [
            "AI-GENERATED",
            "REAL"
        ]

        prediction = class_names[predicted_class]

        st.divider()

        st.subheader("📊 Authentication Result")

        if prediction == "AI-GENERATED":

            st.error(
                f"🤖 AI-GENERATED\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        else:

            st.success(
                f"✅ REAL\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        # ====================================================
        # DETAILS
        # ====================================================

        st.subheader("🔎 Analysis Details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                prediction
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.info(
            f"Model: ResNet-50\n\n"
            f"Processing device: {device}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Art Authentication System | ResNet-50"
)