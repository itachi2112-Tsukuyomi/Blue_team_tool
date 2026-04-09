import os
import sys
import pickle
import torch
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import URLTokenizer
from src.model_builder import ModelBuilder
from src.config import CONFIG
from src.network_monitor import capture_and_analyze
from src.alerting import send_email_alert

st.set_page_config(page_title="Phishlink Detector", layout="wide", page_icon="🎣")

@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = URLTokenizer()
    if os.path.exists("models/tokenizer.json"):
        tokenizer.load("models/tokenizer.json")
    else:
        st.warning("Tokenizer not found! Run training first.")
        
    model = ModelBuilder.build_cnn(num_chars=tokenizer.vocab_size)
    if os.path.exists("models/url_detector.pth"):
        model.load_state_dict(torch.load("models/url_detector.pth", map_location=device))
        model.eval()
        model.to(device)
    else:
        st.warning("Model weights not found! Run training first.")
        
    calibrator = None
    if os.path.exists("models/calibrator.pkl"):
        with open("models/calibrator.pkl", "rb") as f:
            calibrator = pickle.load(f)
            
    return tokenizer, model, calibrator, device

tokenizer, model, calibrator, device = load_assets()

def predict_single(url, threshold=0.5):
    orig_url = url
    url = url.strip().lower()
    if not url.startswith(('http://', 'https://', 'ftp://')):
        url = 'http://' + url
    encoded = tokenizer.encode(url)
    inputs = torch.tensor([encoded], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model(inputs).squeeze(-1)
        raw_prob = torch.sigmoid(logits).cpu().item()
        
    calibrated_prob = raw_prob
    if calibrator:
        X_logits = np.log(np.clip(raw_prob, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_prob, 1e-7, 1 - 1e-7))).reshape(1, -1)
        calibrated_prob = calibrator.predict_proba(X_logits)[0, 1]
        
    label = "Phishing" if calibrated_prob >= threshold else "Benign"
    
    # Risk band
    if calibrated_prob >= 0.90:
        risk = "Very likely phishing"
        color = "red"
    elif calibrated_prob >= 0.50:
        risk = "Suspicious"
        color = "orange"
    else:
        risk = "Probably safe (verify manually)"
        color = "green"
        
    return raw_prob, calibrated_prob, label, risk, color

def predict_batch(urls, threshold=0.5, batch_size=256):
    results = []
    
    # Preprocess urls
    cleaned_urls = []
    for u in urls:
        u = str(u).strip().lower()
        if not u.startswith(('http://', 'https://', 'ftp://')):
            u = 'http://' + u
        cleaned_urls.append(u)
    
    for i in range(0, len(cleaned_urls), batch_size):
        batch_urls = cleaned_urls[i:i+batch_size]
        orig_batch_urls = urls[i:i+batch_size]
        encoded_batch = tokenizer.batch_encode(batch_urls)
        inputs = torch.tensor(encoded_batch, dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = model(inputs).squeeze(-1)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            raw_probs = torch.sigmoid(logits).cpu().numpy()
            
        for j, raw_prob in enumerate(raw_probs):
            calibrated_prob = raw_prob
            if calibrator:
                X_logits = np.log(np.clip(raw_prob, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_prob, 1e-7, 1 - 1e-7))).reshape(1, -1)
                calibrated_prob = calibrator.predict_proba(X_logits)[0, 1]
                
            label = "Phishing" if calibrated_prob >= threshold else "Benign"
            
            if calibrated_prob >= 0.90:
                risk = "Very likely phishing"
            elif calibrated_prob >= 0.50:
                risk = "Suspicious"
            else:
                risk = "Probably safe (verify manually)"
                
            results.append({
                "url": orig_batch_urls[j],
                "raw_prob": float(raw_prob),
                "calibrated_prob": float(calibrated_prob),
                "label": label,
                "risk_band": risk
            })
            
    return results

st.title("🎣 Phishlink: Deep Learning URL Detector")
st.markdown("This dashboard uses a character-level 1D-CNN to detect phishing URLs.")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single URL Prediction", "📂 Batch Processing", "📊 Model Evaluation", "🌐 Network Monitor"])

with tab1:
    st.subheader("Analyze a single URL")
    url_input = st.text_input("Enter a URL to check:")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)
    
    if st.button("Analyze") and url_input:
        raw_p, calib_p, label, risk, color = predict_single(url_input, threshold)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Raw Model Probability", value=f"{raw_p*100:.2f}%")
        with col2:
            st.metric(label="Calibrated Probability", value=f"{calib_p*100:.2f}%")
            
        st.markdown(f"### Verdict: **{label}**")
        st.markdown(f"#### Risk Band: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
        
        # Simple Explainer visualization
        st.markdown("#### Explainability (Gradient-based Character Importance)")
        with st.spinner("Generating saliency..."):
            from src.explain import Explainer
            explainer = Explainer(model, tokenizer, device)
            # Make sure gradients can be tracked
            chars, saliency = explainer.explain_url(url_input, filename="tmp_saliency.png")
            st.image(os.path.join(explainer.explain_dir, "tmp_saliency.png"), width="stretch")

with tab2:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV file with a `url` column.")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'url' not in df.columns:
            st.error("CSV must contain a 'url' column.")
        else:
            with st.spinner("Processing..."):
                urls = df['url'].astype(str).tolist()
                results = predict_batch(urls)
                
                res_df = pd.DataFrame(results)
                st.dataframe(res_df)
                
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

with tab3:
    st.subheader("Model Evaluation & Training Metrics")
    fig_dir = "results/figures"
    
    if os.path.exists(os.path.join(fig_dir, "roc_curve.png")):
        col1, col2 = st.columns(2)
        col1.image(os.path.join(fig_dir, "roc_curve.png"), caption="ROC Curve", width="stretch")
        col2.image(os.path.join(fig_dir, "precision_recall.png"), caption="Precision-Recall Curve", width="stretch")
        
        col3, col4 = st.columns(2)
        if os.path.exists(os.path.join(fig_dir, "conf_matrix.png")):
            col3.image(os.path.join(fig_dir, "conf_matrix.png"), caption="Confusion Matrix", width="stretch")
        if os.path.exists(os.path.join(fig_dir, "calibration_plot.png")):
            col4.image(os.path.join(fig_dir, "calibration_plot.png"), caption="Calibration Plot", width="stretch")
        
        if os.path.exists(os.path.join(fig_dir, "training_curve.png")):
            st.image(os.path.join(fig_dir, "training_curve.png"), caption="Training Curves", width="stretch")
    else:
        st.info("Evaluation figures not found. Run the training and evaluation pipeline first.")

with tab4:
    st.subheader("🌐 Network Monitoring (Wireshark Integration)")
    st.markdown("Monitor network traffic for accesses to identified phishing urls and send email alerts.")
    
    st.info("⚠️ Ensure Wireshark/tshark is installed on your system or pyshark will not work. You may also need administrative privileges.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Network Configuration**")
        interface = st.text_input("Network Interface (e.g., 'Wi-Fi', 'Ethernet', 'eth0')", value="Wi-Fi")
        duration = st.number_input("Monitoring Duration (seconds)", min_value=5, max_value=600, value=15)
        
        enable_ai_scan = st.toggle("Enable Live AI Phishing Detection (Scan all traffic)", value=True, 
                                   help="If enabled, all intercepted traffic will be passed through the Phishlink Deep Learning model. Alerts will trigger automatically if a high-risk score is detected.")
                                   
        target_domains_str = st.text_area("Specific Phishing Domains to Monitor (comma-separated)", value="example-phish.com", disabled=enable_ai_scan)
        
        target_domains = set()
        if not enable_ai_scan:
            target_domains = {d.strip() for d in target_domains_str.split(",") if d.strip()}
        
    with col2:
        st.markdown("**Alerting Configuration**")
        sender_email = st.text_input("Sender Gmail Address", value="", help="Enter the Gmail address used to send the alerts.")
        sender_password = st.text_input("Sender App Password", value="", type="password", help="Use a Google App Password (not your main password) if 2FA is enabled.")
        recipient_email = st.text_input("Recipient Email for Alerts", value="")

        
    if st.button("Start Network Monitoring", type="primary"):
        if not enable_ai_scan and not target_domains:
            st.error("Please provide at least one target domain to monitor, or enable AI scanning.")
        else:
            with st.spinner(f"Capturing packets on '{interface}' for {duration} seconds..."):
                try:
                    raw_events = capture_and_analyze(interface, target_domains, int(duration))
                except Exception as e:
                    raw_events = []
                    st.error(f"Error starting capture: {e}")
            
            # Post-processing: Filter raw events through AI if AI scanning is enabled
            events = []
            if raw_events:
                if enable_ai_scan:
                    with st.spinner("Analyzing captured traffic with Phishlink AI..."):
                        for event in raw_events:
                            domain = event.get("domain", "")
                            if domain:
                                raw_p, calib_p, label, risk, color = predict_single(domain, threshold=0.5)
                                if label == "Phishing":
                                    # Modify event to include AI score
                                    event["ai_risk_score"] = float(calib_p) * 100.0
                                    event["risk_band"] = risk
                                    events.append(event)
                else:
                    # Conventional matching, everything returned by capture_and_analyze is already filtered
                    events = raw_events
            
            if events:
                if enable_ai_scan:
                    st.error(f"🚨 AI Detected {len(events)} suspicious domain access(es)!")
                else:
                    st.warning(f"🚨 Detected {len(events)} access(es) to known phishing domains!")
                st.dataframe(events)
                
                # Send email if configured
                if sender_email and sender_password and recipient_email:
                    with st.spinner("Sending email alert..."):
                        success, msg = send_email_alert(sender_email, sender_password, recipient_email, events)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                else:
                    st.info("Email not configured. Skipping alert.")
            else:
                if raw_events and enable_ai_scan:
                    st.success(f"✅ Intercepted {len(raw_events)} domains, but the AI classified all of them as safe.")
                else:
                    st.success("✅ No accesses to the specified phishing domains detected during the capturing window.")

