# ═══════════════════════════════════════════════════════════════════════════
# app/app.py  –  Car Advisor AI  (ZHAW AI Applications Project)
# ═══════════════════════════════════════════════════════════════════════════
#
# Memory strategy
# ───────────────
# CV  (torch/ResNet18)       : loaded inside cv_predict(), deleted by caller
# ML  (sklearn/joblib)       : loaded inside ml_predict(), deleted before return
# RAG (faiss + transformers) : loaded once on first need, kept in session_state
# Rule: CV and RAG are never in memory at the same time
# ═══════════════════════════════════════════════════════════════════════════

import gc
import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv

# Env vars must come before any torch import
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Import torch NOW at startup so it is fully initialised before Streamlit's
# file-watcher scans torch.classes.__path__.  On Apple Silicon, the watcher
# access corrupts torch's C++ class registry if torch is only half-loaded,
# causing a fatal segfault the first time cv_predict() tries to use it.
import torch
torch.set_default_device('cpu')   # all ops default to CPU — no MPS usage anywhere

warnings.filterwarnings('ignore')
print("App starting...", flush=True)

# ── Paths ────────────────────────────────────────────────────────────────────
# Auto-detect layout:
#   Local  : app.py lives in project/app/ → models at project/models/   (../models)
#   HF     : app.py lives in project/     → models at project/models/   (models)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = (
    os.path.join(BASE_DIR, 'models')
    if os.path.isdir(os.path.join(BASE_DIR, 'models'))
    else os.path.normpath(os.path.join(BASE_DIR, '..', 'models'))
)
DATA_DIR = (
    os.path.join(BASE_DIR, 'data')
    if os.path.isdir(os.path.join(BASE_DIR, 'data'))
    else os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))
)

def mp(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)

def dp(filename: str) -> str:
    return os.path.join(DATA_DIR, 'processed', filename)

# ── Load .env (no Streamlit calls yet) ──────────────────────────────────────
load_dotenv(os.path.join(BASE_DIR, '.env'))
if not os.environ.get('OPENAI_API_KEY'):
    load_dotenv(os.path.join(BASE_DIR, '..', '.env'))  # also try project root

# ── Page config — MUST be the very first Streamlit call ─────────────────────
st.set_page_config(
    page_title='Car Advisor AI',
    page_icon='🏎️',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
/* ── Hide/fix Streamlit top header ── */
header[data-testid="stHeader"] {
    background-color: #f0f4f0 !important;
    border-bottom: none !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ── Background ── */
.stApp {
    background-color: #f0f4f0;
    color: #1a1a1a;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3a1a 0%, #2d5a2d 100%);
    border-right: 3px solid #c9a84c;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #f0f0f0 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #c9a84c !important;
}

/* ── Titles ── */
h1 {
    color: #c9a84c !important;
    font-family: 'Arial Black', sans-serif;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 3px solid #2d5a2d;
    padding-bottom: 8px;
}
h2, h3 {
    color: #2d5a2d !important;
    font-weight: bold !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #2d5a2d, #3d7a3d) !important;
    color: #c9a84c !important;
    border: 2px solid #c9a84c !important;
    border-radius: 6px !important;
    font-weight: bold !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #c9a84c, #d4b86a) !important;
    color: #1a3a1a !important;
    box-shadow: 0 4px 15px rgba(201,168,76,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #e8f0e8 !important;
    border-bottom: 3px solid #2d5a2d !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #555555 !important;
    font-weight: bold !important;
    border-radius: 6px 6px 0 0 !important;
    transition: all 0.3s ease !important;
    padding: 10px 20px !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #c9a84c !important;
    background-color: #2d5a2d !important;
    transform: translateY(-2px) !important;
}
.stTabs [aria-selected="true"] {
    color: #c9a84c !important;
    background-color: #2d5a2d !important;
    border-bottom: 3px solid #c9a84c !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: white;
    border: 2px solid #2d5a2d;
    border-top: 4px solid #c9a84c;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
[data-testid="stMetricLabel"] {
    color: #2d5a2d !important;
    font-weight: bold !important;
}
[data-testid="stMetricValue"] {
    color: #c9a84c !important;
    font-weight: bold !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: white !important;
    border: 1px solid #2d5a2d !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr {
    border-color: #2d5a2d !important;
    opacity: 0.3;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: white !important;
    border: 2px dashed #c9a84c !important;
    border-radius: 10px !important;
}

/* ── Progress bars ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #2d5a2d, #c9a84c) !important;
}

/* ── Info boxes ── */
[data-testid="stInfo"] {
    background-color: #e8f4e8 !important;
    border-left: 4px solid #2d5a2d !important;
}
</style>
""", unsafe_allow_html=True)

# ── API key — st.secrets only tried if a secrets file actually exists ────────
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not OPENAI_API_KEY:
    _secrets_candidates = [
        os.path.expanduser('~/.streamlit/secrets.toml'),
        os.path.join(BASE_DIR, '.streamlit', 'secrets.toml'),
        os.path.join(BASE_DIR, '..', '.streamlit', 'secrets.toml'),
    ]
    if any(os.path.exists(p) for p in _secrets_candidates):
        try:
            OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY', '')
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════
# Helper functions — heavy imports are deferred inside each function
# ═══════════════════════════════════════════════════════════════════════════

def lakh_to_chf(lakh_price: float) -> int:
    """Convert Indian Lakh INR to CHF, rounded to nearest 10."""
    return round(lakh_price * 100_000 * 0.011 / 10) * 10


def get_brand_list() -> list:
    """Load brand encoder, extract class names, free encoder immediately."""
    import joblib
    enc   = joblib.load(mp('brand_encoder.pkl'))
    names = list(enc.classes_)
    del enc
    gc.collect()
    return names


def cv_predict(image: Image.Image, top_k: int = 3):
    """
    Load ResNet18, run inference, return (predictions, model).
    Caller MUST: del model; gc.collect()
    """
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models, transforms

    # Explicitly pin to CPU — MPS (Apple Silicon) causes a fatal segfault
    device = torch.device('cpu')

    with open(mp('class_names.json')) as f:
        class_names = json.load(f)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(image.convert('RGB')).unsqueeze(0).to(device)

    net = tv_models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(class_names))
    net.load_state_dict(
        torch.load(mp('car_brand_classifier.pth'), map_location=device, weights_only=False)
    )
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        probs = torch.softmax(net(tensor), dim=1)[0]
    topk = torch.topk(probs, top_k)
    predictions = [
        (class_names[i.item()], topk.values[j].item())
        for j, i in enumerate(topk.indices)
    ]
    return predictions, net


def ml_predict(brand, year, km_driven, fuel_type, transmission,
               mileage, engine, power, seats, owner_type) -> float:
    """
    Load ML models, predict resale price in Lakh INR, delete models before return.
    All sklearn/joblib objects are freed inside this function.
    """
    import joblib

    predictor     = joblib.load(mp('price_predictor.pkl'))
    scaler        = joblib.load(mp('scaler.pkl'))
    brand_encoder = joblib.load(mp('brand_encoder.pkl'))
    with open(dp('feature_columns.json')) as f:
        feature_columns = json.load(f)

    car_age = 2024 - year
    raw_num = np.array([[km_driven, engine, power, mileage, car_age]], dtype=float)
    scaled  = scaler.transform(raw_num)[0]
    km_s, eng_s, pwr_s, mil_s, age_s = scaled

    try:
        brand_enc = float(brand_encoder.transform([brand])[0])
    except (ValueError, AttributeError):
        brand_enc = float(len(brand_encoder.classes_) // 2)

    row = pd.DataFrame([{
        'Kilometers_Driven':         km_s,
        'Mileage':                   mil_s,
        'Engine':                    eng_s,
        'Power':                     pwr_s,
        'Seats':                     float(seats),
        'Brand':                     brand_enc,
        'Car_Age':                   age_s,
        'Fuel_Type_Diesel':          float(fuel_type == 'Diesel'),
        'Fuel_Type_Petrol':          float(fuel_type == 'Petrol'),
        'Transmission_Automatic':    float(transmission == 'Automatic'),
        'Transmission_Manual':       float(transmission == 'Manual'),
        'Owner_Type_First':          float(owner_type == 'First'),
        'Owner_Type_Fourth & Above': float(owner_type == 'Fourth & Above'),
        'Owner_Type_Second':         float(owner_type == 'Second'),
        'Owner_Type_Third':          float(owner_type == 'Third'),
    }])[feature_columns]

    price = round(float(predictor.predict(row)[0]), 2)

    del predictor, scaler, brand_encoder
    gc.collect()
    return price


def load_rag() -> dict:
    """
    Load all RAG components. Returns a dict stored in session_state['rag'].
    Called at most once per session.
    """
    import faiss
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI

    idx = faiss.read_index(mp('faiss_index.bin'))
    with open(mp('rag_documents.json'), encoding='utf-8') as f:
        docs = json.load(f)
    with open(mp('rag_config.json')) as f:
        config = json.load(f)
    # device='cpu' — MPS (Apple Silicon GPU) segfaults during encode() on this torch version
    embedder = SentenceTransformer(config.get('embedding_model', 'all-MiniLM-L6-v2'), device='cpu')
    client   = OpenAI(api_key=OPENAI_API_KEY)
    return {'idx': idx, 'docs': docs, 'embedder': embedder, 'client': client}


def rag_answer(question: str, rag: dict, k: int = 3):
    """Retrieve top-k docs then generate a grounded answer. Returns (answer, context_docs)."""
    import faiss as _faiss

    q_vec = rag['embedder'].encode([question], convert_to_numpy=True).astype('float32')
    _faiss.normalize_L2(q_vec)
    _, indices     = rag['idx'].search(q_vec, k)
    context_docs   = [rag['docs'][i] for i in indices[0] if i < len(rag['docs'])]
    context_text   = '\n\n'.join(context_docs)

    system_msg = (
        'You are an expert car advisor assistant specialising in the Indian used-car market. '
        'Provide concise, accurate, and helpful answers based on the provided context.'
    )
    user_msg = (
        f'<context>\n{context_text}\n</context>\n\n'
        f'<question>\n{question}\n</question>\n\n'
        'Answer based on the context above. Be concise and helpful. '
        'If the context lacks information, share general knowledge clearly.'
    )
    try:
        resp = rag['client'].chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user',   'content': user_msg},
            ],
            max_tokens=350,
            temperature=0.3,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as exc:
        answer = f'[API Error] {exc}'

    return answer, context_docs


def ensure_rag() -> bool:
    """
    Load RAG into session_state['rag'] if not already loaded.
    Returns True on success, False on failure (error stored in session_state['rag_error']).
    """
    if 'rag' in st.session_state:
        return True
    if 'rag_error' in st.session_state:
        return False
    if not OPENAI_API_KEY:
        st.session_state['rag_error'] = 'OPENAI_API_KEY is not set.'
        return False
    try:
        with st.spinner('Loading AI advisor (one-time setup)…'):
            st.session_state['rag'] = load_rag()
        return True
    except Exception as exc:
        st.session_state['rag_error'] = str(exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('## 🚗 Car Advisor AI')
    st.markdown(
        'An end-to-end AI pipeline combining **Computer Vision**, '
        '**Machine Learning**, and **Natural Language Processing** '
        'to help you make smarter used-car decisions.'
    )
    st.divider()

    st.markdown('### How it works')
    st.markdown(
        '**Step 1 — 📷 Photo Analysis**  \n'
        'Upload a car photo. ResNet18 identifies the brand from the image.\n\n'
        '**Step 2 — 💰 Price Estimation**  \n'
        'Enter car specs. A Random Forest model (R² = 0.91) predicts the fair resale price in CHF.\n\n'
        '**Step 3 — 🤖 AI Explanation**  \n'
        'GPT-3.5-turbo, backed by a FAISS knowledge base, explains the prediction and gives tailored buying advice.'
    )
    st.divider()

    st.markdown('### Dataset note')
    st.markdown(
        'Image classification uses the **Auto_Bilder** dataset (42 car brands).  \n'
        'Price prediction uses the **Indian Used Car Market** dataset (~5,700 listings).'
    )
    st.caption('Prices converted from Indian market data to CHF')
    st.divider()
   


# ═══════════════════════════════════════════════════════════════════════════
# Page header
# ═══════════════════════════════════════════════════════════════════════════

st.title('🏎️ Car Advisor AI')
st.markdown('*AI-powered brand recognition, price prediction, and buying advice — all in one place.*')
st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# Tabs — all dynamic content is wrapped in a catch-all try/except
# ═══════════════════════════════════════════════════════════════════════════

try:
    tab1, tab2, tab3, tab4 = st.tabs([
        '🔍 Car Analyzer',
        '💬 Car Advisor Chatbot',
        '📊 Dataset Insights',
        'ℹ️ About',
    ])

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1 — Car Analyzer
    # ─────────────────────────────────────────────────────────────────────

    with tab1:
        st.header('Car Analyzer')
        st.markdown(
            'Upload a car photo to identify the brand, enter the car\'s specs, '
            'then get an estimated resale price and AI explanation.'
        )

        uploaded_file = st.file_uploader(
            'Upload a car image (JPG or PNG)',
            type=['jpg', 'jpeg', 'png'],
            label_visibility='visible',
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            col_img, col_cv = st.columns([1, 1], gap='large')

            with col_img:
                st.image(image, caption='Uploaded car image', use_column_width=True)

            with col_cv:
                st.subheader('Brand Recognition')
                if st.button('🔍 Analyze Car', type='primary', use_container_width=True):
                    try:
                        # Unload RAG before loading CV to keep both out of memory simultaneously
                        if 'rag' in st.session_state:
                            del st.session_state['rag']
                            gc.collect()

                        with st.spinner('Loading CV model and analysing image…'):
                            predictions, cv_model = cv_predict(image, top_k=3)
                        del cv_model
                        gc.collect()

                        top_brand, top_conf = predictions[0]
                        st.success(f'**Identified brand: {top_brand}**')

                        st.markdown('**Top-3 predictions:**')
                        for brand_name, conf in predictions:
                            st.markdown(f'`{brand_name}`')
                            st.progress(float(conf), text=f'{conf:.1%}')

                        st.session_state['cv_brand'] = top_brand
                        st.session_state['cv_conf']  = top_conf
                        st.session_state['cv_done']  = True
                    except Exception as exc:
                        st.error(f'Brand recognition failed: {exc}')

            # ── Price estimation ──────────────────────────────────────────
            if st.session_state.get('cv_done'):
                st.divider()
                st.subheader('Price Estimation')
                st.markdown('Fill in the car\'s details below, then click **Estimate Price**.')

                # Cache the brand list (list of strings, tiny) in session_state
                if 'ml_brands' not in st.session_state:
                    try:
                        st.session_state['ml_brands'] = get_brand_list()
                    except Exception:
                        st.session_state['ml_brands'] = []
                ml_brands = st.session_state['ml_brands'] or [
                    'Audi', 'BMW', 'Honda', 'Hyundai', 'Maruti', 'Toyota'
                ]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('**General**')
                    cv_brand    = st.session_state.get('cv_brand', ml_brands[0])
                    default_idx = ml_brands.index(cv_brand) if cv_brand in ml_brands else 0
                    brand_sel   = st.selectbox('Brand', ml_brands, index=default_idx)
                    year        = st.slider('Manufacturing Year', 2000, 2024, 2018, step=1)
                    km_driven   = st.slider('Kilometers Driven', 0, 300_000, 50_000, step=1_000,
                                            format='%d km')
                    owner_type  = st.selectbox('Owner Type',
                                               ['First', 'Second', 'Third', 'Fourth & Above'])

                with col2:
                    st.markdown('**Fuel & Transmission**')
                    fuel_type    = st.selectbox('Fuel Type',
                                                ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
                    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
                    seats        = st.selectbox('Seats', [2, 4, 5, 6, 7, 8], index=2)

                with col3:
                    st.markdown('**Engine Specs**')
                    mileage = st.number_input('Mileage (kmpl)', 5.0, 40.0, 18.5, step=0.5,
                                              help='Fuel efficiency in km per litre.')
                    engine  = st.number_input('Engine CC', 500.0, 5_000.0, 1_497.0, step=50.0,
                                              help='Displacement in cubic centimetres.')
                    power   = st.number_input('Power (BHP)', 30.0, 600.0, 108.5, step=5.0,
                                              help='Maximum power in brake horsepower.')

                if st.button('💰 Estimate Price', type='primary', use_container_width=True):
                    try:
                        with st.spinner('Loading ML model and predicting price…'):
                            price = ml_predict(
                                brand_sel, year, km_driven, fuel_type, transmission,
                                mileage, engine, power, seats, owner_type,
                            )
                        st.session_state['last_price']     = price
                        st.session_state['last_brand_sel'] = brand_sel
                        st.session_state['last_specs']     = dict(
                            year=year, km_driven=km_driven, fuel_type=fuel_type,
                            transmission=transmission, mileage=mileage,
                            engine=engine, power=power, seats=seats,
                        )
                    except Exception as exc:
                        st.error(f'Price prediction failed: {exc}')

                if 'last_price' in st.session_state:
                    price      = st.session_state['last_price']
                    brand_used = st.session_state.get('last_brand_sel', brand_sel)
                    specs      = st.session_state.get('last_specs', {})

                    st.divider()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric('Estimated Price (CHF)', f'CHF {lakh_to_chf(price):,.0f}')
                    m2.metric('Car Age', f'{2024 - specs.get("year", 2018)} years')
                    m3.metric('CV Brand (image)', st.session_state.get('cv_brand', '—'))
                    m4.metric('ML Brand (selected)', brand_used)

                    # RAG explanation
                    st.divider()
                    st.subheader('AI Explanation')
                    if not OPENAI_API_KEY:
                        st.warning('Set OPENAI_API_KEY to enable AI explanations.')
                    elif ensure_rag():
                        with st.spinner('Generating AI explanation…'):
                            rag_query = (
                                f'I have a {specs.get("year")} {brand_used} with '
                                f'{specs.get("km_driven"):,} km driven, '
                                f'{specs.get("fuel_type")} fuel, '
                                f'{specs.get("transmission")} transmission, '
                                f'{specs.get("engine")} CC engine, '
                                f'{specs.get("power")} BHP power. '
                                f'The estimated resale price is CHF {lakh_to_chf(price):,.0f}. '
                                f'Is this price fair and should I buy it?'
                            )
                            answer, context_docs = rag_answer(rag_query, st.session_state['rag'])
                        st.info(answer)
                        with st.expander('View retrieved knowledge-base documents'):
                            for j, doc in enumerate(context_docs, 1):
                                st.markdown(f'**[{j}]** {doc}')
                    else:
                        st.warning(
                            f'AI explanation unavailable: {st.session_state.get("rag_error", "unknown error")}'
                        )
        else:
            st.info('Upload a car image above to get started with the analysis.')

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2 — Car Advisor Chatbot
    # ─────────────────────────────────────────────────────────────────────

    with tab2:
        st.header('Car Advisor Chatbot')
        st.markdown(
            'Ask any car-related question. The chatbot retrieves relevant facts from our '
            'knowledge base and uses GPT-3.5-turbo to generate a grounded answer.'
        )

        # Load RAG on first visit to this tab; cached in session_state afterwards
        rag_ok = ensure_rag()

        if not rag_ok:
            st.error(f'Chatbot unavailable: {st.session_state.get("rag_error", "unknown error")}')
            st.info(
                'To enable the chatbot:\n'
                '1. Set `OPENAI_API_KEY` in `.env` or HuggingFace Secrets.\n'
                '2. Run `04_nlp_rag.ipynb` to build the FAISS knowledge base.\n'
                '3. Restart the app.'
            )
        else:
            rag = st.session_state['rag']

            st.markdown('**Quick questions — click to ask:**')
            example_qs = [
                'What should I look for when buying a used car?',
                'Is a diesel or petrol car better for city driving?',
                'How does car age affect resale value?',
            ]
            q_cols = st.columns(3)
            for i, (col, q) in enumerate(zip(q_cols, example_qs)):
                with col:
                    if st.button(q, key=f'ex_q_{i}', use_container_width=True):
                        st.session_state['pending_chat'] = q

            st.divider()

            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            for msg in st.session_state.chat_history:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])
                    if msg['role'] == 'assistant' and msg.get('context'):
                        with st.expander('View source documents'):
                            for j, doc in enumerate(msg['context'], 1):
                                st.markdown(f'**[{j}]** {doc}')

            pending    = st.session_state.pop('pending_chat', None)
            user_input = st.chat_input('Ask a car question…') or pending

            if user_input:
                st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                with st.chat_message('user'):
                    st.markdown(user_input)

                with st.chat_message('assistant'):
                    with st.spinner('Thinking…'):
                        answer, context_docs = rag_answer(user_input, rag)
                    st.markdown(answer)
                    with st.expander('View source documents'):
                        for j, doc in enumerate(context_docs, 1):
                            st.markdown(f'**[{j}]** {doc}')

                st.session_state.chat_history.append({
                    'role':    'assistant',
                    'content': answer,
                    'context': context_docs,
                })

            if st.session_state.get('chat_history'):
                st.markdown('')
                if st.button('🗑️ Clear conversation', key='clear_chat'):
                    st.session_state.chat_history = []
                    st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3 — Dataset Insights
    # ─────────────────────────────────────────────────────────────────────

    with tab3:
        st.header('Dataset Insights')
        st.markdown('Explore the Indian used-car dataset that powers the ML price prediction model.')

        try:
            df = pd.read_csv(dp('used_cars_clean.csv'))
            bool_cols = [c for c in df.columns
                         if c.startswith(('Fuel_Type_', 'Transmission_', 'Owner_Type_'))]
            df[bool_cols] = df[bool_cols].replace({'True': 1, 'False': 0}).astype(int)
        except Exception as data_err:
            st.error(f'Dataset error: {data_err}')
            df = None

        if df is not None:
            try:
                brand_list = get_brand_list()
                df['Brand_Name'] = df['Brand'].apply(
                    lambda x: brand_list[int(x)] if int(x) < len(brand_list) else f'Brand {int(x)}'
                )
            except Exception:
                df['Brand_Name'] = df['Brand'].astype(int).astype(str)

            df['Fuel_Type'] = 'Other / CNG'
            df.loc[df['Fuel_Type_Diesel'] == 1, 'Fuel_Type'] = 'Diesel'
            df.loc[df['Fuel_Type_Petrol'] == 1, 'Fuel_Type'] = 'Petrol'
            df['Price_CHF'] = df['Price'].apply(lakh_to_chf)

            total_cars = len(df)
            avg_price  = df['Price'].mean()
            top_brand  = df['Brand_Name'].value_counts().index[0]
            top_fuel   = df['Fuel_Type'].value_counts().index[0]

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric('Total Listings', f'{total_cars:,}')
            mc2.metric('Average Resale Price (CHF)', f'CHF {lakh_to_chf(avg_price):,.0f}')
            mc3.metric('Most Common Brand', top_brand)
            mc4.metric('Most Common Fuel', top_fuel)

            st.divider()

            st.subheader('Price Distribution')
            fig_hist = px.histogram(
                df, x='Price_CHF', nbins=60,
                title='Distribution of Used Car Resale Prices',
                labels={'Price_CHF': 'Price (CHF)', 'count': 'Number of Cars'},
                color_discrete_sequence=['#636EFA'],
                template='plotly_white',
            )
            fig_hist.update_layout(
                bargap=0.05, xaxis_title='Price (CHF)', yaxis_title='Number of Cars'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader('Top 10 Brands by Listing Count')
            brand_counts = df['Brand_Name'].value_counts().head(10).reset_index()
            brand_counts.columns = ['Brand', 'Count']
            fig_brands = px.bar(
                brand_counts, x='Brand', y='Count',
                title='Top 10 Car Brands in Dataset',
                color='Count', color_continuous_scale='Blues',
                template='plotly_white', text='Count',
            )
            fig_brands.update_traces(textposition='outside')
            fig_brands.update_layout(coloraxis_showscale=False, yaxis_title='Number of Listings')
            st.plotly_chart(fig_brands, use_container_width=True)

            st.subheader('Price Distribution by Fuel Type')
            fig_box = px.box(
                df, x='Fuel_Type', y='Price_CHF',
                title='Resale Price by Fuel Type',
                labels={'Price_CHF': 'Price (CHF)', 'Fuel_Type': 'Fuel Type'},
                color='Fuel_Type',
                color_discrete_sequence=px.colors.qualitative.Set2,
                template='plotly_white',
            )
            fig_box.update_layout(showlegend=False, yaxis_title='Price (CHF)')
            st.plotly_chart(fig_box, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 4 — About
    # ─────────────────────────────────────────────────────────────────────

    with tab4:
        st.header('About This Project')

        col_left, col_right = st.columns([3, 2], gap='large')

        with col_left:
            st.subheader('Project Description')
            st.markdown(
                'This is a ZHAW university AI course project that demonstrates how three different '
                'AI techniques can be combined into one coherent, end-to-end pipeline.\n\n'
                'The application answers a common real-world question:  \n'
                '**"I have a car photo and some basic specifications — what brand is it, '
                'what is it worth, and should I buy it?"**'
            )

            st.subheader('Three-Block Architecture')
            st.markdown(
                '**Block 1 — Computer Vision (ResNet18)**  \n'
                'Transfer learning on ImageNet-pretrained ResNet18, fine-tuned on the '
                'Auto_Bilder dataset. The model classifies car brand from a single image '
                'across 42 brands with 82.44% top-1 accuracy.\n\n'
                '**Block 2 — Machine Learning (Random Forest)**  \n'
                'A Random Forest regressor trained on ~5,700 Indian used-car listings. '
                'Features include mileage, engine size, power, car age, fuel type, '
                'transmission, and brand. Achieves R² = 0.91 on the test set. '
                'Prices are converted from Indian Lakh (₹) to CHF for display.\n\n'
                '**Block 3 — NLP / RAG (GPT-3.5-turbo + FAISS)**  \n'
                'Retrieval-Augmented Generation using `all-MiniLM-L6-v2` embeddings '
                'stored in a FAISS vector index. GPT-3.5-turbo receives the retrieved '
                'context via XML-tagged prompts and generates grounded, factual answers.'
            )

            st.subheader('Model Performance')
            perf_df = pd.DataFrame({
                'Block':  ['CV (ResNet18)', 'ML (Random Forest)', 'NLP (RAG)'],
                'Task':   ['Brand Classification', 'Price Prediction (CHF)', 'Question Answering'],
                'Metric': ['Top-1 Accuracy', 'R² Score', 'Retrieval Model'],
                'Result': ['82.44 %', '0.91', 'all-MiniLM-L6-v2 + FAISS'],
            })
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

        with col_right:
            st.subheader('Technology Stack')
            st.markdown(
                '| Layer | Technology |\n'
                '|-------|------------|\n'
                '| CV model | PyTorch + ResNet18 |\n'
                '| ML model | scikit-learn Random Forest |\n'
                '| Embeddings | sentence-transformers |\n'
                '| Vector store | FAISS (IndexFlatIP) |\n'
                '| LLM | OpenAI GPT-3.5-turbo |\n'
                '| Frontend | Streamlit |\n'
                '| Charts | Plotly Express |\n'
            )

            st.subheader('Integration Flow')
            st.code(
                'car_image.jpg  +  user specs\n'
                '       |\n'
                '  [Block 1 — CV]\n'
                '  ResNet18 → brand name\n'
                '       |\n'
                '  [Block 2 — ML]\n'
                '  Random Forest → price (CHF)\n'
                '       |\n'
                '  [Block 3 — RAG]\n'
                '  FAISS retrieve → GPT-3.5-turbo → explanation\n'
                '       |\n'
                '  Streamlit UI → user',
                language='text',
            )

            st.subheader('Dataset Sources')
            st.markdown(
                '- **Image data:** https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset  \n'
                '- **Tabular data:** https://www.kaggle.com/datasets/tunguz/used-car-auction-prices  \n'
                '- **Knowledge base:** Built from dataset statistics + car-buying tips'
            )

        st.divider()
        st.subheader('Author & Course')
        st.markdown(
            '**Course:** AI Applications — ZHAW School of Engineering  \n'
            '**Author:** Daniele Magnano  \n'
            '**Year:** 2026  \n'
            '**Goal:** Demonstrate an end-to-end AI pipeline (image → brand → price → explanation)'
        )

except Exception as e:
    st.error(f"App error: {e}")
    st.stop()
