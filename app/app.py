# ═══════════════════════════════════════════════════════════════════════════
# app/app.py  –  Car Advisor AI  (ZHAW AI Applications Project)
# ═══════════════════════════════════════════════════════════════════════════

import gc
import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Graceful FAISS import — both 'faiss' and 'faiss-cpu' install under the same 'faiss' namespace
try:
    import faiss as _faiss_lib
    FAISS_AVAILABLE = True
except ImportError:
    _faiss_lib = None
    FAISS_AVAILABLE = False

# ── Path helpers ────────────────────────────────────────────────────────────
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(APP_DIR, '..', 'models'))
DATA_DIR   = os.path.normpath(os.path.join(APP_DIR, '..', 'data'))

def mp(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)

def dp(*parts: str) -> str:
    return os.path.join(DATA_DIR, *parts)

# Load .env (OPENAI_API_KEY)
load_dotenv(os.path.join(APP_DIR, '..', '.env'))
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# ── Page config (must be FIRST Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title='Car Advisor AI',
    page_icon='🚗',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ═══════════════════════════════════════════════════════════════════════════
# Cached model loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_cv_model():
    """Load ResNet18 in eval mode. Called lazily on button click; caller must del the model."""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models

        weights_path = mp('car_brand_classifier.pth')
        names_path   = mp('class_names.json')

        if not os.path.exists(weights_path):
            return None, None, 'car_brand_classifier.pth not found — run 02_cv_model.ipynb first.'
        if not os.path.exists(names_path):
            return None, None, 'class_names.json not found — run 02_cv_model.ipynb first.'

        with open(names_path) as f:
            class_names = json.load(f)

        device = torch.device('cpu')
        try:
            net = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            net = tv_models.resnet18(pretrained=False)

        net.fc = nn.Linear(net.fc.in_features, len(class_names))
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.to(device).eval()
        return net, class_names, None
    except Exception as exc:
        return None, None, str(exc)


@st.cache_resource(show_spinner='Loading ML models…')
def load_ml_models():
    """Load price predictor, scaler, brand encoder, feature columns.
    Returns (predictor, scaler, brand_encoder, feature_columns, error)."""
    try:
        import joblib

        for fname in ('price_predictor.pkl', 'scaler.pkl', 'brand_encoder.pkl'):
            if not os.path.exists(mp(fname)):
                return None, None, None, None, f'{fname} not found — run 03_ml_numeric.ipynb first.'

        fc_path = dp('processed', 'feature_columns.json')
        if not os.path.exists(fc_path):
            return None, None, None, None, 'feature_columns.json not found — run 01_eda.ipynb first.'

        predictor     = joblib.load(mp('price_predictor.pkl'))
        scaler        = joblib.load(mp('scaler.pkl'))
        brand_encoder = joblib.load(mp('brand_encoder.pkl'))
        with open(fc_path) as f:
            feature_columns = json.load(f)

        return predictor, scaler, brand_encoder, feature_columns, None
    except Exception as exc:
        return None, None, None, None, str(exc)


@st.cache_resource(show_spinner='Loading RAG components…')
def load_rag_components():
    """Load FAISS index, documents, embedding model, OpenAI client.
    Returns (faiss_index, documents, embedder, openai_client, error)."""
    try:
        if not FAISS_AVAILABLE:
            return None, None, None, None, (
                'faiss is not installed. Run: pip install faiss-cpu'
            )
        from sentence_transformers import SentenceTransformer
        from openai import OpenAI

        for fname in ('faiss_index.bin', 'rag_documents.json', 'rag_config.json'):
            if not os.path.exists(mp(fname)):
                return None, None, None, None, f'{fname} not found — run 04_nlp_rag.ipynb first.'

        if not OPENAI_API_KEY:
            return None, None, None, None, 'OPENAI_API_KEY not set in .env file.'

        idx = _faiss_lib.read_index(mp('faiss_index.bin'))
        with open(mp('rag_documents.json'), encoding='utf-8') as f:
            docs = json.load(f)
        with open(mp('rag_config.json')) as f:
            config = json.load(f)

        embedder = SentenceTransformer(config.get('embedding_model', 'all-MiniLM-L6-v2'))
        client   = OpenAI(api_key=OPENAI_API_KEY)
        return idx, docs, embedder, client, None
    except Exception as exc:
        return None, None, None, None, str(exc)


@st.cache_data(show_spinner='Loading dataset…')
def load_dataset():
    """Load and preprocess used_cars_clean.csv. Returns (df, error)."""
    try:
        csv_path = dp('processed', 'used_cars_clean.csv')
        if not os.path.exists(csv_path):
            return None, 'used_cars_clean.csv not found — run 01_eda.ipynb first.'

        df = pd.read_csv(csv_path)
        bool_cols = [c for c in df.columns
                     if c.startswith(('Fuel_Type_', 'Transmission_', 'Owner_Type_'))]
        df[bool_cols] = df[bool_cols].replace({'True': 1, 'False': 0}).astype(int)
        return df, None
    except Exception as exc:
        return None, str(exc)


# ═══════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ═══════════════════════════════════════════════════════════════════════════

def cv_predict(image: Image.Image, top_k: int = 3):
    """Load ResNet18, run inference on image, return (predictions, model).
    Caller must `del model` and call `gc.collect()` after use."""
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models, transforms

    weights_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'car_brand_classifier.pth')
    names_path   = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_names.json')

    with open(names_path) as f:
        class_names = json.load(f)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(image.convert('RGB')).unsqueeze(0)

    net = tv_models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(class_names))
    net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    net.eval()

    with torch.no_grad():
        probs = torch.softmax(net(tensor), dim=1)[0]
    topk = torch.topk(probs, top_k)
    predictions = [(class_names[i.item()], topk.values[j].item()) for j, i in enumerate(topk.indices)]
    return predictions, net


def ml_predict(
        brand: str, year: int, km_driven: float,
        fuel_type: str, transmission: str,
        mileage: float, engine: float, power: float,
        seats: int, owner_type: str,
        predictor, scaler, brand_encoder, feature_columns: list) -> float:
    """Predict resale price (returned in Lakh INR; callers convert to CHF via lakh_to_chf)."""
    car_age = 2024 - year
    # Scaler fit order: km_driven, engine, power, mileage, car_age
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

    return round(float(predictor.predict(row)[0]), 2)


def rag_retrieve(query: str, faiss_idx, docs: list, embedder, k: int = 3) -> list:
    """Return top-k most relevant documents for query."""
    q_vec = embedder.encode([query], convert_to_numpy=True).astype('float32')
    _faiss_lib.normalize_L2(q_vec)
    _, indices = faiss_idx.search(q_vec, k)
    return [docs[i] for i in indices[0] if i < len(docs)]


def rag_generate(question: str, context_docs: list, openai_client) -> str:
    """Generate GPT answer grounded in retrieved context (XML-tagged prompt)."""
    context_text = '\n\n'.join(context_docs)
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
        response = openai_client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user',   'content': user_msg},
            ],
            max_tokens=350,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f'[API Error] {exc}'


def rag_full(question: str, faiss_idx, docs, embedder, openai_client, k: int = 3):
    """Full RAG pipeline: retrieve → augment → generate. Returns (answer, context_docs)."""
    context_docs = rag_retrieve(question, faiss_idx, docs, embedder, k)
    answer       = rag_generate(question, context_docs, openai_client)
    return answer, context_docs


def lakh_to_chf(lakh_price: float) -> int:
    # 1 Lakh = 100,000 INR, 1 INR = 0.011 CHF; round to nearest 10 CHF
    return round(lakh_price * 100_000 * 0.011 / 10) * 10


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
    st.caption('💱 Prices converted from Indian market data to CHF')
    st.divider()
    st.caption('ZHAW AI Applications Project · 2024')


# ═══════════════════════════════════════════════════════════════════════════
# Page header
# ═══════════════════════════════════════════════════════════════════════════

st.title('🚗 Car Advisor AI')
st.markdown('*AI-powered brand recognition, price prediction, and buying advice — all in one place.*')
st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    '🔍 Car Analyzer',
    '💬 Car Advisor Chatbot',
    '📊 Dataset Insights',
    'ℹ️ About',
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Car Analyzer
# ═══════════════════════════════════════════════════════════════════════════

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
                    with st.spinner('Loading brand classifier and analysing image…'):
                        predictions, cv_model = cv_predict(image, top_k=3)
                    del cv_model
                    gc.collect()

                    top_brand, top_conf = predictions[0]
                    st.success(f'**Identified brand: {top_brand}**')

                    st.markdown('**Top-3 predictions:**')
                    for brand, conf in predictions:
                        st.markdown(f'`{brand}`')
                        st.progress(float(conf), text=f'{conf:.1%}')

                    st.session_state['cv_brand'] = top_brand
                    st.session_state['cv_conf']  = top_conf
                    st.session_state['cv_done']  = True
                except Exception as exc:
                    st.error(f'Brand recognition failed: {exc}')

        # ── Price estimation section ────────────────────────────────────────
        if st.session_state.get('cv_done'):
            st.divider()
            st.subheader('Price Estimation')
            st.markdown('Fill in the car\'s details below, then click **Estimate Price**.')

            predictor, scaler, brand_encoder, feature_columns, ml_err = load_ml_models()

            if ml_err:
                st.error(f'ML model error: {ml_err}')
            else:
                ml_brands = list(brand_encoder.classes_)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('**General**')
                    # Try to pre-select the CV brand if it exists in ML brands
                    cv_brand = st.session_state.get('cv_brand', ml_brands[0])
                    default_brand_idx = ml_brands.index(cv_brand) if cv_brand in ml_brands else 0
                    brand_sel  = st.selectbox('Brand', ml_brands, index=default_brand_idx)
                    year       = st.slider('Manufacturing Year', 2000, 2024, 2018, step=1)
                    km_driven  = st.slider('Kilometers Driven', 0, 300_000, 50_000, step=1_000,
                                           format='%d km')
                    owner_type = st.selectbox('Owner Type',
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
                    with st.spinner('Predicting resale price…'):
                        try:
                            price = ml_predict(
                                brand_sel, year, km_driven, fuel_type, transmission,
                                mileage, engine, power, seats, owner_type,
                                predictor, scaler, brand_encoder, feature_columns,
                            )
                            gc.collect()
                            st.session_state['last_price']      = price
                            st.session_state['last_brand_sel']  = brand_sel
                            st.session_state['last_specs']      = dict(
                                year=year, km_driven=km_driven, fuel_type=fuel_type,
                                transmission=transmission, mileage=mileage,
                                engine=engine, power=power, seats=seats,
                            )
                        except Exception as exc:
                            st.error(f'Price prediction failed: {exc}')
                            st.stop()

                if 'last_price' in st.session_state:
                    price       = st.session_state['last_price']
                    brand_used  = st.session_state.get('last_brand_sel', brand_sel)
                    specs       = st.session_state.get('last_specs', {})

                    st.divider()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric('Estimated Price (CHF)', f'CHF {lakh_to_chf(price):,.0f}')
                    m2.metric('Car Age', f'{2024 - specs.get("year", 2018)} years')
                    m3.metric('CV Brand (image)', st.session_state.get('cv_brand', '—'))
                    m4.metric('ML Brand (selected)', brand_used)

                    # RAG explanation
                    st.divider()
                    st.subheader('AI Explanation')
                    faiss_idx, docs, embedder, openai_client, rag_err = load_rag_components()

                    if rag_err:
                        st.warning(f'AI explanation unavailable: {rag_err}')
                    else:
                        with st.spinner('Generating AI explanation…'):
                            rag_query = (
                                f'I have a {specs.get("year")} {brand_used} with '
                                f'{specs.get("km_driven"):,} km driven, '
                                f'{specs.get("fuel_type")} fuel, {specs.get("transmission")} transmission, '
                                f'{specs.get("engine")} CC engine, {specs.get("power")} BHP power. '
                                f'The estimated resale price is CHF {lakh_to_chf(price):,.0f}. '
                                f'Is this price fair and should I buy it?'
                            )
                            answer, context_docs = rag_full(
                                rag_query, faiss_idx, docs, embedder, openai_client
                            )

                        st.info(answer)
                        with st.expander('View retrieved knowledge-base documents'):
                            for j, doc in enumerate(context_docs, 1):
                                st.markdown(f'**[{j}]** {doc}')
    else:
        st.info('Upload a car image above to get started with the analysis.')


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Car Advisor Chatbot
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header('Car Advisor Chatbot')
    st.markdown(
        'Ask any car-related question. The chatbot retrieves relevant facts from our '
        'knowledge base and uses GPT-3.5-turbo to generate a grounded answer.'
    )

    faiss_idx, docs, embedder, openai_client, rag_err = load_rag_components()

    if rag_err:
        st.error(f'Chatbot unavailable: {rag_err}')
        st.info(
            'To enable the chatbot:\n'
            '1. Make sure `OPENAI_API_KEY` is set in your `.env` file.\n'
            '2. Run `04_nlp_rag.ipynb` to build the FAISS knowledge base.\n'
            '3. Restart the Streamlit app.'
        )
    else:
        # Example questions
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

        # Initialise chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Render existing messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if msg['role'] == 'assistant' and msg.get('context'):
                    with st.expander('View source documents'):
                        for j, doc in enumerate(msg['context'], 1):
                            st.markdown(f'**[{j}]** {doc}')

        # Consume pending question (from button click) or chat input
        pending    = st.session_state.pop('pending_chat', None)
        user_input = st.chat_input('Ask a car question…') or pending

        if user_input:
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            with st.chat_message('user'):
                st.markdown(user_input)

            with st.chat_message('assistant'):
                with st.spinner('Thinking…'):
                    answer, context_docs = rag_full(
                        user_input, faiss_idx, docs, embedder, openai_client
                    )
                st.markdown(answer)
                with st.expander('View source documents'):
                    for j, doc in enumerate(context_docs, 1):
                        st.markdown(f'**[{j}]** {doc}')

            st.session_state.chat_history.append({
                'role':    'assistant',
                'content': answer,
                'context': context_docs,
            })

        # Clear button (only visible when there is history)
        if st.session_state.get('chat_history'):
            st.markdown('')
            if st.button('🗑️ Clear conversation', key='clear_chat'):
                st.session_state.chat_history = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset Insights
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.header('Dataset Insights')
    st.markdown('Explore the Indian used-car dataset that powers the ML price prediction model.')

    df, data_err = load_dataset()

    if data_err:
        st.error(f'Dataset error: {data_err}')
    else:
        # Decode Brand integers to readable names using brand_encoder
        _, _, brand_encoder, _, ml_err = load_ml_models()
        if not ml_err and brand_encoder is not None:
            brand_list = list(brand_encoder.classes_)
            df = df.copy()
            df['Brand_Name'] = df['Brand'].apply(
                lambda x: brand_list[int(x)] if int(x) < len(brand_list) else f'Brand {int(x)}'
            )
        else:
            df = df.copy()
            df['Brand_Name'] = df['Brand'].astype(int).astype(str)

        # Derive readable Fuel_Type column from boolean dummies
        df['Fuel_Type'] = 'Other / CNG'
        df.loc[df['Fuel_Type_Diesel'] == 1, 'Fuel_Type'] = 'Diesel'
        df.loc[df['Fuel_Type_Petrol'] == 1, 'Fuel_Type'] = 'Petrol'

        # Convert raw prices to CHF for all chart and metric displays
        df['Price_CHF'] = df['Price'].apply(lakh_to_chf)

        # ── Key metrics row ────────────────────────────────────────────────
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

        # ── Price distribution histogram ────────────────────────────────────
        st.subheader('Price Distribution')
        fig_hist = px.histogram(
            df, x='Price_CHF', nbins=60,
            title='Distribution of Used Car Resale Prices',
            labels={'Price_CHF': 'Price (CHF)', 'count': 'Number of Cars'},
            color_discrete_sequence=['#636EFA'],
            template='plotly_white',
        )
        fig_hist.update_layout(
            bargap=0.05,
            xaxis_title='Price (CHF)',
            yaxis_title='Number of Cars',
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Top 10 brands bar chart ─────────────────────────────────────────
        st.subheader('Top 10 Brands by Listing Count')
        brand_counts = df['Brand_Name'].value_counts().head(10).reset_index()
        brand_counts.columns = ['Brand', 'Count']

        fig_brands = px.bar(
            brand_counts, x='Brand', y='Count',
            title='Top 10 Car Brands in Dataset',
            color='Count',
            color_continuous_scale='Blues',
            template='plotly_white',
            text='Count',
        )
        fig_brands.update_traces(textposition='outside')
        fig_brands.update_layout(coloraxis_showscale=False, yaxis_title='Number of Listings')
        st.plotly_chart(fig_brands, use_container_width=True)

        # ── Price by fuel type boxplot ──────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.header('About This Project')

    col_left, col_right = st.columns([3, 2], gap='large')

    with col_left:
        st.subheader('Project Description')
        st.markdown(
            'This is a ZHAW university AI course project that demonstrates how three different '
            'AI techniques can be combined into one coherent, end-to-end pipeline.\n\n'
            'The application answers a common real-world question:  \n'
            '**"I have a car photo and some basic specs — what brand is it, '
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
            '- **Image data:** Auto_Bilder (Stanford Cars-style), 42 brands  \n'
            '- **Tabular data:** Indian Used Car Market, ~5,700 listings  \n'
            '- **Knowledge base:** Built from dataset statistics + car-buying tips'
        )

    st.divider()
    st.subheader('Author & Course')
    st.markdown(
        '**Course:** AI Applications — ZHAW School of Engineering  \n'
        '**Year:** 2024  \n'
        '**Goal:** Demonstrate an end-to-end AI pipeline (image → brand → price → explanation)'
    )
