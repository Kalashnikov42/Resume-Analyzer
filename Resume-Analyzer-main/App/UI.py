import streamlit as st 

def load_css():
    st.markdown("""
    <style>
    :root {
        --primary-color: #505081;
        --secondary-color: #8686AC;
        --accent-color: #0F0E47;
        --text-color: #FFFFFF;
        --bg-color: #272757;
        --card-bg: #2E2E4D;
        --input-bg: #1E1E3F;
        --input-border: #8686AC;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: var(--bg-color);
        color: var(--text-color);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-color) !important;
    }

    .stMarkdown, .stText, p, div, span, label {
        color: var(--text-color) !important;
    }

    .custom-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .custom-header h1,
    .custom-header p {
        color: white !important;
    }

    .info-card {
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--secondary-color);
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        color: var(--text-color);
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > textarea {
        background-color: var(--input-bg);
        color: var(--text-color) !important;
        border: 1px solid var(--input-border);
        border-radius: 6px;
    }

    .stFileUploader > div {
        background-color: var(--input-bg);
        border: 1px dashed var(--input-border);
        color: var(--text-color) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    .css-1d391kg {
        background: linear-gradient(180deg, var(--primary-color), var(--accent-color));
    }

    .css-1d391kg .css-1v0mbdj, 
    .css-1d391kg .css-1v0mbdj label,
    .css-1d391kg .stMarkdown, 
    .css-1d391kg p, 
    .css-1d391kg div {
        color: white !important;
    }

    .metric-card {
        background: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        border-top: 3px solid var(--accent-color);
    }
    </style>
    """, unsafe_allow_html=True)