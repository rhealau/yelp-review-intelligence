"""
Yelp Review Intelligence System
Streamlit Web Application

A comprehensive AI-powered platform to analyze restaurant reviews using TWO AI models:
1. DistilBERT for sentiment analysis and quality classification
2. FLAN-T5 for generating analysis text and recommendations
"""

import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
import os
from typing import Dict, List
from collections import Counter

# ========================================
# Page Configuration
# ========================================
st.set_page_config(
    page_title="Yelp Review Intelligence System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# Custom CSS
# ========================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF1744;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .analysis-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .analysis-box h3 {
        margin: 0 0 1rem 0;
        font-size: 1.3rem;
    }
    .analysis-box p {
        margin: 0;
        line-height: 1.6;
        font-size: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF1744;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #D50000;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .model-info {
        background-color: #F3E5F5;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #9C27B0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# Model Loading - TWO MODELS!
# ========================================
@st.cache_resource
def load_models():
    """Load all required models - TWO PIPELINES for the assignment!"""
    try:
        with st.spinner("🤖 Loading AI models... This may take a minute..."):
            # Check if GPU is available
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                st.success("🚀 GPU available! Using GPU acceleration")
                gpu_name = torch.cuda.get_device_name(0)
                st.info(f"GPU: {gpu_name}")
            else:
                st.info("💻 GPU not available, using CPU")
            
            # ========================================
            # PIPELINE 1: DistilBERT Classification
            # ========================================
            st.write("📊 Loading Pipeline 1: DistilBERT for sentiment and quality classification...")
            
            # 1a. Sentiment Analysis Model
            sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device
            )
            st.success("✅ Sentiment model loaded")
            
            # 1b. Quality Classification Model (Your trained model)
            model_candidates = [
                ("RLau33/yelp-review-quality-v2", "Your fine-tuned model (Hugging Face)"),
                ("RLau33/yelpreviewproject", "Your original model"),
                ("./yelp_review_quality_final", "Locally saved model"),
                ("distilbert-base-uncased", "Original pretrained model"),
            ]
            
            quality_model = None
            loaded_model_name = ""
            loaded_model_source = ""
            
            for model_name, description in model_candidates:
                try:
                    st.info(f"Trying: {description}")
                    model = pipeline(
                        "text-classification",
                        model=model_name,
                        device=device
                    )
                    
                    # Test if model works
                    test_result = model("The food was great!")[0]
                    
                    if 'label' in test_result and 'score' in test_result:
                        quality_model = model
                        loaded_model_name = model_name
                        loaded_model_source = description
                        st.success(f"✅ Quality model loaded: {description}")
                        break
                        
                except Exception as e:
                    continue
            
            if quality_model is None:
                st.warning("⚠️ Quality model failed to load, using rule-based fallback")
                loaded_model_source = "Rule-based fallback"
            
            # ========================================
            # PIPELINE 2: FLAN-T5 Text Generation
            # ========================================
            st.write("✍️ Loading Pipeline 2: FLAN-T5 for analysis text generation...")
            
            text_generator = None
            generator_source = ""
            
            generator_candidates = [
                ("RLau33/flan-t5-yelp-analysis", "Your fine-tuned FLAN-T5"),
                ("google/flan-t5-small", "Base FLAN-T5-small"),
            ]
            
            for gen_model_name, gen_description in generator_candidates:
                try:
                    st.info(f"Trying: {gen_description}")
                    generator = pipeline(
                        "text2text-generation",
                        model=gen_model_name,
                        device=device
                    )
                    
                    # Test if model works
                    test_output = generator("Test: The food was great!", max_length=50)[0]
                    
                    if 'generated_text' in test_output:
                        text_generator = generator
                        generator_source = gen_description
                        st.success(f"✅ Text generator loaded: {gen_description}")
                        break
                        
                except Exception as e:
                    st.warning(f"Failed to load {gen_description}: {str(e)}")
                    continue
            
            if text_generator is None:
                st.error("❌ Text generator failed to load")
                generator_source = "Not available"
            
            # Display model info in sidebar
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"""
            <div class="model-info">
            <strong>🤖 Pipeline 1 (Classification):</strong><br>
            Quality Model: {loaded_model_source}<br>
            <small>Model: {loaded_model_name}</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.sidebar.markdown(f"""
            <div class="model-info">
            <strong>✍️ Pipeline 2 (Generation):</strong><br>
            Text Generator: {generator_source}
            </div>
            """, unsafe_allow_html=True)
            
        return sentiment_model, quality_model, text_generator, loaded_model_source, generator_source
        
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, "", ""

# Load models
sentiment_model, quality_model, text_generator, model_source, generator_source = load_models()
models_loaded = sentiment_model is not None and text_generator is not None

# ========================================
# Helper Functions
# ========================================
def analyze_sentiment(text: str) -> Dict:
    """Analyze review sentiment using DistilBERT"""
    if not sentiment_model:
        return {"label": "UNKNOWN", "score": 0.0}
    
    try:
        result = sentiment_model(text[:512])[0]
        return result
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")
        return {"label": "ERROR", "score": 0.0}

def analyze_quality(text: str) -> Dict:
    """Analyze review quality using your trained DistilBERT model"""
    if not quality_model:
        # Fallback: Rule-based assessment
        length = len(text)
        if length < 30:
            return {"label": "LABEL_0", "score": 0.8}
        elif length < 100:
            return {"label": "LABEL_1", "score": 0.7}
        else:
            return {"label": "LABEL_2", "score": 0.75}
    
    try:
        result = quality_model(text[:512])[0]
        
        if st.session_state.get('show_debug', False):
            st.write(f"🔍 Raw quality model output: {result}")
            
        return result
    except Exception as e:
        st.error(f"Quality analysis error: {e}")
        return {"label": "LABEL_1", "score": 0.5}

def generate_analysis_text(review_text: str, sentiment_result: Dict, quality_result: Dict) -> str:
    """
    Generate analysis text and recommendations using FLAN-T5 (PIPELINE 2)
    This is the second model required by the assignment!
    """
    if not text_generator:
        # Fallback: Template-based generation
        sentiment = sentiment_result.get("label", "UNKNOWN").lower()
        quality_label = quality_result.get("label", "UNKNOWN")
        quality_name, _, _, _ = map_quality_label(quality_label)
        
        return f"This review shows {sentiment} sentiment with {quality_name.lower()}. The restaurant owner should review this feedback and take appropriate action based on the customer's experience."
    
    try:
        # Construct prompt for FLAN-T5
        prompt = f"Analyze this restaurant review and provide recommendations for the owner: {review_text[:400]}"
        
        # Generate analysis using FLAN-T5
        result = text_generator(
            prompt,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )[0]
        
        generated_text = result['generated_text']
        
        if st.session_state.get('show_debug', False):
            st.write(f"🔍 Generated analysis: {generated_text}")
        
        return generated_text
        
    except Exception as e:
        st.error(f"Text generation error: {e}")
        # Fallback
        return "Analysis generation failed. Please try again."

def map_quality_label(label: str) -> tuple:
    """Map quality label to readable format"""
    label_str = str(label).strip().upper()
    
    if label_str in ['LABEL_0', '0', 'LOW_QUALITY', 'LOW']:
        return ('Low Quality', '🔴', 30, 'Short, uninformative reviews')
    elif label_str in ['LABEL_1', '1', 'MEDIUM_QUALITY', 'MEDIUM']:
        return ('Medium Quality', '🟡', 60, 'Average length with some details')
    elif label_str in ['LABEL_2', '2', 'HIGH_QUALITY', 'HIGH']:
        return ('High Quality', '🟢', 90, 'Detailed, informative reviews')
    else:
        return ('Unknown Quality', '⚪', 50, 'Cannot determine quality level')

def extract_keywords(text: str, top_n: int = 5) -> List[tuple]:
    """Extract keywords from text"""
    words = text.lower().split()
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'is', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                  'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those'}
    
    words = [w for w in words if w not in stop_words and len(w) > 3]
    word_freq = Counter(words)
    keywords = word_freq.most_common(top_n)
    
    if keywords:
        max_freq = keywords[0][1]
        keywords = [(word, freq / max_freq) for word, freq in keywords]
    
    return keywords

# ========================================
# Sidebar Navigation
# ========================================
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Home", "📝 Single Analysis", "🧪 Model Testing"],
    index=0
)

# Add debug option
debug_mode = st.sidebar.checkbox("🔍 Debug Mode", value=False)
if debug_mode:
    st.session_state['show_debug'] = True
    st.sidebar.info("Debug mode enabled")
else:
    st.session_state['show_debug'] = False

st.sidebar.markdown("---")
st.sidebar.info("""
**About this app:**

AI-powered Yelp Review Intelligence System

**Two-Model Architecture:**
1. 🎭 **Pipeline 1**: DistilBERT
   - Sentiment Analysis
   - Quality Classification

2. ✍️ **Pipeline 2**: FLAN-T5
   - Analysis Text Generation
   - Owner Recommendations

**Tech Stack:**
- Classification: DistilBERT
- Generation: FLAN-T5-small
- Frontend: Streamlit
- Visualization: Plotly

**Developed for:**
ISOM5240 Course Project
HKUST Business School
""")

# GitHub link
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center;'>
    <a href='https://github.com/rhealau' target='_blank'>
        <img src='https://img.shields.io/badge/GitHub-rhealau-blue?logo=github' />
    </a>
</div>
""", unsafe_allow_html=True)

# ========================================
# PAGE: Home
# ========================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🍽️ Yelp Review Intelligence System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the AI-Powered Review Analysis Platform!
    
    This system uses **two distinct AI models** to provide comprehensive review analysis:
    
    #### 🤖 Two-Model Architecture
    
    **Pipeline 1: DistilBERT Classification**
    - Sentiment analysis (positive/negative)
    - Review quality classification (low/medium/high)
    - Fast and accurate classification
    
    **Pipeline 2: FLAN-T5 Text Generation**
    - Generates detailed analysis text
    - Provides actionable recommendations for restaurant owners
    - Focuses on operations, marketing, and customer engagement
    
    #### 🎯 Key Features
    
    - **Single Review Analysis**: Analyze individual reviews with detailed insights
    - **AI-Generated Recommendations**: Get specific suggestions for restaurant improvement
    - **Model Testing**: Compare both models side-by-side
    - **Real-time Processing**: Instant analysis with GPU acceleration (when available)
    
    #### 🚀 Get Started
    
    1. Navigate to **"📝 Single Analysis"** to analyze a review
    2. Enter or paste a restaurant review
    3. Get instant analysis from both AI models
    4. View generated recommendations for the restaurant owner
    
    #### 📊 Technology Stack
    
    - **Model 1**: DistilBERT (fine-tuned on Yelp reviews)
    - **Model 2**: FLAN-T5-small (fine-tuned for analysis generation)
    - **Dataset**: Yelp Review Full (650k+ reviews)
    - **Framework**: Hugging Face Transformers
    - **Deployment**: Streamlit Community Cloud
    """)
    
    # Display model status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎭 Pipeline 1: Classification</h3>
            <p>DistilBERT Model</p>
            <p>Status: <strong>""" + ("✅ Active" if sentiment_model and quality_model else "❌ Inactive") + """</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>✍️ Pipeline 2: Generation</h3>
            <p>FLAN-T5 Model</p>
            <p>Status: <strong>""" + ("✅ Active" if text_generator else "❌ Inactive") + """</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📚 About This Project
    
    This application demonstrates the integration of two different types of AI models:
    
    1. **Classification Models** (DistilBERT): Categorize and label data
    2. **Generation Models** (FLAN-T5): Create new text based on input
    
    By combining both approaches, we create a comprehensive analysis system that not only
    identifies patterns in reviews but also generates human-readable insights and recommendations.
    
    **Course**: ISOM5240 - AI and Business Applications  
    **Institution**: HKUST Business School  
    **Author**: RLau33
    """)

# ========================================
# PAGE: Single Analysis
# ========================================
elif page == "📝 Single Analysis":
    st.markdown('<h1 class="main-header">📝 Single Review Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Analyze a Single Review with AI
    
    Enter a restaurant review below to get:
    - **Sentiment analysis** (positive/negative)
    - **Quality classification** (low/medium/high)
    - **AI-generated analysis and recommendations** for the restaurant owner
    """)
    
    # Sample reviews for quick testing
    sample_reviews = {
        "Select a sample...": "",
        "5-Star Review (Excellent)": "This restaurant is absolutely amazing! The food was delicious, service was impeccable, and the atmosphere was perfect. The staff went above and beyond to make our anniversary dinner special. We had the ribeye steak and it was cooked to perfection. The wine selection was excellent and our server gave great recommendations. Will definitely be coming back and recommending to all our friends!",
        "1-Star Review (Poor)": "Terrible experience. Waited over an hour for cold food. The server was rude and never checked on us. The restaurant was dirty and the bathroom was disgusting. Food was overpriced for the poor quality. Manager didn't even apologize. Never coming back and warning everyone to stay away.",
        "3-Star Review (Average)": "It was okay. Nothing special but nothing terrible either. Food was average, service was acceptable. Prices were reasonable. Might come back if in the area but wouldn't go out of my way."
    }
    
    selected_sample = st.selectbox("Quick Test (Optional)", list(sample_reviews.keys()))
    
    # Text input
    review_text = st.text_area(
        "Enter Restaurant Review:",
        value=sample_reviews[selected_sample],
        height=150,
        placeholder="Paste or type a restaurant review here..."
    )
    
    if st.button("🔍 Analyze Review", type="primary"):
        if not review_text.strip():
            st.warning("⚠️ Please enter a review to analyze.")
        elif not models_loaded:
            st.error("❌ Models are not loaded. Please refresh the page.")
        else:
            with st.spinner("🤖 Analyzing review with both AI models..."):
                # PIPELINE 1: Classification
                st.markdown("### 🎭 Pipeline 1: Classification Results")
                
                col1, col2 = st.columns(2)
                
                # Sentiment Analysis
                with col1:
                    sentiment_result = analyze_sentiment(review_text)
                    sentiment_label = sentiment_result.get("label", "UNKNOWN")
                    sentiment_score = sentiment_result.get("score", 0.0)
                    
                    sentiment_emoji = "😊" if sentiment_label == "POSITIVE" else "😞"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{sentiment_emoji} Sentiment</h3>
                        <h2>{sentiment_label}</h2>
                        <p>Confidence: {sentiment_score:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Quality Classification
                with col2:
                    quality_result = analyze_quality(review_text)
                    quality_label = quality_result.get("label", "UNKNOWN")
                    quality_score = quality_result.get("score", 0.0)
                    
                    quality_name, quality_emoji, _, quality_desc = map_quality_label(quality_label)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{quality_emoji} Quality</h3>
                        <h2>{quality_name}</h2>
                        <p>Confidence: {quality_score:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # PIPELINE 2: Text Generation
                st.markdown("### ✍️ Pipeline 2: AI-Generated Analysis")
                
                with st.spinner("Generating analysis and recommendations..."):
                    analysis_text = generate_analysis_text(review_text, sentiment_result, quality_result)
                
                st.markdown(f"""
                <div class="analysis-box">
                    <h3>🎯 Analysis & Recommendations for Restaurant Owner</h3>
                    <p>{analysis_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Additional Insights
                st.markdown("### 📊 Additional Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Review Statistics:**")
                    st.write(f"- **Length**: {len(review_text)} characters")
                    st.write(f"- **Words**: {len(review_text.split())} words")
                    st.write(f"- **Sentences**: {review_text.count('.')} (approx)")
                
                with col2:
                    st.markdown("**Top Keywords:**")
                    keywords = extract_keywords(review_text, top_n=5)
                    if keywords:
                        for word, score in keywords:
                            st.write(f"- **{word}**: {score:.0%}")
                    else:
                        st.write("No significant keywords found")
                
                st.success("✅ Analysis complete! Both models have processed the review.")

# ========================================
# PAGE: Model Testing
# ========================================
elif page == "🧪 Model Testing":
    st.markdown('<h1 class="main-header">🧪 Model Testing & Comparison</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Test Both AI Models
    
    This page allows you to test and compare the two models:
    - **Pipeline 1**: DistilBERT (Classification)
    - **Pipeline 2**: FLAN-T5 (Text Generation)
    """)
    
    st.markdown("---")
    
    # Model Status
    st.markdown("### 📊 Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎭 Pipeline 1: DistilBERT")
        st.write(f"**Sentiment Model**: {'✅ Loaded' if sentiment_model else '❌ Not loaded'}")
        st.write(f"**Quality Model**: {'✅ Loaded' if quality_model else '❌ Not loaded'}")
        st.write(f"**Source**: {model_source}")
    
    with col2:
        st.markdown("#### ✍️ Pipeline 2: FLAN-T5")
        st.write(f"**Text Generator**: {'✅ Loaded' if text_generator else '❌ Not loaded'}")
        st.write(f"**Source**: {generator_source}")
    
    st.markdown("---")
    
    # Test Input
    st.markdown("### 🧪 Test Input")
    
    test_text = st.text_area(
        "Enter test review:",
        value="The food was amazing and the service was excellent!",
        height=100
    )
    
    if st.button("🚀 Run Both Models"):
        if test_text.strip():
            st.markdown("### 📊 Results")
            
            # Pipeline 1
            st.markdown("#### 🎭 Pipeline 1: Classification")
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_result = analyze_sentiment(test_text)
                st.json(sentiment_result)
            
            with col2:
                quality_result = analyze_quality(test_text)
                st.json(quality_result)
            
            # Pipeline 2
            st.markdown("#### ✍️ Pipeline 2: Text Generation")
            analysis = generate_analysis_text(test_text, sentiment_result, quality_result)
            st.success(analysis)
        else:
            st.warning("Please enter text to test")
    
    st.markdown("---")
    
    # Model Information
    st.markdown("### 📚 Model Information")
    
    with st.expander("🎭 Pipeline 1: DistilBERT Details"):
        st.markdown("""
        **Model Architecture**: DistilBERT (Distilled BERT)
        
        **Tasks**:
        1. Sentiment Analysis: Classify reviews as POSITIVE or NEGATIVE
        2. Quality Classification: Classify reviews as LOW, MEDIUM, or HIGH quality
        
        **Training**:
        - Base model: `distilbert-base-uncased`
        - Fine-tuned on: Yelp Review Full dataset
        - Training samples: ~50,000 reviews
        
        **Performance**:
        - Fast inference (~50ms per review)
        - High accuracy on sentiment classification
        - Reliable quality assessment
        """)
    
    with st.expander("✍️ Pipeline 2: FLAN-T5 Details"):
        st.markdown("""
        **Model Architecture**: FLAN-T5-small (Text-to-Text)
        
        **Task**: Generate analysis text and recommendations for restaurant owners
        
        **Training**:
        - Base model: `google/flan-t5-small` (77M parameters)
        - Fine-tuned on: Yelp Review Full dataset
        - Training samples: ~50,000 reviews
        - Output format: Analysis text (up to 256 tokens)
        
        **Capabilities**:
        - Generate contextual analysis
        - Provide actionable recommendations
        - Focus on operations, marketing, and customer engagement
        
        **Performance**:
        - Inference time: ~200-500ms per review
        - Generates human-readable text
        - Contextually relevant recommendations
        """)

# ========================================
# Footer
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Yelp Review Intelligence System</strong></p>
    <p>Powered by DistilBERT + FLAN-T5 | Built with Streamlit</p>
    <p>ISOM5240 Course Project | HKUST Business School</p>
    <p>© 2024 RLau33 | <a href='https://github.com/rhealau'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
