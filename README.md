# 🍽️ Yelp Review Intelligence System

An AI-powered platform for analyzing restaurant reviews using **two distinct machine learning models** to provide comprehensive insights and actionable recommendations.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://your-app-url.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project implements a **two-model architecture** for restaurant review analysis:

### Pipeline 1: DistilBERT Classification
- **Sentiment Analysis**: Classify reviews as positive or negative
- **Quality Classification**: Categorize reviews as low, medium, or high quality
- **Model**: Fine-tuned DistilBERT on Yelp Review Full dataset

### Pipeline 2: FLAN-T5 Text Generation
- **Analysis Generation**: Create detailed analysis text from review content
- **Recommendations**: Generate actionable suggestions for restaurant owners
- **Model**: Fine-tuned FLAN-T5-small on Yelp Review Full dataset

## ✨ Features

- 📊 **Dual-Model Analysis**: Combines classification and generation models
- 🎭 **Sentiment Detection**: Identifies positive and negative reviews
- ⭐ **Quality Assessment**: Evaluates review informativeness
- ✍️ **AI-Generated Insights**: Produces human-readable analysis and recommendations
- 🚀 **Real-time Processing**: Instant analysis with GPU acceleration support
- 🎨 **Interactive UI**: Built with Streamlit for easy interaction

## 🏗️ Architecture

```
User Input (Review)
       ↓
┌──────────────────────────────────┐
│   Pipeline 1: DistilBERT         │
│   - Sentiment Analysis           │
│   - Quality Classification       │
└──────────────────────────────────┘
       ↓
┌──────────────────────────────────┐
│   Pipeline 2: FLAN-T5            │
│   - Analysis Text Generation     │
│   - Owner Recommendations        │
└──────────────────────────────────┘
       ↓
Output (Comprehensive Analysis)
```

## 📦 Installation

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/rhealau/yelp-review-intelligence.git
cd yelp-review-intelligence
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🚀 Deployment

### Streamlit Community Cloud

1. **Fork this repository** to your GitHub account

2. **Go to** [Streamlit Community Cloud](https://streamlit.io/cloud)

3. **Click "New app"** and select:
   - Repository: `your-username/yelp-review-intelligence`
   - Branch: `main`
   - Main file path: `app.py`

4. **Click "Deploy"** and wait for deployment to complete

5. **Your app will be live** at `https://your-app-name.streamlit.app`

## 🤖 Model Training

### Train FLAN-T5 Model

1. **Open Google Colab**
   - Upload `train_flan_t5_yelp.ipynb` to Google Colab

2. **Enable GPU**
   - Runtime → Change runtime type → GPU (T4)

3. **Run all cells**
   - The notebook will:
     - Load Yelp Review Full dataset
     - Fine-tune FLAN-T5-small model
     - Upload trained model to Hugging Face Hub

4. **Training time**: Approximately 2-4 hours on Colab T4 GPU

5. **Model will be available** at `https://huggingface.co/RLau33/flan-t5-yelp-analysis`

### Train DistilBERT Model (Already Done)

The DistilBERT quality classification model is already trained and available at:
- `RLau33/yelp-review-quality-v2`

## 📊 Dataset

**Yelp Review Full Dataset**
- Source: [Hugging Face Datasets](https://huggingface.co/datasets/Yelp/yelp_review_full)
- Training samples: 650,000 reviews
- Test samples: 50,000 reviews
- Star ratings: 1-5 stars (mapped to labels 0-4)

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Classification Model** | DistilBERT |
| **Generation Model** | FLAN-T5-small |
| **ML Framework** | Hugging Face Transformers |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Deployment** | Streamlit Community Cloud |

## 📁 Project Structure

```
yelp-review-intelligence/
├── app.py                          # Main Streamlit application
├── train_flan_t5_yelp.ipynb       # FLAN-T5 training notebook
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── .gitignore                      # Git ignore file
```

## 🎓 Course Information

**Course**: ISOM5240 - AI and Business Applications  
**Institution**: HKUST Business School  
**Semester**: Fall 2024  
**Author**: RLau33

## 📝 Assignment Requirements

This project fulfills the following requirements:

✅ **Two Model Pipelines**
- Pipeline 1: DistilBERT for classification
- Pipeline 2: FLAN-T5 for text generation

✅ **Hugging Face Integration**
- Uses pre-trained models from Hugging Face
- Fine-tuned on Yelp Review Full dataset
- Models uploaded to Hugging Face Hub

✅ **Deployment**
- Deployed on Streamlit Community Cloud
- Connected to GitHub repository
- Publicly accessible web application

✅ **Documentation**
- Comprehensive README
- Training notebooks with comments
- Code documentation

## 🔗 Links

- **Live App**: [https://your-app-url.streamlit.app](https://your-app-url.streamlit.app)
- **GitHub**: [https://github.com/rhealau](https://github.com/rhealau)
- **FLAN-T5 Model**: [https://huggingface.co/RLau33/flan-t5-yelp-analysis](https://huggingface.co/RLau33/flan-t5-yelp-analysis)
- **DistilBERT Model**: [https://huggingface.co/RLau33/yelp-review-quality-v2](https://huggingface.co/RLau33/yelp-review-quality-v2)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yelp** for providing the review dataset
- **Hugging Face** for the Transformers library and model hosting
- **Google** for FLAN-T5 model
- **Streamlit** for the deployment platform
- **HKUST** for the course and guidance

## 📧 Contact

For questions or feedback, please contact:
- **GitHub**: [@rhealau](https://github.com/rhealau)
- **Email**: Your email (optional)

---

**Built with ❤️ for ISOM5240 | HKUST Business School**
