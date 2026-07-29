# 💊 Intelligent Drug Analysis Platform — Streamlit UI

An intelligent drug analysis platform that uses ChromaDB and AI to search, analyze, and compare pharmaceutical information.

## 🚀 Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

#### English Version

```bash
streamlit run streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

#### Vietnamese Version

```bash
streamlit run streamlit_app_vi.py --server.port 8502
```

The application will be available at:

```text
http://localhost:8502
```

## 🎯 Key Features

### 1. Semantic Drug Search

- Search for medications using natural language
- Describe symptoms or the type of medication you are looking for
- View similarity scores and detailed drug information
- Search by composition, indications, or side effects

Example queries:

- `pain relief for headache`
- `antibiotic for infection`
- `medication for sleep disorders`

### 2. Drug Substitution

- Find alternative medications with similar effects
- Filter out medications with severe side effects
- Prioritize medications with better user reviews
- Sort results by similarity score or review rating

### 3. Side Effect Analysis

- Analyze potential interactions between multiple medications
- Detect and display possible side effects
- Visualize side effects by organ system
- Select and compare multiple medications

### 4. Medical Q&A Chatbot

- Ask natural-language questions about medications and general health topics
- Uses a Retrieval-Augmented Generation system powered by ChromaDB
- Provides medication suggestions based on reported symptoms
- Stores conversation history during the session

### 5. Manufacturer Analytics

- Analyze pharmaceutical manufacturers
- View statistics on medication quantity and quality
- Compare different manufacturers
- Visualize review score distributions

### 6. Dashboard Overview

- View an overview of the entire system
- Explore top manufacturers and medication categories
- Review medication quality statistics
- Analyze the overall dataset

## 🔧 Data Structure

The application uses four ChromaDB collections:

- `drugs_main`: General medication information
- `drugs_side_effects`: Medication side effects
- `drugs_composition`: Medication ingredients and composition
- `drugs_reviews`: User reviews and ratings

## 📱 User Interface

- **Sidebar Navigation:** Select features from the left-side menu
- **Responsive Design:** Adapts to different screen sizes
- **Interactive Charts:** Data visualizations powered by Plotly
- **Real-Time Search:** Search and analyze medication data in real time

## 🎨 Customization

Custom CSS is embedded in `streamlit_app.py` and includes:

- A healthcare-inspired color theme
- Custom medication cards
- Responsive layouts
- Hover effects
- Improved spacing and visual hierarchy

## ⚡ Performance Optimization

- AI models are cached using `@st.cache_resource`
- ChromaDB uses a persistent client connection
- Queries are optimized with result limits
- Resource-intensive operations are processed efficiently
- Repeated model loading is minimized

## 🔒 Medical Disclaimer

> **Important:** The information provided by this application is for educational and reference purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult a qualified doctor, pharmacist, or healthcare professional before taking or changing any medication.

## 🐛 Troubleshooting

### ChromaDB Connection Error

```text
Error connecting to ChromaDB
```

**Solution:** Make sure the `./chroma_db` directory exists and contains valid vectorized data.

### AI Model Loading Error

```text
Error loading model
```

**Solution:** Check your internet connection. The model may need to be downloaded when the application runs for the first time.

### Slow Performance

Try the following solutions:

- Install Watchdog:

```bash
pip install watchdog
```

- Allocate more RAM if possible
- Reduce the `n_results` value used in search queries
- Avoid loading unnecessary collections
- Restart the Streamlit application after updating the database

## 📞 Support

If you encounter an issue, verify that:

1. All dependencies have been installed successfully
2. The ChromaDB database exists and contains data
3. Your internet connection is stable
4. Ports `8501` and `8502` are available
5. The required AI model can be loaded successfully
6. You are using a compatible Python version
