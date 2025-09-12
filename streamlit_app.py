#!/usr/bin/env python3
"""
Nền tảng Phân tích Thuốc Thông minh - Streamlit UI
=================================================

Nền tảng tìm kiếm và phân tích thuốc toàn diện sử dụng ChromaDB và AI.
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Nền tảng Phân tích Thuốc Thông minh",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .search-box {
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .drug-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .similarity-score {
        color: #1f77b4;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_chromadb():
    """Thiết lập ChromaDB client và collections"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = {
            'drugs_main': client.get_collection("drugs_main"),
            'drugs_side_effects': client.get_collection("drugs_side_effects"),
            'drugs_composition': client.get_collection("drugs_composition"),
            'drugs_reviews': client.get_collection("drugs_reviews")
        }
        return client, collections
    except Exception as e:
        st.error(f"Lỗi kết nối ChromaDB: {e}")
        return None, None

@st.cache_resource
def load_model():
    """Tải mô hình sentence transformer"""
    try:
        with st.spinner("Đang tải mô hình AI..."):
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model
    except Exception as e:
        st.error(f"Lỗi tải mô hình: {e}")
        return None

def search_medicines(collection, query_text: str, model, n_results: int = 5):
    """Tìm kiếm thuốc sử dụng độ tương đồng ngữ nghĩa"""
    try:
        query_embedding = model.encode(query_text).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        return results
    except Exception as e:
        st.error(f"Lỗi tìm kiếm: {e}")
        return None

def format_similarity(distance: float) -> str:
    """Định dạng điểm tương đồng thành phần trăm"""
    return f"{(1 - distance) * 100:.1f}%"

def semantic_search_page(collections, model):
    """Trang 1: Tìm kiếm Ngữ nghĩa"""
    st.markdown('<div class="main-header">🔍 Tìm kiếm Ngữ nghĩa</div>', unsafe_allow_html=True)
    
    st.markdown("### Tìm thuốc bằng mô tả ngôn ngữ tự nhiên")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Describe your symptoms or medicine needs:",
            placeholder="e.g., pain relief, headache, antibiotics, heart medication",
            key="search_query"
        )
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Search options
    with st.expander("Search Options"):
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider("Number of results", 5, 20, 10)
        with col2:
            search_collection = st.selectbox(
                "Search in:",
                ["Main Database", "By Composition", "By Side Effects"],
                index=0
            )
    
    if search_button and query:
        with st.spinner("Searching medicines..."):
            # Select collection
            if search_collection == "Main Database":
                collection = collections['drugs_main']
            elif search_collection == "Theo thành phần":
                collection = collections['drugs_composition']
            else:
                collection = collections['drugs_side_effects']
            
            results = search_medicines(collection, query, model, num_results)
            
            if results and results['metadatas'][0]:
                st.success(f"Tìm thấy {len(results['metadatas'][0])} thuốc phù hợp với tìm kiếm của bạn")
                
                # Hiển thị kết quả
                for i, metadata in enumerate(results['metadatas'][0]):
                    distance = results['distances'][0][i]
                    similarity = format_similarity(distance)
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="drug-card">
                            <h4>💊 {metadata['medicine_name']}</h4>
                            <p><strong>Độ tương đồng:</strong> <span class="similarity-score">{similarity}</span></p>
                            <p><strong>🧪 Thành phần:</strong> {metadata['composition'][:100]}...</p>
                            <p><strong>🎯 Công dụng:</strong> {metadata['uses'][:150]}...</p>
                            <p><strong>🏭 Hãng sản xuất:</strong> {metadata['manufacturer']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Chỉ số đánh giá
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Đánh giá xuất sắc", f"{metadata['excellent_review']}%", delta=None)
                        with col2:
                            st.metric("Đánh giá trung bình", f"{metadata['average_review']}%", delta=None)
                        with col3:
                            st.metric("Đánh giá kém", f"{metadata['poor_review']}%", delta=None)
                        
                        st.markdown("---")
            else:
                st.warning("Không tìm thấy thuốc nào phù hợp với tiêu chí tìm kiếm của bạn.")

def drug_substitution_page(collections, model):
    """Trang 2: Thay thế Thuốc"""
    st.markdown('<div class="main-header">🔄 Thay thế Thuốc</div>', unsafe_allow_html=True)
    
    st.markdown("### Tìm thuốc thay thế có tác dụng tương tự")
    
    # Nhập liệu
    medicine_name = st.text_input(
        "Nhập tên thuốc cần tìm thay thế:",
        placeholder="Ví dụ: Paracetamol, Aspirin, Amoxicillin"
    )
    
    # Tùy chọn lọc
    col1, col2 = st.columns(2)
    with col1:
        avoid_side_effects = st.checkbox("Tránh thuốc có tác dụng phụ mạnh")
    with col2:
        prefer_good_reviews = st.checkbox("Uu tiên thuốc có đánh giá tốt")
    
    if st.button("Tìm Thay thế", type="primary") and medicine_name:
        with st.spinner("Đang tìm thuốc thay thế..."):
            # Search in composition collection
            results = search_medicines(collections['drugs_composition'], medicine_name, model, 10)
            
            if results and results['metadatas'][0]:
                alternatives = []
                for i, metadata in enumerate(results['metadatas'][0]):
                    distance = results['distances'][0][i]
                    
                    # Apply filters
                    if avoid_side_effects and metadata.get('poor_review', 0) > 30:
                        continue
                    if prefer_good_reviews and metadata.get('excellent_review', 0) < 50:
                        continue
                    
                    alternatives.append({
                        'name': metadata['medicine_name'],
                        'similarity': (1 - distance) * 100,
                        'composition': metadata['composition'],
                        'uses': metadata['uses'],
                        'manufacturer': metadata['manufacturer'],
                        'excellent_review': metadata.get('excellent_review', 0),
                        'poor_review': metadata.get('poor_review', 0)
                    })
                
                if alternatives:
                    st.success(f"Found {len(alternatives)} alternative medicines")
                    
                    # Sort options
                    sort_by = st.selectbox("Sort by:", ["Similarity", "Review Score"])
                    
                    if sort_by == "Review Score":
                        alternatives.sort(key=lambda x: x['excellent_review'], reverse=True)
                    else:
                        alternatives.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Display alternatives
                    for alt in alternatives:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"""
                            **💊 {alt['name']}**  
                            *Similarity: {alt['similarity']:.1f}%*  
                            🧪 {alt['composition'][:80]}...  
                            🎯 {alt['uses'][:100]}...  
                            🏭 {alt['manufacturer']}
                            """)
                        with col2:
                            st.metric("Good Reviews", f"{alt['excellent_review']}%")
                        st.markdown("---")
                else:
                    st.warning("No alternatives found matching your filter criteria.")
            else:
                st.warning("No alternatives found for this medicine.")

def side_effects_analysis_page(collections, model):
    """Page 3: Side Effects Analysis"""
    st.markdown('<div class="main-header">⚠️ Side Effect & Interaction Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("### Analyze potential side effects and drug interactions")
    
    # Get all medicine names for multi-select
    try:
        all_medicines = collections['drugs_main'].get(include=["metadatas"])
        medicine_names = [meta['medicine_name'] for meta in all_medicines['metadatas']]
        medicine_names = sorted(list(set(medicine_names)))[:1000]  # Limit for performance
    except:
        medicine_names = []
    
    # Multi-select for medicines
    selected_medicines = st.multiselect(
        "Select medicines to analyze (or type to search):",
        options=medicine_names,
        help="Select multiple medicines to check for interactions"
    )
    
    if st.button("Analyze Interactions", type="primary") and selected_medicines:
        with st.spinner("Analyzing side effects and interactions..."):
            st.success(f"Analyzing {len(selected_medicines)} selected medicines")
            
            # Display selected medicines
            st.markdown("### Selected Medicines:")
            for i, med in enumerate(selected_medicines, 1):
                st.markdown(f"{i}. **{med}**")
            
            # Mock analysis (in real implementation, you'd analyze actual interactions)
            st.markdown("### ⚠️ Potential Interactions:")
            
            if len(selected_medicines) > 1:
                st.warning("⚠️ Multiple medicines detected - please consult with healthcare provider")
                
                # Mock side effects analysis
                common_side_effects = ["Nausea", "Dizziness", "Headache", "Fatigue", "Stomach upset"]
                side_effects_data = {
                    'Side Effect': common_side_effects,
                    'Frequency': np.random.randint(10, 80, len(common_side_effects))
                }
                
                df_side_effects = pd.DataFrame(side_effects_data)
                
                # Bar chart of side effects
                fig = px.bar(df_side_effects, x='Side Effect', y='Frequency',
                           title='Common Side Effects Frequency',
                           color='Frequency',
                           color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                
                # Categories analysis
                categories = ['Liver', 'Kidney', 'Heart', 'Nervous System', 'Digestive']
                category_counts = np.random.randint(1, 5, len(categories))
                
                fig_pie = px.pie(values=category_counts, names=categories,
                               title='Side Effects by System')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Select multiple medicines to analyze interactions")

def chatbot_page(collections, model):
    """Page 4: Medical Q&A Chatbot"""
    st.markdown('<div class="main-header">💬 Medical Q&A Chatbot</div>', unsafe_allow_html=True)
    
    st.markdown("### Ask medical questions in natural language")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about medicines, symptoms, or health conditions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search for relevant medicines
                results = search_medicines(collections['drugs_main'], prompt, model, 3)
                
                if results and results['metadatas'][0]:
                    response = f"Based on your query about '{prompt}', here are some relevant medicines:\n\n"
                    
                    for i, metadata in enumerate(results['metadatas'][0][:3]):
                        distance = results['distances'][0][i]
                        similarity = format_similarity(distance)
                        
                        response += f"**{i+1}. {metadata['medicine_name']}** (Similarity: {similarity})\n"
                        response += f"- *Composition*: {metadata['composition'][:60]}...\n"
                        response += f"- *Uses*: {metadata['uses'][:80]}...\n"
                        response += f"- *Manufacturer*: {metadata['manufacturer']}\n\n"
                    
                    response += "\n⚠️ **Disclaimer**: This information is for educational purposes only. Always consult with a healthcare professional before making medical decisions."
                else:
                    response = "I couldn't find specific medicines for your query. Could you provide more details about your symptoms or the type of medicine you're looking for?"
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

def manufacturer_analytics_page(collections):
    """Page 5: Manufacturer Analytics"""
    st.markdown('<div class="main-header">🏭 Manufacturer Analytics</div>', unsafe_allow_html=True)
    
    st.markdown("### Analyze pharmaceutical manufacturers")
    
    try:
        # Get all medicines data
        all_medicines = collections['drugs_main'].get(include=["metadatas"])
        
        # Create DataFrame
        medicines_df = pd.DataFrame([
            {
                'name': meta['medicine_name'],
                'manufacturer': meta['manufacturer'],
                'composition': meta['composition'],
                'uses': meta['uses'],
                'excellent_review': meta.get('excellent_review', 0),
                'average_review': meta.get('average_review', 0),
                'poor_review': meta.get('poor_review', 0)
            }
            for meta in all_medicines['metadatas']
        ])
        
        # Get top manufacturers
        manufacturer_counts = medicines_df['manufacturer'].value_counts()
        top_manufacturers = manufacturer_counts.head(20).index.tolist()
        
        # Manufacturer selection
        selected_manufacturer = st.selectbox(
            "Select a manufacturer to analyze:",
            options=top_manufacturers
        )
        
        if selected_manufacturer:
            # Filter data for selected manufacturer
            manufacturer_data = medicines_df[medicines_df['manufacturer'] == selected_manufacturer]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Medicines", len(manufacturer_data))
            with col2:
                avg_excellent = manufacturer_data['excellent_review'].mean()
                st.metric("Avg Excellent Reviews", f"{avg_excellent:.1f}%")
            with col3:
                avg_poor = manufacturer_data['poor_review'].mean()
                st.metric("Avg Poor Reviews", f"{avg_poor:.1f}%")
            with col4:
                market_share = len(manufacturer_data) / len(medicines_df) * 100
                st.metric("Market Share", f"{market_share:.2f}%")
            
            # Review distribution
            review_data = {
                'Review Type': ['Excellent', 'Average', 'Poor'],
                'Percentage': [
                    manufacturer_data['excellent_review'].mean(),
                    manufacturer_data['average_review'].mean(),
                    manufacturer_data['poor_review'].mean()
                ]
            }
            
            fig = px.pie(values=review_data['Percentage'], names=review_data['Review Type'],
                        title=f'Review Distribution - {selected_manufacturer}',
                        color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Top manufacturers comparison
            st.markdown("### Top Manufacturers Comparison")
            
            comparison_data = []
            for mfr in top_manufacturers[:10]:
                mfr_data = medicines_df[medicines_df['manufacturer'] == mfr]
                comparison_data.append({
                    'Manufacturer': mfr,
                    'Medicine Count': len(mfr_data),
                    'Avg Excellent Review': mfr_data['excellent_review'].mean(),
                    'Avg Poor Review': mfr_data['poor_review'].mean()
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig_bar = px.bar(comparison_df, x='Manufacturer', y='Medicine Count',
                           title='Medicine Count by Top Manufacturers',
                           text='Medicine Count')
            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading manufacturer data: {e}")

def dashboard_overview_page(collections):
    """Page 6: Dashboard Overview"""
    st.markdown('<div class="main-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    
    st.markdown("### System Overview and Statistics")
    
    try:
        # Get collection counts
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            main_count = collections['drugs_main'].count()
            st.metric("Total Medicines", f"{main_count:,}")
        
        with col2:
            se_count = collections['drugs_side_effects'].count()
            st.metric("Side Effects Records", f"{se_count:,}")
        
        with col3:
            comp_count = collections['drugs_composition'].count()
            st.metric("Composition Records", f"{comp_count:,}")
        
        with col4:
            review_count = collections['drugs_reviews'].count()
            st.metric("Review Records", f"{review_count:,}")
        
        # Get sample data for analysis
        sample_data = collections['drugs_main'].get(
            limit=1000,
            include=["metadatas"]
        )
        
        medicines_df = pd.DataFrame([
            {
                'name': meta['medicine_name'],
                'manufacturer': meta['manufacturer'],
                'excellent_review': meta.get('excellent_review', 0),
                'uses': meta['uses']
            }
            for meta in sample_data['metadatas']
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top manufacturers
            st.markdown("### 🏭 Top 10 Manufacturers")
            top_mfrs = medicines_df['manufacturer'].value_counts().head(10)
            
            fig_mfr = px.bar(
                x=top_mfrs.values,
                y=top_mfrs.index,
                orientation='h',
                title='Medicines Count by Manufacturer'
            )
            fig_mfr.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_mfr, use_container_width=True)
        
        with col2:
            # Review distribution
            st.markdown("### ⭐ Review Score Distribution")
            
            # Create review categories
            review_categories = []
            for score in medicines_df['excellent_review']:
                if score >= 70:
                    review_categories.append('Excellent (70%+)')
                elif score >= 50:
                    review_categories.append('Good (50-70%)')
                elif score >= 30:
                    review_categories.append('Average (30-50%)')
                else:
                    review_categories.append('Poor (<30%)')
            
            review_dist = pd.Series(review_categories).value_counts()
            
            fig_review = px.pie(
                values=review_dist.values,
                names=review_dist.index,
                title='Medicine Quality Distribution'
            )
            st.plotly_chart(fig_review, use_container_width=True)
        
        # Medicine categories analysis
        st.markdown("### 💊 Medicine Categories Analysis")
        
        # Extract categories from uses (simplified)
        categories = []
        for uses in medicines_df['uses']:
            if 'pain' in uses.lower():
                categories.append('Pain Relief')
            elif 'infection' in uses.lower() or 'bacterial' in uses.lower():
                categories.append('Anti-infectious')
            elif 'diabetes' in uses.lower():
                categories.append('Diabetes')
            elif 'blood pressure' in uses.lower() or 'hypertension' in uses.lower():
                categories.append('Cardiovascular')
            elif 'asthma' in uses.lower() or 'respiratory' in uses.lower():
                categories.append('Respiratory')
            else:
                categories.append('Other')
        
        category_dist = pd.Series(categories).value_counts()
        
        fig_cat = px.bar(
            x=category_dist.index,
            y=category_dist.values,
            title='Medicine Distribution by Category'
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("💊 Drug Intelligence Platform")
    st.sidebar.markdown("---")
    
    # Initialize data
    if 'initialized' not in st.session_state:
        with st.spinner("Initializing system..."):
            client, collections = setup_chromadb()
            model = load_model()
            
            if client and collections and model:
                st.session_state.client = client
                st.session_state.collections = collections
                st.session_state.model = model
                st.session_state.initialized = True
            else:
                st.error("Failed to initialize system. Please check your setup.")
                return
    
    # Navigation menu
    pages = {
        "🔍 Semantic Search": semantic_search_page,
        "🔄 Drug Substitution": drug_substitution_page,
        "⚠️ Side Effect Analysis": side_effects_analysis_page,
        "💬 Medical Q&A Chatbot": chatbot_page,
        "🏭 Manufacturer Analytics": manufacturer_analytics_page,
        "📊 Dashboard Overview": dashboard_overview_page
    }
    
    selected_page = st.sidebar.selectbox("Choose a function:", list(pages.keys()))
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.success("✅ ChromaDB Connected")
    st.sidebar.success("✅ AI Model Loaded")
    st.sidebar.info(f"📊 {st.session_state.collections['drugs_main'].count():,} medicines available")
    
    # Display selected page
    if selected_page in ["💬 Medical Q&A Chatbot"]:
        pages[selected_page](st.session_state.collections, st.session_state.model)
    elif selected_page in ["🏭 Manufacturer Analytics", "📊 Dashboard Overview"]:
        pages[selected_page](st.session_state.collections)
    else:
        pages[selected_page](st.session_state.collections, st.session_state.model)

if __name__ == "__main__":
    main()