#!/usr/bin/env python3
"""
Nền tảng Phân tích Thuốc Thông minh - Streamlit UI
=================================================

Nền tảng tìm kiếm và phân tích thuốc toàn diện sử dụng ChromaDB và AI.
"""

import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
# Cấu hình trang
st.set_page_config(
    page_title="Nền tảng Phân tích Thuốc Thông minh",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
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
        background-color: ##112838;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .drug-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: ##112838;
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

@st.cache_resource
def get_openai_client():
    key = get_openai_api_key()
    if not key:
        return None
    return OpenAI(api_key=key)

def _reviews_from_main(name, collections, model):
    """Lấy % đánh giá từ drugs_main theo tên thuốc (n_results=1)."""
    try:
        if not name or "drugs_main" not in collections:
            return None, None, None
        hits = search_medicines(collections["drugs_main"], name, model, n_results=1)
        metas = (hits or {}).get("metadatas", [[]])
        if metas and metas[0]:
            m = metas[0][0] or {}
            return m.get("excellent_review"), m.get("average_review"), m.get("poor_review")
    except Exception:
        pass
    return None, None, None

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
    
    # Ô tìm kiếm
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Mô tả triệu chứng hoặc loại thuốc cần tìm:",
            placeholder="Ví dụ: giảm đau, đau đầu, kháng sinh, thuốc tim",
            key="search_query"
        )
    with col2:
        search_button = st.button("🔍 Tìm kiếm", type="primary")
    
    # Tùy chọn tìm kiếm
    with st.expander("Tùy chọn tìm kiếm"):
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider("Số kết quả", 5, 20, 10)
        with col2:
            search_collection = st.selectbox(
                "Tìm kiếm trong:",
                ["Cơ sở dữ liệu chính", "Theo thành phần", "Theo tác dụng phụ"],
                index=0
            )
    
    if search_button and query:
        with st.spinner("Đang tìm kiếm thuốc..."):
            # Chọn collection
            if search_collection == "Cơ sở dữ liệu chính":
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
        prefer_good_reviews = st.checkbox("Ưu tiên thuốc có đánh giá tốt")
    
    if st.button("Tìm Thay thế", type="primary") and medicine_name:
        with st.spinner("Đang tìm thuốc thay thế..."):
            # Tìm trong collection thành phần
            results = search_medicines(collections['drugs_composition'], medicine_name, model, 10)
            
            if results and results['metadatas'][0]:
                alternatives = []
                for i, metadata in enumerate(results['metadatas'][0]):
                    distance = results['distances'][0][i]
                    
                    # Áp dụng bộ lọc
                    if avoid_side_effects and metadata.get('poor_review', 0) > 30:
                        continue
                    if prefer_good_reviews and metadata.get('excellent_review', 0) < 50:
                        continue

                    alternatives.append({
                        'name': metadata.get('medicine_name', '(Không rõ tên)'),
                        'similarity': (1 - distance) * 100,
                        'composition': metadata.get('composition') or metadata.get('ingredients') \
                                       or (results['documents'][0][i] if results.get('documents') else ""),  # FIX
                        'uses': metadata.get('uses', ''),  # FIX
                        'manufacturer': metadata.get('manufacturer', ''),  # FIX
                        'excellent_review': metadata.get('excellent_review', 0),  # FIX
                        'poor_review': metadata.get('poor_review', 0),  # FIX
                    })

                # Nếu % đánh giá trống (hoặc =0) trong composition → hỏi nhanh sang main để bồi dữ liệu
                for alt in alternatives:
                    has_reviews = any([
                        alt.get("excellent_review"),
                        alt.get("average_review"),
                        alt.get("poor_review")
                    ])
                    if not has_reviews:
                        ex, av, po = _reviews_from_main(alt.get("name"), collections, model)
                        if any([ex, av, po]):
                            alt["excellent_review"] = ex or 0
                            alt["average_review"] = av or 0
                            alt["poor_review"] = po or 0

                if alternatives:
                    st.success(f"Tìm thấy {len(alternatives)} thuốc thay thế")
                    
                    # Tùy chọn sắp xếp
                    sort_by = st.selectbox("Sắp xếp theo:", ["Độ tương đồng", "Điểm đánh giá"])
                    
                    if sort_by == "Điểm đánh giá":
                        alternatives.sort(key=lambda x: x['excellent_review'], reverse=True)
                    else:
                        alternatives.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Hiển thị thuốc thay thế
                    for alt in alternatives:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"""
                            **💊 {alt.get('name')}**  
                            *Độ tương đồng: {alt.get('similarity'):.1f}%*  
                            🧪 {alt.get('composition')[:80]}...  
                            🎯 {alt.get('uses')[:100]}...  
                            🏭 {alt.get('manufacturer')}
                            """)
                        with col2:
                            st.metric("Đánh giá tốt", f"{alt.get('excellent_review',0)}%")
                            st.metric("Trung bình", f"{alt.get('average_review', 0)}%")
                            st.metric("Kém", f"{alt.get('poor_review', 0)}%")

                        st.markdown("---")
                else:
                    st.warning("Không tìm thấy thuốc thay thế nào phù hợp với tiêu chí lọc của bạn.")
            else:
                st.warning("Không tìm thấy thuốc thay thế cho loại thuốc này.")

def side_effects_analysis_page(collections):
    """Trang 3: Phân tích Tác dụng Phụ"""
    st.markdown('<div class="main-header">⚠️ Phân tích Tác dụng Phụ & Tương tác</div>', unsafe_allow_html=True)
    
    st.markdown("### Phân tích tác dụng phụ tiềm ẩn và tương tác thuốc")
    
    # Lấy tất cả tên thuốc để multi-select
    try:
        all_medicines = collections['drugs_main'].get(include=["metadatas"])
        medicine_names = [meta['medicine_name'] for meta in all_medicines['metadatas']]
        medicine_names = sorted(list(set(medicine_names)))[:1000]  # Giới hạn để tăng hiệu suất
    except:
        medicine_names = []
    
    # Multi-select cho thuốc
    selected_medicines = st.multiselect(
        "Chọn thuốc để phân tích (hoặc gõ để tìm kiếm):",
        options=medicine_names,
        help="Chọn nhiều thuốc để kiểm tra tương tác"
    )
    
    if st.button("Phân tích Tương tác", type="primary") and selected_medicines:
        with st.spinner("Đang phân tích tác dụng phụ và tương tác..."):
            st.success(f"Đang phân tích {len(selected_medicines)} thuốc đã chọn")
            
            # Hiển thị thuốc đã chọn
            st.markdown("### Thuốc đã chọn:")
            for i, med in enumerate(selected_medicines, 1):
                st.markdown(f"{i}. **{med}**")
            
            # Phân tích giả lập (trong thực tế, bạn sẽ phân tích tương tác thực)
            st.markdown("### ⚠️ Tương tác Tiềm ẩn:")
            
            if len(selected_medicines) > 1:
                st.warning("⚠️ Phát hiện nhiều thuốc - vui lòng tham khảo ý kiến bác sĩ")
                
                # Phân tích tác dụng phụ giả lập
                common_side_effects = ["Buồn nôn", "Chóng mặt", "Đau đầu", "Mệt mỏi", "Đau dạ dày"]
                side_effects_data = {
                    'Tác dụng phụ': common_side_effects,
                    'Tần suất': np.random.randint(10, 80, len(common_side_effects))
                }
                
                df_side_effects = pd.DataFrame(side_effects_data)
                
                # Biểu đồ cột tác dụng phụ
                fig = px.bar(df_side_effects, x='Tác dụng phụ', y='Tần suất',
                           title='Tần suất Tác dụng Phụ Thường gặp',
                           color='Tần suất',
                           color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                
                # Phân tích theo hệ cơ quan
                categories = ['Gan', 'Thận', 'Tim', 'Hệ thần kinh', 'Tiêu hóa']
                category_counts = np.random.randint(1, 5, len(categories))
                
                fig_pie = px.pie(values=category_counts, names=categories,
                               title='Tác dụng Phụ theo Hệ Cơ quan')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Chọn nhiều thuốc để phân tích tương tác")

def get_openai_api_key():
    try:
        if hasattr(st, "secrets") and "openai_api_key" in st.secrets:
            return st.secrets["openai_api_key"]
    except Exception:
        pass
    if "openai_api_key" in st.session_state:
        return st.session_state["openai_api_key"]
    return os.getenv("OPENAI_API_KEY")

def generate_vi_answer_openai(user_q: str, results: dict, model_name: str = "gpt-5-mini"):
    client = get_openai_client()
    if client is None:
        return None  # không có key -> fallback DB

    # ... build context như bạn đã làm ...
    rsp = client.responses.create(
        model=model_name,              # "gpt-5-mini" hoặc "gpt-5"
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    try:
        return rsp.output_text.strip()
    except Exception:
        return None

def chatbot_page(collections, model):
    """Chatbot Y tế Q&A: ưu tiên trả lời bằng GPT-5 (tiếng Việt) trên ngữ cảnh nội bộ;
    nếu API lỗi thì rơi về chế độ chỉ truy hồi trong DB và tóm tắt kết quả.
    """
    import re

    st.markdown('<div class="main-header">💬 Chatbot Y tế Q&A</div>', unsafe_allow_html=True)
    st.markdown("### Hỏi đáp về thuốc và sức khỏe bằng ngôn ngữ tự nhiên")

    # --- hiển thị lịch sử
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # --- ô nhập
    user_q = st.chat_input("Hỏi tôi về thuốc, triệu chứng, hoặc tình trạng sức khỏe...")
    if not user_q:
        return
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # --- kiểm tra tài nguyên
    if model is None or collections is None or "drugs_main" not in collections:
        reply = "Hệ thống chưa sẵn sàng (thiếu model hoặc collection)."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    # --- truy hồi ngữ cảnh nội bộ
    results = search_medicines(collections["drugs_main"], user_q, model, n_results=6)

    # --- 1) Thử gọi ChatGPT (GPT-5) để trả lời tiếng Việt từ NGỮ CẢNH nội bộ
    answer = None
    try:
        # 'gpt-5-mini' nhanh & tiết kiệm; bạn có thể đổi thành 'gpt-5'
        answer = generate_vi_answer_openai(user_q, results, model_name="gpt-5-mini")
    except Exception as e:
        st.warning(f"Không gọi được ChatGPT: {e}")

    if answer:
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        return

    # --- 2) Fallback: chỉ DB (không gọi API) ---------------------------------

    # helpers nhỏ để làm sạch & “dịch nhẹ” field uses
    def _normalize_uses_en(txt: str) -> str:
        if not isinstance(txt, str):
            return ""
        t = re.sub(r'(?<=stroke)Treatment', '. Treatment', txt, flags=re.I)  # "strokeTreatment" -> "stroke. Treatment"
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    _USES_EN_VI = [
        (r'\bTreatment of\b', 'Điều trị'),
        (r'\bPrevention of\b', 'Phòng ngừa'),
        (r'\bManagement of\b', 'Kiểm soát'),
        (r'\bHypertension\b', 'tăng huyết áp'),
        (r'\bhigh blood pressure\b', 'cao huyết áp'),
        (r'\bHeart failure\b', 'suy tim'),
        (r'\bheart attack\b', 'nhồi máu cơ tim'),
        (r'\bstroke\b', 'đột quỵ'),
        (r'\bConstipation\b', 'táo bón'),
        (r'\bPain\b', 'đau'),
        (r'\bFever\b', 'sốt'),
        (r'\bCough\b', 'ho'),
        (r'\bAllergy\b', 'dị ứng'),
        (r'\bInfection\b', 'nhiễm khuẩn'),
        (r'\bDiarrhea\b', 'tiêu chảy'),
    ]
    def _vi_translate_uses(en_text: str) -> str:
        t = _normalize_uses_en(en_text or "")
        for en, vi in _USES_EN_VI:
            t = re.sub(en, vi, t, flags=re.I)
        return (t[:1].upper() + t[1:]) if t else t

    metas = (results or {}).get("metadatas", [[]])
    dists = (results or {}).get("distances", [[]])
    meta_rows = metas[0] if metas else []
    dist_rows = dists[0] if dists else []
    n = min(len(meta_rows), len(dist_rows)) if meta_rows and dist_rows else 0

    if n == 0:
        reply = "Chưa tìm thấy kết quả phù hợp trong cơ sở dữ liệu. Bạn mô tả chi tiết hơn về triệu chứng hoặc tên thuốc nhé?"
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    lines = []
    for i, meta in enumerate(meta_rows[:3]):
        meta = meta or {}
        dist = dist_rows[i] if i < len(dist_rows) else None
        sim = format_similarity(dist) if isinstance(dist, (int, float)) else "—"

        name = meta.get("medicine_name", "(Không rõ tên)")
        comp = (meta.get("composition") or meta.get("ingredients", "")).strip()
        uses_en = _normalize_uses_en(meta.get("uses", ""))
        uses_vi = _vi_translate_uses(uses_en) if uses_en else ""
        manu = meta.get("manufacturer", "Không rõ")
        ex = meta.get("excellent_review", 0)
        av = meta.get("average_review", 0)
        po = meta.get("poor_review", 0)

        block = [
            f"**{i+1}. {name}** *(Độ tương đồng: {sim})*",
            f"- **Thành phần:** {comp[:160] + ('...' if len(comp) > 160 else '')}" if comp else "- **Thành phần:** (chưa có dữ liệu)",
            f"- **Công dụng (VI):** {uses_vi}" if uses_vi else "- **Công dụng:** (chưa có dữ liệu)",
        ]
        if uses_en and uses_vi and uses_vi.lower() != uses_en.lower():
            block.append(f"  <span style='opacity:0.7'>EN: {uses_en}</span>")
        block += [
            f"- **Nhà sản xuất:** {manu}",
            f"- **Đánh giá:** xuất sắc {ex}%, trung bình {av}%, kém {po}%",
        ]

        # gợi ý thuốc tương tự theo thành phần
        related_names = []
        if comp and "drugs_composition" in collections:
            rel_hits = search_medicines(collections["drugs_composition"], comp, model, n_results=8)
            r_metas = (rel_hits or {}).get("metadatas", [[]])
            if r_metas and r_metas[0]:
                for m2 in r_metas[0]:
                    nm = (m2 or {}).get("medicine_name")
                    if nm and nm != name and nm not in related_names:
                        related_names.append(nm)
                    if len(related_names) >= 3:
                        break
        if related_names:
            block.append(f"- **Thuốc tương tự (thành phần gần giống):** {', '.join(related_names)}")

        lines.append("\n".join(block))

    lines.append("\n⚠️ *Thông tin chỉ mang tính tham khảo. Không tự ý dùng/đổi thuốc; hãy hỏi dược sĩ/bác sĩ.*")
    reply = "\n\n".join(lines)

    with st.chat_message("assistant"):
        st.markdown(reply, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": reply})

def manufacturer_analytics_page(collections):
    """Trang 5: Phân tích Nhà sản xuất"""
    st.markdown('<div class="main-header">🏭 Phân tích Nhà sản xuất</div>', unsafe_allow_html=True)
    
    st.markdown("### Phân tích các công ty dược phẩm")
    
    try:
        # Lấy tất cả dữ liệu thuốc
        all_medicines = collections['drugs_main'].get(include=["metadatas"])
        
        # Tạo DataFrame
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
        
        # Lấy top nhà sản xuất
        manufacturer_counts = medicines_df['manufacturer'].value_counts()
        top_manufacturers = manufacturer_counts.head(20).index.tolist()
        
        # Chọn nhà sản xuất
        selected_manufacturer = st.selectbox(
            "Chọn nhà sản xuất để phân tích:",
            options=top_manufacturers
        )
        
        if selected_manufacturer:
            # Lọc dữ liệu cho nhà sản xuất đã chọn
            manufacturer_data = medicines_df[medicines_df['manufacturer'] == selected_manufacturer]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tổng số thuốc", len(manufacturer_data))
            with col2:
                avg_excellent = manufacturer_data['excellent_review'].mean()
                st.metric("TB Đánh giá Xuất sắc", f"{avg_excellent:.1f}%")
            with col3:
                avg_poor = manufacturer_data['poor_review'].mean()
                st.metric("TB Đánh giá Kém", f"{avg_poor:.1f}%")
            with col4:
                market_share = len(manufacturer_data) / len(medicines_df) * 100
                st.metric("Thị phần", f"{market_share:.2f}%")
            
            # Phân bố đánh giá
            review_data = {
                'Loại Đánh giá': ['Xuất sắc', 'Trung bình', 'Kém'],
                'Phần trăm': [
                    manufacturer_data['excellent_review'].mean(),
                    manufacturer_data['average_review'].mean(),
                    manufacturer_data['poor_review'].mean()
                ]
            }
            
            fig = px.pie(values=review_data['Phần trăm'], names=review_data['Loại Đánh giá'],
                        title=f'Phân bố Đánh giá - {selected_manufacturer}',
                        color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
            
            # So sánh top nhà sản xuất
            st.markdown("### So sánh Top Nhà sản xuất")
            
            comparison_data = []
            for mfr in top_manufacturers[:10]:
                mfr_data = medicines_df[medicines_df['manufacturer'] == mfr]
                comparison_data.append({
                    'Nhà sản xuất': mfr,
                    'Số lượng Thuốc': len(mfr_data),
                    'TB Đánh giá Xuất sắc': mfr_data['excellent_review'].mean(),
                    'TB Đánh giá Kém': mfr_data['poor_review'].mean()
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig_bar = px.bar(comparison_df, x='Nhà sản xuất', y='Số lượng Thuốc',
                           title='Số lượng Thuốc theo Top Nhà sản xuất',
                           text='Số lượng Thuốc')
            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
            
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu nhà sản xuất: {e}")

def dashboard_overview_page(collections):
    """Trang 6: Tổng quan Dashboard"""
    st.markdown('<div class="main-header">📊 Tổng quan Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("### Tổng quan Hệ thống và Thống kê")
    
    try:
        # Lấy số lượng collection
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            main_count = collections['drugs_main'].count()
            st.metric("Tổng số Thuốc", f"{main_count:,}")
        
        with col2:
            se_count = collections['drugs_side_effects'].count()
            st.metric("Bản ghi Tác dụng Phụ", f"{se_count:,}")
        
        with col3:
            comp_count = collections['drugs_composition'].count()
            st.metric("Bản ghi Thành phần", f"{comp_count:,}")
        
        with col4:
            review_count = collections['drugs_reviews'].count()
            st.metric("Bản ghi Đánh giá", f"{review_count:,}")
        
        # Lấy dữ liệu mẫu để phân tích
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
            # Top nhà sản xuất
            st.markdown("### 🏭 Top 10 Nhà sản xuất")
            top_mfrs = medicines_df['manufacturer'].value_counts().head(10)
            
            fig_mfr = px.bar(
                x=top_mfrs.values,
                y=top_mfrs.index,
                orientation='h',
                title='Số lượng Thuốc theo Nhà sản xuất'
            )
            fig_mfr.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_mfr, use_container_width=True)
        
        with col2:
            # Phân bố đánh giá
            st.markdown("### ⭐ Phân bố Điểm Đánh giá")
            
            # Tạo các danh mục đánh giá
            review_categories = []
            for score in medicines_df['excellent_review']:
                if score >= 70:
                    review_categories.append('Xuất sắc (70%+)')
                elif score >= 50:
                    review_categories.append('Tốt (50-70%)')
                elif score >= 30:
                    review_categories.append('Trung bình (30-50%)')
                else:
                    review_categories.append('Kém (<30%)')
            
            review_dist = pd.Series(review_categories).value_counts()
            
            fig_review = px.pie(
                values=review_dist.values,
                names=review_dist.index,
                title='Phân bố Chất lượng Thuốc'
            )
            st.plotly_chart(fig_review, use_container_width=True)
        
        # Phân tích danh mục thuốc
        st.markdown("### 💊 Phân tích Danh mục Thuốc")
        
        # Trích xuất danh mục từ uses (đơn giản hóa)
        categories = []
        for uses in medicines_df['uses']:
            if 'đau' in uses.lower() or 'pain' in uses.lower():
                categories.append('Giảm đau')
            elif 'nhiễm khuẩn' in uses.lower() or 'infection' in uses.lower() or 'bacterial' in uses.lower():
                categories.append('Chống nhiễm khuẩn')
            elif 'tiểu đường' in uses.lower() or 'diabetes' in uses.lower():
                categories.append('Tiểu đường')
            elif 'huyết áp' in uses.lower() or 'blood pressure' in uses.lower() or 'hypertension' in uses.lower():
                categories.append('Tim mạch')
            elif 'hen suyễn' in uses.lower() or 'asthma' in uses.lower() or 'respiratory' in uses.lower():
                categories.append('Hô hấp')
            else:
                categories.append('Khác')
        
        category_dist = pd.Series(categories).value_counts()
        
        fig_cat = px.bar(
            x=category_dist.index,
            y=category_dist.values,
            title='Phân bố Thuốc theo Danh mục'
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu dashboard: {e}")

def main():
    """Ứng dụng chính"""
    # Navigation sidebar
    st.sidebar.title("💊 Nền tảng Phân tích Thuốc Thông minh")
    st.sidebar.markdown("---")

    # Khởi tạo dữ liệu
    if 'initialized' not in st.session_state:
        with st.spinner("Đang khởi tạo hệ thống..."):
            client, collections = setup_chromadb()
            model = load_model()
            
            if client and collections and model:
                st.session_state.client = client
                st.session_state.collections = collections
                st.session_state.model = model
                st.session_state.initialized = True
            else:
                st.error("Không thể khởi tạo hệ thống. Vui lòng kiểm tra thiết lập.")
                return
    
    # Menu navigation
    pages = {
        "🔍 Tìm kiếm Ngữ nghĩa": semantic_search_page,
        "🔄 Thay thế Thuốc": drug_substitution_page,
        "⚠️ Phân tích Tác dụng Phụ": side_effects_analysis_page,
        "💬 Chatbot Y tế Q&A": chatbot_page,
        "🏭 Phân tích Nhà sản xuất": manufacturer_analytics_page,
        "📊 Tổng quan Dashboard": dashboard_overview_page
    }
    
    selected_page = st.sidebar.selectbox("Chọn chức năng:", list(pages.keys()))
    
    # Trạng thái hệ thống
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Trạng thái Hệ thống")
    st.sidebar.success("✅ ChromaDB Đã kết nối")
    st.sidebar.success("✅ Mô hình AI")
    st.sidebar.info(f"📊 {st.session_state.collections['drugs_main'].count():,} thuốc có sẵn")
    
    # Hiển thị trang đã chọn
    if selected_page in ["💬 Chatbot Y tế Q&A"]:
        pages[selected_page](st.session_state.collections, st.session_state.model)
    elif selected_page in ["🏭 Phân tích Nhà sản xuất", "📊 Tổng quan Dashboard"]:
        pages[selected_page](st.session_state.collections)
    elif selected_page in ["⚠️ Phân tích Tác dụng Phụ"]:
        pages[selected_page](st.session_state.collections)
    else:
        pages[selected_page](st.session_state.collections, st.session_state.model)

if __name__ == "__main__":
    main()