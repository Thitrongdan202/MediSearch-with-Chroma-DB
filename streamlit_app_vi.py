#!/usr/bin/env python3
"""
Nền tảng Phân tích Thuốc Thông minh - Streamlit UI
=================================================

Nền tảng tìm kiếm và phân tích thuốc toàn diện sử dụng ChromaDB và AI.
"""

import chromadb
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import inspect

def run_page(fn, collections, model):
    try:
        params = [
            p for p in inspect.signature(fn).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(params) >= 2:
            return fn(collections, model)
        elif len(params) == 1:
            return fn(collections)
        else:
            return fn()
    except Exception:
        return fn(collections, model)

# Symptom
def _lc(x): return (x or "").lower()
def _has_any(text, kws): return any(k in text for k in kws)

SYMPTOM_MAP = {
    # ——— Tiêu hoá ———
    "constipation": {
        "aliases": ["táo bón"],
        "pos": [
            "constipation","laxative","stool softener","bulk-forming","osmotic","stimulant laxative",
            "bisacodyl","senna","sennoside","lactulose","polyethylene glycol","macrogol",
            "psyllium","ispaghula","docusate","glycerin","magnesium hydroxide"
        ],
        "neg": ["hypertension","high blood pressure","antihypertensive","telmisartan","amlodipine","losartan",
                "valsartan","phenytoin","carbamazepine","valproate"]
    },
    "diarrhea": {
        "aliases": ["tiêu chảy","đi ngoài lỏng","phân lỏng"],
        "pos": [
            "diarrhea","antidiarrheal","loperamide","racecadotril","bismuth subsalicylate",
            "oral rehydration salts","ors","rehydration","zinc",
            # nhiễm khuẩn / virus
            "infectious diarrhea","bacterial diarrhea","viral diarrhea"
        ],
        "neg": ["constipation","laxative","antihypertensive","antiepileptic"]
    },

    # ——— Sốt / đau ———
    "fever": {
        "aliases": ["sốt","hạ sốt","giảm sốt","đau nhức","giảm đau","panadol","paradon","paracetamol"],
        "pos": [
            "fever","antipyretic","analgesic","pain reliever","paracetamol","acetaminophen",
            "ibuprofen","naproxen","aspirin","caffeine","cold & flu"
        ],
        "neg": ["antiepileptic","antihypertensive","phenytoin","carbamazepine","telmisartan"]
    },
    "headache": {
        "aliases": ["đau đầu","nhức đầu","migraine","đau nửa đầu"],
        "pos": [
            "headache","migraine","analgesic","pain reliever","paracetamol","acetaminophen",
            "ibuprofen","naproxen","diclofenac","aspirin","triptan","sumatriptan"
        ],
        "neg": ["antiepileptic","antihypertensive","phenytoin","carbamazepine","valproate","telmisartan"]
    },

    # ——— Hô hấp / dị ứng ———
    "runny_nose": {
        "aliases": ["sổ mũi","xổ mũi","chảy mũi","ngạt mũi","viêm mũi","cảm lạnh","hắt hơi","dị ứng"],
        "pos": [
            "runny nose","rhinitis","allergic rhinitis","sneezing","allergy",
            "antihistamine","decongestant","cetirizine","loratadine","fexofenadine",
            "chlorpheniramine","diphenhydramine","phenylephrine","pseudoephedrine",
            "oxymetazoline","xylometazoline","azelastine","guaifenesin","bromhexine","ambroxol"
        ],
        "neg": ["antihypertensive","antiepileptic"]
    },
    "cough": {
        "aliases": ["ho","ho khan","ho có đờm"],
        "pos": [
            "cough","dry cough","productive cough","expectorant","mucolytic",
            "dextromethorphan","guaifenesin","ambroxol","bromhexine","acetylcysteine","codeine","levodropropizine"
        ],
        "neg": ["antihypertensive","antiepileptic"]
    },

    # ——— Dạ dày ———
    "stomach_pain": {
        "aliases": ["đau dạ dày","đau thượng vị","khó tiêu","ợ nóng","trào ngược"],
        "pos": [
            "epigastric pain","stomach ache","gastric pain","indigestion","dyspepsia","heartburn",
            "gastritis","peptic ulcer","duodenal ulcer","gastric ulcer","acid reflux","GERD",
            "antacid","proton pump inhibitor","ppi","h2 blocker",
            "omeprazole","pantoprazole","esomeprazole","rabeprazole","lansoprazole"
        ],
        "neg": ["antihypertensive","antiepileptic","acne","isotretinoin","benzoyl peroxide"]
    },

    # ——— Cơ – xương – khớp ———
    "muscle_spasm": {
        "aliases": ["đau cơ","co thắt cơ","giãn cơ","chuột rút"],
        "pos": [
            "muscular pain","pain due to muscle spasm","muscle spasm","muscle relaxation",
            "muscle relaxant","tizanidine","baclofen","eperisone","tolperisone","thiocolchicoside"
        ],
        "neg": ["antihypertensive","antiepileptic"]
    },

    # ——— Nhiễm trùng ———
    "bacterial_infection": {
        "aliases": ["nhiễm khuẩn"],
        "pos": [
            "bacterial infection","antibiotic","amoxicillin","amoxicillin clavulanate","azithromycin","clarithromycin",
            "ciprofloxacin","levofloxacin","moxifloxacin","cephalexin","cefuroxime","ceftriaxone","metronidazole"
        ],
        "neg": ["acne","isotretinoin","benzoyl peroxide"]
    },
    "fungal_skin": {
        "aliases": ["nhiễm nấm da","hắc lào","lang ben"],
        "pos": ["fungal skin infection","tinea","dermatophyte","clotrimazole","ketoconazole","terbinafine",
                "miconazole","fluconazole","griseofulvin"],
        "neg": []
    },
    "parasitic_infection": {
        "aliases": ["nhiễm ký sinh trùng","giun","sán"],
        "pos": ["parasitic infection","anthelmintic","albendazole","mebendazole","ivermectin","praziquantel"],
        "neg": []
    },

    # ——— Da liễu (để loại trừ khi tìm “thuốc tim”) ———
    "acne": {
        "aliases": ["mụn","mụn trứng cá","acne"],
        "pos": ["acne","isotretinoin","adapalene","benzoyl peroxide","clindamycin gel","tretinoin"],
        "neg": []
    },

    # ——— TIM MẠCH ———
    "cardio": {
        "aliases": [
            "thuốc tim","thuoc tim","tim mạch","thuốc tim mạch","tăng huyết áp","cao huyết áp",
            "huyết áp cao","suy tim","đau thắt ngực","nhồi máu cơ tim","đột quỵ","bệnh tim"
        ],
        "pos": [
            "hypertension","high blood pressure","antihypertensive",
            "ace inhibitor","arb","beta blocker","calcium channel blocker","diuretic",
            "telmisartan","losartan","valsartan","olmesartan","irbesartan",
            "amlodipine","nifedipine","diltiazem","verapamil",
            "atenolol","metoprolol","bisoprolol","carvedilol","nebivolol",
            "hydrochlorothiazide","furosemide","spironolactone","indapamide",
            "enalapril","lisinopril","perindopril","ramipril",
            "ivabradine","nitroglycerin","isosorbide mononitrate","ranolazine",
            "statin","atorvastatin","rosuvastatin","simvastatin",
            "antiplatelet","aspirin","clopidogrel","ticagrelor",
            "anticoagulant","warfarin","apixaban","rivaroxaban","dabigatran",
            "heart failure","angina","myocardial infarction","stroke","arrhythmia","amiodarone","sacubitril"
        ],
        "neg": [
            "acne","isotretinoin","benzoyl peroxide","fungal skin infection","tinea","clotrimazole",
            "paracetamol","cold & flu","sneezing","allergic rhinitis"
        ]
    },
}

def detect_symptoms(q: str):
    ql = _lc(q)
    hits = []
    for key, cfg in SYMPTOM_MAP.items():
        if any(al in ql for al in cfg["aliases"] + [key]):
            hits.append(key)
    return hits

def detect_symptom(q: str):
    syms = detect_symptoms(q)
    return syms[0] if syms else None

def symptom_search_rows(symptom: str, collections, model, top_each=12):
    cfg = SYMPTOM_MAP[symptom]
    queries = [" ".join(cfg["pos"]), " ".join(cfg["aliases"]), " ".join(cfg["pos"][:6])]
    seen, rows = set(), []
    for q in queries:
        hits = search_medicines(collections["drugs_main"], q, model, n_results=top_each)
        metas = (hits or {}).get("metadatas", [[]]); dists = (hits or {}).get("distances", [[]])
        if not metas or not metas[0]: continue
        for i, m in enumerate(metas[0]):
            if not isinstance(m, dict): continue
            name = m.get("medicine_name") or ""
            uses = m.get("uses") or ""
            comp = m.get("composition") or m.get("ingredients") or ""
            blob = _lc(f"{name} {uses} {comp}")
            if _has_any(blob, cfg["neg"]):  # loại sai nhóm
                continue
            key = _lc(name) or f"{i}-{hash(blob)}"
            if key in seen: continue
            seen.add(key)
            dist = dists[0][i] if dists and dists[0] and i < len(dists[0]) else None
            rows.append((m, dist))
    return rows

def rerank_symptom(symptom: str, rows):
    cfg = SYMPTOM_MAP[symptom]
    pos = [_lc(x) for x in cfg["pos"]]
    out = []
    for m, dist in rows:
        uses = _lc(m.get("uses") or "")
        comp = _lc(m.get("composition") or m.get("ingredients") or "")
        base = 0.5 if not isinstance(dist,(int,float)) else max(0.0, min(1.0, 1-float(dist)))
        boost = 0.0
        if _has_any(uses, pos): boost += 0.35
        if _has_any(comp, pos): boost += 0.35
        out.append((base+boost, m, dist))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

import re

# Chỉ giữ 1 bộ quy tắc (rút gọn ở đây; bạn có thể giữ bộ dài hiện tại)
USES_EN_VI = [
    # 1) Dị ứng / hắt hơi / sổ mũi
    (r"\bSneezing and (runny nose|nasal discharge|sổ mũi)\s+(due to|caused by)\s+allerg(y|ies)\b",
     "hắt hơi và sổ mũi do dị ứng"),
    (r"\bSneezing\b", "hắt hơi"),
    (r"\bdue to allergies\b", "do dị ứng"),
    (r"\ballerg(y|ies)\b", "dị ứng"),

    # 2) Nhiễm khuẩn / nấm / ký sinh trùng
    (r"\bFungal skin infection(s)?\b", "nhiễm nấm da"),
    (r"\bBacterial (skin )?infection(s)?\b", "nhiễm khuẩn"),
    (r"\bParasitic infection(s)?\b", "nhiễm ký sinh trùng"),

    # 3) Tiêu chảy nhiễm khuẩn (đặt trước rule Diarrhea chung)
    (r"\bAcute infectious diarrh(o)?ea\b", "tiêu chảy nhiễm khuẩn cấp"),
    (r"\bInfectious diarrh(o)?ea\b", "tiêu chảy nhiễm khuẩn"),
    (r"\bBacterial diarrh(o)?ea\b", "tiêu chảy do vi khuẩn"),
    (r"\bViral diarrh(o)?ea\b", "tiêu chảy do vi rút"),
    (r"\bDiarrh(o)?ea due to (infection|bacteria|virus)\b", "tiêu chảy do nhiễm trùng"),

    # 4) Cơ – xương – khớp / giãn cơ
    (r"\bPain due to muscle spasm(s)?\b", "đau do co thắt cơ"),
    (r"\bMuscular pain\b", "đau cơ"),
    (r"\bMuscle spasm(s)?\b", "co thắt cơ"),
    (r"\bMuscle relaxation\b", "giãn cơ"),
    (r"\bMuscle relaxant(s)?\b", "thuốc giãn cơ"),

    # 5) Da liễu
    (r"\bAcne\b", "mụn trứng cá"),

    # 6) Tiền phẫu ruột (đủ dị bản)
    (r"\bIntestine\s+preparation\b(.*?before (any )?surgery)?", "chuẩn bị ruột trước phẫu thuật"),
    (r"\bIntestin(al|e)\s+preparation\b(.*?before (any )?surgery)?", "chuẩn bị ruột trước phẫu thuật"),
    (r"\bBowel\s+preparation\b(.*?before (any )?surgery)?", "chuẩn bị ruột trước phẫu thuật"),
    (r"\bPreoperative (bowel|intestinal) preparation\b", "chuẩn bị ruột trước phẫu thuật"),

    # 7) Thiếu hụt dinh dưỡng
    (r"\bNutritional deficienc(y|ies)\b", "thiếu hụt dinh dưỡng"),
    (r"\bNutrient deficienc(y|ies)\b", "thiếu hụt dinh dưỡng"),

    # Khung câu
    (r"\bTreatment of\b", "Điều trị"),
    (r"\bRelief of\b", "Giảm"),
    (r"\bPrevention of\b", "Phòng ngừa"),
    (r"\bManagement of\b", "Kiểm soát"),
    (r"\bProphylaxis of\b", "Dự phòng"),
    (r"\bAs an adjunct to\b", "Hỗ trợ điều trị"),
    (r"\bAdjunct\b", "Hỗ trợ điều trị"),
    (r"\bIndicated for\b", "Chỉ định cho"),
    (r"\bUsed for\b", "Dùng cho"),

    # --- Viêm loét đại tràng / IBS ---
    (r"\bUlcerative colitis\b", "viêm loét đại tràng"),
    (r"\bIrritable bowel syndrome\b", "hội chứng ruột kích thích"),
    (r"\bIBS\b", "hội chứng ruột kích thích"),

    # --- Tiêu chảy do rotavirus (đặt trước rule Diarrhea chung) ---
    (r"\bRotaviral diarrh(o)?ea\b", "tiêu chảy do rotavirus"),
    (r"\bDiarrh(o)?ea due to rotavirus\b", "tiêu chảy do rotavirus"),
    (r"\bdue to rotavirus\b", "do rotavirus"),

    # --- tiêu chảy nhiễm khuẩn & dị bản ---
    (r"\bAcute infectious diarrh(o)?ea\b", "tiêu chảy nhiễm khuẩn cấp"),
    (r"\bInfectious diarrh(o)?ea\b", "tiêu chảy nhiễm khuẩn"),
    (r"\bBacterial diarrh(o)?ea\b", "tiêu chảy do vi khuẩn"),
    (r"\bViral diarrh(o)?ea\b", "tiêu chảy do vi rút"),
    (r"\bDiarrh(o)?ea due to (infection|bacteria|virus)\b", "tiêu chảy do nhiễm trùng"),

    # --- chuẩn bị ruột trước mổ (đủ mọi dị bản hay gặp) ---
    (r"\bIntestine\s+preparation\b(.*?before (any )?surgery)?", "chuẩn bị ruột trước phẫu thuật"),
    (r"\bIntestin(al|e)\s+preparation\b(.*?before (any )?surgery)?", "chuẩn bị ruột trước phẫu thuật"),
    (r"\bBowel\s+preparation\b(.*?before (any )?surgery)?", "chuẩn bị ruột trước phẫu thuật"),
    (r"\bPreoperative (bowel|intestinal) preparation\b", "chuẩn bị ruột trước phẫu thuật"),

    # --- thiếu hụt dinh dưỡng ---
    (r"\bNutritional deficienc(y|ies)\b", "thiếu hụt dinh dưỡng"),
    (r"\bNutrient deficienc(y|ies)\b", "thiếu hụt dinh dưỡng"),

    # --- Nhiễm trùng hô hấp / thần kinh ---
    (r"\bPneumonia\b", "viêm phổi"),
    (r"\bCommunity[- ]acquired pneumonia\b", "viêm phổi mắc phải cộng đồng"),
    (r"\bHospital[- ]acquired pneumonia\b", "viêm phổi bệnh viện"),

    (r"\bMeningitis\b", "viêm màng não"),
    (r"\bBacterial meningitis\b", "viêm màng não do vi khuẩn"),
    (r"\bViral meningitis\b", "viêm màng não do vi rút"),

    # --- Nhiễm trùng máu ---
    (r"\bBloodstream infection(s)?\b", "nhiễm trùng máu"),
    (r"\bBlood infection(s)?\b", "nhiễm trùng máu"),
    (r"\bBacter(ae)?mia\b", "nhiễm khuẩn huyết"),
    (r"\bSeptic(ae)?mia\b", "nhiễm trùng huyết"),
    (r"\bSepsis\b", "nhiễm trùng huyết"),

    # --- Nhiễm trùng tai ---
    (r"\bEar infection(s)?\b", "nhiễm trùng tai"),
    (r"\bOtitis media\b", "viêm tai giữa"),
    (r"\bAcute otitis media\b|\bAOM\b", "viêm tai giữa cấp"),
    (r"\bOtitis externa\b", "viêm ống tai ngoài"),

    # --- Tim mạch - thần kinh ---
    (r"\bStroke\b|\bCerebrovascular accident\b", "đột quỵ"),
    (r"\bHeart failure\b", "suy tim"),
    (r"\bCongestive heart failure\b", "suy tim sung huyết"),

    # --- Ký sinh trùng da ---
    (r"\bScabies\b", "ghẻ"),
    (r"\bSarcoptes scabiei\b", "ghẻ (Sarcoptes scabiei)"),
    # Triệu chứng phổ biến
    (r"\bPain relief\b", "giảm đau"),
    (r"\bFever relief\b", "giảm sốt"),
    (r"\bDiarrh(o)?ea\b", "tiêu chảy"),
    (r"\bConstipation\b", "táo bón"),
    (r"\bFever\b", "sốt"),
    (r"\bCough\b", "ho"),
    (r"\bRunny nose\b", "sổ mũi"),
    (r"\bNasal congestion\b", "nghẹt mũi"),
    (r"\bHeadache\b", "đau đầu"),
    (r"\bMigraine\b", "đau nửa đầu"),
    # --- Tim mạch (ưu tiên cụ thể > tổng quát) ---
    (r"\bHypertension\b(\s*\(high blood pressure\))?", "tăng huyết áp"),
    (r"\bHigh blood pressure\b", "cao huyết áp"),
    (r"\bAngina\b(\s*\(heart[- ]related chest pain\))?", "đau thắt ngực"),
    (r"\bArrhythmia\b", "rối loạn nhịp tim"),
    (r"\bHeart attack\b", "nhồi máu cơ tim"),
    # (đã có) (r"\bMigraine\b","đau nửa đầu"),

    # --- Dạ dày - thực quản ---
    (r"\bHeartburn\b", "ợ nóng"),
    (r"\bGastro[- ]?o?esophageal reflux disease\b(\s*\(Acid reflux\))?", "trào ngược dạ dày thực quản"),
    (r"\bGERD\b", "trào ngược dạ dày thực quản"),
    (r"\bGORD\b", "trào ngược dạ dày thực quản"),
    (r"\bAcid reflux\b", "trào ngược dạ dày thực quản"),
    (r"\bPeptic ulcer disease\b", "loét dạ dày tá tràng"),
    (r"\bZollinger[-–— ]Ellison syndrome\b", "hội chứng Zollinger–Ellison")
]

HEAD_VERBS = (
    "Điều trị","Giảm","Phòng ngừa","Kiểm soát","Dự phòng","Hỗ trợ điều trị",
    "Hạ","Làm giảm","Làm dịu","Liệu pháp","Giãn"
)

COND_TERMS = ["tiêu chảy","tiêu chảy nhiễm khuẩn","táo bón","sốt","ho","sổ mũi","hắt hơi",
              "nghẹt mũi","đau đầu","đau nửa đầu","đau họng","đau răng","đau bụng",
              "đau dạ dày","đau thượng vị","khó tiêu","viêm dạ dày","viêm",
              "nhiễm khuẩn","nhiễm nấm da","nhiễm ký sinh trùng",
              "đau cơ","co thắt cơ","giãn cơ",
              "thiếu hụt dinh dưỡng","trào ngược dạ dày thực quản","loét dạ dày tá tràng",
              "tăng huyết áp","đái tháo đường type 2","mỡ máu cao",
              "đau thắt ngực","rối loạn nhịp tim","nhồi máu cơ tim",
              "ợ nóng","trào ngược dạ dày thực quản","loét dạ dày tá tràng","hội chứng zollinger–ellison"
              "viêm loét đại tràng", "hội chứng ruột kích thích", "tiêu chảy do rotavirus",
              "viêm phổi","viêm màng não","nhiễm trùng máu",
              "nhiễm khuẩn huyết","nhiễm trùng huyết","nhiễm trùng tai","viêm tai giữa","đột quỵ","suy tim","ghẻ"
]

def normalize_uses_en(txt: str) -> str:
    if not isinstance(txt, str): return ""
    t = txt
    # tách các mệnh đề bị dính
    t = re.sub(r'(?<=stroke)Treatment', '. Treatment', t, flags=re.I)
    t = re.sub(r'(?i)([A-Za-z%)])\s*(?=(Treatment|Prevention|Relief|Management|Control|Prophylaxis|Indicated|Used) of\b)', r'\1. ', t)
    t = re.sub(r'(?i)([A-Za-z%)])\s*(?=(Intestin(?:al|e)?\s+preparation|Bowel\s+preparation|Pain relief|Fever relief))', r'\1. ', t)
    t = re.sub(r'\s*&\s*', ', ', t)
    return re.sub(r'\s+',' ', t).strip()

def _add_default_verb_if_missing(t: str) -> str:
    s = t.strip()
    if not s: return s
    if any(re.match(rf"(?i)^{re.escape(v)}\b", s) for v in HEAD_VERBS): return s
    if any(re.match(rf"(?i)^{re.escape(cond)}\b", s) for cond in COND_TERMS): return f"Điều trị {s}"
    return s

def vi_translate_uses(en_text: str) -> str:
    t = normalize_uses_en(en_text or "")
    for pat, repl in USES_EN_VI:
        t = re.sub(pat, repl, t, flags=re.I)

    # 🔒 Guard rails – bắt mọi dị bản còn sót
    t = re.sub(r'(?i)\bintestin(?:al|e)?\s+preparation(?:\s+before(?:\s+any)?\s+surgery)?', 'chuẩn bị ruột trước phẫu thuật', t)
    t = re.sub(r'(?i)\bbowel\s+preparation(?:\s+before(?:\s+any)?\s+surgery)?', 'chuẩn bị ruột trước phẫu thuật', t)
    t = re.sub(r'(?i)\bpain\s+relief\b', 'giảm đau', t)
    t = re.sub(r'(?i)\bfever\s+relief\b', 'giảm sốt', t)
    t = re.sub(r'(?i)\btreatment of\b', 'Điều trị', t)
    t = re.sub(r'(?i)\binfectious\s+tiêu chảy\b', 'tiêu chảy nhiễm khuẩn', t)
    t = re.sub(r'(?i)\btiêu chảy\s+nhiễm trùng\b', 'tiêu chảy nhiễm khuẩn', t)
    t = re.sub(r'(?i)\bnutritional\s+deficiencies\b', 'thiếu hụt dinh dưỡng', t)
    t = re.sub(r'(?i)\binfectious\s+tiêu chảy\b', 'tiêu chảy nhiễm khuẩn', t)
    t = re.sub(r'(?i)\btiêu chảy\s+nhiễm trùng\b', 'tiêu chảy nhiễm khuẩn', t)
    t = re.sub(r'(?i)\bnutritional\s+deficiencies\b', 'thiếu hụt dinh dưỡng', t)
    t = re.sub(r'(?i)\bhắt hơi and sổ mũi\b', 'hắt hơi và sổ mũi', t)
    t = re.sub(r'(?i)\bdiarrh(o)?ea\b', 'tiêu chảy', t)
    t = re.sub(r'(?i)\bconstipation\b', 'táo bón', t)
    t = re.sub(r'(?i)\bfever\b', 'sốt', t)
    t = re.sub(r'\s*\((acid reflux|heart[- ]related chest pain|high blood pressure)\)\s*', '', t, flags=re.I)
    t = re.sub(r'\s+', ' ', t).strip()
    t = _add_default_verb_if_missing(t)
    t = re.sub(r'\s+',' ', t).strip()
    t = re.sub(r'(?i)\bulcerative colitis\b', 'viêm loét đại tràng', t)
    t = re.sub(r'(?i)\birritable bowel syndrome\b', 'hội chứng ruột kích thích', t)
    t = re.sub(r'(?i)\bibs\b', 'hội chứng ruột kích thích', t)
    t = re.sub(r'(?i)\brotaviral diarrh(o)?ea\b', 'tiêu chảy do rotavirus', t)
    t = re.sub(r'(?i)\bdiarrh(o)?ea due to rotavirus\b', 'tiêu chảy do rotavirus', t)
    t = re.sub(r'(?i)\bdue to rotavirus\b', 'do rotavirus', t)
    return (t[:1].upper()+t[1:]) if t else t



# Load environment variables from .env file
load_dotenv()
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
    print(f"OpenAI API Key: {'(có)' if key else '(không)'}")
    if not key:
        return None
    return OpenAI(api_key=key)

def translate_query_openai(user_prompt) -> str:
    # using 'gpt-3.5-turbo' for translation
    client = get_openai_client()
    if client is None:
        return user_prompt  # không có key -> fallback DB
    system_prompt = "Bạn là một trợ lý dịch thuật y tế, hãy dịch câu hỏi sau đây sang tiếng Anh một cách chính xác với các từ ngữ chuyên ngành."
    rsp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return rsp.choices[0].message.content if rsp.choices else user_prompt

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
    st.markdown('<div class="main-header"> Tìm kiếm</div>', unsafe_allow_html=True)
    
    st.markdown("### Tìm thuốc bằng mô tả và thành phần")
    
    # Ô tìm kiếm
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Mô tả triệu chứng hoặc loại thuốc cần tìm:",
            placeholder="Ví dụ: giảm đau, đau đầu, kháng sinh, thuốc tim",
            key="search_query"
        )
    with col2:
        search_button = st.button(" Tìm kiếm", type="primary")
    
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
    # translate query
    # GIỮ nguyên biến 'query' từ ô nhập để nhận diện triệu chứng
    orig_vi = query
    symptom = detect_symptom(orig_vi or "")

    if search_button and orig_vi:
        with st.spinner("Đang tìm kiếm thuốc..."):
            # Chọn collection
            if search_collection == "Cơ sở dữ liệu chính":
                collection = collections['drugs_main']
            elif search_collection == "Theo thành phần":
                collection = collections['drugs_composition']
            else:
                collection = collections['drugs_side_effects']

            if symptom and collection == collections['drugs_main']:
                rows = symptom_search_rows(symptom, collections, model, top_each=num_results * 2)
                ranked = rerank_symptom(symptom, rows)[:num_results]
                st.success(f"Tìm thấy {len(ranked)} thuốc phù hợp với triệu chứng của bạn")

                for score, metadata, dist in ranked:
                    similarity = f"{(float(score) * 100):.1f}%"
                    name = metadata.get('medicine_name', '(Không rõ tên)')
                    comp = (metadata.get('composition') or metadata.get('ingredients', '')) or ''
                    uses_en = normalize_uses_en(metadata.get('uses', ''))
                    uses_vi = vi_translate_uses(uses_en) if uses_en else ''
                    manu = metadata.get('manufacturer', 'Không rõ')

                    st.markdown(f"""
                    <div class="drug-card">
                      <h4> {name}</h4>
                      <p><strong>Độ tương đồng:</strong> <span class="similarity-score">{similarity}</span></p>
                      <p><strong> Thành phần:</strong> {comp[:100]}...</p>
                      <p><strong> Công dụng (VI):</strong> {uses_vi or '(chưa có dữ liệu)'}{"<br><span style='opacity:.7'>EN: " + uses_en + "</span>" if uses_en and uses_vi and uses_vi.lower() != uses_en.lower() else ""}</p>
                      <p><strong> Hãng sản xuất:</strong> {manu}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Đánh giá xuất sắc", f"{metadata.get('excellent_review', 0)}%")
                    c2.metric("Đánh giá trung bình", f"{metadata.get('average_review', 0)}%")
                    c3.metric("Đánh giá kém", f"{metadata.get('poor_review', 0)}%")
                    st.markdown("---")

            else:
                # Chỉ dịch EN khi tìm ở CSDL chính và không phải triệu chứng
                q_for_search = translate_query_openai(orig_vi) if (
                            collection == collections['drugs_main'] and not symptom) else orig_vi

                results = search_medicines(collection, q_for_search, model, num_results)
                if results and results['metadatas'][0]:
                    st.success(f"Tìm thấy {len(results['metadatas'][0])} thuốc phù hợp với tìm kiếm của bạn")
                    for i, metadata in enumerate(results['metadatas'][0]):
                        distance = results['distances'][0][i]
                        similarity = format_similarity(distance)
                        name = metadata.get('medicine_name', '(Không rõ tên)')
                        comp = (metadata.get('composition') or metadata.get('ingredients', '')) or ''
                        uses_en = normalize_uses_en(metadata.get('uses', ''))
                        uses_vi = vi_translate_uses(uses_en) if uses_en else ''
                        manu = metadata.get('manufacturer', 'Không rõ')

                        st.markdown(f"""
                        <div class="drug-card">
                          <h4> {name}</h4>
                          <p><strong>Độ tương đồng:</strong> <span class="similarity-score">{similarity}</span></p>
                          <p><strong> Thành phần:</strong> {comp[:100]}...</p>
                          <p><strong> Công dụng (VI):</strong> {uses_vi or '(chưa có dữ liệu)'}{"<br><span style='opacity:.7'>EN: " + uses_en + "</span>" if uses_en and uses_vi and uses_vi.lower() != uses_en.lower() else ""}</p>
                          <p><strong> Hãng sản xuất:</strong> {manu}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Đánh giá xuất sắc", f"{metadata.get('excellent_review', 0)}%")
                        c2.metric("Đánh giá trung bình", f"{metadata.get('average_review', 0)}%")
                        c3.metric("Đánh giá kém", f"{metadata.get('poor_review', 0)}%")
                        st.markdown("---")
                else:
                    st.warning("Không tìm thấy thuốc nào phù hợp với tiêu chí tìm kiếm của bạn.")
    return


def drug_substitution_page(collections, model):
    st.markdown('<div class="main-header"> Thuốc Thay thế</div>', unsafe_allow_html=True)
    
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
                    
                    alt = {
                        'name': metadata.get('medicine_name', '(Không rõ tên)'),
                        'similarity': (1 - distance) * 100,
                        'composition': metadata.get('composition') or metadata.get('ingredients') \
                                       or (results['documents'][0][i] if results.get('documents') else ""),  # FIX
                        'uses': metadata.get('uses', ''),  # FIX
                        'manufacturer': metadata.get('manufacturer', ''),  # FIX
                        'excellent_review': metadata.get('excellent_review', 0),  # FIX
                        'poor_review': metadata.get('poor_review', 0),  # FIX
                    }
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
                    # Áp dụng bộ lọc
                    if avoid_side_effects and alt.get('poor_review', 0) > 30:
                        continue
                    if prefer_good_reviews and alt.get('excellent_review', 0) < 50:
                        continue
                    alternatives.append(alt)

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
                            title = (alt.get('name') or '(Không rõ tên)').strip()
                            comp = (alt.get('composition') or '').strip()
                            uses = (alt.get('uses') or '').strip()
                            manu = (alt.get('manufacturer') or '').strip()

                            st.markdown(
                                f"**{title}**  \n"
                                f"*Độ tương đồng: {alt.get('similarity'):.1f}%*  \n"
                                f"{comp[:80]}...  \n"
                                f"{uses[:100]}...  \n"
                                f"{manu}"
                            )

                        with col2:
                            st.metric("Đánh giá tốt", f"{alt.get('excellent_review', 0)}%")
                            st.metric("Trung bình", f"{alt.get('average_review', 0)}%")
                            st.metric("Kém", f"{alt.get('poor_review', 0)}%")

                        st.markdown("---")
                else:
                    st.warning("Không tìm thấy thuốc thay thế nào phù hợp với tiêu chí lọc của bạn.")
            else:
                st.warning("Không tìm thấy thuốc thay thế cho loại thuốc này.")

def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")

def generate_vi_answer_openai(user_q: str, results: dict, model_name: str = "gpt-5-mini"):
    client = get_openai_client()
    if client is None:
        return None  # không có key -> fallback DB
    system_prompt = "Bạn là một trợ lý ảo y tế, hãy trả lời câu hỏi sau đây dựa trên ngữ cảnh nội bộ."
    user_prompt = f"""Dựa trên các thông tin thuốc dưới đây, hãy trả lời câu hỏi của người dùng bằng tiếng Việt một cách ngắn gọn và dễ hiểu. Nếu không tìm thấy thông tin phù hợp, hãy nói rằng bạn không chắc chắn và khuyên người dùng hỏi ý kiến bác sĩ hoặc dược sĩ.
    """
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

from openai import OpenAI

def chatbot_page(collections, model):
    import streamlit as st

    st.markdown('<div class="main-header">🤖 Chatbot Y tế Q&A</div>', unsafe_allow_html=True)
    st.markdown("### Hỏi đáp về thuốc, triệu chứng, sức khỏe (ngôn ngữ tự nhiên)")

    # --- lưu & hiển thị lịch sử hội thoại
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)

    # --- nhập câu hỏi
    user_q = st.chat_input("Nhập câu hỏi về thuốc, triệu chứng hoặc tình trạng sức khỏe...")
    if not user_q:
        return
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # --- kiểm tra dữ liệu
    if model is None or collections is None or "drugs_main" not in collections:
        reply = " Hệ thống chưa sẵn sàng (thiếu model hoặc collection)."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    # --- 1) Truy hồi từ ChromaDB
    query_q = "Câu hỏi hiện tại: " + user_q
    for m in st.session_state.messages[:-1]:  # thêm lịch sử hội thoại (trừ câu hỏi hiện tại)
        if m["role"] == "user":
            query_q += "\nNgười dùng đã hỏi: " + m["content"]
        elif m["role"] == "assistant":
            query_q += "\nTrợ lý đã trả lời: " + m["content"]
    query_translated = translate_query_openai(query_q)
    print(f"Query translated: {query_translated}")
    results = search_medicines(collections["drugs_main"], query_translated, model, n_results=6)

    # --- 2) Chuẩn bị ngữ cảnh từ DB
    context_texts = []
    if results and "documents" in results:
        for doc in results["documents"][0][:4]:  # lấy top-4 doc
            if doc:
                context_texts.append(doc.strip())
    context = "\n\n".join(context_texts) if context_texts else "Không tìm thấy thông tin nội bộ."
    print(f"Context prepared: {context}")
    # --- 3) Gọi GPT-5 trực tiếp với thanh loading
    client = OpenAI()

    system_prompt = (
        "Bạn là Chatbot Y tế. "
        "Trả lời ngắn gọn, rõ ràng, bằng tiếng Việt. "
        "Ưu tiên sử dụng dữ liệu nội bộ (context) cung cấp. "
        "Nếu liên quan đến thuốc hoặc điều trị, BẮT BUỘC thêm cảnh báo: "
        "Thông tin chỉ tham khảo, vui lòng hỏi ý kiến bác sĩ/dược sĩ trước khi sử dụng."
    )

    answer = None
    try:
        with st.spinner(" AI đang soạn câu trả lời..."):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Ngữ cảnh nội bộ:\n{context}"},
                {"role": "user", "content": user_q},
            ]
            # thêm lịch sử hội thoại trước đó (trừ câu hỏi hiện tại)
            for m in st.session_state.messages[:-1]:
                if m["role"] in ["user", "assistant"]:
                    messages.append({"role": m["role"], "content": m["content"]})

            resp = client.chat.completions.create(
                model="gpt-5-mini",   # hoặc "gpt-5"
                messages=messages,
            )
            answer = resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Không gọi được GPT-5: {e}")

    # --- 4) Trả lời hoặc fallback
    if answer:
        with st.chat_message("assistant"):
            st.markdown(answer, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        reply = " Hiện tại không thể tạo câu trả lời. Vui lòng thử lại sau."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


def manufacturer_analytics_page(collections):
    st.markdown('<div class="main-header"> Phân tích Nhà sản xuất</div>', unsafe_allow_html=True)
    
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

            fig.update_traces(
                textinfo="percent",
                hovertemplate="<b>%{label}</b>: %{value:.1f}%<extra></extra>"
            )
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
            fig_bar.update_traces(
                texttemplate='%{text}', textposition='outside',
                hovertemplate="<b>%{x}</b><br>Số lượng: %{y}<extra></extra>"
            )
            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
            
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu nhà sản xuất: {e}")

def dashboard_overview_page(collections):
    """Trang 6: Tổng quan Dashboard"""
    st.markdown('<div class="main-header"> Tổng quan Dashboard</div>', unsafe_allow_html=True)
    
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
            st.markdown("###  Top 10 Nhà sản xuất")
            top_mfrs = medicines_df['manufacturer'].value_counts().head(10)

            fig_mfr = px.bar(
                x=top_mfrs.values,
                y=top_mfrs.index,
                orientation='h',
                title='Số lượng Thuốc theo Nhà sản xuất',
                labels={'x': 'Số lượng Thuốc', 'y': 'Nhà sản xuất'}  # 👈 đặt nhãn trục
            )
            fig_mfr.update_yaxes(categoryorder="total ascending")
            fig_mfr.update_traces(hovertemplate="<b>%{y}</b><br>Số lượng: %{x}<extra></extra>")
            fig_mfr.update_layout(
                xaxis_title="Số lượng Thuốc",
                yaxis_title="Nhà sản xuất",
                margin=dict(l=220, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_mfr, use_container_width=True)

        with col2:
            # Phân bố đánh giá
            st.markdown("###  Phân bố Điểm Đánh giá")
            
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

            fig_review.update_traces(
                textinfo="percent",
                hovertemplate="<b>%{label}</b>: %{percent}<extra></extra>"
            )

            st.plotly_chart(fig_review, use_container_width=True)
        
        # Phân tích danh mục thuốc
        st.markdown("###  Phân tích Danh mục Thuốc")
        
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
            title='Phân bố Thuốc theo Danh mục',
            labels={'x': 'Danh mục', 'y': 'Số lượng Thuốc'}
        )

        fig_cat.update_traces(
            hovertemplate="<b>Danh mục:</b> %{x}<br><b>Số lượng:</b> %{y}<extra></extra>"
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
        "Tìm kiếm Thuốc": semantic_search_page,
        "Thuốc Thay thế": drug_substitution_page,
        "Chatbot Y tế Q&A": chatbot_page,
        "Phân tích Nhà sản xuất": manufacturer_analytics_page,
        "Tổng quan Dashboard": dashboard_overview_page
    }
    
    selected_page = st.sidebar.selectbox("Chọn chức năng:", list(pages.keys()))
    
    # Trạng thái hệ thống
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Trạng thái Hệ thống")
    st.sidebar.success(" ChromaDB Đã kết nối")
    st.sidebar.success(" Mô hình AI")
    st.sidebar.info(f" {st.session_state.collections['drugs_main'].count():,} thuốc có sẵn")
    
    # Hiển thị trang đã chọn
    if selected_page in [" Chatbot Y tế Q&A"]:
        pages[selected_page](st.session_state.collections, st.session_state.model)
    elif selected_page in [" Phân tích Nhà sản xuất", " Tổng quan Dashboard"]:
        pages[selected_page](st.session_state.collections)
    elif selected_page in [" Phân tích Tác dụng Phụ"]:
        pages[selected_page](st.session_state.collections)
    else:
        run_page(pages[selected_page], st.session_state.collections, st.session_state.model)


if __name__ == "__main__":
    main()