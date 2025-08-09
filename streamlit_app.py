# streamlit_app.py
# -*- coding: utf-8 -*-
"""
AI Destekli Bireysel Matematik Eğitimi — MVP v1.1 (Auto-Label Entegre)
Kübra için tez prototipi (Streamlit tek dosya)

Yenilikler (v1.1):
- CSV yüklenince otomatik PII temizleme (İsim/Ad-Soyad, Zaman Damgası, Yaş-Sınıf vb.)
- Likert (TR) -> sayısal (5..1) dönüşümü
- Soru -> Tema eşleme (görsel, işitsel, müzik, kinestetik, okuma-yazma, sosyal, bireysel)
- Tema skorlarının hesaplanması (score_*) ve kural-tabanlı "learning_style" etiketi
- Temiz ve etiketlenmiş veri setlerini indir butonları
- Mevcut akışla bütünleşik eğitim/tahmin/rapor

Not: CSV yanıtlarınız 5'li Likert ifadeleri içeriyorsa doğrudan çalışır.
"""

from __future__ import annotations
import io
import json
from datetime import datetime
import re
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

try:
    from fpdf import FPDF  # type: ignore
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# =============== Yardımcılar: Veri Temizleme & Etiketleme ===============
LIKERT_MAP = {
    "kesinlikle katılıyorum": 5,
    "katılıyorum": 4,
    "kararsızım": 3,
    "katılmıyorum": 2,
    "hiç katılmıyorum": 1,
}

PII_KEYWORDS = [
    "isim", "ad", "ad soyad", "soyad", "name", "full name",
    "timestamp", "zaman damgası", "tarih", "oluşturma zamanı",
    "sınıf", "sinif", "class", "grade",
    "yaş", "yas", "age",
    "okul", "school",
]

THEMES = {
    "görsel": ["görsel", "grafik", "şema", "renk", "zihin haritası", "çiz", "resim", "diagram", "gözümde"],
    "işitsel": ["işitsel", "dinleyerek", "sesli", "konuşarak", "anlatarak", "podcast"],
    "müzik": ["müzik", "ritim", "şarkı", "melodi"],
    "kinestetik": ["kinestetik", "somut", "deneyerek", "dokunarak", "uygulama", "materyal", "hareket"],
    "okuma-yazma": ["okuyarak", "yazarak", "yazıya", "not", "defter"],
    "sosyal": ["grup", "birlikte", "arkadaş", "tartış"],
    "bireysel": ["kendi başıma", "tek başıma", "yalnız"],
}

PRIORITY = ["görsel", "işitsel", "müzik", "kinestetik", "okuma-yazma", "sosyal", "bireysel"]


def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def drop_pii_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    norm_cols = {c: normalize_text(c) for c in df.columns}
    drop_cols = []
    for orig, norm in norm_cols.items():
        for kw in PII_KEYWORDS:
            if kw in norm:
                drop_cols.append(orig)
                break
    df_clean = df.drop(columns=list(dict.fromkeys(drop_cols)), errors="ignore")
    return df_clean, drop_cols


def map_likert_series(s: pd.Series) -> pd.Series:
    return s.apply(normalize_text).map(LIKERT_MAP)


def auto_label_dataset(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Likert maddelerini 5..1'e çevir, tema skorlarını hesapla ve learning_style üret."""
    df = df_in.copy()
    # 1) Likert'e çevrilebilen sütunları dönüştür
    likert_cols = []
    for c in df.columns:
        if df[c].dtype == "object":
            mapped = map_likert_series(df[c])
            if mapped.notna().mean() >= 0.5:
                df[c] = mapped
                likert_cols.append(c)

    # 2) Soru -> tema ataması
    col_themes: Dict[str, List[str]] = {c: [] for c in df.columns}
    for c in df.columns:
        cname = normalize_text(c)
        for th, keys in THEMES.items():
            if any(k in cname for k in keys):
                col_themes[c].append(th)

    # 3) Tema skorları
    theme_scores = {}
    theme_used_cols = {}
    for th in THEMES.keys():
        th_cols = [c for c in likert_cols if th in col_themes.get(c, [])]
        theme_used_cols[th] = th_cols
        if th_cols:
            theme_scores[th] = df[th_cols].mean(axis=1)
        else:
            theme_scores[th] = pd.Series([np.nan]*len(df), index=df.index)
    theme_df = pd.DataFrame(theme_scores)

    # 4) Kural-tabanlı etiket (argmax + öncelik)
    def pick_label(row):
        vals = {th: row[th] for th in PRIORITY if not pd.isna(row[th])}
        if not vals:
            return np.nan
        max_val = max(vals.values())
        ties = [th for th, v in vals.items() if v == max_val]
        for th in PRIORITY:
            if th in ties:
                return th
        return ties[0]

    labels = theme_df.apply(pick_label, axis=1)

    # 5) Çıktı birleştir
    df_labeled = df.copy()
    for th in PRIORITY:
        df_labeled[f"score_{th}"] = theme_df.get(th)
    df_labeled["learning_style"] = labels

    return df_labeled, theme_used_cols

# =============== Modelleme Yardımcıları ===============

def _detect_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    features = [c for c in df.columns if c != target]
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    num_cols, cat_cols = _detect_feature_types(df, target)
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model(algorithm: str):
    if algorithm == "Lojistik Regresyon":
        return LogisticRegression(max_iter=300)
    elif algorithm == "Random Forest":
        return RandomForestClassifier(n_estimators=400, random_state=42)
    raise ValueError("Bilinmeyen algoritma")


def train_and_eval(df: pd.DataFrame, target: str, algorithm: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )
    pipe = Pipeline(steps=[("pre", build_preprocessor(df, target)), ("clf", build_model(algorithm))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        report = classification_report(y_test, y_pred, zero_division=0)
    except Exception:
        report = "Sınıflandırma raporu üretilemedi."
    return pipe, {"accuracy": float(acc), "report": report, "classes_": sorted(y.unique())}

# =============== Öneri Motoru ===============

def style_recommendations(predicted_style: str, top_probs: List[Tuple[str, float]]):
    style = (predicted_style or "").lower()
    base = {
        "genel": [
            "Haftalık 2-3 küçük hedef belirleyin ve bitince ödüllendirin.",
            "Pomodoro (25dk çalışma/5dk mola) ritmini deneyin.",
            "Yanlışlar analiz edilip yeniden karşısına çıkarılmalı (spaced repetition).",
        ]
    }
    buckets = {
        "görsel": [
            "Konu özetlerini zihin haritası ve renkli şema ile çıkarın.",
            "Formüller için poster/flashcard hazırlayın; görünür yerlere asın.",
            "Video anlatım + interaktif görsellerle pekiştirme yapın.",
        ],
        "işitsel": [
            "Konuyu sesli anlatarak tekrar (Feynman tekniği).",
            "Podcast/konu anlatımı dinleyerek not çıkarın.",
            "Ritmik ezgilerle formül ve tanımları kodlayın.",
        ],
        "kinestetik": [
            "Somut materyallerle problem modelleyin.",
            "Ayakta çalışma/tahtada çözüm ve kısa hareket molaları ekleyin.",
            "Gerçek hayat görevleri (alışveriş, ölçüm) ile uygulama yapın.",
        ],
        "okuma-yazma": [
            "Kısa paragraf özetleri ve kendi kelimeleriyle formül defteri tutun.",
            "Çözümleri adım adım yazın; bir sonraki derste sadece adımları görün.",
            "Gün sonunda 5 maddeyle öğrenilenleri yazılı özetleyin.",
        ],
        "oyun": [
            "Süreli mini yarışlar ve puan panosu kurun.",
            "Seviye atlamalı görev listeleri ve rozet sistemi kullanın.",
            "İkili-eşli problem çözme oyunları planlayın.",
        ],
        "müzik": [
            "Ritimle tekrar: çarpım tablosu/örüntüler için tempo tutun.",
            "Arka planda düşük sesli lo-fi müzikle odak blokları oluşturun.",
            "Melodik ipuçlarıyla formülleri kodlayın.",
        ],
        "sosyal": [
            "Akranla tartışma ve eşle anlatma seansları planlayın.",
            "Küçük grup projeleriyle sorumluluk paylaşın.",
            "Soru-cevap odaklı mini oturumlar yapın.",
        ],
        "bireysel": [
            "Sessiz, dikkat dağıtıcısız çalışma blokları oluşturun.",
            "Kendi hızında mikro-öğrenme modülleri verin.",
            "Öz-değerlendirme listeleriyle ilerlemeyi takip edin.",
        ],
    }
    key = None
    for k in buckets:
        if k in style:
            key = k
            break
    if key is None:
        if "gorsel" in style: key = "görsel"
        elif "isitsel" in style: key = "işitsel"
        elif "kinestetik" in style: key = "kinestetik"
        elif "okuma" in style or "yazma" in style: key = "okuma-yazma"
        elif "oyun" in style: key = "oyun"
        elif "muzik" in style: key = "müzik"
        elif "sosyal" in style: key = "sosyal"
        elif "bireysel" in style: key = "bireysel"
    personalized = buckets.get(key, [])
    tips = {
        "öğrenci": personalized + base["genel"],
        "veli": [
            "Haftalık planı birlikte gözden geçirin; küçük ama sürekli ilerlemeyi kutlayın.",
            "Gürültüsüz bir çalışma köşesi oluşturun.",
            "Zorlanılan konularda süreç odaklı geri bildirim verin.",
        ],
        "öğretmen": [
            "Ders başı hedef netleştirme, ders sonu 2 dakikalık yansıma.",
            "Stile uygun 1 etkinlik ekleyin (harita/oyun/rol-play vb.).",
            "Yanlış türü analizi ile hedefli ek alıştırma verin.",
        ],
        "olasılıklar": [f"{name}: %{prob*100:.1f}" for name, prob in top_probs],
    }
    return tips


def build_pdf(title: str, student_name: str, predicted_style: str, tips: Dict[str, List[str]]) -> bytes:
    if not FPDF_AVAILABLE:
        raise RuntimeError("FPDF mevcut değil")
    pdf = FPDF()
    pdf.add_page()

    # Unicode fontu repodaki fonts klasöründen yükle
    font_path = os.path.join("fonts", "DejaVuSans.ttf")
    try:
        pdf.add_font('DejaVu', '', font_path, uni=True)
        pdf.set_font('DejaVu', '', 14)
    except Exception as e:
        # Font bulunamazsa Arial'a düş ve bilgilendir
        pdf.set_font('Arial', size=14)

    pdf.cell(0, 10, txt=title, ln=True)
    pdf.set_font_size(12)
    pdf.cell(0, 8, txt=f"Öğrenci: {student_name}", ln=True)
    pdf.cell(0, 8, txt=f"Tahmin edilen öğrenme stili: {predicted_style}", ln=True)
    pdf.ln(4)

    def write_list(header: str, items: List[str]):
        pdf.set_font_size(12)
        pdf.cell(0, 8, txt=header, ln=True)
        pdf.set_font_size(11)
        for it in items:
            pdf.multi_cell(0, 6, txt=f"• {it}")
        pdf.ln(2)

    tips = tips or {}
    write_list("Öğrenci için öneriler:", tips.get("öğrenci", []))
    write_list("Veliye öneriler:", tips.get("veli", []))
    write_list("Öğretmene öneriler:", tips.get("öğretmen", []))
    write_list("Olasılıklar:", tips.get("olasılıklar", []))

    # fpdf2'de S çıktısı string döner; latin-1 encode ile güvenli şekilde bayt'a çeviriyoruz
    out = pdf.output(dest="S").encode("latin-1", "ignore")
    return out


def build_txt(title: str, student_name: str, predicted_style: str, tips: Dict[str, List[str]]) -> bytes:
    lines = [title, f"Öğrenci: {student_name}", f"Tahmin edilen öğrenme stili: {predicted_style}", ""]
    for header in ["öğrenci", "veli", "öğretmen", "olasılıklar"]:
        lines.append(header.upper())
        for it in tips.get(header, []):
            lines.append(f" - {it}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")

# =============== Streamlit Arayüzü ===============

st.set_page_config(page_title="AI Destekli Bireysel Matematik Eğitimi — MVP v1.1", layout="wide")
st.title("AI Destekli Bireysel Matematik Eğitimi — MVP v1.1")
st.caption("Veri yükle → PII temizle → Auto-label → Modeli eğit → Tahmin → Rapor")

if "dataset_raw" not in st.session_state:
    st.session_state.dataset_raw = None
if "dataset_clean" not in st.session_state:
    st.session_state.dataset_clean = None
if "dataset_labeled" not in st.session_state:
    st.session_state.dataset_labeled = None
if "target_col" not in st.session_state:
    st.session_state.target_col = "learning_style"
if "model" not in st.session_state:
    st.session_state.model = None
if "classes" not in st.session_state:
    st.session_state.classes = []
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_probs" not in st.session_state:
    st.session_state.last_probs = []
if "last_student" not in st.session_state:
    st.session_state.last_student = {}
if "theme_used_cols" not in st.session_state:
    st.session_state.theme_used_cols = {}


TAB1, TAB2, TAB3, TAB4 = st.tabs([
    "1) Veri Yükle & Etiketle",
    "2) Model Eğit",
    "3) Öğrenci Tahmini",
    "4) Raporlar & Dışa Aktarım",
])

# ---------- TAB 1 ----------
with TAB1:
    st.subheader("Verinizi yükleyin")
    up = st.file_uploader("CSV dosyası yükle", type=["csv"])
    if up is not None:
        try:
            df_raw = pd.read_csv(up)
        except Exception:
            up.seek(0)
            df_raw = pd.read_csv(up, sep=";")
        st.session_state.dataset_raw = df_raw
        st.success(f"Yüklendi: {df_raw.shape[0]} satır, {df_raw.shape[1]} sütun")
        with st.expander("Ham veri önizleme", expanded=False):
            st.dataframe(df_raw.head(20))

        # PII temizleme
        df_clean, dropped = drop_pii_columns(df_raw)
        st.session_state.dataset_clean = df_clean
        st.info("PII içerikli sütunlar kaldırıldı: " + (", ".join(dropped) if dropped else "(yok)"))
        st.download_button("Temiz veri (CSV) indir", data=df_clean.to_csv(index=False).encode("utf-8"), file_name="dataset_clean.csv", mime="text/csv")

        # Auto-label
        df_labeled, theme_used_cols = auto_label_dataset(df_clean)
        st.session_state.dataset_labeled = df_labeled
        st.session_state.theme_used_cols = theme_used_cols

        st.success("Otomatik etiketleme tamamlandı → 'learning_style' üretildi ve score_* sütunları eklendi.")
        # En yüksek skora göre belirlenen stil zaten 'learning_style' sütununda; özetini gösterelim
        st.write("**Etiket (learning_style) dağılımı:**")
        counts = df_labeled['learning_style'].value_counts(dropna=True)
        st.bar_chart(counts)

        st.dataframe(df_labeled.head(20))
        st.download_button("Etiketlenmiş veri (CSV) indir", data=df_labeled.to_csv(index=False).encode("utf-8"), file_name="dataset_labeled.csv", mime="text/csv")

        with st.expander("Tema bazında kullanılan soru sayıları"):
            for th, cols in theme_used_cols.items():
                st.write(f"**{th}**: {len(cols)} madde")
            for th, cols in theme_used_cols.items():
                st.write(f"**{th}**: {len(cols)} madde")

# ---------- TAB 2 ----------
with TAB2:
    st.subheader("Modelinizi eğitin")
    if st.session_state.dataset_labeled is None:
        st.info("Önce veri yükleyip etiketleyin.")
    else:
        df = st.session_state.dataset_labeled.copy()
        target = st.selectbox("Hedef değişken", options=["learning_style"], index=0)
        st.session_state.target_col = target

        # Varsayılan: hedef dışındaki tüm sütunlar özellik; ancak metin/boş sütunları otomatik ayıkla
        blacklist = [target]
        feat_cols = [c for c in df.columns if c not in blacklist]
        # Streamlit formu ile istendiğinde ayıklama/ekleme
        with st.expander("Özellik sütunlarını düzenle (isteğe bağlı)"):
            feat_cols = st.multiselect("Eğitime dahil edilecek sütunlar", options=[c for c in df.columns if c != target], default=feat_cols)
        Xy = df[feat_cols + [target]].dropna()
        st.session_state.feature_columns = feat_cols

        algo = st.selectbox("Algoritma", ["Lojistik Regresyon", "Random Forest"], index=1)
        test_size = st.slider("Test oranı", 0.1, 0.4, 0.2, 0.05)
        if st.button("Eğit ve Değerlendir", type="primary"):
            with st.spinner("Model eğitiliyor..."):
                pipe, metrics = train_and_eval(Xy, target, algo, test_size)
                st.session_state.model = pipe
                st.session_state.classes = metrics.get("classes_", [])
                st.success(f"Doğruluk (accuracy): {metrics['accuracy']:.3f}")
                st.text("Sınıflandırma Raporu:")
                st.code(metrics.get("report", ""))
                # Model indir
                buf = io.BytesIO()
                joblib.dump(pipe, buf)
                buf.seek(0)
                st.download_button("Modeli .joblib olarak indir", data=buf.read(), file_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib", mime="application/octet-stream")

# ---------- TAB 3 ----------
with TAB3:
    st.subheader("Yeni öğrenci tahmini")
    if st.session_state.model is None or st.session_state.dataset_labeled is None:
        st.info("Önce veri yükleyip bir model eğitin.")
    else:
        df = st.session_state.dataset_labeled
        feat_cols = st.session_state.feature_columns

        # Kullanım modu: Veriden seç veya elle gir
        mode = st.radio("Girdi yöntemi", ["Veriden seç", "Elle gir"], horizontal=True)

        # Varsayılan öğrenci adı
        default_name = "Öğrenci-001"

        if mode == "Veriden seç":
            # Bir kayıt seçtir (PII yoksa indeks üzerinden)
            options = list(df.index)
            if not options:
                st.warning("Veri bulunamadı.")
            else:
                def fmt(i):
                    lbl = str(df.loc[i, st.session_state.target_col]) if st.session_state.target_col in df.columns else "?"
                    return f"Kayıt {i} — etiket: {lbl}"
                selected_idx = st.selectbox("Kayıt seç", options=options, format_func=fmt, index=0)
                # Seçilen kayıttan giriş değerlerini hazırla
                X_row = df.loc[selected_idx, feat_cols]
                inputs = {}
                for col in feat_cols:
                    val = X_row[col]
                    if pd.api.types.is_numeric_dtype(df[col]):
                        inputs[col] = float(0.0 if pd.isna(val) else val)
                    else:
                        inputs[col] = str(val)
                student_name = f"Öğrenci-{int(selected_idx):03d}"
                # Tahmini anında göster butonla
                if st.button("Seçili kayda göre tahmin et", type="primary"):
                    X_new = pd.DataFrame([inputs])
                    model = st.session_state.model
                    try:
                        proba = model.predict_proba(X_new)[0]
                        classes = getattr(model, "classes_", st.session_state.classes)
                        pairs = list(zip(classes, proba))
                        pairs.sort(key=lambda x: x[1], reverse=True)
                        pred = pairs[0][0]
                        top3 = pairs[:3]
                    except Exception:
                        pred = model.predict(X_new)[0]
                        top3 = [(pred, 1.0)]
                    st.session_state.last_pred = pred
                    st.session_state.last_probs = top3
                    st.session_state.last_student = {"name": student_name, "inputs": inputs}
                    st.success(f"Tahmin edilen öğrenme stili: **{pred}**")
                    st.write("**Olasılıklar (top-3):**")
                    st.table(pd.DataFrame({"stil": [p[0] for p in top3], "olasılık": [round(p[1], 4) for p in top3]}))
                    tips = style_recommendations(pred, top3)
                    st.markdown("### Kişiselleştirilmiş Öneriler")
                    st.markdown("**Öğrenci**")
                    for t in tips["öğrenci"]:
                        st.markdown(f"- {t}")
                    st.markdown("**Veli**")
                    for t in tips["veli"]:
                        st.markdown(f"- {t}")
                    st.markdown("**Öğretmen**")
                    for t in tips["öğretmen"]:
                        st.markdown(f"- {t}")
        else:
            # Elle giriş modu (mevcut form korunuyor)
            with st.form("predict_form"):
                student_name = st.text_input("Öğrenci adı/etiketi", value=default_name)
                inputs = {}
                for col in feat_cols:
                    # Özellik alanlarını tipine göre oluştur
                    if pd.api.types.is_numeric_dtype(df[col]):
                        default = float(df[col].median()) if pd.notnull(df[col].median()) else 0.0
                        inputs[col] = st.number_input(col, value=default)
                    else:
                        cats = sorted([str(x) for x in df[col].dropna().unique().tolist()][:50])
                        default = cats[0] if cats else ""
                        inputs[col] = st.selectbox(col, options=cats if cats else [""])
                submitted = st.form_submit_button("Tahmin Et", type="primary")
            if submitted:
                X_new = pd.DataFrame([inputs])
                model = st.session_state.model
                try:
                    proba = model.predict_proba(X_new)[0]
                    classes = getattr(model, "classes_", st.session_state.classes)
                    pairs = list(zip(classes, proba))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    pred = pairs[0][0]
                    top3 = pairs[:3]
                except Exception:
                    pred = model.predict(X_new)[0]
                    top3 = [(pred, 1.0)]
                st.session_state.last_pred = pred
                st.session_state.last_probs = top3
                st.session_state.last_student = {"name": student_name, "inputs": inputs}
                st.success(f"Tahmin edilen öğrenme stili: **{pred}**")
                st.write("**Olasılıklar (top-3):**")
                st.table(pd.DataFrame({"stil": [p[0] for p in top3], "olasılık": [round(p[1], 4) for p in top3]}))
                tips = style_recommendations(pred, top3)
                st.markdown("### Kişiselleştirilmiş Öneriler")
                st.markdown("**Öğrenci**")
                for t in tips["öğrenci"]:
                    st.markdown(f"- {t}")
                st.markdown("**Veli**")
                for t in tips["veli"]:
                    st.markdown(f"- {t}")
                st.markdown("**Öğretmen**")
                for t in tips["öğretmen"]:
                    st.markdown(f"- {t}")

# ---------- TAB 4 ----------

with TAB4:
    st.subheader("Rapor oluştur ve indir")
    if not st.session_state.last_pred:
        st.info("Önce bir tahmin üretin.")
    else:
        pred = st.session_state.last_pred
        top3 = st.session_state.last_probs
        student_name = st.session_state.last_student.get("name", "Öğrenci")
        tips = style_recommendations(pred, top3)
        st.write(f"**Öğrenci:** {student_name}")
        st.write(f"**Tahmin edilen stil:** {pred}")
        title = f"Bireysel Matematik Eğitimi Raporu — {datetime.now().strftime('%Y-%m-%d')}"
        if FPDF_AVAILABLE:
            try:
                pdf_bytes = build_pdf(title, student_name, pred, tips)
                st.download_button(
                    label="PDF indir",
                    data=pdf_bytes,
                    file_name=f"rapor_{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.warning(f"PDF üretilemedi: {e}")
        txt_bytes = build_txt(title, student_name, pred, tips)
        st.download_button(
            label="TXT indir (yedek)",
            data=txt_bytes,
            file_name=f"rapor_{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

st.divider()
st.caption("v1.1 — Auto-label entegre. Sonraki adımlar: model karşılaştırma, hiperparametre arama, önem analizi, veli paneli.")
