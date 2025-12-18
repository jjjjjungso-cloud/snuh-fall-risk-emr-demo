import streamlit as st
import pandas as pd
import datetime
import joblib
import numpy as np
import altair as alt

# 1. 페이지 설정 및 리소스 로딩
st.set_page_config(page_title="SNUH AI Fall Dashboard v2.2", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_ai_model():
    try:
        # imblearn 파이프라인 대응을 위해 joblib 사용
        return joblib.load('risk_score_model.joblib')
    except Exception as e:
        st.error(f"❌ 모델 로드 에러: {e}")
        st.info("requirements.txt의 라이브러리 버전을 확인하세요.")
        return None

new_model = load_ai_model()

# 2. 병원 EMR 스타일 CSS (신호등 시스템 반영)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp { background-color: #1e252b; color: #e0e0e0; font-family: 'Noto Sans KR', sans-serif; }
    .digital-monitor {
        background-color: #000000; border-radius: 12px; padding: 25px;
        text-align: center; border: 4px solid #455a64; transition: all 0.5s;
    }
    .high-risk { border-color: #ff5252 !important; box-shadow: 0 0 25px #ff5252; animation: blink 1s infinite; }
    .mid-risk { border-color: #ffca28 !important; box-shadow: 0 0 15px #ffca28; }
    .low-risk { border-color: #00e5ff !important; }
    @keyframes blink { 50% { opacity: 0.8; } }
    .score-val { font-family: 'Consolas', monospace; font-size: 5rem; font-weight: 900; line-height: 1; }
    .note-box { background: #2c3e50; padding: 12px; border-radius: 5px; border-left: 5px solid #0288d1; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# 3. 환자 데이터 및 상태 관리 (제시해주신 4인 예시)
PATIENTS = [
    {"name": "① 저위험 A", "gender": "F", "age": 58, "severity": 2, "sbp": 120, "dbp": 78, "pr": 78, "rr": 18, "bt": 36.6, "alb": 4.1, "crp": 0.3, "mental": "명료(Alert)"},
    {"name": "② 저위험 B", "gender": "M", "age": 72, "severity": 2, "sbp": 130, "dbp": 82, "pr": 76, "rr": 18, "bt": 36.7, "alb": 3.8, "crp": 0.8, "mental": "명료(Alert)"},
    {"name": "③ 중위험", "gender": "F", "age": 68, "severity": 3, "sbp": 115, "dbp": 75, "pr": 88, "rr": 20, "bt": 37.2, "alb": 3.0, "crp": 4.0, "mental": "기면(Verbal)"},
    {"name": "④ 고위험 (상위 20%)", "gender": "M", "age": 65, "severity": 3, "sbp": 110, "dbp": 70, "pr": 96, "rr": 22, "bt": 37.6, "alb": 2.4, "crp": 6.0, "mental": "혼미(Painful)"}
]

if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'nursing_notes' not in st.session_state: st.session_state.nursing_notes = []

def reset_sim(idx):
    p = PATIENTS[idx]
    for k, v in p.items(): st.session_state[f"v_{k}"] = v
    st.session_state.alarm_done = False

if 'v_age' not in st.session_state: reset_sim(0)

# 4. 사이드바: 11개 변수 실시간 시뮬레이션
with st.sidebar:
    st.header("🏥 환자 시뮬레이션")
    sel = st.radio("환자 리스트", [p['name'] for p in PATIENTS], index=st.session_state.current_idx)
    new_idx = [p['name'] for p in PATIENTS].index(sel)
    if new_idx != st.session_state.current_idx:
        st.session_state.current_idx = new_idx
        reset_sim(new_idx)
        st.rerun()

    st.divider()
    st.subheader("⚡ 11개 지표 실시간 조작")
    st.session_state.v_gender = st.radio("성별", ["M", "F"], index=0 if st.session_state.v_gender=="M" else 1, horizontal=True)
    st.session_state.v_age = st.slider("나이", 0, 100, st.session_state.v_age)
    st.session_state.v_severity = st.select_slider("중증도", options=[1, 2, 3, 4, 5], value=st.session_state.v_severity)
    st.session_state.v_sbp = st.number_input("수축기 혈압 (SBP)", value=st.session_state.v_sbp)
    st.session_state.v_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.v_alb, step=0.1)
    st.session_state.v_crp = st.number_input("CRP (염증 지수)", value=st.session_state.v_crp)
    st.session_state.v_mental = st.selectbox("의식 상태", ["명료(Alert)", "기면(Verbal)", "혼미(Painful)"], 
                                          index=["명료(Alert)", "기면(Verbal)", "혼미(Painful)"].index(st.session_state.v_mental))
    
    col1, col2 = st.columns(2)
    with col1: st.session_state.v_dbp = st.number_input("DBP", value=st.session_state.v_dbp)
    with col2: st.session_state.v_pr = st.number_input("PR", value=st.session_state.v_pr)
    with col1: st.session_state.v_rr = st.number_input("RR", value=st.session_state.v_rr)
    with col2: st.session_state.v_bt = st.number_input("BT", value=st.session_state.v_bt, format="%.1f")

# 5. AI 추론 및 신호등 판정 (Scaling 로직 포함)
def get_ai_prediction():
    if not new_model: return "Error", 0, "low-risk", "#888", 0
    m_map = {"명료(Alert)": 0, "기면(Verbal)": 1, "혼미(Painful)": 2}
    
    # 11개 피처 정렬
    df = pd.DataFrame([{
        '성별': 1 if st.session_state.v_gender == 'M' else 0, '중증도분류': st.session_state.v_severity,
        'SBP': st.session_state.v_sbp, 'DBP': st.session_state.v_dbp, 'RR': st.session_state.v_rr,
        'PR': st.session_state.v_pr, 'BT': st.session_state.v_bt,
        '내원시 반응': m_map.get(st.session_state.v_mental, 0),
        '나이': st.session_state.v_age, 'albumin': st.session_state.v_alb, 'crp': st.session_state.v_crp
    }])
    
    prob = new_model.predict_proba(df)[0][1]
    
    # 임계값 기준 판정
    if prob >= 0.025498: # 고위험
        level, css, color = "고위험 (상위 20%)", "high-risk", "#ff5252"
        score = int(80 + (prob - 0.025498) * 400)
    elif prob >= 0.017725: # 중위험
        level, css, color = "중위험 (상위 40%)", "mid-risk", "#ffca28"
        score = int(50 + (prob - 0.017725) * 1000)
    else: # 저위험
        level, css, color = "저위험 (일반)", "low-risk", "#00e5ff"
        score = int(prob * 1500)
        
    return level, min(score, 99), css, color, prob

lvl, score, css, color, raw_p = get_ai_prediction()

# 6. 메인 레이아웃 및 알람 워크플로우
st.title("🏥 SNUH AI Fall Management CDSS")

c1, c2 = st.columns([1, 1.3])
with c1:
    # 실시간 계기판
    blink_class = css if css == "high-risk" and not st.session_state.get('alarm_done', False) else css
    st.markdown(f"""
    <div class="digital-monitor {blink_class}">
        <div style="color:{color}; font-weight:bold; font-size:1.2rem; margin-bottom:10px;">{lvl}</div>
        <div class="score-val" style="color:{color};">{score}</div>
        <div style="font-size:0.8rem; color:gray; margin-top:15px;">AI Raw Prob: {raw_p:.6f}</div>
    </div>
    """, unsafe_allow_html=True)

    # 고위험군 알람 팝업
    if css == "high-risk" and not st.session_state.get('alarm_done', False):
        @st.dialog("🚨 고위험군 즉각 중재 필요")
        def show_dialog():
            st.warning(f"AI 고위험 감지 ({score}점)")
            i1 = st.checkbox("침상 난간(Side Rail) 고정", value=True)
            i2 = st.checkbox("보호자 동반 보행 교육", value=True)
            i3 = st.checkbox("영양팀 협진 의뢰 (Albumin 저하)", value=(st.session_state.v_alb < 3.0))
            
            if st.button("수행 완료 및 EMR 전송", type="primary", use_container_width=True):
                note = f"[{datetime.datetime.now().strftime('%H:%M')}] AI 고위험군 판정({score}점). 간호중재 시행함."
                st.session_state.nursing_notes.insert(0, note)
                st.session_state.alarm_done = True
                st.rerun()
        show_dialog()

with c2:
    st.subheader("📝 실시간 간호 기록")
    if not st.session_state.nursing_notes:
        st.info("고위험 알람 발생 시 중재 내역이 여기에 기록됩니다.")
    else:
        for n in st.session_state.nursing_notes:
            st.markdown(f'<div class="note-box">{n}</div>', unsafe_allow_html=True)

# 7. 시각화
st.divider()
st.subheader("📊 주요 지표 분석")
chart_df = pd.DataFrame({
    '지표': ['Age', 'Alb', 'SBP', 'PR', 'CRP'],
    '수치': [st.session_state.v_age, st.session_state.v_alb*20, st.session_state.v_sbp/2, st.session_state.v_pr, st.session_state.v_crp*5]
}).set_index('지표')
st.bar_chart(chart_df)
