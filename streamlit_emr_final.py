import streamlit as st
import pandas as pd
import datetime
import joblib
import numpy as np
import altair as alt

# 1. 페이지 설정 및 리소스 로딩
st.set_page_config(page_title="SNUH AI Fall System", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_resources():
    try:
        # imbalanced-learn 파이프라인 대응 로드
        return joblib.load('risk_score_model.joblib')
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

new_model = load_resources()

# 2. 스타일 (CSS) - 기존 EMR 스타일 유지
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp { background-color: #1e252b; color: #e0e0e0; font-family: 'Noto Sans KR', sans-serif; }
    
    .digital-monitor {
        background-color: #000000; border: 2px solid #455a64; border-radius: 8px;
        padding: 20px; text-align: center; box-shadow: inset 0 0 20px rgba(0,0,0,0.9);
    }
    @keyframes blink { 50% { border-color: #ff5252; box-shadow: 0 0 20px #ff5252; } }
    .high-risk-blink { animation: blink 1s infinite; border: 3px solid #ff5252 !important; }
    
    .digital-number { font-family: 'Consolas', monospace; font-size: 52px; font-weight: 900; line-height: 1.0; }
    .status-text { font-size: 18px; font-weight: bold; margin-bottom: 5px; }
    
    .custom-alert-box {
        position: fixed; bottom: 30px; right: 30px; width: 380px;
        background-color: #263238; border-left: 8px solid #ff5252;
        padding: 20px; z-index: 9999; border-radius: 8px; box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# 3. 시연용 환자 데이터 (요청하신 정확한 수치 세팅)
PATIENTS_BASE = [
    {"name": "① 저위험 A", "gender": "F", "age": 58, "severity": 2, "sbp": 120, "dbp": 78, "pr": 78, "rr": 18, "bt": 36.6, "alb": 4.1, "crp": 0.3, "mental": "alert", "diag": "정기검진"},
    {"name": "② 저위험 B", "gender": "M", "age": 72, "severity": 2, "sbp": 130, "dbp": 82, "pr": 76, "rr": 18, "bt": 36.7, "alb": 3.8, "crp": 0.8, "mental": "alert", "diag": "단순 골절"},
    {"name": "③ 중위험", "gender": "F", "age": 68, "severity": 3, "sbp": 115, "dbp": 75, "pr": 88, "rr": 20, "bt": 37.2, "alb": 3.0, "crp": 4.0, "mental": "verbal response", "diag": "대퇴골 골절"},
    {"name": "④ 고위험 (상위 20%)", "gender": "M", "age": 65, "severity": 3, "sbp": 110, "dbp": 70, "pr": 96, "rr": 22, "bt": 37.6, "alb": 2.4, "crp": 6.0, "mental": "painful response", "diag": "충수염"}
]

if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'nursing_notes' not in st.session_state: st.session_state.nursing_notes = []

def update_sim_data(idx):
    p = PATIENTS_BASE[idx]
    st.session_state.s_sex, st.session_state.s_age = p['gender'], p['age']
    st.session_state.s_sev, st.session_state.s_sbp = p['severity'], p['sbp']
    st.session_state.s_dbp, st.session_state.s_pr = p['dbp'], p['pr']
    st.session_state.s_rr, st.session_state.s_bt = p['rr'], p['bt']
    st.session_state.s_alb, st.session_state.s_crp = p['alb'], p['crp']
    st.session_state.s_mental = p['mental']
    st.session_state.alarm_confirmed = False

if 's_age' not in st.session_state: update_sim_data(0)

# 4. 사이드바: 11개 변수 조작 패널
with st.sidebar:
    st.header("🏥 시뮬레이션 설정")
    selected_p = st.radio("환자 리스트", [p['name'] for p in PATIENTS_BASE], index=st.session_state.current_idx)
    new_idx = [p['name'] for p in PATIENTS_BASE].index(selected_p)
    if new_idx != st.session_state.current_idx:
        st.session_state.current_idx = new_idx
        update_sim_data(new_idx)
        st.rerun()

    st.divider()
    st.subheader("⚡ 11개 지표 조작")
    st.session_state.s_sex = st.radio("성별", ["M", "F"], index=0 if st.session_state.s_sex=="M" else 1, horizontal=True)
    st.session_state.s_age = st.slider("나이", 0, 100, st.session_state.s_age)
    st.session_state.s_sev = st.select_slider("중증도", options=[1, 2, 3, 4, 5], value=st.session_state.s_sev)
    st.session_state.s_sbp = st.number_input("SBP", value=st.session_state.s_sbp)
    st.session_state.s_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.s_alb, step=0.1)
    st.session_state.s_crp = st.number_input("CRP", value=st.session_state.s_crp)
    st.session_state.s_mental = st.selectbox("의식 반응", ["alert", "verbal response", "painful response", "unresponsive"], 
                                          index=["alert", "verbal response", "painful response", "unresponsive"].index(st.session_state.s_mental))
    c1, c2 = st.columns(2)
    with c1: st.session_state.s_dbp = st.number_input("DBP", value=st.session_state.s_dbp)
    with c2: st.session_state.s_pr = st.number_input("PR", value=st.session_state.s_pr)
    with c1: st.session_state.s_rr = st.number_input("RR", value=st.session_state.s_rr)
    with c2: st.session_state.s_bt = st.number_input("BT", value=st.session_state.s_bt, format="%.1f")

# 5. 핵심: 리스크 매칭 및 점수 보정(Rescaling) 로직
def get_matched_risk():
    if not new_model: return "오류", 0, "#888", False, 0
    
    # 모델 입력용 DF 생성 (11개 피처)
    df = pd.DataFrame([{
        '성별': 1 if st.session_state.s_sex == 'M' else 0,
        '중증도분류': st.session_state.s_sev,
        'SBP': st.session_state.s_sbp, 'DBP': st.session_state.s_dbp,
        'RR': st.session_state.s_rr, 'PR': st.session_state.s_pr, 'BT': st.session_state.s_bt,
        '내원시 반응': st.session_state.s_mental,
        '나이': st.session_state.s_age,
        'albumin': st.session_state.s_alb,
        'crp': st.session_state.s_crp
    }])
    
    prob = new_model.predict_proba(df)[0][1]
    
    # [핵심 매칭 로직] 팀원 기준값 반영
    # 고위험 >= 0.025498 / 중위험 >= 0.017725
    if prob >= 0.025498:
        level, color, score = "고위험 (상위 20%)", "#ff5252", int(80 + (prob - 0.025498) * 300)
        is_high = True
    elif prob >= 0.017725:
        level, color, score = "중위험 (상위 40%)", "#ffca28", int(50 + (prob - 0.017725) * 1000)
        is_high = False
    else:
        level, color, score = "저위험 (일반)", "#00e5ff", int(prob * 2500)
        is_high = False
        
    return level, min(score, 99), color, is_high, prob

level, display_score, status_color, alert_trigger, raw_prob = get_matched_risk()

# 6. 메인 화면 레이아웃
st.title("🏥 SNUH Smart AI Fall Dashboard")

col1, col2 = st.columns([1.2, 2])

with col1:
    # 디지털 계기판 (매칭된 점수와 색상 반영)
    blink = "high-risk-blink" if alert_trigger and not st.session_state.get('alarm_confirmed', False) else ""
    st.markdown(f"""
    <div class="digital-monitor {blink}">
        <div class="status-text" style="color:{status_color};">{level}</div>
        <div class="digital-number" style="color:{status_color};">{display_score}</div>
        <div style="font-size:11px; color:gray; margin-top:10px;">AI Prob: {raw_prob:.6f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if alert_trigger and not st.session_state.get('alarm_confirmed', False):
        if st.button("🚨 알람 확인 및 간호 중재 기록", type="primary", use_container_width=True):
            st.session_state.alarm_confirmed = True
            # 간호 기록 자동 생성 로직
            note = f"[{datetime.datetime.now().strftime('%H:%M')}] AI 고위험군 감지({display_score}점). 침상난간 확인 및 예방교육 시행함."
            st.session_state.nursing_notes.insert(0, note)
            st.rerun()

with col2:
    st.subheader("📝 실시간 간호 기록 (EMR)")
    if not st.session_state.nursing_notes:
        st.info("고위험군 발생 시 중재 내역이 여기에 자동 기록됩니다.")
    else:
        for n in st.session_state.nursing_notes:
            st.markdown(f'<div style="background:#2c3e50; padding:10px; border-radius:5px; margin-bottom:5px; border-left:4px solid #0288d1;">{n}</div>', unsafe_allow_html=True)

# 7. 리스크 팝업 알람
if alert_trigger and not st.session_state.get('alarm_confirmed', False):
    st.markdown(f"""
    <div class="custom-alert-box">
        <div style="color:#ff5252; font-weight:bold; font-size:1.3em;">🚨 낙상 고위험 감지! ({display_score}점)</div>
        <p style="font-size:0.9em; margin-top:10px;">현재 환자는 <b>상위 20% 고위험군</b>에 속합니다. 즉시 안전 조치를 취하십시오.</p>
    </div>
    """, unsafe_allow_html=True)
