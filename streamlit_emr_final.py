import streamlit as st
import pandas as pd
import datetime
import joblib
import numpy as np
import altair as alt

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 리소스 로딩
# --------------------------------------------------------------------------------
st.set_page_config(page_title="SNUH AI Fall System", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_resources():
    try:
        return joblib.load('risk_score_model.joblib')
    except:
        return None

new_model = load_resources()

# [CSS 스타일] 기존의 세련된 다크모드 UI 유지
st.markdown("""
<style>
    .stApp { background-color: #1e252b; color: #e0e0e0; }
    .digital-monitor {
        background-color: #000000; border: 2px solid #455a64; border-radius: 8px;
        padding: 20px; text-align: center; box-shadow: inset 0 0 20px rgba(0,0,0,0.9);
    }
    @keyframes blink { 50% { border-color: #ff5252; box-shadow: 0 0 15px #ff5252; } }
    .alarm-active { animation: blink 1s infinite; border: 2px solid #ff5252 !important; }
    .digital-number { font-family: 'Consolas', monospace; font-size: 50px; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 2. 환자 데이터 및 상태 관리
# --------------------------------------------------------------------------------
PATIENTS_BASE = [
    {"name": "① 저위험 A", "gender": "F", "age": 58, "severity": 2, "sbp": 120, "dbp": 78, "pr": 78, "rr": 18, "bt": 36.6, "alb": 4.1, "crp": 0.3, "mental": "명료(Alert)"},
    {"name": "② 저위험 B", "gender": "M", "age": 72, "severity": 2, "sbp": 130, "dbp": 82, "pr": 76, "rr": 18, "bt": 36.7, "alb": 3.8, "crp": 0.8, "mental": "명료(Alert)"},
    {"name": "③ 중위험", "gender": "F", "age": 68, "severity": 3, "sbp": 115, "dbp": 75, "pr": 88, "rr": 20, "bt": 37.2, "alb": 3.0, "crp": 4.0, "mental": "기면(Verbal)"},
    {"name": "④ 고위험 (상위 20%)", "gender": "M", "age": 65, "severity": 3, "sbp": 110, "dbp": 70, "pr": 96, "rr": 22, "bt": 37.6, "alb": 2.4, "crp": 6.0, "mental": "혼미(Painful)"}
]

if 'current_idx' not in st.session_state: st.session_state.current_idx = 0

def update_sim_data(idx):
    p = PATIENTS_BASE[idx]
    st.session_state.s_sex = p['gender']
    st.session_state.s_age = p['age']
    st.session_state.s_sev = p['severity']
    st.session_state.s_sbp = p['sbp']
    st.session_state.s_dbp = p['dbp']
    st.session_state.s_pr = p['pr']
    st.session_state.s_rr = p['rr']
    st.session_state.s_bt = p['bt']
    st.session_state.s_alb = p['alb']
    st.session_state.s_crp = p['crp']
    st.session_state.s_mental = p['mental']
    st.session_state.alarm_confirmed = False

if 's_age' not in st.session_state: update_sim_data(0)

# --------------------------------------------------------------------------------
# 3. 사이드바: 11개 입력값 통합 배치
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("🏥 환자 선택 및 시뮬레이션")
    selected = st.radio("예시 환자 로드", [p['name'] for p in PATIENTS_BASE], index=st.session_state.current_idx)
    new_idx = [p['name'] for p in PATIENTS_BASE].index(selected)
    
    if new_idx != st.session_state.current_idx:
        st.session_state.current_idx = new_idx
        update_sim_data(new_idx)
        st.rerun()

    st.divider()
    st.subheader("⚡ 11개 핵심 지표 조정")
    
    # 11개 입력 위젯 배치
    st.session_state.s_sex = st.radio("성별", ["M", "F"], index=0 if st.session_state.s_sex=="M" else 1, horizontal=True)
    st.session_state.s_age = st.slider("나이", 0, 100, st.session_state.s_age)
    st.session_state.s_sev = st.select_slider("중증도분류(KTAS)", options=[1, 2, 3, 4, 5], value=st.session_state.s_sev)
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.session_state.s_sbp = st.number_input("SBP", value=st.session_state.s_sbp, step=5)
        st.session_state.s_pr = st.number_input("PR", value=st.session_state.s_pr, step=5)
        st.session_state.s_bt = st.number_input("BT", value=st.session_state.s_bt, step=0.1, format="%.1f")
    with col_v2:
        st.session_state.s_dbp = st.number_input("DBP", value=st.session_state.s_dbp, step=5)
        st.session_state.s_rr = st.number_input("RR", value=st.session_state.s_rr, step=2)
        st.session_state.s_crp = st.number_input("CRP", value=st.session_state.s_crp, step=0.5)

    st.session_state.s_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.s_alb, step=0.1)
    st.session_state.s_mental = st.selectbox("내원시 반응", ["명료(Alert)", "기면(Verbal)", "혼미(Painful)"], 
                                          index=["명료(Alert)", "기면(Verbal)", "혼미(Painful)"].index(st.session_state.s_mental))

# --------------------------------------------------------------------------------
# 4. 추론 및 메인 화면 표출
# --------------------------------------------------------------------------------
def get_prediction():
    if new_model is None: return "Error", 0, "#888", False, 0
    mental_map = {"명료(Alert)": 0, "기면(Verbal)": 1, "혼미(Painful)": 2}
    
    # 모델 학습 순서에 맞춘 11개 피처 데이터프레임
    df = pd.DataFrame([{
        '성별': 1 if st.session_state.s_sex == 'M' else 0,
        '중증도분류': st.session_state.s_sev,
        'SBP': st.session_state.s_sbp, 'DBP': st.session_state.s_dbp,
        'RR': st.session_state.s_rr, 'PR': st.session_state.s_pr, 'BT': st.session_state.s_bt,
        '내원시 반응': mental_map.get(st.session_state.s_mental, 0),
        '나이': st.session_state.s_age, 'albumin': st.session_state.s_alb, 'crp': st.session_state.s_crp
    }])
    
    prob = new_model.predict_proba(df)[0][1]
    # 팀원 기준값 반영
    if prob >= 0.025498: return "고위험 (TOP 20%)", int(85+prob*10), "#ff5252", True, prob
    elif prob >= 0.017725: return "중위험 (TOP 40%)", int(55+prob*15), "#ffca28", False, prob
    else: return "저위험 (안정)", int(25+prob*15), "#00e5ff", False, prob

level, score, color, alert, raw_p = get_prediction()

# 메인 레이아웃
st.title("🏥 SNUH AI Fall Prevention System v2")
c1, c2 = st.columns([1, 2])

with c1:
    alarm_css = "alarm-active" if alert and not st.session_state.get('alarm_confirmed', False) else ""
    st.markdown(f"""
    <div class="digital-monitor {alarm_css}">
        <div style="color:{color}; font-size:18px; font-weight:bold;">{level}</div>
        <div class="digital-number" style="color:{color};">{score}</div>
        <div style="font-size:12px; color:gray; margin-top:10px;">Prob: {raw_p:.6f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if alert and not st.session_state.get('alarm_confirmed', False):
        if st.button("🚨 알람 확인 (Confirm Intervention)", use_container_width=True, type="primary"):
            st.session_state.alarm_confirmed = True
            st.rerun()

with c2:
    st.subheader(f"📊 {selected} 실시간 분석 리포트")
    st.info(f"이 환자는 현재 상위 {('20%' if alert else '40% 이내' if score > 50 else '관리군')}의 위험도를 보이고 있습니다.")
    
    # 11개 수치 요약 바 차트
    v_data = pd.DataFrame({
        '지표': ['Age', 'Alb', 'SBP', 'PR', 'CRP'],
        '수치': [st.session_state.s_age, st.session_state.s_alb*20, st.session_state.s_sbp/2, st.session_state.s_pr, st.session_state.s_crp*5]
    }).set_index('지표')
    st.bar_chart(v_data)

st.divider()
st.subheader("📝 스마트 간호 기록 (Auto-Charting)")
note = f"[{level}] 낙상위험도 {score}점 확인됨. SBP {st.session_state.s_sbp}, Albumin {st.session_state.s_alb} 등 생체 징후 변화에 따른 집중 모니터링 시행함."
st.text_area("생성된 문구", value=note, height=100)
