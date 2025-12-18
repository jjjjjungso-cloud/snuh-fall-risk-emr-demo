import streamlit as st
import pandas as pd
import datetime
import time
import joblib
import numpy as np
import altair as alt

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 리소스 로딩
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="SNUH AI Fall Dashboard v2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [CSS 스타일] 기존 EMR의 세련된 다크모드 디자인 유지
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp { background-color: #1e252b; color: #e0e0e0; font-family: 'Noto Sans KR', sans-serif; }
    
    .digital-monitor-container {
        background-color: #000000; border: 2px solid #455a64; border-radius: 8px;
        padding: 20px; margin-top: 15px; box-shadow: inset 0 0 20px rgba(0,0,0,0.9);
        display: flex; justify-content: space-around; align-items: center; transition: all 0.5s;
    }
    @keyframes blink { 50% { border-color: #ff5252; box-shadow: 0 0 15px #ff5252; } }
    .alarm-active { animation: blink 1s infinite; border: 2px solid #ff5252 !important; }

    .digital-number { font-family: 'Consolas', monospace; font-size: 48px; font-weight: 900; line-height: 1.0; }
    .monitor-label { color: #90a4ae; font-size: 14px; font-weight: bold; margin-bottom: 5px; }

    .custom-alert-box {
        position: fixed; bottom: 30px; right: 30px; width: 380px;
        background-color: #263238; border-left: 8px solid #ff5252;
        box-shadow: 0 6px 25px rgba(0,0,0,0.7); border-radius: 8px;
        padding: 20px; z-index: 9999;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('risk_score_model.joblib')
        return model
    except:
        return None

new_model = load_resources()

# --------------------------------------------------------------------------------
# 2. 시연용 환자 데이터 세팅 (요청하신 4인)
# --------------------------------------------------------------------------------
PATIENTS_BASE = [
    {"name": "① 저위험 A (정상군)", "gender": "F", "age": 58, "severity": 2, "sbp": 120, "dbp": 78, "pr": 78, "rr": 18, "bt": 36.6, "alb": 4.1, "crp": 0.3, "mental": "명료(Alert)"},
    {"name": "② 저위험 B (정상-고령)", "gender": "M", "age": 72, "severity": 2, "sbp": 130, "dbp": 82, "pr": 76, "rr": 18, "bt": 36.7, "alb": 3.8, "crp": 0.8, "mental": "명료(Alert)"},
    {"name": "③ 중위험 (경계/관찰)", "gender": "F", "age": 68, "severity": 3, "sbp": 115, "dbp": 75, "pr": 88, "rr": 20, "bt": 37.2, "alb": 3.0, "crp": 4.0, "mental": "기면(Verbal)"},
    {"name": "④ 고위험 (상위 20%)", "gender": "M", "age": 65, "severity": 3, "sbp": 110, "dbp": 70, "pr": 96, "rr": 22, "bt": 37.6, "alb": 2.4, "crp": 6.0, "mental": "혼미(Painful)"}
]

# --------------------------------------------------------------------------------
# 3. 세션 상태 관리
# --------------------------------------------------------------------------------
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'alarm_confirmed' not in st.session_state: st.session_state.alarm_confirmed = False

def update_sim_data(idx):
    p = PATIENTS_BASE[idx]
    st.session_state.sim_sex = p['gender']
    st.session_state.sim_age = p['age']
    st.session_state.sim_severity = p['severity']
    st.session_state.sim_sbp = p['sbp']
    st.session_state.sim_dbp = p['dbp']
    st.session_state.sim_pr = p['pr']
    st.session_state.sim_rr = p['rr']
    st.session_state.sim_bt = p['bt']
    st.session_state.sim_alb = p['alb']
    st.session_state.sim_crp = p['crp']
    st.session_state.sim_mental = p['mental']
    st.session_state.alarm_confirmed = False

if 'sim_age' not in st.session_state: update_sim_data(0)

# --------------------------------------------------------------------------------
# 4. 추론 로직 (팀원 기준값 적용)
# --------------------------------------------------------------------------------
def run_inference():
    if new_model is None: return "Error", 0, "#888", False, 0
    
    mental_map = {"명료(Alert)": 0, "기면(Verbal)": 1, "혼미(Painful)": 2}
    # 11개 피처 순서 맞춤
    features = pd.DataFrame([{
        '성별': 1 if st.session_state.sim_sex == 'M' else 0,
        '중증도분류': st.session_state.sim_severity,
        'SBP': st.session_state.sim_sbp,
        'DBP': st.session_state.sim_dbp,
        'RR': st.session_state.sim_rr,
        'PR': st.session_state.sim_pr,
        'BT': st.session_state.sim_bt,
        '내원시 반응': mental_map.get(st.session_state.sim_mental, 0),
        '나이': st.session_state.sim_age,
        'albumin': st.session_state.sim_alb,
        'crp': st.session_state.sim_crp
    }])
    
    prob = new_model.predict_proba(features)[0][1]
    
    # 팀원 기준값: 고위험 >= 0.025498, 중위험 >= 0.017725
    if prob >= 0.025498:
        return "고위험 (상위 20%)", int(80 + prob*15), "#ff5252", True, prob
    elif prob >= 0.017725:
        return "중위험 (상위 40%)", int(50 + prob*15), "#ffca28", False, prob
    else:
        return "저위험 (안정)", int(20 + prob*15), "#00e5ff", False, prob

# --------------------------------------------------------------------------------
# 5. UI 메인 레이아웃
# --------------------------------------------------------------------------------
col_side, col_main = st.columns([2.5, 7.5])

with col_side:
    st.markdown("### 🏥 재원 환자")
    selected_name = st.radio("목록", [p['name'] for p in PATIENTS_BASE], index=st.session_state.current_idx, label_visibility="collapsed")
    new_idx = [p['name'] for p in PATIENTS_BASE].index(selected_name)
    if new_idx != st.session_state.current_idx:
        st.session_state.current_idx = new_idx
        update_sim_data(new_idx)
        st.rerun()

    st.divider()
    st.markdown("### ⚡ 실시간 수치 시뮬레이션")
    st.session_state.sim_age = st.slider("나이", 0, 100, st.session_state.sim_age)
    st.session_state.sim_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.sim_alb, step=0.1)
    st.session_state.sim_mental = st.selectbox("의식 반응", ["명료(Alert)", "기면(Verbal)", "혼미(Painful)"], index=["명료(Alert)", "기면(Verbal)", "혼미(Painful)"].index(st.session_state.sim_mental))
    st.session_state.sim_sbp = st.number_input("SBP (혈압)", value=st.session_state.sim_sbp)

with col_main:
    # 추론 실행
    status, score, color, is_alert, raw_p = run_inference()
    
    st.title("SNUH Smart AI Fall Dashboard")
    
    # 상단 정보 바
    st.info(f"**현재 환자:** {selected_name} | **성별:** {st.session_state.sim_sex} | **CRP:** {st.session_state.sim_crp}")

    # 디지털 계기판
    alarm_class = "alarm-active" if is_alert and not st.session_state.alarm_confirmed else ""
    st.markdown(f"""
    <div class="digital-monitor-container {alarm_class}">
        <div style="text-align:center;">
            <div class="monitor-label">RISK STATUS</div>
            <div style="color:{color}; font-weight:bold; font-size:24px;">{status}</div>
        </div>
        <div style="width:2px; height:60px; background-color:#444;"></div>
        <div style="text-align:center;">
            <div class="monitor-label">FALL SCORE</div>
            <div class="digital-number" style="color:{color};">{score}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if is_alert and not st.session_state.alarm_confirmed:
        if st.button("🚨 알람 확인 (Confirm Intervention)", use_container_width=True, type="primary"):
            st.session_state.alarm_confirmed = True
            st.rerun()

    # 시각화 차트 (XAI 대용)
    st.divider()
    st.markdown("##### 📊 주요 위험 지표 실시간 추이")
    chart_data = pd.DataFrame({
        '지표': ['SBP', 'BT', 'PR', 'Albumin', 'Age'],
        '수치': [st.session_state.sim_sbp/2, st.session_state.sim_bt*2, st.session_state.sim_pr, st.session_state.sim_alb*20, st.session_state.sim_age]
    })
    st.line_chart(chart_data.set_index('지표'))

# 고위험 팝업
if is_alert and not st.session_state.alarm_confirmed:
    st.markdown(f"""
    <div class="custom-alert-box">
        <div style="color:#ff5252; font-weight:bold; font-size:1.3em;">🚨 낙상 고위험 감지!</div>
        <p style="margin-top:10px; font-size:0.95em;">현재 환자는 <b>상위 20% 고위험군</b>에 해당합니다.<br>즉시 침상 난간을 확인하고 낙상 예방 교육을 실시하십시오.</p>
        <div style="font-size:0.8em; color:#90a4ae; margin-top:10px;">(Raw Probability: {raw_p:.6f})</div>
    </div>
    """, unsafe_allow_html=True)
