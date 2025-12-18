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
    page_title="SNUH Ward EMR - AI Fall System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_resources():
    res = {}
    try:
        # 모델 및 백분위 계산용 참조 데이터 로드
        res['model'] = joblib.load('risk_score_model.joblib')
        ref_data = np.load('train_score_ref.npz')
        # .npz 파일 내부의 키값 확인 (일반적으로 'train_scores_sorted.npy')
        res['ref_scores'] = ref_data['train_scores_sorted.npy'] 
    except Exception as e:
        st.error(f"리소스 로드 실패 (파일 확인 필요): {e}")
        return None
    return res

artifacts = load_resources()

# --------------------------------------------------------------------------------
# 2. 스타일 (CSS) - 기존 껍데기 디자인 유지
# --------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp { background-color: #1e252b; color: #e0e0e0; font-family: 'Noto Sans KR', sans-serif; }

    /* 헤더 */
    .header-container {
        background-color: #263238; padding: 10px 20px; border-radius: 5px;
        border-top: 3px solid #0288d1; box-shadow: 0 2px 5px rgba(0,0,0,0.3); margin-bottom: 10px;
    }

    /* 디지털 계기판 */
    .digital-monitor-container {
        background-color: #000000; border: 2px solid #455a64; border-radius: 8px;
        padding: 15px; margin-top: 15px; display: flex; justify-content: space-around; align-items: center;
    }
    @keyframes blink { 50% { border-color: #ff5252; box-shadow: 0 0 15px #ff5252; } }
    .alarm-active { animation: blink 1s infinite; border: 2px solid #ff5252 !important; }
    .digital-number { font-family: 'Consolas', monospace; font-size: 40px; font-weight: 900; line-height: 1.0; }

    /* 하단 알람 박스 */
    .custom-alert-box {
        position: fixed; bottom: 30px; right: 30px; width: 380px; height: auto;
        background-color: #263238; border-left: 8px solid #ff5252;
        box-shadow: 0 6px 25px rgba(0,0,0,0.7); border-radius: 8px; padding: 20px; z-index: 9999;
    }
    
    .note-entry { background-color: #2c3e50; padding: 15px; border-radius: 5px; border-left: 4px solid #0288d1; margin-bottom: 10px; }
    div.stButton > button { width: 100%; background-color: #d32f2f; color: white; font-weight: bold; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 3. 환자 데이터 (데모용 수정안 A/B 세팅)
# --------------------------------------------------------------------------------
PATIENTS = [
    {
        "name": "A안: 염증/영양 악화 케이스", "age": 65, "gender": "M", "severity": 2,
        "sbp": 120, "dbp": 80, "pr": 75, "rr": 18, "bt": 36.5,
        "alb": 4.0, "crp": 0.2, "mental": "alert", "id": "12345678", "diag": "Pneumonia R/O"
    },
    {
        "name": "B안: 고령/반응 저하 케이스", "age": 82, "gender": "F", "severity": 3,
        "sbp": 125, "dbp": 80, "pr": 85, "rr": 20, "bt": 37.0,
        "alb": 3.2, "crp": 3.0, "mental": "verbal response", "id": "87654321", "diag": "General Weakness"
    }
]

if 'pt_idx' not in st.session_state: st.session_state.pt_idx = 0
if 'nursing_notes' not in st.session_state: st.session_state.nursing_notes = []
if 'alarm_confirmed' not in st.session_state: st.session_state.alarm_confirmed = False

def update_simulation_values(idx):
    p = PATIENTS[idx]
    st.session_state.s_age, st.session_state.s_sex = p['age'], p['gender']
    st.session_state.s_sev, st.session_state.s_sbp = p['severity'], p['sbp']
    st.session_state.s_dbp, st.session_state.s_pr = p['dbp'], p['pr']
    st.session_state.s_rr, st.session_state.s_bt = p['rr'], p['bt']
    st.session_state.s_alb, st.session_state.s_crp = p['alb'], p['crp']
    st.session_state.s_mental = p['mental']
    st.session_state.alarm_confirmed = False

if 's_age' not in st.session_state: update_simulation_values(0)

# --------------------------------------------------------------------------------
# 4. 핵심 로직: 백분위 기반 낙상 위험도 계산
# --------------------------------------------------------------------------------
def calculate_fall_risk():
    if not artifacts: return 0, 0
    
    # 11개 피처 입력 구성
    input_df = pd.DataFrame([{
        '성별': 1 if st.session_state.s_sex == 'M' else 0,
        '중증도분류': st.session_state.s_sev,
        'SBP': st.session_state.s_sbp, 'DBP': st.session_state.s_dbp,
        'RR': st.session_state.s_rr, 'PR': st.session_state.s_pr, 'BT': st.session_state.s_bt,
        '내원시 반응': st.session_state.s_mental,
        '나이': st.session_state.s_age, 'albumin': st.session_state.s_alb, 'crp': st.session_state.s_crp
    }])
    
    prob = artifacts['model'].predict_proba(input_df)[0][1]
    
    # 백분위 계산 (0~100점 스케일링)
    # 전체 환자 분포 중 현재 환자의 확률보다 낮은 데이터의 비율을 점수화
    percentile = np.searchsorted(artifacts['ref_scores'], prob) / len(artifacts['ref_scores']) * 100
    
    return int(percentile), prob

# --------------------------------------------------------------------------------
# 5. 메인 레이아웃
# --------------------------------------------------------------------------------
col_side, col_main = st.columns([2, 8])

with col_side:
    st.markdown("### 🏥 환자 리스트")
    sel_name = st.radio("환자", [p['name'] for p in PATIENTS], index=st.session_state.pt_idx, label_visibility="collapsed")
    new_idx = [p['name'] for p in PATIENTS].index(sel_name)
    if new_idx != st.session_state.pt_idx:
        st.session_state.pt_idx = new_idx
        update_simulation_values(new_idx)
        st.rerun()
    
    # 리스크 계산 실행
    fall_score, raw_prob = calculate_fall_risk()
    is_high_risk = fall_score >= 80 # 상위 20% 진입 시

    # 디지털 계기판
    alarm_css = "alarm-active" if is_high_risk and not st.session_state.alarm_confirmed else ""
    f_color = "#ff5252" if is_high_risk else ("#ffca28" if fall_score >= 60 else "#00e5ff")
    
    st.markdown(f"""
    <div class="digital-monitor-container {alarm_css}">
        <div style="text-align:center; width:100%;">
            <div style="color:#90a4ae; font-size:12px; font-weight:bold;">FALL RISK SCORE</div>
            <div class="digital-number" style="color:{f_color};">{fall_score}</div>
            <div style="color:{f_color}; font-size:12px; font-weight:bold;">{"TOP 20% (HIGH)" if is_high_risk else "NORMAL"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### ⚡ 실시간 수치 조정")
    st.session_state.s_age = st.slider("나이", 0, 100, st.session_state.s_age)
    st.session_state.s_sev = st.select_slider("중증도", options=[1, 2, 3, 4, 5], value=st.session_state.s_sev)
    st.session_state.s_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.s_alb, step=0.1)
    st.session_state.s_crp = st.number_input("CRP", value=st.session_state.s_crp, step=0.5)
    st.session_state.s_mental = st.selectbox("의식 상태", ["alert", "verbal response", "painful response", "unresponsive"], 
                                          index=["alert", "verbal response", "painful response", "unresponsive"].index(st.session_state.s_mental))

with col_main:
    curr_p = PATIENTS[st.session_state.pt_idx]
    st.markdown(f"""
    <div class="header-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div><span style="font-size:1.5em; font-weight:bold; color:white;">🏥 SNUH Ward AI</span>
            <span style="margin-left:20px;"><b>{curr_p['name']}</b> ({st.session_state.s_sex}/{st.session_state.s_age}세)</span></div>
            <div style="color:#b0bec5; font-size:0.9em;">ID: {curr_p['id']} | 진단: {curr_p['diag']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["🛡️ AI Simulation View", "📝 간호기록(Auto-Charting)"])
    
    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("##### 🔍 입력 데이터 요약")
            st.write(f"• **Vital Signs:** {st.session_state.s_sbp}/{st.session_state.s_dbp} - {st.session_state.s_pr} - {st.session_state.s_bt}℃")
            st.write(f"• **Lab Results:** Albumin {st.session_state.s_alb} / CRP {st.session_state.s_crp}")
            st.write(f"• **Consciousness:** {st.session_state.s_mental}")
        with c2:
            st.markdown("##### 📊 판단 근거 (Percentile)")
            st.info(f"AI 확률: **{raw_prob:.6f}**\n\n현재 전체 재원 환자 중 **상위 {100-fall_score}%**에 해당하는 위험도입니다.")

    with t2:
        for note in st.session_state.nursing_notes:
            st.markdown(f'<div class="note-entry"><small>{note["time"]}</small><br>{note["content"]}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 6. 고정 알람 박스 & 중재 워크플로우
# --------------------------------------------------------------------------------
if is_high_risk and not st.session_state.alarm_confirmed:
    st.markdown(f"""
    <div class="custom-alert-box">
        <div style="color:#ff5252; font-weight:bold; font-size:1.2em;">🚨 낙상 고위험군 감지! ({fall_score}점)</div>
        <div style="font-size:0.9em; margin-top:10px;">상태 변화로 인해 위험도가 <b>상위 20%</b> 이내로 급격히 상승했습니다. 즉시 중재를 시행하십시오.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚨 중재 수행 및 확인 (EMR 자동 기록)"):
        note_text = f"[{datetime.datetime.now().strftime('%H:%M')}] 낙상 고위험 감지({fall_score}점). 상기 환자 상태 변화(Albumin {st.session_state.s_alb}, CRP {st.session_state.s_crp})에 따라 침상난간 확인 및 낙상 예방 교육 재시행함."
        st.session_state.nursing_notes.insert(0, {"time": datetime.datetime.now().strftime('%H:%M'), "content": note_text})
        st.session_state.alarm_confirmed = True
        st.rerun()
