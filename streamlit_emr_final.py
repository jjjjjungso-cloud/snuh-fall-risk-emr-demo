import streamlit as st
import pandas as pd
import datetime
import time
import joblib
import numpy as np
import altair as alt

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 리소스 로딩 (기존 동일)
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
        # 모델 및 참조 데이터 로드 (파일이 있어야 작동합니다)
        res['model'] = joblib.load('risk_score_model.joblib')
        ref_data = np.load('train_score_ref.npz')
        res['ref_scores'] = ref_data['train_scores_sorted'] # 상위 % 계산용
    except Exception as e:
        st.error(f"리소스 로드 실패: {e} (모델 파일 확인 필요)")
        return None
    return res

artifacts = load_resources()

# --------------------------------------------------------------------------------
# 2. 스타일 (기존 껍데기 디자인 유지)
# --------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp { background-color: #1e252b; color: #e0e0e0; font-family: 'Noto Sans KR', sans-serif; }
    .header-container { background-color: #263238; padding: 10px 20px; border-radius: 5px; border-top: 3px solid #0288d1; margin-bottom: 10px; }
    .digital-monitor-container {
        background-color: #000000; border: 2px solid #455a64; border-radius: 8px;
        padding: 15px; margin-top: 15px; display: flex; justify-content: space-around; align-items: center;
    }
    @keyframes blink { 50% { border-color: #ff5252; box-shadow: 0 0 15px #ff5252; } }
    .alarm-active { animation: blink 1s infinite; border: 2px solid #ff5252 !important; }
    .digital-number { font-family: 'Consolas', monospace; font-size: 40px; font-weight: 900; line-height: 1.0; }
    .custom-alert-box {
        position: fixed; bottom: 30px; right: 30px; width: 380px; background-color: #263238; 
        border-left: 8px solid #ff5252; padding: 20px; z-index: 9999; border-radius: 8px;
    }
    .note-entry { background-color: #2c3e50; padding: 15px; border-radius: 5px; border-left: 4px solid #0288d1; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 3. 환자 데이터 (요청하신 A안/B안 세팅)
# --------------------------------------------------------------------------------
PATIENTS = [
    {
        "name": "Case A: 염증/영양 악화 의심", 
        "age": 65, "gender": "M", "severity": 2,
        "sbp": 120, "dbp": 80, "pr": 72, "rr": 18, "bt": 36.6,
        "alb": 4.0, "crp": 0.2, "mental": "alert", 
        "id": "2025-A65", "diag": "R/O Sepsis, Malnutrition"
    },
    {
        "name": "Case B: 고령 및 반응 저하 관찰", 
        "age": 82, "gender": "F", "severity": 2,
        "sbp": 115, "dbp": 70, "pr": 88, "rr": 20, "bt": 37.2,
        "alb": 4.0, "crp": 0.2, "mental": "alert", 
        "id": "2025-B82", "diag": "General Weakness"
    }
]

# 세션 상태 초기화
if 'pt_idx' not in st.session_state: st.session_state.pt_idx = 0
if 'nursing_notes' not in st.session_state: st.session_state.nursing_notes = []
if 'alarm_confirmed' not in st.session_state: st.session_state.alarm_confirmed = False

def update_sim(idx):
    p = PATIENTS[idx]
    st.session_state.s_age, st.session_state.s_sex = p['age'], p['gender']
    st.session_state.s_sev, st.session_state.s_sbp = p['severity'], p['sbp']
    st.session_state.s_dbp, st.session_state.s_pr = p['dbp'], p['pr']
    st.session_state.s_rr, st.session_state.s_bt = p['rr'], p['bt']
    st.session_state.s_alb, st.session_state.s_crp = p['alb'], p['crp']
    st.session_state.s_mental = p['mental']
    st.session_state.alarm_confirmed = False

if 's_age' not in st.session_state: update_sim(0)

# --------------------------------------------------------------------------------
# 4. 핵심 로직: 낙상 위험도 계산 (기존 동일)
# --------------------------------------------------------------------------------
def get_fall_risk():
    if not artifacts: return 50, 0.05 # 모델 없을 시 더미 데이터
    
    df = pd.DataFrame([{
        '성별': 1 if st.session_state.s_sex == 'M' else 0,
        '중증도분류': st.session_state.s_sev,
        'SBP': st.session_state.s_sbp, 'DBP': st.session_state.s_dbp,
        'RR': st.session_state.s_rr, 'PR': st.session_state.s_pr, 'BT': st.session_state.s_bt,
        '내원시 반응': st.session_state.s_mental,
        '나이': st.session_state.s_age, 'albumin': st.session_state.s_alb, 'crp': st.session_state.s_crp
    }])
    
    prob = artifacts['model'].predict_proba(df)[0][1]
    percentile = np.searchsorted(artifacts['ref_scores'], prob) / len(artifacts['ref_scores']) * 100
    return int(percentile), prob

# --------------------------------------------------------------------------------
# 5. 메인 레이아웃 (데이터 바인딩 최적화)
# --------------------------------------------------------------------------------
col_side, col_main = st.columns([3, 7])

with col_side:
    st.markdown("### 🏥 대상 환자 선택")
    sel_name = st.selectbox("환자 리스트", [p['name'] for p in PATIENTS], index=st.session_state.pt_idx)
    new_idx = [p['name'] for p in PATIENTS].index(sel_name)
    
    if new_idx != st.session_state.pt_idx:
        st.session_state.pt_idx = new_idx
        update_sim(new_idx)
        st.rerun()
    
    fall_score, raw_prob = get_fall_risk()
    is_high = fall_score >= 80 
    
    # 디지털 대시보드 표시
    alarm_css = "alarm-active" if is_high and not st.session_state.alarm_confirmed else ""
    f_color = "#ff5252" if is_high else ("#ffca28" if fall_score >= 60 else "#00e5ff")
    
    st.markdown(f"""
    <div class="digital-monitor-container {alarm_css}">
        <div style="text-align:center; width:100%;">
            <div style="color:#90a4ae; font-size:12px; font-weight:bold;">FALL RISK SCORE (Percentile)</div>
            <div class="digital-number" style="color:{f_color};">{fall_score}</div>
            <div style="color:{f_color}; font-size:14px; font-weight:bold;">
                {"⚠️ HIGH RISK (TOP 20%)" if is_high else "✅ STABLE"}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### ⚡ 실시간 수치 조정")
    # 아래 슬라이더들을 조절하며 상위 20% 진입 여부를 테스트할 수 있습니다.
    st.session_state.s_age = st.slider("나이 (Age)", 0, 100, st.session_state.s_age)
    st.session_state.s_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.s_alb, step=0.1)
    st.session_state.s_crp = st.number_input("CRP", value=st.session_state.s_crp, step=0.1)
    st.session_state.s_mental = st.selectbox("의식 상태 (Mental)", ["alert", "verbal response", "painful response", "unresponsive"], 
                                         index=["alert", "verbal response", "painful response", "unresponsive"].index(st.session_state.s_mental))

with col_main:
    curr_p = PATIENTS[st.session_state.pt_idx]
    st.markdown(f"""
    <div class="header-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div><span style="font-size:1.5em; font-weight:bold; color:white;">🏥 AI Fall Risk Monitor</span>
            <span style="margin-left:20px;"><b>{curr_p['name']}</b></span></div>
            <div style="color:#b0bec5;">ID: {curr_p['id']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["📊 시뮬레이션 분석", "📝 간호기록"])
    
    with t1:
        st.subheader("AI 예측 근거")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("현재 예측 확률 (Raw Prob)", f"{raw_prob:.4f}")
            st.caption("모델이 계산한 0~1 사이의 원시 확률값입니다.")
        with c2:
            st.metric("전체 대비 위험 순위", f"상위 {100-fall_score}%")
            st.caption("기존 학습 데이터셋과 비교한 상대적 위험도입니다.")
            
        st.info(f"**임상적 제언:** 현재 환자는 {st.session_state.s_age}세이며, Albumin {st.session_state.s_alb}인 상태입니다. "
                "수치를 미세하게 조정하여 'High Risk' 알람이 발생하는 임계점을 확인해보세요.")

    with t2:
        if st.button("📝 현재 상태 기록 남기기"):
            note = f"[{datetime.datetime.now().strftime('%H:%M')}] 낙상 위험 점수 {fall_score}점 확인. (Alb:{st.session_state.s_alb}, CRP:{st.session_state.s_crp})"
            st.session_state.nursing_notes.insert(0, {"time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), "content": note})
        
        for note in st.session_state.nursing_notes:
            st.markdown(f'<div class="note-entry"><small>{note["time"]}</small><br>{note["content"]}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 6. 고정 알람 박스
# --------------------------------------------------------------------------------
if is_high and not st.session_state.alarm_confirmed:
    st.markdown(f"""
    <div class="custom-alert-box">
        <div style="color:#ff5252; font-weight:bold; font-size:1.2em;">🚨 낙상 고위험 감지!</div>
        <div style="font-size:0.9em; margin-top:10px;">환자의 수치가 낙상 고위험군(상위 20%)에 도달했습니다. <b>침대 난간 확인 및 낙상 예방 간호</b>가 필요합니다.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚨 중재 완료 (Confirm Alarm)"):
        st.session_state.alarm_confirmed = True
        st.rerun()
