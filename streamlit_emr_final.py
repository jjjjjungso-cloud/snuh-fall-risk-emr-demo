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
    page_title="SNUH Ward EMR - AI System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_resources():
    res = {}
    try:
        # 모델 및 백분위 참조 데이터 로드
        res['model'] = joblib.load('risk_score_model.joblib')
        ref_data = np.load('train_score_ref.npz')
        # 파일 내 키값 확인 후 로드 (일반적으로 첫번째 키 사용)
        key = list(ref_data.keys())[0]
        res['ref_scores'] = ref_data[key]
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return None
    return res

artifacts = load_resources()

# --------------------------------------------------------------------------------
# 2. 스타일 (기존 껍데기 UI - 다크모드 & 디지털 계기판)
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
    .digital-number { font-family: 'Consolas', monospace; font-size: 42px; font-weight: 900; line-height: 1.0; }

    /* 알람 박스 */
    .custom-alert-box {
        position: fixed; bottom: 30px; right: 30px; width: 380px; height: auto;
        background-color: #263238; border-left: 8px solid #ff5252;
        box-shadow: 0 6px 25px rgba(0,0,0,0.7); border-radius: 8px; padding: 20px; z-index: 9999;
    }
    
    .note-entry { background-color: #2c3e50; padding: 15px; border-radius: 5px; border-left: 4px solid #0288d1; margin-bottom: 10px; }
    .risk-tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; margin: 2px; border: 1px solid #ff5252; color: #ff867c; }
    
    div.stButton > button { width: 100%; background-color: #d32f2f; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 3. 환자 데이터 및 상태 관리 (수정안 A/B안 적용)
# --------------------------------------------------------------------------------
PATIENTS = [
    {
        "name": "A안: 염증/영양 악화 케이스", "age": 65, "gender": "M", "severity": 2,
        "sbp": 120, "dbp": 80, "pr": 75, "rr": 18, "bt": 36.5,
        "alb": 4.0, "crp": 0.2, "mental": "alert", "id": "12345678", "diag": "Pneumonia R/O"
    },
    {
        "name": "B안: 고령/반응 저하 케이스", "age": 82, "gender": "F", "severity": 2,
        "sbp": 125, "dbp": 80, "pr": 85, "rr": 20, "bt": 37.0,
        "alb": 4.0, "crp": 0.2, "mental": "alert", "id": "87654321", "diag": "General Weakness"
    }
]

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
# 4. 리스크 계산 & 요인 감지 로직
# --------------------------------------------------------------------------------
def get_risk_analysis():
    if not artifacts: return 0, 0, []
    
    # 11개 피처 정렬
    df = pd.DataFrame([{
        '성별': 1 if st.session_state.s_sex == 'M' else 0,
        '중증도분류': st.session_state.s_sev,
        'SBP': st.session_state.s_sbp, 'DBP': st.session_state.s_dbp,
        'RR': st.session_state.s_rr, 'PR': st.session_state.s_pr, 'BT': st.session_state.s_bt,
        '내원시 반응': st.session_state.s_mental,
        '나이': st.session_state.s_age, 'albumin': st.session_state.s_alb, 'crp': st.session_state.s_crp
    }])
    
    prob = artifacts['model'].predict_proba(df)[0][1]
    # 백분위 점수 계산
    score = int(np.searchsorted(artifacts['ref_scores'], prob) / len(artifacts['ref_scores']) * 100)
    
    # 위험 요인 감지 (껍데기 UI용)
    factors = []
    if st.session_state.s_alb < 3.0: factors.append("알부민 저하")
    if st.session_state.s_crp > 5.0: factors.append("염증 수치 상승")
    if st.session_state.s_mental != 'alert': factors.append("의식 상태 변화")
    if st.session_state.s_age >= 75: factors.append("고령(고위험)")
    if st.session_state.s_sbp < 100: factors.append("저혈압 경향")
    
    return score, prob, factors

fall_score, raw_prob, detected_factors = get_risk_analysis()
is_high_risk = fall_score >= 80

# --------------------------------------------------------------------------------
# 5. 메인 레이아웃 구성
# --------------------------------------------------------------------------------
col_side, col_main = st.columns([2, 8])

with col_side:
    st.markdown("### 🏥 재원 환자 리스트")
    sel_name = st.radio("선택", [p['name'] for p in PATIENTS], index=st.session_state.pt_idx, label_visibility="collapsed")
    new_idx = [p['name'] for p in PATIENTS].index(sel_name)
    if new_idx != st.session_state.pt_idx:
        st.session_state.pt_idx = new_idx
        update_sim(new_idx)
        st.rerun()

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
    st.markdown("### ⚡ 실시간 수치 조작")
    # 데모용 주요 변수 슬라이더
    st.session_state.s_alb = st.slider("Albumin (영양)", 1.0, 5.0, st.session_state.s_alb, step=0.1)
    st.session_state.s_crp = st.number_input("CRP (염증)", value=st.session_state.s_crp, step=0.5)
    st.session_state.s_mental = st.selectbox("의식 상태", ["alert", "verbal response", "painful response", "unresponsive"], 
                                          index=["alert", "verbal response", "painful response", "unresponsive"].index(st.session_state.s_mental))
    st.session_state.s_age = st.slider("나이", 0, 100, st.session_state.s_age)
    st.session_state.s_sev = st.select_slider("중증도분류", options=[1, 2, 3, 4, 5], value=st.session_state.s_sev)

with col_main:
    curr_p = PATIENTS[st.session_state.pt_idx]
    # 헤더
    st.markdown(f"""
    <div class="header-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div><span style="font-size:1.5em; font-weight:bold; color:white;">🏥 SNUH Ward AI</span>
            <span style="margin-left:20px;"><b>{curr_p['name']}</b> ({st.session_state.s_sex}/{st.session_state.s_age}세)</span></div>
            <div style="color:#b0bec5; font-size:0.9em;">ID: {curr_p['id']} | 진단: {curr_p['diag']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["🛡️ AI Simulation View", "💊 오더", "📝 간호기록(Auto-Note)"])
    
    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("##### 🔍 실시간 감지 리스크 요인")
            if detected_factors:
                for f in detected_factors:
                    st.markdown(f"<span class='risk-tag'>{f}</span>", unsafe_allow_html=True)
            else:
                st.info("현재 특이 위험 요인 없음")
            
            st.markdown("---")
            st.markdown("##### 📊 판단 근거 (Raw Probability)")
            st.code(f"AI 확률값: {raw_prob:.6f}\n위험군 판정: {'고위험군 진입' if is_high_risk else '안정권'}")
        
        with c2:
            st.markdown("##### ✅ 추천 간호 중재 (맞춤형)")
            # 요인에 따른 중재 자동 체크박스
            chk_rail = st.checkbox("침상 난간(Side Rail) 고정 확인", value=is_high_risk)
            chk_edu = st.checkbox("낙상 예방 교육 및 호출기 위치 안내", value=True)
            chk_nutri = st.checkbox("🥩 영양팀 협진 의뢰", value=("알부민 저하" in detected_factors))
            chk_round = st.checkbox("🕒 1시간 간격 집중 라운딩", value=("의식 상태 변화" in detected_factors))

            if st.button("간호 수행 완료 및 EMR 저장", type="primary"):
                actions = []
                if chk_rail: actions.append("난간고정")
                if chk_edu: actions.append("예방교육")
                if chk_nutri: actions.append("영양협진")
                if chk_round: actions.append("집중모니터링")
                
                note_text = f"[{datetime.datetime.now().strftime('%H:%M')}] 낙상위험평가({fall_score}점). 요인({', '.join(detected_factors)}) 확인되어 중재({', '.join(actions)}) 시행함."
                st.session_state.nursing_notes.insert(0, {"time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), "content": note_text})
                st.toast("기록이 저장되었습니다.")
                st.rerun()

    with t3:
        for note in st.session_state.nursing_notes:
            st.markdown(f'<div class="note-entry"><small>{note["time"]}</small><br>{note["content"]}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 6. 하단 고정 알람 박스
# --------------------------------------------------------------------------------
if is_high_risk and not st.session_state.alarm_confirmed:
    f_str = "<br>• ".join(detected_factors) if detected_factors else "복합적 요인"
    st.markdown(f"""
    <div class="custom-alert-box">
        <div style="color:#ff5252; font-weight:bold; font-size:1.2em;">🚨 낙상 고위험 감지! ({fall_score}점)</div>
        <div style="font-size:0.95em; margin-top:10px; color:#eceff1;">상태 변화로 인해 <b>상위 20% 고위험군</b>에 진입했습니다.</div>
        <div style="background:#3e2723; padding:10px; border-radius:5px; margin-top:10px; color:#ffcdd2; font-size:0.9em; border:1px solid #ff5252;">
            <b>[주요 위험 요인]</b><br>• {f_str}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚨 알람 확인 (Confirm Intervention)", key="confirm_btn"):
        st.session_state.alarm_confirmed = True
        st.rerun()
