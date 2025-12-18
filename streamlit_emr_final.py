import streamlit as st
import pandas as pd
import datetime
import joblib
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="SNUH AI Fall Dashboard v2", layout="wide")

# 2. 모델 로드
@st.cache_resource
def load_model():
    try:
        # 파일명이 정확히 일치해야 합니다.
        return joblib.load('risk_score_model.joblib')
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

model = load_model()

# 3. 스타일 정의 (신호등 및 알람)
st.markdown("""
<style>
    .stApp { background-color: #1e252b; color: #e0e0e0; }
    .digital-monitor {
        background-color: #000000; border-radius: 12px; padding: 25px;
        text-align: center; border: 4px solid #455a64;
        transition: all 0.5s;
    }
    /* 신호등 효과 */
    .high-risk { border-color: #ff5252 !important; box-shadow: 0 0 20px #ff5252; animation: blink 1s infinite; }
    .mid-risk { border-color: #ffca28 !important; }
    .low-risk { border-color: #00e5ff !important; }
    @keyframes blink { 50% { opacity: 0.7; } }
    .digital-number { font-family: 'Consolas', monospace; font-size: 5rem; font-weight: 900; line-height: 1; }
</style>
""", unsafe_allow_html=True)

# 4. 환자 데이터 세팅
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
    for k, v in p.items(): st.session_state[f"s_{k}"] = v
    st.session_state.alarm_done = False

if 's_age' not in st.session_state: reset_sim(0)

# 5. 사이드바 조작 (11개 변수)
with st.sidebar:
    st.header("🏥 시뮬레이션")
    sel = st.radio("환자 선택", [p['name'] for p in PATIENTS], index=st.session_state.current_idx)
    new_idx = [p['name'] for p in PATIENTS].index(sel)
    if new_idx != st.session_state.current_idx:
        st.session_state.current_idx = new_idx
        reset_sim(new_idx)
        st.rerun()
    
    st.divider()
    st.session_state.s_age = st.slider("나이", 0, 100, st.session_state.s_age)
    st.session_state.s_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.s_alb, step=0.1)
    st.session_state.s_mental = st.selectbox("반응", ["명료(Alert)", "기면(Verbal)", "혼미(Painful)"], 
                                          index=["명료(Alert)", "기면(Verbal)", "혼미(Painful)"].index(st.session_state.s_mental))
    st.session_state.s_sbp = st.number_input("SBP", value=st.session_state.s_sbp)
    st.session_state.s_crp = st.number_input("CRP", value=st.session_state.s_crp)
    st.session_state.s_severity = st.selectbox("중증도", [1,2,3,4,5], index=st.session_state.s_severity-1)
    # 나머지 5개 변수 (모델 입력을 위해 필요)
    st.session_state.s_gender = st.radio("성별", ["M", "F"], index=0 if st.session_state.s_gender=="M" else 1, horizontal=True)
    st.session_state.s_dbp = st.number_input("DBP", value=st.session_state.s_dbp)
    st.session_state.s_pr = st.number_input("PR", value=st.session_state.s_pr)
    st.session_state.s_rr = st.number_input("RR", value=st.session_state.s_rr)
    st.session_state.s_bt = st.number_input("BT", value=st.session_state.s_bt, format="%.1f")

# 6. 추론 로직 및 신호등 판정
def run_model():
    if not model: return "Error", 0, "low-risk", "#888", 0
    
    m_map = {"명료(Alert)": 0, "기면(Verbal)": 1, "혼미(Painful)": 2}
    # 팀원 모델의 11개 피처 이름 및 순서 (중요: 학습 데이터와 일치해야 함)
    input_df = pd.DataFrame([{
        '성별': 1 if st.session_state.s_gender == 'M' else 0,
        '중증도분류': st.session_state.s_severity,
        'SBP': st.session_state.s_sbp, 'DBP': st.session_state.s_dbp,
        'RR': st.session_state.s_rr, 'PR': st.session_state.s_pr, 'BT': st.session_state.s_bt,
        '내원시 반응': m_map.get(st.session_state.s_mental, 0),
        '나이': st.session_state.s_age, 'albumin': st.session_state.s_alb, 'crp': st.session_state.s_crp
    }])
    
    prob = model.predict_proba(input_df)[0][1] # 낙상군(1)일 확률
    
    # [3단계 신호등 판정 기준]
    if prob >= 0.025498: # 고위험
        return "상위 20% (고위험)", int(85 + prob*10), "high-risk", "#ff5252", prob
    elif prob >= 0.017725: # 중위험
        return "상위 40% (중위험)", int(55 + prob*15), "mid-risk", "#ffca28", prob
    else: # 저위험
        return "일반군 (저위험)", int(20 + prob*15), "low-risk", "#00e5ff", prob

lvl, score, css_class, color, raw_p = run_model()

# 7. 메인 화면 및 팝업 중재
st.title("🏥 SNUH AI Fall Prevention CDSS")

c1, c2 = st.columns([1, 1.5])
with c1:
    st.markdown(f"""
    <div class="digital-monitor {css_class}">
        <div style="color:{color}; font-weight:bold; font-size:1.2rem;">{lvl}</div>
        <div class="digital-number" style="color:{color};">{score}</div>
        <div style="font-size:0.8rem; color:gray; margin-top:10px;">Raw Prob: {raw_p:.6f}</div>
    </div>
    """, unsafe_allow_html=True)

    # 고위험군 진입 시 중재 다이얼로그
    if css_class == "high-risk" and not st.session_state.get('alarm_done', False):
        @st.dialog("🚨 고위험 중재 가이드")
        def show_guide():
            st.error(f"낙상 위험 점수 {score}점 감지!")
            i1 = st.checkbox("침상 난간(Side Rail) 고정", value=True)
            i2 = st.checkbox("영양팀 협진 의뢰 (Albumin 저하)", value=(st.session_state.s_alb < 3.0))
            if st.button("수행 완료 및 차팅"):
                note = f"[{datetime.datetime.now().strftime('%H:%M')}] AI 고위험 감지({score}점). 난간고정/영양협진 시행함."
                st.session_state.nursing_notes.insert(0, note)
                st.session_state.alarm_done = True
                st.rerun()
        show_guide()

with c2:
    st.subheader("📝 실시간 간호 기록 (Auto-Charting)")
    for n in st.session_state.nursing_notes:
        st.markdown(f'<div style="background:#2c3e50; padding:10px; border-radius:5px; margin-bottom:5px;">{n}</div>', unsafe_allow_html=True)

# 시뮬레이션 데이터 확인용 (디버깅)
with st.expander("🔍 모델 입력 데이터 확인 (점수가 안 바뀔 때 확인하세요)"):
    st.write("현재 모델로 전송되는 데이터 프레임:")
    # 위에서 정의한 input_df를 다시 보여줌
    m_map = {"명료(Alert)": 0, "기면(Verbal)": 1, "혼미(Painful)": 2}
    debug_df = pd.DataFrame([{
        '성별': 1 if st.session_state.s_gender == 'M' else 0, '중증도분류': st.session_state.s_severity,
        'SBP': st.session_state.s_sbp, 'DBP': st.session_state.s_dbp, 'RR': st.session_state.s_rr,
        'PR': st.session_state.s_pr, 'BT': st.session_state.s_bt,
        '내원시 반응': m_map.get(st.session_state.s_mental, 0),
        '나이': st.session_state.s_age, 'albumin': st.session_state.s_alb, 'crp': st.session_state.s_crp
    }])
    st.table(debug_df)
