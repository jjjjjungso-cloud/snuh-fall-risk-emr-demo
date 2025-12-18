import streamlit as st
import pandas as pd
import datetime
import joblib
import numpy as np
import altair as alt

# 1. 페이지 설정 및 리소스 로딩
st.set_page_config(page_title="SNUH AI Smart Fall System", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_resources():
    try:
        return joblib.load('risk_score_model.joblib')
    except:
        return None

new_model = load_resources()

# 2. 스타일 (CSS) - 기존 껍데기 유지
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
    .note-entry { background-color: #2c3e50; padding: 12px; border-radius: 5px; border-left: 4px solid #0288d1; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# 3. 환자 데이터 및 상태 관리
PATIENTS_BASE = [
    {"name": "① 저위험 A", "gender": "F", "age": 58, "severity": 2, "sbp": 120, "dbp": 78, "pr": 78, "rr": 18, "bt": 36.6, "alb": 4.1, "crp": 0.3, "mental": "명료(Alert)"},
    {"name": "② 저위험 B", "gender": "M", "age": 72, "severity": 2, "sbp": 130, "dbp": 82, "pr": 76, "rr": 18, "bt": 36.7, "alb": 3.8, "crp": 0.8, "mental": "명료(Alert)"},
    {"name": "③ 중위험", "gender": "F", "age": 68, "severity": 3, "sbp": 115, "dbp": 75, "pr": 88, "rr": 20, "bt": 37.2, "alb": 3.0, "crp": 4.0, "mental": "기면(Verbal)"},
    {"name": "④ 고위험 (상위 20%)", "gender": "M", "age": 65, "severity": 3, "sbp": 110, "dbp": 70, "pr": 96, "rr": 22, "bt": 37.6, "alb": 2.4, "crp": 6.0, "mental": "혼미(Painful)"}
]

if 'nursing_notes' not in st.session_state: st.session_state.nursing_notes = []
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0

def reset_patient(idx):
    p = PATIENTS_BASE[idx]
    st.session_state.s_sex, st.session_state.s_age = p['gender'], p['age']
    st.session_state.s_sev, st.session_state.s_sbp = p['severity'], p['sbp']
    st.session_state.s_dbp, st.session_state.s_pr = p['dbp'], p['pr']
    st.session_state.s_rr, st.session_state.s_bt = p['rr'], p['bt']
    st.session_state.s_alb, st.session_state.s_crp = p['alb'], p['crp']
    st.session_state.s_mental = p['mental']
    st.session_state.alarm_shown = False # 팝업 중복 방지

if 's_age' not in st.session_state: reset_patient(0)

# 4. 사이드바 조작 패널 (11개 변수)
with st.sidebar:
    st.header("🏥 시뮬레이션 설정")
    sel = st.radio("환자 선택", [p['name'] for p in PATIENTS_BASE], index=st.session_state.current_idx)
    new_i = [p['name'] for p in PATIENTS_BASE].index(sel)
    if new_i != st.session_state.current_idx:
        st.session_state.current_idx = new_i
        reset_patient(new_i)
        st.rerun()

    st.divider()
    st.subheader("⚡ 실시간 수치 조작")
    st.session_state.s_sex = st.radio("성별", ["M", "F"], index=0 if st.session_state.s_sex=="M" else 1, horizontal=True)
    st.session_state.s_age = st.slider("나이", 0, 100, st.session_state.s_age)
    st.session_state.s_sev = st.select_slider("중증도(KTAS)", options=[1, 2, 3, 4, 5], value=st.session_state.s_sev)
    st.session_state.s_sbp = st.number_input("SBP", value=st.session_state.s_sbp, step=5)
    st.session_state.s_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.s_alb, step=0.1)
    st.session_state.s_crp = st.number_input("CRP", value=st.session_state.s_crp, step=0.5)
    st.session_state.s_mental = st.selectbox("의식 반응", ["명료(Alert)", "기면(Verbal)", "혼미(Painful)"], 
                                          index=["명료(Alert)", "기면(Verbal)", "혼미(Painful)"].index(st.session_state.s_mental))
    # 나머지 4개 생략 가능하나 완벽을 위해 추가 (사이드바 공간 절약 위해 columns 사용)
    c1, c2 = st.columns(2)
    with c1: st.session_state.s_dbp = st.number_input("DBP", value=st.session_state.s_dbp)
    with c2: st.session_state.s_pr = st.number_input("PR", value=st.session_state.s_pr)
    with c1: st.session_state.s_rr = st.number_input("RR", value=st.session_state.s_rr)
    with c2: st.session_state.s_bt = st.number_input("BT", value=st.session_state.s_bt, format="%.1f")

# 5. AI 추론 로직
def get_risk():
    if not new_model: return "Error", 0, False, 0
    m_map = {"명료(Alert)": 0, "기면(Verbal)": 1, "혼미(Painful)": 2}
    df = pd.DataFrame([{
        '성별': 1 if st.session_state.s_sex == 'M' else 0, '중증도분류': st.session_state.s_sev,
        'SBP': st.session_state.s_sbp, 'DBP': st.session_state.s_dbp, 'RR': st.session_state.s_rr,
        'PR': st.session_state.s_pr, 'BT': st.session_state.s_bt,
        '내원시 반응': m_map.get(st.session_state.s_mental, 0),
        '나이': st.session_state.s_age, 'albumin': st.session_state.s_alb, 'crp': st.session_state.s_crp
    }])
    prob = new_model.predict_proba(df)[0][1]
    
    # 임계값: 고위험 >= 0.025498, 중위험 >= 0.017725
    if prob >= 0.025498: return "고위험", int(85 + prob*10), True, prob
    elif prob >= 0.017725: return "중위험", int(55 + prob*15), False, prob
    else: return "저위험", int(25 + prob*15), False, prob

res_lvl, res_score, is_high, raw_prob = get_risk()

# 6. 알람 팝업 및 간호 중재 다이얼로그
@st.dialog("🚨 낙상 고위험군 즉각 중재 필요")
def show_intervention_dialog(score, prob):
    st.warning(f"환자의 낙상 위험도가 급증하였습니다. (AI Score: {score}점)")
    st.write("감지된 위험 요인에 따라 필수 간호 중재를 선택하십시오.")
    
    # 위험 요인별 중재 제안
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**필수 안전**")
        i1 = st.checkbox("침상 난간(Side Rail) 고정", value=True)
        i2 = st.checkbox("낙상 주의 표지판 부착", value=True)
    with c2:
        st.markdown("**맞춤형 케어**")
        i3 = st.checkbox("영양팀 협진 의뢰 (Albumin 저하)", value=(st.session_state.s_alb < 3.0))
        i4 = st.checkbox("수면제/이뇨제 복용 주의 교육", value=True)

    if st.button("중재 완료 및 차팅 저장", type="primary", use_container_width=True):
        selected = []
        if i1: selected.append("난간고정")
        if i2: selected.append("주의표지부착")
        if i3: selected.append("영양협진")
        if i4: selected.append("약물교육")
        
        note = f"[{datetime.datetime.now().strftime('%H:%M')}] 낙상 고위험 감지({score}점). 간호중재({', '.join(selected)}) 시행함."
        st.session_state.nursing_notes.insert(0, note)
        st.session_state.alarm_shown = True
        st.rerun()

# 7. 메인 화면 구성
st.title("🏥 SNUH AI Fall Management Workflow")
col_gauge, col_chart = st.columns([1, 2])

with col_gauge:
    # 실시간 위험도 계기판
    blink = "alarm-active" if is_high and not st.session_state.get('alarm_shown', False) else ""
    color = "#ff5252" if is_high else "#ffca28" if res_lvl=="중위험" else "#00e5ff"
    st.markdown(f"""
    <div class="digital-monitor {blink}">
        <div style="color:{color}; font-weight:bold;">{res_lvl} STATUS</div>
        <div class="digital-number" style="color:{color};">{res_score}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 팝업 트리거: 고위험군 진입 시 자동 실행
    if is_high and not st.session_state.get('alarm_shown', False):
        show_intervention_dialog(res_score, raw_prob)

with col_chart:
    st.subheader("📋 실시간 간호 기록 (EMR 연동)")
    if not st.session_state.nursing_notes:
        st.info("기록된 중재 내역이 없습니다. 수치를 조작하여 알람을 발생시켜 보세요.")
    else:
        for n in st.session_state.nursing_notes:
            st.markdown(f'<div class="note-entry">{n}</div>', unsafe_allow_html=True)

# 시각화 추가 (변수 영향력 시뮬레이션)
st.divider()
st.subheader("📊 주요 지표 실시간 분석")
v_df = pd.DataFrame({
    '항목': ['SBP', 'BT', 'Alb', 'Age', 'CRP'],
    '수치': [st.session_state.s_sbp/2, st.session_state.s_bt*2, st.session_state.s_alb*20, st.session_state.s_age, st.session_state.s_crp*10]
})
st.line_chart(v_df.set_index('항목'))
