import streamlit as st
import pandas as pd
import datetime
import joblib
import numpy as np

# 1. 페이지 설정 및 리소스 로딩
st.set_page_config(page_title="SNUH Smart AI v2.1", layout="wide")

@st.cache_resource
def load_new_model():
    try:
        # 파일명이 다르면 에러가 나므로 확인 필수 (risk_score_model.joblib)
        return joblib.load('risk_score_model.joblib')
    except Exception as e:
        st.error(f"모델 파일 로드 에러: {e}")
        return None

model = load_new_model()

# 2. 스타일 (신호등 및 알람 디자인)
st.markdown("""
<style>
    .stApp { background-color: #1e252b; color: #e0e0e0; }
    .digital-monitor {
        background-color: #000000; border-radius: 12px; padding: 25px;
        text-align: center; border: 4px solid #455a64; transition: all 0.5s;
    }
    /* 신호등 클래스 */
    .high-risk { border-color: #ff5252 !important; box-shadow: 0 0 25px #ff5252; animation: blink 1s infinite; }
    .mid-risk { border-color: #ffca28 !important; box-shadow: 0 0 15px #ffca28; }
    .low-risk { border-color: #00e5ff !important; }
    
    @keyframes blink { 50% { opacity: 0.8; } }
    .score-val { font-family: 'Consolas', monospace; font-size: 5rem; font-weight: 900; line-height: 1; }
    .note-box { background: #2c3e50; padding: 10px; border-radius: 5px; border-left: 5px solid #0288d1; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# 3. 환자 데이터 및 상태 관리
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

# 4. 사이드바 조작 패널 (11개 필수 변수)
with st.sidebar:
    st.header("🏥 시뮬레이션 설정")
    sel = st.radio("환자 선택", [p['name'] for p in PATIENTS], index=st.session_state.current_idx)
    new_idx = [p['name'] for p in PATIENTS].index(sel)
    if new_idx != st.session_state.current_idx:
        st.session_state.current_idx = new_idx
        reset_sim(new_idx)
        st.rerun()

    st.divider()
    # 11개 변수 실시간 조작
    st.session_state.v_gender = st.radio("성별", ["M", "F"], index=0 if st.session_state.v_gender=="M" else 1, horizontal=True)
    st.session_state.v_age = st.slider("나이", 0, 100, st.session_state.v_age)
    st.session_state.v_severity = st.select_slider("중증도", options=[1,2,3,4,5], value=st.session_state.v_severity)
    st.session_state.v_sbp = st.number_input("수축기 혈압 (SBP)", value=st.session_state.v_sbp)
    st.session_state.v_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.v_alb, step=0.1)
    st.session_state.v_crp = st.number_input("CRP (염증수치)", value=st.session_state.v_crp)
    st.session_state.v_mental = st.selectbox("의식상태", ["명료(Alert)", "기면(Verbal)", "혼미(Painful)"], 
                                          index=["명료(Alert)", "기면(Verbal)", "혼미(Painful)"].index(st.session_state.v_mental))
    # 보조 지표
    c1, c2 = st.columns(2)
    with c1: st.session_state.v_dbp = st.number_input("DBP", value=st.session_state.v_dbp)
    with c2: st.session_state.v_pr = st.number_input("PR", value=st.session_state.v_pr)
    with c1: st.session_state.v_rr = st.number_input("RR", value=st.session_state.v_rr)
    with c2: st.session_state.v_bt = st.number_input("BT", value=st.session_state.v_bt, format="%.1f")

# 5. [중요] AI 추론 및 등급 변환 로직
def get_prediction():
    if not model: return "Error", 0, "low-risk", "#888", 0
    
    m_map = {"명료(Alert)": 0, "기면(Verbal)": 1, "혼미(Painful)": 2}
    # 팀원 모델의 11개 피처 이름 및 순서 완벽 매핑
    input_df = pd.DataFrame([{
        '성별': 1 if st.session_state.v_gender == 'M' else 0,
        '중증도분류': st.session_state.v_severity,
        'SBP': st.session_state.v_sbp, 'DBP': st.session_state.v_dbp,
        'RR': st.session_state.v_rr, 'PR': st.session_state.v_pr, 'BT': st.session_state.v_bt,
        '내원시 반응': m_map.get(st.session_state.v_mental, 0),
        '나이': st.session_state.v_age, 'albumin': st.session_state.v_alb, 'crp': st.session_state.v_crp
    }])
    
    prob = model.predict_proba(input_df)[0][1]
    
    # [신호등 시스템 판정]
    if prob >= 0.025498: # 고위험
        level, css, color = "고위험 (상위 20%)", "high-risk", "#ff5252"
        # 점수 스케일링: 임계값을 80점으로 맵핑하여 가시성 확보
        display_score = int(80 + (prob - 0.025498) * 400) 
    elif prob >= 0.017725: # 중위험
        level, css, color = "중위험 (상위 40%)", "mid-risk", "#ffca28"
        display_score = int(50 + (prob - 0.017725) * 1000)
    else: # 저위험
        level, css, color = "저위험 (일반관리)", "low-risk", "#00e5ff"
        display_score = int(prob * 1500)
        
    return level, min(display_score, 99), css, color, prob

lvl, score, css, color, raw_p = get_prediction()

# 6. 메인 화면 및 워크플로우
st.title("🏥 SNUH AI Fall Prevention CDSS v2.1")

col_gauge, col_note = st.columns([1, 1.2])

with col_gauge:
    # 실시간 계기판
    blink_class = css if css == "high-risk" and not st.session_state.get('alarm_done', False) else css
    st.markdown(f"""
    <div class="digital-monitor {blink_class}">
        <div style="color:{color}; font-weight:bold; font-size:1.3rem; margin-bottom:10px;">{lvl}</div>
        <div class="score-val" style="color:{color};">{score}</div>
        <div style="font-size:0.8rem; color:gray; margin-top:15px;">AI Raw Prob: {raw_p:.6f}</div>
    </div>
    """, unsafe_allow_html=True)

    # [워크플로우] 고위험군 진입 시 중재 팝업
    if css == "high-risk" and not st.session_state.get('alarm_done', False):
        @st.dialog("🚨 고위험군 즉각 간호 중재")
        def intervention():
            st.warning(f"위험 요인 감지: Albumin({st.session_state.v_alb}), 의식({st.session_state.v_mental})")
            c1, c2 = st.columns(2)
            with c1:
                i1 = st.checkbox("침상 난간(Side Rail) 고정", value=True)
                i2 = st.checkbox("낙상 주의 표지판 부착", value=True)
            with c2:
                i3 = st.checkbox("영양팀 협진 의뢰", value=(st.session_state.v_alb < 3.0))
                i4 = st.checkbox("보호자 동반 교육 시행", value=True)
            
            if st.button("수행 완료 및 차팅 저장", type="primary", use_container_width=True):
                notes = []
                if i1: notes.append("난간고정")
                if i3: notes.append("영양협진")
                if i4: notes.append("보호자교육")
                
                log = f"[{datetime.datetime.now().strftime('%H:%M')}] AI 고위험 감지({score}점). 중재({', '.join(notes)}) 시행함."
                st.session_state.nursing_notes.insert(0, log)
                st.session_state.alarm_done = True
                st.rerun()
        intervention()

with col_note:
    st.subheader("📝 간호 기록 (EMR 연동)")
    if not st.session_state.nursing_notes:
        st.info("고위험 상황이 발생하면 여기에 중재 기록이 남습니다.")
    else:
        for n in st.session_state.nursing_notes:
            st.markdown(f'<div class="note-box">{n}</div>', unsafe_allow_html=True)

# 7. 변수 영향력 시각화
st.divider()
st.subheader("📊 주요 지표 실시간 시뮬레이션")
chart_data = pd.DataFrame({
    '지표': ['Age', 'Albumin', 'SBP', 'PR', 'CRP'],
    '수치': [st.session_state.v_age, st.session_state.v_alb*20, st.session_state.v_sbp/2, st.session_state.v_pr, st.session_state.v_crp*5]
}).set_index('지표')
st.bar_chart(chart_data)
