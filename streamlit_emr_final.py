import streamlit as st
import pandas as pd
import datetime
import joblib
import numpy as np
import altair as alt

# 1. 페이지 설정 및 리소스 로딩
st.set_page_config(
    page_title="SNUH Ward EMR - AI Fall System v2.1",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_new_brain():
    try:
        # imblearn 파이프라인 대응을 위해 joblib으로 로드
        # requirements.txt에 imbalanced-learn이 반드시 포함되어야 합니다.
        model = joblib.load('risk_score_model.joblib')
        return model
    except Exception as e:
        st.error(f"❌ 모델 로드 에러: {e}")
        return None

new_model = load_new_brain()

# 2. 세련된 병원 EMR 스타일 (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp { background-color: #1e252b; color: #e0e0e0; font-family: 'Noto Sans KR', sans-serif; }

    /* 디지털 계기판 디자인 */
    .digital-monitor {
        background-color: #000000; border: 2px solid #455a64; border-radius: 12px;
        padding: 25px; text-align: center; box-shadow: inset 0 0 20px rgba(0,0,0,0.9);
        transition: all 0.5s ease;
    }
    
    /* 신호등 시스템 클래스 */
    .high-risk { border: 4px solid #ff5252 !important; box-shadow: 0 0 25px #ff5252; animation: blink 1s infinite; }
    .mid-risk { border: 4px solid #ffca28 !important; box-shadow: 0 0 15px #ffca28; }
    .low-risk { border: 4px solid #00e5ff !important; }
    
    @keyframes blink { 50% { opacity: 0.7; } }
    .digital-number { font-family: 'Consolas', monospace; font-size: 5rem; font-weight: 900; line-height: 1.0; }
    .status-label { font-size: 1.2rem; font-weight: bold; margin-bottom: 10px; }

    /* 간호 기록 박스 */
    .note-entry { background-color: #2c3e50; padding: 15px; border-radius: 5px; border-left: 5px solid #0288d1; margin-bottom: 10px; }
    .header-container { background-color: #263238; padding: 15px; border-radius: 8px; border-top: 4px solid #0288d1; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# 3. 시연용 환자 데이터 및 세션 상태 관리
PATIENTS_BASE = [
    {"name": "① 저위험 A (정상군)", "gender": "F", "age": 58, "severity": 2, "sbp": 120, "dbp": 78, "pr": 78, "rr": 18, "bt": 36.6, "alb": 4.1, "crp": 0.3, "mental": "명료(Alert)"},
    {"name": "② 저위험 B (정상-고령)", "gender": "M", "age": 72, "severity": 2, "sbp": 130, "dbp": 82, "pr": 76, "rr": 18, "bt": 36.7, "alb": 3.8, "crp": 0.8, "mental": "명료(Alert)"},
    {"name": "③ 중위험 (경계/관찰)", "gender": "F", "age": 68, "severity": 3, "sbp": 115, "dbp": 75, "pr": 88, "rr": 20, "bt": 37.2, "alb": 3.0, "crp": 4.0, "mental": "기면(Verbal)"},
    {"name": "④ 고위험 (상위 20%)", "gender": "M", "age": 65, "severity": 3, "sbp": 110, "dbp": 70, "pr": 96, "rr": 22, "bt": 37.6, "alb": 2.4, "crp": 6.0, "mental": "혼미(Painful)"}
]

if 'nursing_notes' not in st.session_state: st.session_state.nursing_notes = []
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0

def reset_to_patient(idx):
    p = PATIENTS_BASE[idx]
    st.session_state.v_gender, st.session_state.v_age = p['gender'], p['age']
    st.session_state.v_sev, st.session_state.v_sbp = p['severity'], p['sbp']
    st.session_state.v_dbp, st.session_state.v_pr = p['dbp'], p['pr']
    st.session_state.v_rr, st.session_state.v_bt = p['rr'], p['bt']
    st.session_state.v_alb, st.session_state.v_crp = p['alb'], p['crp']
    st.session_state.v_mental = p['mental']
    st.session_state.alarm_done = False # 새로운 환자 선택 시 알람 리셋

if 'v_age' not in st.session_state: reset_to_patient(0)

# 4. 사이드바: 11개 입력 변수 시뮬레이션
with st.sidebar:
    st.header("🏥 시뮬레이션 설정")
    selected_p = st.radio("환자 선택", [p['name'] for p in PATIENTS_BASE], index=st.session_state.current_idx)
    new_i = [p['name'] for p in PATIENTS_BASE].index(selected_p)
    
    if new_i != st.session_state.current_idx:
        st.session_state.current_idx = new_i
        reset_to_patient(new_i)
        st.rerun()

    st.divider()
    st.subheader("⚡ 11개 실시간 지표 조정")
    st.session_state.v_gender = st.radio("성별", ["M", "F"], index=0 if st.session_state.v_gender=="M" else 1, horizontal=True)
    st.session_state.v_age = st.slider("나이", 0, 100, st.session_state.v_age)
    st.session_state.v_sev = st.select_slider("중증도분류(KTAS)", options=[1, 2, 3, 4, 5], value=st.session_state.v_sev)
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.session_state.v_sbp = st.number_input("SBP", value=st.session_state.v_sbp, step=5)
        st.session_state.v_pr = st.number_input("PR", value=st.session_state.v_pr, step=5)
        st.session_state.v_bt = st.number_input("BT", value=st.session_state.v_bt, step=0.1, format="%.1f")
    with col_v2:
        st.session_state.v_dbp = st.number_input("DBP", value=st.session_state.v_dbp, step=5)
        st.session_state.v_rr = st.number_input("RR", value=st.session_state.v_rr, step=2)
        st.session_state.v_crp = st.number_input("CRP", value=st.session_state.v_crp, step=0.5)

    st.session_state.v_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.v_alb, step=0.1)
    st.session_state.v_mental = st.selectbox("내원시 반응", ["명료(Alert)", "기면(Verbal)", "혼미(Painful)"], 
                                          index=["명료(Alert)", "기면(Verbal)", "혼미(Painful)"].index(st.session_state.v_mental))

# 5. 핵심 로직: 11개 피처 AI 추론 및 등급 변환 (Scaling 적용)
def get_ai_prediction():
    if new_model is None: return "Error", 0, "low-risk", "#888", 0
    
    m_map = {"명료(Alert)": 0, "기면(Verbal)": 1, "혼미(Painful)": 2}
    # 팀원 모델의 11개 피처 순서와 이름 완벽 일치
    df = pd.DataFrame([{
        '성별': 1 if st.session_state.v_gender == 'M' else 0,
        '중증도분류': st.session_state.v_sev,
        'SBP': st.session_state.v_sbp, 'DBP': st.session_state.v_dbp,
        'RR': st.session_state.v_rr, 'PR': st.session_state.v_pr, 'BT': st.session_state.v_bt,
        '내원시 반응': m_map.get(st.session_state.v_mental, 0),
        '나이': st.session_state.v_age, 'albumin': st.session_state.v_alb, 'crp': st.session_state.v_crp
    }])
    
    try:
        prob = new_model.predict_proba(df)[0][1]
        
        # [신호등 시스템 판정 및 점수 보정]
        if prob >= 0.025498: # 고위험 상위 20%
            level, css, color = "고위험 (상위 20%)", "high-risk", "#ff5252"
            display_score = int(80 + (prob - 0.025498) * 400) # 80~99점대로 맵핑
        elif prob >= 0.017725: # 중위험 상위 40%
            level, css, color = "중위험 (상위 40%)", "mid-risk", "#ffca28"
            display_score = int(50 + (prob - 0.017725) * 1000) # 50~79점대로 맵핑
        else:
            level, css, color = "저위험 (일반관리)", "low-risk", "#00e5ff"
            display_score = int(prob * 1500) # 0~49점대로 맵핑
            
        return level, min(display_score, 99), css, color, prob
    except:
        return "추론 오류", 0, "low-risk", "#888", 0

lvl, score, css_class, status_color, raw_p = get_ai_prediction()

# 6. 메인 레이아웃 및 워크플로우
# 상단 환자 정보 바
st.markdown(f"""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:1.5em; font-weight:bold;">🏥 SNUH Smart AI Fall Dashboard</span>
        <span style="color:#cfd8dc;">환자: <b>{selected_p}</b> ({st.session_state.v_gender}/{st.session_state.v_age}세) | CRP: {st.session_state.v_crp}</span>
    </div>
</div>
""", unsafe_allow_html=True)

col_monitor, col_notes = st.columns([1, 1.2])

with col_monitor:
    # 실시간 디지털 계기판
    blink = css_class if css_class == "high-risk" and not st.session_state.get('alarm_done', False) else css_class
    st.markdown(f"""
    <div class="digital-monitor {blink}">
        <div class="status-label" style="color:{status_color};">{lvl}</div>
        <div class="digital-number" style="color:{status_color};">{score}</div>
        <div style="font-size:0.8rem; color:gray; margin-top:15px;">AI Raw Prob: {raw_p:.6f}</div>
    </div>
    """, unsafe_allow_html=True)

    # [워크플로우] 고위험군 진입 시 중재 팝업 다이얼로그
    if css_class == "high-risk" and not st.session_state.get('alarm_done', False):
        @st.dialog("🚨 낙상 고위험군 즉각 중재")
        def show_intervention():
            st.warning(f"AI 분석 결과 고위험군으로 판정되었습니다. (Score: {score})")
            st.write("환자 맞춤형 간호 중재를 선택해 주세요.")
            
            c1, c2 = st.columns(2)
            with c1:
                i1 = st.checkbox("침상 난간(Side Rail) 상시 고정", value=True)
                i2 = st.checkbox("낙상 주의 표지판 부착", value=True)
            with c2:
                i3 = st.checkbox("영양팀 협진 의뢰", value=(st.session_state.v_alb < 3.0))
                i4 = st.checkbox("보호자 동반 보행 교육", value=True)
                
            if st.button("중재 수행 완료 및 EMR 저장", type="primary", use_container_width=True):
                selected = []
                if i1: selected.append("난간고정")
                if i2: selected.append("표지판부착")
                if i3: selected.append("영양협진")
                if i4: selected.append("보호자교육")
                
                log = f"[{datetime.datetime.now().strftime('%H:%M')}] {lvl} 감지({score}점). 간호중재({', '.join(selected)}) 시행함. (Albumin: {st.session_state.v_alb})"
                st.session_state.nursing_notes.insert(0, log)
                st.session_state.alarm_done = True
                st.rerun()
        show_intervention()

with col_notes:
    st.subheader("📝 간호 기록 (EMR Auto-Note)")
    if not st.session_state.nursing_notes:
        st.info("수치를 조작하여 고위험 알람을 발생시키면 중재 기록이 여기에 남습니다.")
    else:
        for n in st.session_state.nursing_notes:
            st.markdown(f'<div class="note-entry">{n}</div>', unsafe_allow_html=True)

# 7. 변수 영향력 분석 시각화
st.divider()
st.subheader("📊 주요 지표 실시간 시뮬레이션")
chart_data = pd.DataFrame({
    '지표': ['Age', 'Albumin', 'SBP', 'PR', 'CRP'],
    '수치': [st.session_state.v_age, st.session_state.v_alb*20, st.session_state.v_sbp/2, st.session_state.v_pr, st.session_state.v_crp*5]
}).set_index('지표')
st.bar_chart(chart_data)
