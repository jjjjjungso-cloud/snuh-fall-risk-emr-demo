import streamlit as st
import pandas as pd
import datetime
import time
import joblib
import numpy as np
import altair as alt

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 상태 관리
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="SNUH Ward EMR - AI System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# 2. 스타일 (CSS) - 기존 디자인 100% 유지
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
    .header-info-text { font-size: 1.1em; color: #eceff1; margin-right: 15px; }

    /* 디지털 계기판 */
    .digital-monitor-container {
        background-color: #000000; border: 2px solid #455a64; border-radius: 8px;
        padding: 15px; margin-top: 15px; margin-bottom: 5px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.9); transition: border 0.3s;
        display: flex !important; flex-direction: row !important;
        justify-content: space-around !important; align-items: center !important;
    }
    @keyframes blink { 50% { border-color: #ff5252; box-shadow: 0 0 15px #ff5252; } }
    .alarm-active { animation: blink 1s infinite; border: 2px solid #ff5252 !important; }

    .score-box { text-align: center; width: 45%; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .digital-number { font-family: 'Consolas', monospace; font-size: 36px; font-weight: 900; line-height: 1.0; text-shadow: 0 0 10px rgba(255,255,255,0.4); margin-top: 5px; }
    .monitor-label { color: #90a4ae; font-size: 12px; font-weight: bold; letter-spacing: 1px; }
    .divider-line { width: 1px; height: 50px; background-color: #444; }

    /* 알람 박스 */
    .custom-alert-box {
        position: fixed; bottom: 30px; right: 30px; width: 380px; height: auto;
        background-color: #263238; border-left: 8px solid #ff5252;
        box-shadow: 0 6px 25px rgba(0,0,0,0.7); border-radius: 8px;
        padding: 20px; z-index: 9999; animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn { from { transform: translateX(120%); } to { transform: translateX(0); } }
    .alert-title { color: #ff5252; font-weight: bold; font-size: 1.4em; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }
    .alert-content { color: #eceff1; font-size: 1.0em; margin-bottom: 15px; line-height: 1.5; }
    .alert-factors { background-color: #3e2723; padding: 12px; border-radius: 6px; margin-bottom: 20px; color: #ffcdd2; font-size: 0.95em; border: 1px solid #ff5252; }

    /* 기타 UI */
    .note-entry { background-color: #2c3e50; padding: 15px; border-radius: 5px; border-left: 4px solid #0288d1; margin-bottom: 10px; }
    .risk-tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; margin: 2px; border: 1px solid #ff5252; color: #ff867c; }
    
    div.stButton > button {
        width: 100%; background-color: #d32f2f; color: white; border: none;
        padding: 12px 0; border-radius: 6px; font-weight: bold; font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 4. 리소스 로딩 (새로운 모델 및 참조 데이터)
# --------------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    resources = {}
    try:
        # 새로운 모델 로드
        resources['model'] = joblib.load('risk_score_model.joblib')
        # 중요도 데이터 (기존 형식 유지하여 차트용으로 사용)
        try: resources['importance'] = pd.read_csv('rf_feature_importance_top10.csv')
        except: resources['importance'] = None
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None
    return resources

res = load_resources()

# --------------------------------------------------------------------------------
# 5. 상태 초기화
# --------------------------------------------------------------------------------
if 'nursing_notes' not in st.session_state:
    st.session_state.nursing_notes = [{"time": "2025-12-19 08:00", "writer": "김분당", "content": "새로운 AI 모델(v2) 적용됨. 실시간 모니터링 중."}]
if 'current_pt_idx' not in st.session_state: st.session_state.current_pt_idx = 0
if 'alarm_confirmed' not in st.session_state: st.session_state.alarm_confirmed = False

def confirm_alarm():
    st.session_state.alarm_confirmed = True

# 4인의 시연용 예시 데이터 적용
PATIENTS_BASE = [
    {"id": "12345678", "bed": "04-01", "name": "① 저위험 A", "gender": "F", "age": 58, "severity": 2, "sbp": 120, "dbp": 78, "pr": 78, "rr": 18, "bt": 36.6, "alb": 4.1, "crp": 0.3, "mental": "alert", "diag": "Pneumonia"},
    {"id": "87654321", "bed": "04-02", "name": "② 저위험 B", "gender": "M", "age": 72, "severity": 2, "sbp": 130, "dbp": 82, "pr": 76, "rr": 18, "bt": 36.7, "alb": 3.8, "crp": 0.8, "mental": "alert", "diag": "Stomach Cancer"},
    {"id": "11223344", "bed": "05-01", "name": "③ 중위험", "gender": "F", "age": 68, "severity": 3, "sbp": 115, "dbp": 75, "pr": 88, "rr": 20, "bt": 37.2, "alb": 3.0, "crp": 4.0, "mental": "verbal response", "diag": "Femur Fracture"},
    {"id": "99887766", "bed": "05-02", "name": "④ 고위험 (상위 20%)", "gender": "M", "age": 65, "severity": 3, "sbp": 110, "dbp": 70, "pr": 96, "rr": 22, "bt": 37.6, "alb": 2.4, "crp": 6.0, "mental": "painful response", "diag": "Appendicitis"},
]

# 현재 선택된 환자의 기본값으로 시뮬레이션 변수 초기화
p = PATIENTS_BASE[st.session_state.current_pt_idx]
defaults = {
    'sim_sbp': p['sbp'], 'sim_dbp': p['dbp'], 'sim_pr': p['pr'], 'sim_rr': p['rr'], 
    'sim_bt': p['bt'], 'sim_alb': p['alb'], 'sim_crp': p['crp'], 
    'sim_mental': p['mental'], 'sim_severity': p['severity']
}
for key, val in defaults.items():
    if key not in st.session_state: st.session_state[key] = val

# --------------------------------------------------------------------------------
# 6. 새로운 모델 기반 예측 로직 (11개 피처 사용)
# --------------------------------------------------------------------------------
def calculate_risk_score(pt_static):
    if res and 'model' in res:
        model = res['model']
        # 11개 피처 순서 맞춤 (팀원 모델 요구사항)
        input_data = {
            '성별': 1 if pt_static['gender'] == 'M' else 0,
            '중증도분류': st.session_state.sim_severity,
            'SBP': st.session_state.sim_sbp,
            'DBP': st.session_state.sim_dbp,
            'RR': st.session_state.sim_rr,
            'PR': st.session_state.sim_pr,
            'BT': st.session_state.sim_bt,
            '내원시 반응': st.session_state.sim_mental, # 'alert', 'verbal response' 등 문자열 그대로
            '나이': pt_static['age'],
            'albumin': st.session_state.sim_alb,
            'crp': st.session_state.sim_crp
        }
        
        try:
            input_df = pd.DataFrame([input_data])
            prob = model.predict_proba(input_df)[0][1] # 고위험군 확률
            
            # 확률값을 0-100 점수로 변환 (스케일링)
            # 기준: 0.025498(상위 20%)을 85점으로 맵핑하여 가독성 증대
            if prob >= 0.025498:
                display_score = int(85 + (prob - 0.025498) * 400)
            elif prob >= 0.017725:
                display_score = int(55 + (prob - 0.017725) * 1000)
            else:
                display_score = int(prob * 2000)
            
            return min(display_score, 99), prob
        except:
            return 10, 0.01
    return 10, 0.01

# --------------------------------------------------------------------------------
# 7. 상세 분석 팝업 (Section 7 다이얼로그)
# --------------------------------------------------------------------------------
@st.dialog("낙상 위험도 정밀 분석", width="large")
def show_risk_details(name, factors, current_score):
    st.info(f"🕒 **{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}** 기준 분석 결과입니다.")
    tab1, tab2 = st.tabs(["🛡️ 맞춤형 간호중재", "📊 AI 판단 근거"])
    
    with tab1:
        c1, c2, c3 = st.columns([1, 0.2, 1])
        with c1:
            st.markdown("##### 🚨 감지된 위험요인")
            with st.container(border=True):
                if factors:
                    for f in factors: st.error(f"• {f}")
                else: st.write("특이 위험 요인 없음")
        with c2:
            st.markdown("<div style='display:flex; height:200px; align-items:center; justify-content:center; font-size:40px;'>➡</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("##### ✅ 필수 간호 진술문")
            with st.container(border=True):
                chk_rail = st.checkbox("침상 난간(Side Rail) 올림 확인", value=(current_score >= 60))
                chk_med = st.checkbox("💊 고위험 약물(수면제 등) 주의 교육", value=(st.session_state.sim_mental != 'alert'))
                chk_nutri = st.checkbox("🥩 영양팀 협진 의뢰 (Albumin 저조)", value=(st.session_state.sim_alb < 3.0))
                chk_edu = st.checkbox("📢 낙상 예방 교육 및 호출기 안내", value=True)

        if st.button("간호 수행 완료 및 기록 저장", type="primary", use_container_width=True):
            actions = []
            if chk_rail: actions.append("난간고정")
            if chk_nutri: actions.append("영양협진")
            if chk_edu: actions.append("예방교육")
            note_content = f"낙상위험평가({current_score}점) -> 위험요인({', '.join(factors)}) 확인 -> 중재({', '.join(actions)}) 시행함."
            st.session_state.nursing_notes.insert(0, {"time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), "writer": "김분당", "content": note_content})
            st.toast("기록되었습니다!")
            time.sleep(1)
            st.rerun()

    with tab2:
        st.markdown("##### 🔍 환자 맞춤형 변수 기여도")
        if res['importance'] is not None:
            df_imp = res['importance'].copy().sort_values('importance', ascending=True).tail(10)
            chart = alt.Chart(df_imp).mark_bar(color='#0288d1').encode(
                x=alt.X('importance', title='기여도'),
                y=alt.Y('feature', sort='-x', title='변수명')
            ).properties(height=350)
            st.altair_chart(chart, use_container_width=True)

# --------------------------------------------------------------------------------
# 8. 메인 레이아웃 및 Flow 구성
# --------------------------------------------------------------------------------
col_sidebar, col_main = st.columns([2, 8])

with col_sidebar:
    st.selectbox("근무 DUTY", ["Day", "Evening", "Night"])
    st.divider()
    st.markdown("### 🏥 재원 환자")
    idx = st.radio("환자 리스트", range(len(PATIENTS_BASE)), format_func=lambda i: f"[{PATIENTS_BASE[i]['bed']}] {PATIENTS_BASE[i]['name']}", label_visibility="collapsed")
    
    if idx != st.session_state.current_pt_idx:
        st.session_state.current_pt_idx = idx
        st.session_state.alarm_confirmed = False
        # 세션 데이터 리셋
        p_new = PATIENTS_BASE[idx]
        st.session_state.sim_sbp, st.session_state.sim_dbp = p_new['sbp'], p_new['dbp']
        st.session_state.sim_pr, st.session_state.sim_rr = p_new['pr'], p_new['rr']
        st.session_state.sim_bt, st.session_state.sim_alb = p_new['bt'], p_new['alb']
        st.session_state.sim_crp, st.session_state.sim_mental = p_new['crp'], p_new['mental']
        st.session_state.sim_severity = p_new['severity']
        st.rerun()

    curr_pt = PATIENTS_BASE[idx]
    
    # 뇌(AI) 가동
    fall_score, raw_prob = calculate_risk_score(curr_pt)
    
    # 계기판 알람 상태 결정 (확률 기준 0.025498 이상)
    is_high_risk = raw_prob >= 0.025498
    alarm_class = "alarm-active" if is_high_risk and not st.session_state.alarm_confirmed else ""
    f_color = "#ff5252" if is_high_risk else ("#ffca28" if raw_prob >= 0.017725 else "#00e5ff")

    st.markdown(f"""
    <div class="digital-monitor-container {alarm_class}">
        <div class="score-box">
            <div class="monitor-label">FALL RISK</div>
            <div class="digital-number" style="color: {f_color};">{fall_score}</div>
        </div>
        <div class="divider-line"></div>
        <div class="score-box">
            <div class="monitor-label">STATUS</div>
            <div style="color:{f_color}; font-weight:bold; font-size:14px;">{"HIGH" if is_high_risk else "NORMAL"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 위험 요인 추출
    factors = []
    if curr_pt['age'] >= 65: factors.append("고령")
    if st.session_state.sim_alb < 3.0: factors.append("알부민 저조")
    if st.session_state.sim_sbp < 100: factors.append("저혈압 경향")
    if st.session_state.sim_mental != 'alert': factors.append("의식상태 변화")
    if st.session_state.sim_crp > 5.0: factors.append("급성 염증상태")

    if st.button("🔍 상세 분석 및 중재 기록", type="primary", use_container_width=True):
        show_risk_details(curr_pt['name'], factors, fall_score)

with col_main:
    # 헤더
    st.markdown(f"""
    <div class="header-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center;">
                <span style="font-size:1.5em; font-weight:bold; color:white; margin-right:20px;">🏥 SNUH AI</span>
                <span class="header-info-text"><b>{curr_pt['name']}</b> ({curr_pt['gender']}/{curr_pt['age']}세)</span>
                <span class="header-info-text">ID: {curr_pt['id']}</span>
                <span class="header-info-text">진단: <span style="color:#4fc3f7;">{curr_pt['diag']}</span></span>
            </div>
            <div style="color:#b0bec5; font-size:0.9em;">김분당 간호사 | {datetime.datetime.now().strftime('%Y-%m-%d')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🛡️ AI Simulation", "💊 오더", "📝 간호기록"])

    with tab1:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("##### ⚡ 실시간 데이터 시뮬레이션")
            with st.container(border=True):
                st.select_slider("중증도분류", options=[1, 2, 3, 4, 5], key="sim_severity")
                r1, r2 = st.columns(2)
                with r1: st.number_input("SBP", step=5, key="sim_sbp")
                with r2: st.number_input("DBP", step=5, key="sim_dbp")
                r3, r4 = st.columns(2)
                with r3: st.number_input("PR", step=5, key="sim_pr")
                with r4: st.number_input("BT", step=0.1, key="sim_bt")
                st.slider("Albumin", 1.0, 5.5, key="sim_alb")
                st.selectbox("의식 상태", ["alert", "verbal response", "painful response", "unresponsive"], key="sim_mental")
                st.number_input("CRP", step=0.5, key="sim_crp")
        with c2:
            st.markdown("##### 📊 환자 상태 요약")
            st.markdown(f"""
            <div style="background-color:#263238; padding:15px; border-radius:8px;">
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; text-align:center;">
                    <div><div style="color:#aaa; font-size:12px;">BP</div><div style="font-weight:bold; font-size:18px;">{st.session_state.sim_sbp}/{st.session_state.sim_dbp}</div></div>
                    <div><div style="color:#aaa; font-size:12px;">PR</div><div style="font-weight:bold; font-size:18px;">{st.session_state.sim_pr}</div></div>
                    <div><div style="color:#aaa; font-size:12px;">BT</div><div style="font-weight:bold; font-size:18px;">{st.session_state.sim_bt}</div></div>
                    <div><div style="color:#aaa; font-size:12px;">ALB</div><div style="font-weight:bold; font-size:18px;">{st.session_state.sim_alb}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if factors:
                st.markdown("<br>", unsafe_allow_html=True)
                for f in factors: st.markdown(f"<span class='risk-tag'>{f}</span>", unsafe_allow_html=True)

    with tab3:
        for note in st.session_state.nursing_notes:
            st.markdown(f"""<div class="note-entry"><small>{note['time']} | {note['writer']}</small><br>{note['content']}</div>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 9. 고정 알람 박스 로직 (Flow 유지)
# --------------------------------------------------------------------------------
if is_high_risk and not st.session_state.alarm_confirmed:
    f_str = "<br>• ".join(factors) if factors else "복합적 위험요인"
    st.markdown(f"""
    <div class="custom-alert-box">
        <div class="alert-title">🚨 낙상 고위험 감지! ({fall_score}점)</div>
        <div class="alert-content">환자의 상태 변화로 낙상 위험이 급증했습니다. 즉시 확인하십시오.</div>
        <div class="alert-factors"><b>[감지된 주요 위험 요인]</b><br>• {f_str}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='margin-top:-8px'></div>", unsafe_allow_html=True)
    if st.button("확인 (Confirm Intervention)", key="confirm_btn"):
        confirm_alarm()
        st.rerun()
