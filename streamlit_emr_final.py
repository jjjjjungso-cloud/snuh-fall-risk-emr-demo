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
        res['model'] = joblib.load('risk_score_model.joblib')
        ref_data = np.load('train_score_ref.npz')
        res['ref_scores'] = ref_data['train_scores_sorted']
    except:
        return None
    return res

artifacts = load_resources()

# --------------------------------------------------------------------------------
# 2. 스타일 (CSS)
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
    /* 깜빡임 애니메이션 */
    @keyframes blink { 50% { border-color: #ff5252; box-shadow: 0 0 15px #ff5252; } }
    .alarm-active { animation: blink 1s infinite; border: 2px solid #ff5252 !important; }
    
    .digital-number { font-family: 'Consolas', monospace; font-size: 36px; font-weight: 900; line-height: 1.0; }
    .monitor-label { color: #90a4ae; font-size: 12px; font-weight: bold; }

    .custom-alert-box {
        position: fixed; bottom: 30px; right: 30px; width: 380px; background-color: #263238; 
        border-left: 8px solid #ff5252; padding: 20px; z-index: 9999; border-radius: 8px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.7);
    }
    .note-entry { background-color: #2c3e50; padding: 15px; border-radius: 5px; border-left: 4px solid #0288d1; margin-bottom: 10px; }
    
    div.stButton > button { width: 100%; background-color: #d32f2f; color: white; font-weight: bold; border: none; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 3. 상태 관리 및 데이터 초기화
# --------------------------------------------------------------------------------
if 'nursing_log' not in st.session_state:
    st.session_state.nursing_log = [{"time": "2025-12-19 08:00", "content": "신규 입원 환자 낙상 예방 교육 완료함."}]
if 'alarm_confirmed' not in st.session_state: st.session_state.alarm_confirmed = False

intervention_options = {
    "공통/기본": ["침대 난간(Side Rail) 상시 고정", "낙상 예방 표지판 부착", "호출벨 위치 확인 및 교육"],
    "저혈압/어지럼증": ["체위 변경 시 천천히 움직이도록 교육", "보행 시 반드시 보호자 동행", "기립성 저혈압 모니터링"],
    "영양부족/근력약화": ["고단백 식이 권장", "재활의학과 협진(근력 강화)", "침상 옆 보조기구 배치"],
    "염증/발열": ["수분 섭취 권장", "I/O 체크 및 탈수 모니터링", "활력징후 2시간 간격 모니터링"],
    "의식저하/인지장애": ["환자 근거리 배치(Station 앞)", "보호자 상주 교육", "섬망 예방 중재(시계/달력 비치)"],
    "고령(고위험군)": ["야간 조명 유지", "비끄럼 방지 양말 착용 확인", "화장실 이동 시 보조"]
}

# --------------------------------------------------------------------------------
# 4. 분석 로직
# --------------------------------------------------------------------------------
def get_analysis_results():
    risks = ["공통/기본"]
    if st.session_state.sim_sbp < 100: risks.append("저혈압/어지럼증")
    if st.session_state.sim_alb < 3.5: risks.append("영양부족/근력약화")
    if st.session_state.sim_crp > 1.0 or st.session_state.sim_bt >= 37.8: risks.append("염증/발열")
    if st.session_state.sim_mental != "alert": risks.append("의식저하/인지장애")
    if st.session_state.sim_age >= 75: risks.append("고령(고위험군)")
    
    fall_score = 25
    if artifacts:
        try:
            df = pd.DataFrame([{
                '성별': 1 if "남성" in st.session_state.sim_gender else 0, '중증도분류': st.session_state.sim_sev,
                'SBP': st.session_state.sim_sbp, 'DBP': st.session_state.sim_dbp, 'RR': st.session_state.sim_rr,
                'PR': st.session_state.sim_pr, 'BT': st.session_state.sim_bt, '내원시 반응': st.session_state.sim_mental,
                '나이': st.session_state.sim_age, 'albumin': st.session_state.sim_alb, 'crp': st.session_state.sim_crp
            }])
            prob = artifacts['model'].predict_proba(df)[0][1]
            fall_score = int(np.searchsorted(artifacts['ref_scores'], prob) / len(artifacts['ref_scores']) * 100)
        except: pass
        
    return fall_score, risks

# --------------------------------------------------------------------------------
# 5. 다이얼로그 (중재 및 기록)
# --------------------------------------------------------------------------------
@st.dialog("🛡️ 고위험군 맞춤 간호 중재", width="large")
def show_intervention_dialog(score, detected_risks):
    st.write(f"**낙상 위험도: {score}점 (고위험군)**")
    st.markdown("수행한 중재 항목을 선택하여 EMR로 전송하십시오.")
    st.divider()
    
    selected_actions = []
    cols = st.columns(len(detected_risks))
    for i, risk in enumerate(detected_risks):
        with cols[i]:
            st.markdown(f"**[{risk}]**")
            for action in intervention_options.get(risk, []):
                if st.checkbox(action, key=f"int_{risk}_{action}"):
                    selected_actions.append(action)
                    
    if st.button("기록 저장 및 알람 해제", type="primary"):
        if selected_actions:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            log = f"[{now}] [AI 고위험군 알람: {score}점] 중재({', '.join(selected_actions)}) 시행함."
            st.session_state.nursing_log.insert(0, {"time": now, "content": log})
            st.session_state.alarm_confirmed = True # 알람 확인 처리
            st.rerun()
        else:
            st.warning("중재 항목을 선택해주세요.")

# --------------------------------------------------------------------------------
# 6. 메인 화면 구성
# --------------------------------------------------------------------------------
col_side, col_main = st.columns([2, 8])

with col_side:
    st.markdown("### 🏥 담당 환자")
    st.info("김분당 (ID: 12345678)")
    
    st.session_state.sim_age = 45
    st.session_state.sim_gender = "여성 (F)"
    st.session_state.sim_sev = st.selectbox("중증도분류", [1,2,3,4,5], index=4)
    
    score, risks = get_analysis_results()
    
    # [수정] 알람 깜빡이 기준: 80점 이상 (고위험군)
    alarm_css = "alarm-active" if score >= 80 and not st.session_state.alarm_confirmed else ""
    f_color = "#ff5252" if score >= 80 else ("#ffca28" if score >= 60 else "#00e5ff")
    
    st.markdown(f"""
    <div class="digital-monitor-container {alarm_css}">
        <div style="text-align:center;">
            <div class="monitor-label">FALL RISK</div>
            <div class="digital-number" style="color:{f_color};">{score}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 상세 분석/기록", use_container_width=True):
        show_intervention_dialog(score, risks)

with col_main:
    st.markdown(f"""
    <div class="header-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:1.3em; font-weight:bold; color:white;">🏥 SNUH AI EMR</div>
            <div style="color:#b0bec5;">환자: <b>김분당</b> | {datetime.datetime.now().strftime('%Y-%m-%d')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🛡️ 실시간 시뮬레이션", "📝 간호기록"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### ⚡ 데이터 입력")
            with st.container(border=True):
                st.session_state.sim_sbp = st.number_input("SBP", 50, 250, 120)
                st.session_state.sim_dbp = st.number_input("DBP", 30, 150, 80)
                st.session_state.sim_pr = st.number_input("PR", 20, 200, 75)
                st.session_state.sim_rr = st.number_input("RR", 5, 50, 18)
                st.session_state.sim_bt = st.number_input("BT", 30.0, 45.0, 36.5, step=0.1)
                st.session_state.sim_alb = st.slider("Albumin", 1.0, 5.0, 4.5, step=0.1)
                st.session_state.sim_crp = st.number_input("CRP", 0.0, 50.0, 0.1)
                st.session_state.sim_mental = st.selectbox("의식 상태", ["alert", "verbal response", "painful response", "unresponsive"])
        
        with c2:
            st.markdown("##### 📊 감지된 위험 요인")
            for r in risks:
                st.error(f"⚠️ {r}")
            st.info("점수가 80점 이상이 되면 고위험 알람이 활성화됩니다.")

    with tab2:
        for log in st.session_state.nursing_log:
            st.markdown(f'<div class="note-entry"><small>{log["time"]}</small><br>{log["content"]}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 7. [수정] 고위험군 전용 알람 (80점 이상)
# --------------------------------------------------------------------------------
if score >= 80 and not st.session_state.alarm_confirmed:
    st.markdown(f"""
    <div class="custom-alert-box">
        <div style="color:#ff5252; font-weight:bold; font-size:1.4em; margin-bottom:10px;">🚨 낙상 고위험군 감지!</div>
        <div style="font-size:1.0em; color:#eceff1; margin-bottom:15px;">환자가 상위 20% 이내인 <b>{score}점</b>에 도달했습니다. 즉각적인 중재가 필요합니다.</div>
        <div style="background-color:#3e2723; padding:10px; border-radius:6px; color:#ffcdd2; font-size:0.95em; border:1px solid #ff5252;">
            <b>주요 위험 요인:</b> {', '.join(risks)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:-10px'></div>", unsafe_allow_html=True)
    if st.button("🚨 중재 수행 및 알람 해제", key="alarm_confirm_btn"):
        show_intervention_dialog(score, risks)

# 점수가 안전권으로 내려가면 알람 확인 상태 리셋 (나중에 다시 위험해지면 또 떠야 하므로)
if score < 60:
    st.session_state.alarm_confirmed = False
