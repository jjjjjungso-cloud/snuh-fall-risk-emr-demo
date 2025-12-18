import streamlit as st
import pandas as pd
import datetime
import time
import joblib
import numpy as np

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 리소스 로딩
# --------------------------------------------------------------------------------
st.set_page_config(page_title="SNUH AI System", page_icon="🏥", layout="wide")

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('risk_score_model.joblib')
        ref_data = np.load('train_score_ref.npz')
        return model, ref_data['train_scores_sorted']
    except: return None, None

model, ref_scores = load_resources()

# --------------------------------------------------------------------------------
# 2. [에러 해결] 세션 상태(Session State) 초기화 로직
# --------------------------------------------------------------------------------
# 앱이 처음 실행될 때 필요한 모든 변수를 '김분당' 환자 기준으로 미리 설정합니다.
if 'init_done' not in st.session_state:
    st.session_state.sim_age = 45
    st.session_state.sim_gender = "여성 (F)"
    st.session_state.sim_sev = 5
    st.session_state.sim_sbp = 120
    st.session_state.sim_dbp = 80
    st.session_state.sim_pr = 75
    st.session_state.sim_rr = 18
    st.session_state.sim_bt = 36.5
    st.session_state.sim_alb = 4.5
    st.session_state.sim_crp = 0.1
    st.session_state.sim_mental = "alert"
    st.session_state.nursing_log = []
    st.session_state.alarm_confirmed = False
    st.session_state.init_done = True

# --------------------------------------------------------------------------------
# 3. 스타일 (CSS) - 고위험군 알람 및 다크모드 유지
# --------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    .stApp { background-color: #1e252b; color: #e0e0e0; font-family: 'Noto Sans KR', sans-serif; }
    .header-container { background-color: #263238; padding: 10px 20px; border-radius: 5px; border-top: 3px solid #0288d1; margin-bottom: 10px; }
    .digital-monitor-container { background-color: #000000; border: 2px solid #455a64; border-radius: 8px; padding: 15px; margin-top: 15px; display: flex; justify-content: space-around; }
    @keyframes blink { 50% { border-color: #ff5252; box-shadow: 0 0 15px #ff5252; } }
    .alarm-active { animation: blink 1s infinite; border: 2px solid #ff5252 !important; }
    .digital-number { font-family: 'Consolas', monospace; font-size: 36px; font-weight: 900; line-height: 1.0; }
    .custom-alert-box { position: fixed; bottom: 30px; right: 30px; width: 380px; background-color: #263238; border-left: 8px solid #ff5252; padding: 20px; z-index: 9999; border-radius: 8px; box-shadow: 0 6px 25px rgba(0,0,0,0.7); }
    .note-entry { background-color: #2c3e50; padding: 15px; border-radius: 5px; border-left: 4px solid #0288d1; margin-bottom: 10px; }
    div.stButton > button { width: 100%; background-color: #d32f2f; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 4. 중재 옵션 및 분석 함수
# --------------------------------------------------------------------------------
intervention_options = {
    "공통/기본": ["침대 난간(Side Rail) 상시 고정", "낙상 예방 표지판 부착", "호출벨 위치 확인 및 교육"],
    "저혈압/어지럼증": ["체위 변경 시 천천히 움직이도록 교육", "보행 시 보호자 동행", "기립성 저혈압 모니터링"],
    "영양부족/근력약화": ["고단백 식이 권장", "재활의학과 협진(근력 강화)", "침상 옆 보조기구 배치"],
    "염증/발열": ["수분 섭취 권장", "I/O 체크 및 탈수 모니터링", "활력징후 2시간 간격 모니터링"],
    "의식저하/인지장애": ["환자 근거리 배치(Station 앞)", "보호자 상주 교육", "섬망 예방 중재"],
    "고령(고위험군)": ["야간 조명 유지", "미끄럼 방지 양말 착용 확인", "화장실 이동 시 보조"]
}

def get_analysis_results():
    risks = ["공통/기본"]
    # 초기화된 세션 상태 값을 사용하여 에러 방지
    if st.session_state.sim_sbp < 100: risks.append("저혈압/어지럼증")
    if st.session_state.sim_alb < 3.5: risks.append("영양부족/근력약화")
    if st.session_state.sim_crp > 1.0 or st.session_state.sim_bt >= 37.8: risks.append("염증/발열")
    if st.session_state.sim_mental != "alert": risks.append("의식저하/인지장애")
    if st.session_state.sim_age >= 75: risks.append("고령(고위험군)")
    
    score = 25
    if model:
        df = pd.DataFrame([{
            '성별': 1 if "남성" in st.session_state.sim_gender else 0, '중증도분류': st.session_state.sim_sev,
            'SBP': st.session_state.sim_sbp, 'DBP': st.session_state.sim_dbp, 'RR': st.session_state.sim_rr,
            'PR': st.session_state.sim_pr, 'BT': st.session_state.sim_bt, '내원시 반응': st.session_state.sim_mental,
            '나이': st.session_state.sim_age, 'albumin': st.session_state.sim_alb, 'crp': st.session_state.sim_crp
        }])
        prob = model.predict_proba(df)[0][1]
        score = int(np.searchsorted(ref_scores, prob) / len(ref_scores) * 100)
    return score, risks

# --------------------------------------------------------------------------------
# 5. 메인 레이아웃
# --------------------------------------------------------------------------------
@st.dialog("🛡️ 맞춤형 간호 중재")
def show_interventions(score, risks):
    st.write(f"낙상 위험도: **{score}점**")
    selected = []
    for r in risks:
        st.markdown(f"**[{r}]**")
        for opt in intervention_options.get(r, []):
            if st.checkbox(opt, key=f"int_{opt}"): selected.append(opt)
    if st.button("기록 전송"):
        now = datetime.datetime.now().strftime('%H:%M')
        st.session_state.nursing_log.insert(0, {"time": now, "content": f"[AI 점수: {score}] {', '.join(selected)} 시행함."})
        st.session_state.alarm_confirmed = True
        st.rerun()

col_side, col_main = st.columns([2, 8])

with col_side:
    st.markdown("### 🏥 담당 환자")
    st.info(f"김분당 (ID: 12345678)")
    # 사이드바에서 중증도만 바로 조정 가능하게 설정
    st.session_state.sim_sev = st.selectbox("중증도분류", [1,2,3,4,5], index=st.session_state.sim_sev-1)
    
    score, risks = get_analysis_results()
    
    # 80점 이상일 때만 깜빡이는 효과 (고위험군)
    alarm_css = "alarm-active" if score >= 80 and not st.session_state.alarm_confirmed else ""
    f_color = "#ff5252" if score >= 80 else ("#ffca28" if score >= 60 else "#00e5ff")
    
    st.markdown(f"""
    <div class="digital-monitor-container {alarm_css}">
        <div style="text-align:center;">
            <div style="color:#90a4ae; font-size:12px;">FALL RISK</div>
            <div class="digital-number" style="color:{f_color};">{score}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 상세 분석 및 기록"): show_interventions(score, risks)

with col_main:
    st.markdown(f'<div class="header-container"><div style="font-size:1.2em; color:white;"><b>SNUH AI EMR</b> | 환자: 김분당</div></div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["🛡️ Simulation", "📝 Nursing Notes"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            # key 값을 부여하여 세션 상태와 위젯을 직접 연결 (에러 방지의 핵심)
            st.session_state.sim_sbp = st.number_input("SBP", 50, 250, st.session_state.sim_sbp, key="sbp_input")
            st.session_state.sim_dbp = st.number_input("DBP", 30, 150, st.session_state.sim_dbp, key="dbp_input")
            st.session_state.sim_pr = st.number_input("PR", 20, 200, st.session_state.sim_pr, key="pr_input")
            st.session_state.sim_bt = st.number_input("BT", 30.0, 45.0, st.session_state.sim_bt, key="bt_input")
            st.session_state.sim_alb = st.slider("Albumin", 1.0, 5.0, st.session_state.sim_alb, key="alb_input")
            st.session_state.sim_crp = st.number_input("CRP", 0.0, 50.0, st.session_state.sim_crp, key="crp_input")
            st.session_state.sim_mental = st.selectbox("의식 상태", ["alert", "verbal response", "painful response", "unresponsive"], index=0, key="mental_input")
        with c2:
            st.markdown("##### 📊 감지된 위험 요인")
            for r in risks: st.error(f"⚠️ {r}")

    with t2:
        for log in st.session_state.nursing_log:
            st.markdown(f'<div class="note-entry"><small>{log["time"]}</small><br>{log["content"]}</div>', unsafe_allow_html=True)

# 80점 이상일 때만 고정 알람 박스 노출
if score >= 80 and not st.session_state.alarm_confirmed:
    st.markdown(f"""
    <div class="custom-alert-box">
        <div style="color:#ff5252; font-weight:bold; font-size:1.2em;">🚨 낙상 고위험군 감지!</div>
        <div style="color:#eceff1; margin-top:10px;">환자가 상위 20% 이내인 <b>{score}점</b>에 도달했습니다.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚨 알람 확인 및 중재"): show_interventions(score, risks)

# 점수가 안전권으로 내려가면 다시 알람 활성화 준비
if score < 60: st.session_state.alarm_confirmed = False
