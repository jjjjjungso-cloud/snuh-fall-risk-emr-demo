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
        # 실제 모델 파일 로드 (파일명 확인 필요)
        res['model'] = joblib.load('risk_score_model.joblib')
        ref_data = np.load('train_score_ref.npz')
        res['ref_scores'] = ref_data['train_scores_sorted']
    except:
        return None
    return res

res_data = load_resources()

# --------------------------------------------------------------------------------
# 2. 스타일 (CSS) - 기존 껍데기 디자인 유지
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
    .digital-number { font-family: 'Consolas', monospace; font-size: 36px; font-weight: 900; line-height: 1.0; }
    .custom-alert-box {
        position: fixed; bottom: 30px; right: 30px; width: 380px; background-color: #263238; 
        border-left: 8px solid #ff5252; padding: 20px; z-index: 9999; border-radius: 8px;
    }
    .note-entry { background-color: #2c3e50; padding: 15px; border-radius: 5px; border-left: 4px solid #0288d1; margin-bottom: 10px; }
    div.stButton > button { width: 100%; background-color: #d32f2f; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 3. 데이터 및 상태 초기화 (김분당 기준)
# --------------------------------------------------------------------------------
if 'nursing_notes' not in st.session_state:
    st.session_state.nursing_notes = [{"time": "2025-12-19 08:00", "writer": "김분당", "content": "입원 시 낙상 예방 교육 시행함."}]
if 'alarm_confirmed' not in st.session_state: st.session_state.alarm_confirmed = False

# 중재 옵션 정의 (선생님이 주신 리스트)
INTERVENTION_OPTIONS = {
    "공통/기본": ["침대 난간(Side Rail) 상시 고정", "낙상 예방 표지판 부착", "호출벨 위치 확인 및 교육"],
    "저혈압/어지럼증": ["체위 변경 시 천천히 움직이도록 교육", "보행 시 반드시 보호자 동행", "기립성 저혈압 모니터링"],
    "영양부족/근력약화": ["고단백 식이 권장", "재활의학과 협진(근력 강화)", "침상 옆 보조기구 배치"],
    "염증/발열": ["수분 섭취 권장", "I/O 체크 및 탈수 모니터링", "활력징후 2시간 간격 모니터링"],
    "의식저하/인지장애": ["환자 근거리 배치(Station 앞)", "보호자 상주 교육", "섬망 예방 중재(시계/달력 비치)"],
    "고령(고위험군)": ["야간 조명 유지", "미끄럼 방지 양말 착용 확인", "화장실 이동 시 보조"]
}

# --------------------------------------------------------------------------------
# 4. 핵심 로직: 위험 요인 감지 및 점수 계산
# --------------------------------------------------------------------------------
def get_analysis():
    # 11개 입력값에 기반한 위험 요인 추출
    risks = ["공통/기본"]
    if st.session_state.sim_sbp < 100: risks.append("저혈압/어지럼증")
    if st.session_state.sim_alb < 3.5: risks.append("영양부족/근력약화")
    if st.session_state.sim_crp > 1.0 or st.session_state.sim_bt >= 37.8: risks.append("염증/발열")
    if st.session_state.sim_mental != "명료(Alert)": risks.append("의식저하/인지장애")
    if st.session_state.sim_age >= 75: risks.append("고령(고위험군)")
    
    # 모델 점수 계산 (Percentile 기반)
    fall_score = 25 # 기본값
    if res_data:
        try:
            input_df = pd.DataFrame([{
                '성별': 1 if "남성" in st.session_state.sim_gender else 0, '중증도분류': st.session_state.sim_sev,
                'SBP': st.session_state.sim_sbp, 'DBP': st.session_state.sim_dbp, 'RR': st.session_state.sim_rr,
                'PR': st.session_state.sim_pr, 'BT': st.session_state.sim_bt, '내원시 반응': st.session_state.sim_mental,
                '나이': st.session_state.sim_age, 'albumin': st.session_state.sim_alb, 'crp': st.session_state.sim_crp
            }])
            prob = res_data['model'].predict_proba(input_df)[0][1]
            fall_score = int(np.searchsorted(res_data['ref_scores'], prob) / len(res_data['ref_scores']) * 100)
        except: pass
    return fall_score, risks

# --------------------------------------------------------------------------------
# 5. 상세 중재 팝업 (Dialog)
# --------------------------------------------------------------------------------
@st.dialog("📋 맞춤형 간호중재 및 Auto-Charting", width="large")
def show_intervention_dialog(score, detected_risks):
    st.write(f"현재 AI 위험 점수: **{score}점** | 감지된 위험군: {', '.join(detected_risks)}")
    
    st.markdown("##### ✅ 수행할 간호 중재를 선택하세요")
    selected_actions = []
    
    # 위험 요인별 동적 체크박스 생성
    cols = st.columns(len(detected_risks))
    for i, risk in enumerate(detected_risks):
        with cols[i]:
            st.markdown(f"**[{risk}]**")
            for action in INTERVENTION_OPTIONS.get(risk, []):
                if st.checkbox(action, key=f"chk_{risk}_{action}"):
                    selected_actions.append(action)
    
    st.divider()
    if st.button("간호 수행 완료 및 기록 저장", type="primary"):
        if selected_actions:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            note = f"[AI 낙상평가: {score}점] 위험요인({', '.join(detected_risks)}) 확인되어 중재({', '.join(selected_actions)}) 시행함."
            st.session_state.nursing_notes.insert(0, {"time": now, "writer": "김분당", "content": note})
            st.session_state.alarm_confirmed = True
            st.rerun()
        else:
            st.warning("수행한 중재를 하나 이상 선택해주세요.")

# --------------------------------------------------------------------------------
# 6. 메인 레이아웃
# --------------------------------------------------------------------------------
col_side, col_main = st.columns([2, 8])

# [좌측 패널]
with col_side:
    st.markdown("### 🏥 담당 환자")
    st.info("김분당 (F/45세) [04-01]")
    st.divider()
    
    # 입력 세션 상태 관리
    st.session_state.sim_age = 45
    st.session_state.sim_gender = "여성 (F)"
    st.session_state.sim_sev = st.selectbox("중증도", [1,2,3,4,5], index=4)
    
    fall_score, detected_risks = get_analysis()
    
    # 디지털 계기판
    alarm_class = "alarm-active" if fall_score >= 60 and not st.session_state.alarm_confirmed else ""
    f_color = "#ff5252" if fall_score >= 80 else ("#ffca28" if fall_score >= 60 else "#00e5ff")
    
    st.markdown(f"""
    <div class="digital-monitor-container {alarm_class}">
        <div style="text-align:center;">
            <div style="color:#90a4ae; font-size:12px; font-weight:bold;">FALL RISK</div>
            <div class="digital-number" style="color:{f_color};">{fall_score}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 상세 분석 및 중재 기록", type="primary", use_container_width=True):
        show_intervention_dialog(fall_score, detected_risks)

# [우측 메인 패널]
with col_main:
    st.markdown(f"""
    <div class="header-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:1.2em; color:white;"><b>SNUH AI EMR</b> | 환자: 김분당 (ID: 12345678)</div>
            <div style="color:#b0bec5; font-size:0.9em;">{datetime.datetime.now().strftime('%Y-%m-%d')} | 근무: Day</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["🛡️ 통합 시뮬레이션", "📝 간호기록"])
    
    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("##### ⚡ 실시간 데이터 입력 (11개 변수)")
            with st.container(border=True):
                st.session_state.sim_sbp = st.number_input("SBP", 50, 250, 120, key="sbp_in")
                st.session_state.sim_dbp = st.number_input("DBP", 30, 150, 80, key="dbp_in")
                st.session_state.sim_pr = st.number_input("PR", 20, 200, 75, key="pr_in")
                st.session_state.sim_rr = st.number_input("RR", 5, 50, 18, key="rr_in")
                st.session_state.sim_bt = st.number_input("BT", 30.0, 45.0, 36.5, step=0.1, key="bt_in")
                st.session_state.sim_alb = st.slider("Albumin", 1.0, 5.0, 4.5, step=0.1, key="alb_in")
                st.session_state.sim_crp = st.number_input("CRP", 0.0, 50.0, 0.1, key="crp_in")
                st.session_state.sim_mental = st.selectbox("의식 상태", ["명료(Alert)", "기면(Drowsy)", "혼미(Stupor)"], key="men_in")
        
        with c2:
            st.markdown("##### 📊 감지된 위험 요인")
            for r in detected_risks:
                st.error(f"⚠️ {r}")
            st.info("데이터를 변경하면 AI가 실시간으로 낙상 위험 요인을 분석합니다.")

    with t2:
        for note in st.session_state.nursing_log if 'nursing_log' in st.session_state else st.session_state.nursing_notes:
            st.markdown(f"""
            <div class="note-entry">
                <small>{note['time']} | 작성자: {note['writer']}</small><br>{note['content']}
            </div>
            """, unsafe_allow_html=True)

# [알람 팝업]
if fall_score >= 60 and not st.session_state.alarm_confirmed:
    st.markdown(f"""
    <div class="custom-alert-box">
        <div style="color:#ff5252; font-weight:bold; font-size:1.3em;">🚨 낙상 위험 감지! ({fall_score}점)</div>
        <p style="color:#eceff1; margin-top:10px;">환자의 상태 변화로 위험도가 상승했습니다. 즉시 맞춤형 중재를 수행하십시오.</p>
        <div style="background:#3e2723; padding:10px; border-radius:5px; color:#ffcdd2; font-size:0.9em;">
            감지 요인: {', '.join(detected_risks)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:-10px'></div>", unsafe_allow_html=True)
    if st.button("확인 및 중재 수행", key="confirm_btn"):
        show_intervention_dialog(fall_score, detected_risks)
