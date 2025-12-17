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
# 2. (수정) 쿼리 파라미터 방식 제거: Streamlit 버튼으로 상태 유지
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 3. 스타일 (CSS) - 알람 박스 디자인 수정됨
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

    /* [수정] 알람 박스 디자인 개선 (높이 자동 조절) */
    .custom-alert-box {
        position: fixed; 
        bottom: 30px; 
        right: 30px; 
        width: 380px;
        height: auto; /* 높이 자동 조절 */
        background-color: #263238; 
        border-left: 8px solid #ff5252;
        box-shadow: 0 6px 25px rgba(0,0,0,0.7); 
        border-radius: 8px;
        padding: 20px; 
        z-index: 9999; 
        animation: slideIn 0.5s ease-out;
        font-family: 'Noto Sans KR', sans-serif;
    }
    @keyframes slideIn { from { transform: translateX(120%); } to { transform: translateX(0); } }
    
    .alert-title { color: #ff5252; font-weight: bold; font-size: 1.4em; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }
    .alert-content { color: #eceff1; font-size: 1.0em; margin-bottom: 15px; line-height: 1.5; }
    .alert-factors { background-color: #3e2723; padding: 12px; border-radius: 6px; margin-bottom: 20px; color: #ffcdd2; font-size: 0.95em; border: 1px solid #ff5252; }
    
    /* HTML 버튼 스타일링 */
    a.btn-confirm {
        display: block; 
        width: 100%;
        background-color: #d32f2f; 
        color: white !important; 
        text-align: center; 
        padding: 12px 0; 
        border-radius: 6px; 
        font-weight: bold; 
        font-size: 1.1em;
        text-decoration: none !important;
        transition: background-color 0.3s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    a.btn-confirm:hover { background-color: #b71c1c; transform: translateY(-1px); }

    /* 기타 UI */
    .note-entry { background-color: #2c3e50; padding: 15px; border-radius: 5px; border-left: 4px solid #0288d1; margin-bottom: 10px; }
    .risk-tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; margin: 2px; border: 1px solid #ff5252; color: #ff867c; }
    .legend-item { display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-size: 0.75em; font-weight: bold; color: white; text-align: center; }
    
    div[data-testid="stDialog"] { background-color: #263238; color: #eceff1; }
    .stButton > button { background-color: #37474f; color: white; border: 1px solid #455a64; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { background-color: #263238; color: #b0bec5; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #0277bd; color: white; }

/* (수정) Streamlit 버튼을 Confirm 버튼처럼 보이게 */
div.stButton > button {
    width: 100%;
    background-color: #d32f2f;
    color: white;
    border: none;
    padding: 12px 0;
    border-radius: 6px;
    font-weight: bold;
    font-size: 1.1em;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: background-color 0.3s, transform 0.2s;
}
div.stButton > button:hover {
    background-color: #b71c1c;
    transform: translateY(-1px);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 4. 리소스 로딩
# --------------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    resources = {}
    try:
        resources['model'] = joblib.load('rf_fall_model.joblib')
        df_cols = pd.read_csv('rf_model_feature_columns.csv')
        resources['features'] = df_cols['feature'].tolist()
        try:
            resources['importance'] = pd.read_csv('rf_feature_importance_top10.csv')
        except:
            resources['importance'] = None
    except Exception as e:
        return None
    return resources

res = load_resources()

# --------------------------------------------------------------------------------
# 5. 상태 초기화 (데이터 유지)
# --------------------------------------------------------------------------------
if 'nursing_notes' not in st.session_state:
    st.session_state.nursing_notes = [{"time": "2025-12-12 08:00", "writer": "김분당", "content": "활력징후 측정함. 특이사항 없음."}]
if 'current_pt_idx' not in st.session_state: st.session_state.current_pt_idx = 0
if 'alarm_confirmed' not in st.session_state: st.session_state.alarm_confirmed = False
if 'last_detected_factors' not in st.session_state: st.session_state.last_detected_factors = []
if 'last_fall_score' not in st.session_state: st.session_state.last_fall_score = None
if 'last_confirmed_factors' not in st.session_state: st.session_state.last_confirmed_factors = []
if 'last_confirmed_score' not in st.session_state: st.session_state.last_confirmed_score = None

def confirm_alarm():
    """알람 확인 처리: 세션 상태를 유지한 채로 알람만 확인 처리합니다."""
    st.session_state.alarm_confirmed = True
    # 확인 당시 요인/점수 스냅샷 저장
    st.session_state.last_confirmed_factors = st.session_state.get('last_detected_factors', [])
    st.session_state.last_confirmed_score = st.session_state.get('last_fall_score', None)


# 시뮬레이션 변수 초기화 (개별 키 사용)
defaults = {
    'sim_sbp': 120, 'sim_dbp': 80, 'sim_pr': 80, 'sim_rr': 20, 
    'sim_bt': 36.5, 'sim_alb': 4.0, 'sim_crp': 0.5, 
    'sim_mental': '명료(Alert)', 'sim_meds': False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

PATIENTS_BASE = [
    {"id": "12345678", "bed": "04-01", "name": "김수면", "gender": "M", "age": 78, "diag": "Pneumonia", "doc": "김뇌혈", "nurse": "이간호"},
    {"id": "87654321", "bed": "04-02", "name": "이영희", "gender": "F", "age": 65, "diag": "Stomach Cancer", "doc": "박위장", "nurse": "최간호"},
    {"id": "11223344", "bed": "05-01", "name": "박민수", "gender": "M", "age": 82, "diag": "Femur Fracture", "doc": "최정형", "nurse": "김간호"},
    {"id": "99887766", "bed": "05-02", "name": "정수진", "gender": "F", "age": 32, "diag": "Appendicitis", "doc": "이외과", "nurse": "박간호"},
]

# --------------------------------------------------------------------------------
# 6. 예측 및 보정 함수
# --------------------------------------------------------------------------------
def calculate_risk_score(pt_static):
    # Session State의 최신 값을 바로 가져옴
    input_vals = {
        'sbp': st.session_state.sim_sbp,
        'dbp': st.session_state.sim_dbp,
        'pr': st.session_state.sim_pr,
        'rr': st.session_state.sim_rr,
        'bt': st.session_state.sim_bt,
        'albumin': st.session_state.sim_alb,
        'crp': st.session_state.sim_crp,
        'mental': st.session_state.sim_mental,
        'meds': st.session_state.sim_meds
    }

    # 1. AI 모델 예측
    base_score = 0
    if res and 'model' in res:
        model = res['model']
        feature_cols = res['features']
        
        input_data = {col: 0 for col in feature_cols}
        
        input_data['나이'] = pt_static['age']
        input_data['성별'] = 1 if pt_static['gender'] == 'M' else 0
        input_data['SBP'] = input_vals['sbp']
        input_data['DBP'] = input_vals['dbp']
        input_data['PR'] = input_vals['pr']
        input_data['RR'] = input_vals['rr']
        input_data['BT'] = input_vals['bt']
        input_data['albumin'] = input_vals['albumin']
        input_data['crp'] = input_vals['crp']
        
        mental_map = {"명료(Alert)": "alert", "기면(Drowsy)": "verbal response", "혼미(Stupor)": "painful response"}
        m_val = mental_map.get(input_vals['mental'], "alert")
        if f"내원시 반응_{m_val}" in input_data: input_data[f"내원시 반응_{m_val}"] = 1

        try:
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_cols]
            prob = model.predict_proba(input_df)[0][1]
            base_score = int(prob * 100)
        except:
            base_score = 10 

    # 2. 보정 로직 (가산점)
    calibration_score = 0
    
    if input_vals['albumin'] < 3.0: calibration_score += 30
    if input_vals['meds']: calibration_score += 30
    if pt_static['age'] >= 70: calibration_score += 10
    
    if input_vals['sbp'] < 90 or input_vals['sbp'] > 180: calibration_score += 15
    if input_vals['pr'] > 100: calibration_score += 10
    if input_vals['bt'] > 37.5: calibration_score += 5

    final_score = base_score + calibration_score
    return min(final_score, 99)

# --------------------------------------------------------------------------------
# 7. 팝업창
# --------------------------------------------------------------------------------
@st.dialog("낙상/욕창 위험도 정밀 분석", width="large")
def show_risk_details(name, factors, current_score):
    st.info(f"🕒 **{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}** 기준, {name} 님의 분석 결과입니다.")
    
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
                chk_rail = st.checkbox("침상 난간(Side Rail) 올림 확인", value=(current_score >= 40))
                chk_med = st.checkbox("💊 수면제 투여 후 30분 관찰", value=st.session_state.sim_meds)
                chk_nutri = st.checkbox("🥩 영양팀 협진 의뢰", value=(st.session_state.sim_alb < 3.0))
                chk_edu = st.checkbox("📢 낙상 예방 교육 및 호출기 위치 안내", value=True)

        st.markdown("---")
        if st.button("간호 수행 완료 및 기록 저장 (Auto-Charting)", type="primary", use_container_width=True):
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            risk_str = ", ".join(factors) if factors else "없음"
            actions = []
            if chk_rail: actions.append("침상난간 올림 확인")
            if chk_med: actions.append("투약 후 관찰")
            if chk_nutri: actions.append("영양팀 협진")
            if chk_edu: actions.append("예방 교육")
            
            note_content = f"낙상위험평가({current_score}점) -> 위험요인({risk_str}) 확인 -> 중재({', '.join(actions)}) 시행함."
            st.session_state.nursing_notes.insert(0, {"time": current_time, "writer": "김분당", "content": note_content})
            st.toast("저장되었습니다!")
            time.sleep(1)
            st.rerun()

    with tab2:
        st.markdown("##### 🔍 환자 맞춤형 위험 요인 (Top 10)")
        if res and res['importance'] is not None:
            df_imp = res['importance'].copy().sort_values('importance', ascending=True).tail(10)
            colors = []
            for feature in df_imp['feature']:
                color = "#e0e0e0"
                if feature == "나이" and PATIENTS_BASE[st.session_state.current_pt_idx]['age'] >= 65: color = "#ff5252"
                elif feature == "albumin" and st.session_state.sim_alb < 3.0: color = "#ff5252"
                elif feature == "SBP" and (st.session_state.sim_sbp < 100 or st.session_state.sim_sbp > 160): color = "#ff5252"
                elif feature == "PR" and st.session_state.sim_pr > 100: color = "#ff5252"
                colors.append(color)
            df_imp['color'] = colors
            
            chart = alt.Chart(df_imp).mark_bar().encode(
                x=alt.X('importance', title='기여도'),
                y=alt.Y('feature', sort='-x', title='변수명'),
                color=alt.Color('color', scale=None)
            ).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("중요도 데이터가 없습니다.")

# --------------------------------------------------------------------------------
# 8. 메인 레이아웃 구성
# --------------------------------------------------------------------------------
col_sidebar, col_main = st.columns([2, 8])
curr_pt_base = PATIENTS_BASE[st.session_state.current_pt_idx]

# [좌측 패널]
with col_sidebar:
    st.selectbox("근무 DUTY", ["Day", "Evening", "Night"])
    st.divider()

    st.markdown("### 🏥 재원 환자")
    idx = st.radio("환자 리스트", range(len(PATIENTS_BASE)), format_func=lambda i: f"[{PATIENTS_BASE[i]['bed']}] {PATIENTS_BASE[i]['name']}", label_visibility="collapsed")
    
    # 환자 변경 시 리셋
    if idx != st.session_state.current_pt_idx:
        st.session_state.current_pt_idx = idx
        st.session_state.alarm_confirmed = False 
        
        st.session_state.sim_sbp = 120
        st.session_state.sim_dbp = 80
        st.session_state.sim_pr = 80
        st.session_state.sim_rr = 20
        st.session_state.sim_bt = 36.5
        st.session_state.sim_alb = 4.0
        st.session_state.sim_crp = 0.5
        st.session_state.sim_mental = '명료(Alert)'
        st.session_state.sim_meds = False
        st.rerun()
    
    curr_pt_base = PATIENTS_BASE[idx]
    
    st.markdown("---")
    
    # 점수 계산
    fall_score = calculate_risk_score(curr_pt_base)
    sore_score = 15
    
    # 점수가 60 미만으로 떨어지면 알람 상태 리셋 (다시 위험해지면 뜨게)
    if fall_score < 60:
        st.session_state.alarm_confirmed = False

    f_color = "#ff5252" if fall_score >= 60 else ("#ffca28" if fall_score >= 30 else "#00e5ff")
    s_color = "#ff5252" if sore_score >= 18 else ("#ffca28" if sore_score >= 15 else "#00e5ff")
    
    alarm_class = ""
    if fall_score >= 60 and not st.session_state.alarm_confirmed:
        alarm_class = "alarm-active"

    # 가로형 계기판
    st.markdown(f"""
    <div class="digital-monitor-container {alarm_class}">
        <div class="score-box">
            <div class="monitor-label">FALL RISK</div>
            <div class="digital-number" style="color: {f_color};">{fall_score}</div>
        </div>
        <div class="divider-line"></div>
        <div class="score-box">
            <div class="monitor-label">SORE RISK</div>
            <div class="digital-number" style="color: {s_color};">{sore_score}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 위험 요인 텍스트
    detected_factors = []
    if curr_pt_base['age'] >= 65: detected_factors.append("고령")
    if st.session_state.sim_alb < 3.0: detected_factors.append("알부민 저하")
    if st.session_state.sim_meds: detected_factors.append("고위험 약물")
    if st.session_state.sim_sbp < 100: detected_factors.append("저혈압")
    if st.session_state.sim_pr > 100: detected_factors.append("빈맥")
    
    if st.button("🔍 상세 분석 및 중재 기록 열기", type="primary", use_container_width=True):
        show_risk_details(curr_pt_base['name'], detected_factors, fall_score)

# [우측 메인 패널]
with col_main:
    st.markdown(f"""
    <div class="header-container">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div style="display:flex; align-items:center;">
                <span style="font-size:1.5em; font-weight:bold; color:white; margin-right:20px;">🏥 SNUH</span>
                <span class="header-info-text"><span class="header-label">환자명:</span> <b>{curr_pt_base['name']}</b> ({curr_pt_base['gender']}/{curr_pt_base['age']}세)</span>
                <span class="header-info-text"><span class="header-label">ID:</span> {curr_pt_base['id']}</span>
                <span class="header-info-text"><span class="header-label">진단명:</span> <span style="color:#4fc3f7;">{curr_pt_base['diag']}</span></span>
            </div>
            <div style="color:#b0bec5; font-size:0.9em;">김분당 간호사 | {datetime.datetime.now().strftime('%Y-%m-%d')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🛡️ 통합뷰 (AI Simulation)", "💊 오더", "📝 간호기록(Auto-Note)"])

    with tab1:
        c1, c2 = st.columns([1.2, 1])
        
        with c1:
            st.markdown("##### ⚡ 실시간 데이터 입력 (Simulation)")
            with st.container(border=True):
                # [핵심] 위젯의 key를 session state와 1:1 매핑 -> 데이터 유지 및 즉시 반영
                r1, r2 = st.columns(2)
                st.number_input("SBP (수축기)", step=10, key="sim_sbp")
                st.number_input("DBP (이완기)", step=10, key="sim_dbp")
                r3, r4 = st.columns(2)
                st.number_input("PR (맥박)", step=5, key="sim_pr")
                st.number_input("RR (호흡)", step=2, key="sim_rr")
                st.number_input("BT (체온)", step=0.1, format="%.1f", key="sim_bt")
                
                st.slider("Albumin (영양)", 1.0, 5.5, key="sim_alb")
                st.selectbox("의식 상태", ["명료(Alert)", "기면(Drowsy)", "혼미(Stupor)"], key="sim_mental")
                st.checkbox("💊 고위험 약물(수면제 등) 복용", key="sim_meds")

        with c2:
            st.markdown("##### 📊 환자 상태 요약")
            st.markdown(f"""
            <div style="background-color:#263238; padding:15px; border-radius:8px; margin-bottom:15px;">
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; text-align:center;">
                    <div><div style="color:#aaa; font-size:12px;">BP</div><div style="font-weight:bold; font-size:18px;">{st.session_state.sim_sbp}/{st.session_state.sim_dbp}</div></div>
                    <div><div style="color:#aaa; font-size:12px;">PR</div><div style="font-weight:bold; font-size:18px;">{st.session_state.sim_pr}</div></div>
                    <div><div style="color:#aaa; font-size:12px;">RR</div><div style="font-weight:bold; font-size:18px;">{st.session_state.sim_rr}</div></div>
                    <div><div style="color:#aaa; font-size:12px;">BT</div><div style="font-weight:bold; font-size:18px;">{st.session_state.sim_bt}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**[감지된 위험 요인]**")
            if detected_factors:
                for f in detected_factors:
                    st.markdown(f"<span class='risk-tag'>{f}</span>", unsafe_allow_html=True)
            else:
                st.info("특이 사항 없음")

    with tab2: st.write("오더 화면입니다.")

    with tab3:
        st.markdown("##### 📋 간호진술문 (Nursing Note)")
        for note in st.session_state.nursing_notes:
            st.markdown(f"""
            <div class="note-entry">
                <div class="note-time">📅 {note['time']} | 작성자: {note['writer']}</div>
                <div>{note['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.text_area("추가 기록", height=100)
        st.button("저장")

# [NEW] 알람 (알람 박스 + Confirm 버튼: 시각적으로 박스 내부처럼 보이게, 상태 리셋 없음)
if fall_score >= 60 and not st.session_state.alarm_confirmed:
    factors_str = "<br>• ".join(detected_factors) if detected_factors else "복합적 요인"

    # 알람 박스 (HTML)
    st.markdown(f"""
    <div class="custom-alert-box">
        <div class="alert-title">🚨 낙상 고위험 감지! ({fall_score}점)</div>
        <div class="alert-content">
            환자의 상태 변화로 인해 낙상 위험도가 급격히 상승했습니다. 즉시 확인이 필요합니다.
        </div>
        <div class="alert-factors">
            <b>[감지된 주요 위험 요인]</b><br>
            • {factors_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ▶ 시각적으로 알람 박스 내부 버튼처럼 보이게 처리 (fixed 박스 아래에 붙이기)
    st.markdown("<div style='margin-top:-8px'></div>", unsafe_allow_html=True)

    if st.button("확인 (Confirm)", key="confirm_alarm_btn", use_container_width=True):
        confirm_alarm()
        st.rerun()

st.markdown("---")

legends = [("수술전","#e57373"), ("수술중","#ba68c8"), ("검사후","#7986cb"), ("퇴원","#81c784"), ("신규오더","#ffb74d")]
html = '<div style="display:flex; gap:10px;">' + "".join([f'<span class="legend-item" style="background:{c}">{l}</span>' for l,c in legends]) + '</div>'
st.markdown(html, unsafe_allow_html=True)
