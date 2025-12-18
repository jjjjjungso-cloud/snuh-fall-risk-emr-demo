import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 1. 페이지 설정 및 데이터 로드
st.set_page_config(page_title="SNUH AI Fall Monitor", layout="wide")

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('risk_score_model.joblib')
        ref_scores = np.load('train_score_ref.npz')['train_scores_sorted']
        return model, ref_scores
    except: return None, None

model, ref_scores = load_resources()

# 세션 상태(간호기록 저장용) 초기화
if 'nursing_log' not in st.session_state:
    st.session_state.nursing_log = []

# --------------------------------------------------------------------------------
# 2. 사이드바: 11개 입력창 (데이터 입력)
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 환자 데이터 입력")
    age = st.number_input("나이 (Age)", 0, 120, 65)
    gender = st.selectbox("성별", ["남성 (M)", "여성 (F)"])
    severity = st.selectbox("중증도분류", [1, 2, 3, 4, 5], index=1)
    
    c1, c2 = st.columns(2)
    sbp = c1.number_input("SBP (수축기)", 50, 250, 120)
    dbp = c2.number_input("DBP (이완기)", 30, 150, 80)
    
    c3, c4, c5 = st.columns(3)
    pr = c3.number_input("PR (맥박)", 20, 200, 75)
    rr = c4.number_input("RR (호흡)", 5, 50, 18)
    bt = c5.number_input("BT (체온)", 30.0, 45.0, 36.5, step=0.1)
    
    mental = st.selectbox("내원시 반응", ["alert", "verbal response", "painful response", "unresponsive"])
    alb = st.slider("Albumin (영양)", 1.0, 5.0, 4.0, step=0.1)
    crp = st.number_input("CRP (염증)", 0.0, 50.0, 0.2, step=0.1)

# --------------------------------------------------------------------------------
# 3. 위험 요인 분석 로직
# --------------------------------------------------------------------------------
detected_risks = []
if sbp < 100 or dbp < 60: detected_risks.append("저혈압/어지럼증")
if alb < 3.5: detected_risks.append("영양부족/근력약화")
if crp > 1.0 or bt >= 37.8: detected_risks.append("염증/발열")
if mental != "alert": detected_risks.append("의식저하/인지장애")
if age >= 75: detected_risks.append("고령(고위험군)")

# 중재 옵션 정의
intervention_options = {
    "공통/기본": ["침대 난간(Side Rail) 상시 고정", "낙상 예방 표지판 부착", "호출벨 위치 확인 및 교육"],
    "저혈압/어지럼증": ["체위 변경 시 천천히 움직이도록 교육", "보행 시 반드시 보호자 동행", "기립성 저혈압 모니터링"],
    "영양부족/근력약화": ["고단백 식이 권장", "재활의학과 협진(근력 강화)", "침상 옆 보조기구 배치"],
    "염증/발열": ["수분 섭취 권장", "I/O 체크 및 탈수 모니터링", "활력징후 2시간 간격 모니터링"],
    "의식저하/인지장애": ["환자 근거리 배치(Station 앞)", "보호자 상주 교육", "섬망 예방 중재(시계/달력 비치)"],
    "고령(고위험군)": ["야간 조명 유지", "비끄럼 방지 양말 착용 확인", "화장실 이동 시 보조"]
}

# --------------------------------------------------------------------------------
# 4. 메인 화면: 결과 노출 및 중재 선택
# --------------------------------------------------------------------------------
st.title("🏥 AI 기반 낙상 위험 중재 시스템")

# [결과 노출 영역]
input_df = pd.DataFrame([{'성별': 1 if "남성" in gender else 0, '중증도분류': severity, 'SBP': sbp, 'DBP': dbp, 'RR': rr, 'PR': pr, 'BT': bt, '내원시 반응': mental, '나이': age, 'albumin': alb, 'crp': crp}])

if model:
    prob = model.predict_proba(input_df)[0][1]
    fall_score = int(np.searchsorted(ref_scores, prob) / len(ref_scores) * 100)
else:
    fall_score = 45 # 더미 데이터

c_res, c_gauge = st.columns([6, 4])
with c_res:
    st.subheader("📊 낙상 위험 분석 결과")
    if fall_score >= 80:
        st.error(f"### 고위험군 (상위 {100-fall_score}%) - 점수: {fall_score}점")
    elif fall_score >= 60:
        st.warning(f"### 주의군 (상위 {100-fall_score}%) - 점수: {fall_score}점")
    else:
        st.success(f"### 저위험군 (상위 {100-fall_score}%) - 점수: {fall_score}점")

# [중재 선택 영역]
st.divider()
st.subheader("💉 위험 요인별 맞춤 간호 중재 선택")
st.info(f"💡 분석된 위험 요인: {', '.join(detected_risks) if detected_risks else '특이요인 없음'}")

selected_interventions = []

# 위험 요인별로 체크박스 생성
cols = st.columns(len(detected_risks) + 1)
with cols[0]:
    st.write("**[공통 중재]**")
    for action in intervention_options["공통/기본"]:
        if st.checkbox(action, key=f"base_{action}"):
            selected_interventions.append(action)

for i, risk in enumerate(detected_risks):
    with cols[i+1]:
        st.write(f"**[{risk}]**")
        for action in intervention_options[risk]:
            if st.checkbox(action, key=f"{risk}_{action}"):
                selected_interventions.append(action)

# --------------------------------------------------------------------------------
# 5. 간호기록 연동 (자동 텍스트 생성)
# --------------------------------------------------------------------------------
st.divider()
if st.button("📝 간호기록 전송 및 저장", use_container_width=True):
    if not selected_interventions:
        st.warning("수행한 중재 내용을 선택해주세요.")
    else:
        # 간호기록 텍스트 생성
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        interventions_text = ", ".join(selected_interventions)
        record = f"[{timestamp}] [AI 낙상스크리닝: {fall_score}점] {interventions_text} 시행함."
        
        # 세션에 저장 (기록 리스트 상단에 추가)
        st.session_state.nursing_log.insert(0, record)
        st.balloons()
        st.success("간호기록이 성공적으로 저장되었습니다.")

# [저장된 간호기록 리스트 표시]
st.subheader("📄 최근 간호기록 (Nursing Note History)")
for log in st.session_state.nursing_log:
    st.text_area(label="Log Item", value=log, height=70, label_visibility="collapsed")
