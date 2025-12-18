import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 1. 초기 설정
st.set_page_config(page_title="SNUH AI Fall Monitor", layout="wide")

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('risk_score_model.joblib')
        ref_scores = np.load('train_score_ref.npz')['train_scores_sorted']
        return model, ref_scores
    except: return None, None

model, ref_scores = load_resources()

if 'nursing_log' not in st.session_state:
    st.session_state.nursing_log = []

# --------------------------------------------------------------------------------
# 2. 중재 옵션 정의 (선생님이 주신 리스트 반영)
# --------------------------------------------------------------------------------
intervention_options = {
    "공통/기본": ["침대 난간(Side Rail) 상시 고정", "낙상 예방 표지판 부착", "호출벨 위치 확인 및 교육"],
    "저혈압/어지럼증": ["체위 변경 시 천천히 움직이도록 교육", "보행 시 반드시 보호자 동행", "기립성 저혈압 모니터링"],
    "영양부족/근력약화": ["고단백 식이 권장", "재활의학과 협진(근력 강화)", "침상 옆 보조기구 배치"],
    "염증/발열": ["수분 섭취 권장", "I/O 체크 및 탈수 모니터링", "활력징후 2시간 간격 모니터링"],
    "의식저하/인지장애": ["환자 근거리 배치(Station 앞)", "보호자 상주 교육", "섬망 예방 중재(시계/달력 비치)"],
    "고령(고위험군)": ["야간 조명 유지", "미끄럼 방지 양말 착용 확인", "화장실 이동 시 보조"]
}

# --------------------------------------------------------------------------------
# 3. 사이드바: 김분당 환자 데이터 입력
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("👤 환자: 김분당")
    st.divider()
    
    age = st.number_input("나이 (Age)", 0, 120, 45)
    gender = st.selectbox("성별", ["남성 (M)", "여성 (F)"], index=1)
    severity = st.selectbox("중증도분류", [1, 2, 3, 4, 5], index=4)
    
    c1, c2 = st.columns(2)
    sbp = c1.number_input("SBP", 50, 250, 120)
    dbp = c2.number_input("DBP", 30, 150, 80)
    
    c3, c4, c5 = st.columns(3)
    pr = c3.number_input("PR", 20, 200, 75)
    rr = c4.number_input("RR", 5, 50, 18)
    bt = c5.number_input("BT", 30.0, 45.0, 36.5, step=0.1)
    
    mental = st.selectbox("내원시 반응", ["alert", "verbal response", "painful response", "unresponsive"], index=0)
    alb = st.slider("Albumin", 1.0, 5.0, 4.5, step=0.1)
    crp = st.number_input("CRP", 0.0, 50.0, 0.1, step=0.1)

# --------------------------------------------------------------------------------
# 4. 실시간 위험 요인 감지 로직
# --------------------------------------------------------------------------------
detected_risks = ["공통/기본"] # 항상 기본으로 포함
if sbp < 100 or dbp < 60: detected_risks.append("저혈압/어지럼증")
if alb < 3.5: detected_risks.append("영양부족/근력약화")
if crp > 0.5 or bt >= 37.8: detected_risks.append("염증/발열")
if mental != "alert": detected_risks.append("의식저하/인지장애")
if age >= 75: detected_risks.append("고령(고위험군)")

# --------------------------------------------------------------------------------
# 5. 메인 화면: 결과 표출
# --------------------------------------------------------------------------------
st.title("🏥 SNUH AI 낙상 모니터링 & 맞춤형 중재")

# 점수 계산
input_df = pd.DataFrame([{'성별': 1 if "남성" in gender else 0, '중증도분류': severity, 'SBP': sbp, 'DBP': dbp, 'RR': rr, 'PR': pr, 'BT': bt, '내원시 반응': mental, '나이': age, 'albumin': alb, 'crp': crp}])

if model:
    prob = model.predict_proba(input_df)[0][1]
    fall_score = int(np.searchsorted(ref_scores, prob) / len(ref_scores) * 100)
else:
    fall_score = 25 # 모델 없을 시 기본 점수

# 대시보드 출력
c_res, c_gauge = st.columns([6, 4])
with c_res:
    if fall_score >= 80:
        st.error(f"## 분석 결과: 고위험군 ({fall_score}점)")
    elif fall_score >= 60:
        st.warning(f"## 분석 결과: 주의군 ({fall_score}점)")
    else:
        st.success(f"## 분석 결과: 저위험군 ({fall_score}점)")
    st.write(f"현재 감지된 위험 요인: **{', '.join(detected_risks)}**")

# --------------------------------------------------------------------------------
# 6. 맞춤형 중재 선택 및 기록 연동 (선생님의 요청 사항)
# --------------------------------------------------------------------------------
st.divider()
st.subheader("💊 맞춤형 간호 중재 선택")
st.caption("환자의 상태에 따라 필요한 중재 옵션이 자동으로 활성화됩니다.")

selected_actions = []

# 감지된 위험 요인별로 섹션을 나누어 중재 옵션 표시
num_cols = min(len(detected_risks), 3)
cols = st.columns(num_cols)

for i, risk in enumerate(detected_risks):
    with cols[i % num_cols]:
        st.markdown(f"**[{risk}]**")
        for action in intervention_options.get(risk, []):
            if st.checkbox(action, key=f"{risk}_{action}"):
                selected_actions.append(action)

# 간호기록 전송 버튼
st.write("")
if st.button("📝 선택한 중재를 간호기록(EMR)으로 전송", use_container_width=True):
    if not selected_actions:
        st.warning("수행한 중재 항목을 선택해주세요.")
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        # 간호기록 문구 생성
        note = f"[{timestamp}] [AI 낙상점수: {fall_score}점] {', '.join(selected_actions)} 시행함."
        st.session_state.nursing_log.insert(0, note)
        st.success("기록이 성공적으로 전송되었습니다.")

# 기록 히스토리
if st.session_state.nursing_log:
    st.divider()
    st.subheader("📄 간호기록 히스토리")
    for log in st.session_state.nursing_log[:5]:
        st.info(log)
