import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 1. 초기 설정 및 모델 로드
st.set_page_config(page_title="SNUH AI Fall Monitor", layout="wide")

@st.cache_resource
def load_resources():
    try:
        # 실제 모델 파일이 있는 경우 로드, 없는 경우 None 반환
        model = joblib.load('risk_score_model.joblib')
        ref_scores = np.load('train_score_ref.npz')['train_scores_sorted']
        return model, ref_scores
    except: 
        return None, None

model, ref_scores = load_resources()

if 'nursing_log' not in st.session_state:
    st.session_state.nursing_log = []

# --------------------------------------------------------------------------------
# 2. 환자 데이터 설정 (김분당 - 기존 C안 데이터)
# --------------------------------------------------------------------------------
# 나이 45, 알부민 4.5, CRP 0.1, 의식 alert, 중증도 5, 여성
patient_data = {
    "name": "김분당",
    "age": 45,
    "alb": 4.5,
    "crp": 0.1,
    "mental": "alert",
    "sev": 5,
    "gender": "여성 (F)"
}

# --------------------------------------------------------------------------------
# 3. 사이드바: 11개 입력창 (김분당 데이터 기본값)
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header(f"👤 환자: {patient_data['name']}")
    st.write("실시간 수치를 조정하여 위험도를 확인하세요.")
    st.divider()
    
    # 기본값으로 김분당 데이터 세팅
    age = st.number_input("나이 (Age)", 0, 120, patient_data["age"])
    gender = st.selectbox("성별", ["남성 (M)", "여성 (F)"], index=1) # 여성 기본
    severity = st.selectbox("중증도분류", [1, 2, 3, 4, 5], index=4) # 5단계 기본
    
    c1, c2 = st.columns(2)
    sbp = c1.number_input("SBP (수축기)", 50, 250, 120)
    dbp = c2.number_input("DBP (이완기)", 30, 150, 80)
    
    c3, c4, c5 = st.columns(3)
    pr = c3.number_input("PR (맥박)", 20, 200, 75)
    rr = c4.number_input("RR (호흡)", 5, 50, 18)
    bt = c5.number_input("BT (체온)", 30.0, 45.0, 36.5, step=0.1)
    
    mental = st.selectbox("내원시 반응", ["alert", "verbal response", "painful response", "unresponsive"], index=0)
    alb = st.slider("Albumin (영양)", 1.0, 5.0, patient_data["alb"], step=0.1)
    crp = st.number_input("CRP (염증)", 0.0, 50.0, patient_data["crp"], step=0.1)

# --------------------------------------------------------------------------------
# 4. AI 분석 및 결과 계산
# --------------------------------------------------------------------------------
input_df = pd.DataFrame([{
    '성별': 1 if "남성" in gender else 0, 
    '중증도분류': severity, 
    'SBP': sbp, 'DBP': dbp, 'RR': rr, 'PR': pr, 'BT': bt, 
    '내원시 반응': mental, 
    '나이': age, 'albumin': alb, 'crp': crp
}])

if model is not None and ref_scores is not None:
    prob = model.predict_proba(input_df)[0][1]
    fall_score = int(np.searchsorted(ref_scores, prob) / len(ref_scores) * 100)
else:
    # 모델 파일이 없을 경우 시뮬레이션을 위한 임시 계산식 (정상일 때 낮은 점수 유지)
    base_score = 20
    if age > 70: base_score += 30
    if alb < 3.0: base_score += 20
    if severity < 3: base_score += 15
    fall_score = min(base_score, 100)

# --------------------------------------------------------------------------------
# 5. 메인 화면 출력
# --------------------------------------------------------------------------------
st.title("🏥 SNUH AI 낙상 위험 대시보드")
st.subheader(f"현재 환자: {patient_data['name']} (Baseline)")

col_res, col_info = st.columns([5, 5])

with col_res:
    # 결과 시각화
    if fall_score >= 80:
        st.error(f"## 분석 결과: 고위험군 ({fall_score}점)")
        status_text = "🚩 즉각적인 예방 중재가 필요한 상태입니다."
    elif fall_score >= 60:
        st.warning(f"## 분석 결과: 주의군 ({fall_score}점)")
        status_text = "⚠️ 수치 변화를 주의 깊게 관찰하십시오."
    else:
        st.success(f"## 분석 결과: 저위험군 ({fall_score}점)")
        status_text = "✅ 현재 매우 안정적인 상태입니다."
    
    st.write(status_text)

with col_info:
    st.info("💡 **환자 상태 요약**")
    st.write(f"- **영양/염증:** Albumin {alb} / CRP {crp}")
    st.write(f"- **활력징후:** BP {sbp}/{dbp} | PR {pr} | BT {bt}℃")
    st.write(f"- **인적요인:** {age}세 | 중증도 {severity}단계 | {mental}")

# --------------------------------------------------------------------------------
# 6. 간호 중재 및 기록
# --------------------------------------------------------------------------------
st.divider()
st.subheader("📝 간호 중재 선택 및 기록")

# 위험 요인에 따른 중재 제안
st.write("해당 환자에게 시행한 중재를 선택하세요:")
c_int1, c_int2 = st.columns(2)

with c_int1:
    i1 = st.checkbox("침대 난간(Side Rail) 고정 확인")
    i2 = st.checkbox("낙상 예방 표지판 부착")
with c_int2:
    i3 = st.checkbox("호출벨 사용법 재교육")
    i4 = st.checkbox("야간 조명 및 바닥 환경 확인")

if st.button("간호기록(Nursing Note) 전송", use_container_width=True):
    selected = []
    if i1: selected.append("Side Rail 고정")
    if i2: selected.append("예방 표지판 부착")
    if i3: selected.append("호출벨 교육")
    if i4: selected.append("환경 점검")
    
    if not selected:
        st.warning("선택된 중재 항목이 없습니다.")
    else:
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        log_entry = f"[{now}] [AI 낙상점수: {fall_score}점] {', '.join(selected)} 시행함."
        st.session_state.nursing_log.insert(0, log_entry)
        st.success("기록이 저장되었습니다.")

# 기록 히스토리 표시
if st.session_state.nursing_log:
    st.write("---")
    st.write("**최근 간호기록 히스토리**")
    for log in st.session_state.nursing_log[:5]: # 최근 5개만 표시
        st.caption(log)
