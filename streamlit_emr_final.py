import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 1. 초기 설정
st.set_page_config(page_title="SNUH AI Fall Monitor", layout="wide")

# 모델 로드 (생략 가능, 파일 없을 시 더미 점수 활용)
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
# 2. 환자 케이스 정의 (선생님께서 주신 A/B안)
# --------------------------------------------------------------------------------
CASE_PRESETS = {
    "A안: 염증/영양 악화 케이스": {
        "age": 65, "alb": 4.0, "crp": 0.2, "mental": "alert", "sev": 2, "gender": "남성 (M)"
    },
    "B안: 반응 저하 + 고령 케이스": {
        "age": 82, "alb": 4.0, "crp": 0.2, "mental": "alert", "sev": 2, "gender": "여성 (F)"
    }
}

# --------------------------------------------------------------------------------
# 3. 사이드바: 11개 입력창
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 환자 데이터 입력")
    
    # 케이스 선택 버튼
    selected_case = st.radio("시뮬레이션 케이스 선택", list(CASE_PRESETS.keys()))
    preset = CASE_PRESETS[selected_case]
    
    st.divider()
    
    # 프리셋 데이터 바인딩
    age = st.number_input("나이 (Age)", 0, 120, preset["age"])
    gender = st.selectbox("성별", ["남성 (M)", "여성 (F)"], index=0 if preset["gender"] == "남성 (M)" else 1)
    severity = st.selectbox("중증도분류", [1, 2, 3, 4, 5], index=preset["sev"]-1)
    
    c1, c2 = st.columns(2)
    sbp = c1.number_input("SBP", 50, 250, 120)
    dbp = c2.number_input("DBP", 30, 150, 80)
    
    c3, c4, c5 = st.columns(3)
    pr = c3.number_input("PR", 20, 200, 75)
    rr = c4.number_input("RR", 5, 50, 18)
    bt = c5.number_input("BT", 30.0, 45.0, 36.5, step=0.1)
    
    mental = st.selectbox("내원시 반응", ["alert", "verbal response", "painful response", "unresponsive"], 
                          index=["alert", "verbal response", "painful response", "unresponsive"].index(preset["mental"]))
    
    alb = st.slider("Albumin", 1.0, 5.0, preset["alb"], step=0.1)
    crp = st.number_input("CRP", 0.0, 50.0, preset["crp"], step=0.1)

# --------------------------------------------------------------------------------
# 4. 분석 로직 (위험 요인 및 점수)
# --------------------------------------------------------------------------------
# 위험 요인 감지
detected_risks = []
if age >= 75: detected_risks.append("고령(High Age)")
if alb < 3.5: detected_risks.append("저알부민혈증(Albumin ↓)")
if crp > 0.5: detected_risks.append("염증 수치 상승(CRP ↑)")
if sbp < 100: detected_risks.append("저혈압/어지럼증 위험")
if mental != "alert": detected_risks.append("의식/인지 변화")

# 점수 계산 (모델 기반)
input_df = pd.DataFrame([{'성별': 1 if "남성" in gender else 0, '중증도분류': severity, 'SBP': sbp, 'DBP': dbp, 'RR': rr, 'PR': pr, 'BT': bt, '내원시 반응': mental, '나이': age, 'albumin': alb, 'crp': crp}])

if model:
    prob = model.predict_proba(input_df)[0][1]
    fall_score = int(np.searchsorted(ref_scores, prob) / len(ref_scores) * 100)
else:
    # 모델 없을 시 데모용 가중치 (고령일수록, Alb 낮을수록 상승)
    base = 40
    if age > 80: base += 35
    if alb < 3.5: base += 20
    fall_score = min(base, 99)

# --------------------------------------------------------------------------------
# 5. 메인 화면
# --------------------------------------------------------------------------------
st.title("🏥 AI 낙상 위험 분석 및 간호중재")

# [결과 섹션]
st.subheader(f"🔍 {selected_case} 분석")
c_res, c_risk = st.columns([4, 6])

with c_res:
    if fall_score >= 80:
        st.error(f"## 위험도: 고위험군 ({fall_score}점)")
        st.write("👉 **상위 20% 이내**의 낙상 위험군입니다.")
    else:
        st.warning(f"## 위험도: 일반관리군 ({fall_score}점)")
        st.write("👉 수치 변화에 따른 지속적인 모니터링이 필요합니다.")

with c_risk:
    st.markdown("**감지된 임상 위험 요인:**")
    if detected_risks:
        for r in detected_risks:
            st.markdown(f"- ⚠️ {r}")
    else:
        st.write("- 특이 위험 요인 없음 (기본 예방 수칙 준수)")



# [중재 섹션]
st.divider()
st.subheader("💉 맞춤형 간호 중재 선택")

# 중재 데이터베이스
intervention_db = {
    "기본": ["침대 난간(Side Rail) 고정", "낙상 예방 표지판 부착", "취침 전 배뇨 확인"],
    "고령(High Age)": ["야간 조명 유지", "미끄럼 방지 양말 착용 확인", "휠체어 이동 시 보조"],
    "저알부민혈증(Albumin ↓)": ["고단백 식이 교육", "근력 약화에 따른 보행 보조", "침상 옆 호출벨 위치 재확인"],
    "염증 수치 상승(CRP ↑)": ["활력징후 집중 모니터링", "염증 완화 시까지 거동 제한 교육"],
    "저혈압/어지럼증 위험": ["기립성 저혈압 예방 교육", "체위 변경 시 단계적 이동"],
    "의식/인지 변화": ["보호자 상주 강화", "환자 근거리 배치", "지남력 확인"]
}

selected_interventions = []
cols = st.columns(3)

# 1. 공통 중재
with cols[0]:
    st.write("**[공통 중재]**")
    for act in intervention_db["기본"]:
        if st.checkbox(act, key=act): selected_interventions.append(act)

# 2. 감지된 위험 요인별 중재 (동적 생성)
for i, risk in enumerate(detected_risks):
    with cols[(i + 1) % 3]:
        st.write(f"**[{risk} 맞춤 중재]**")
        for act in intervention_db.get(risk, []):
            if st.checkbox(act, key=f"{risk}_{act}"): selected_interventions.append(act)

# [간호기록 연동 섹션]
st.divider()
if st.button("📝 선택한 중재 간호기록으로 전송", use_container_width=True):
    if not selected_interventions:
        st.warning("중재 항목을 선택해주세요.")
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        note = f"[{timestamp}] [AI 낙상스크리닝: {fall_score}점] {', '.join(selected_interventions)} 시행함."
        st.session_state.nursing_log.insert(0, note)
        st.success("간호기록이 연동되었습니다.")

st.subheader("📄 간호기록 히스토리 (EMR)")
for log in st.session_state.nursing_log:
    st.info(log)
