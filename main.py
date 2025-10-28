# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="서울 연도별 평균기온 추이", page_icon="🌤️", layout="wide")

CSV_NAME = "seoult.csv"

@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    # 인코딩 유연하게 처리
    encodings = ["utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            continue
    if last_err and 'df' not in locals():
        raise last_err

    # 컬럼 존재 여부 점검 (필수 컬럼)
    required_cols = ["날짜", "지점", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 날짜 전처리: 탭/공백 제거 후 datetime 변환
    df["날짜"] = df["날짜"].astype(str).str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)

    # 숫자 컬럼을 float로 통일
    num_cols = ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

@st.cache_data(show_spinner=False)
def impute_and_aggregate(df: pd.DataFrame):
    """
    결측치 처리:
      - 날짜 인덱스 기준으로 시간 보간 (interpolate(method='time'))
      - 앞/뒤 가장자리 결측은 ffill/bfill로 보완
    집계:
      - 일평균 '평균기온(℃)'을 연도별 평균으로 집계
    """
    df2 = df.copy()
    df2 = df2.set_index("날짜").asfreq("D")  # 혹시 누락된 날짜가 있으면 캘린더를 데일리로 맞춤
    num_cols = ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]

    # 시간 축 보간
    df2[num_cols] = df2[num_cols].interpolate(method="time", limit_direction="both")

    # 남은 결측 가장자리 보완
    df2[num_cols] = df2[num_cols].ffill().bfill()

    # (선택) 평균기온 결측이 여전히 있으면 (최고+최저)/2로 보정
    need_patch = df2["평균기온(℃)"].isna()
    if need_patch.any():
        df2.loc[need_patch, "평균기온(℃)"] = (
            df2.loc[need_patch, "최고기온(℃)"] + df2.loc[need_patch, "최저기온(℃)"]
        ) / 2.0

    df2 = df2.reset_index()

    # 연도 컬럼 추가
    df2["연도"] = df2["날짜"].dt.year

    # 연도별 평균기온 집계
    yearly = (
        df2.groupby("연도", as_index=False)["평균기온(℃)"]
        .mean()
        .rename(columns={"평균기온(℃)": "연도별 평균기온(℃)"})
    )

    return df2, yearly

def make_chart(yearly: pd.DataFrame, year_min: int, year_max: int):
    base = alt.Chart(
        yearly[(yearly["연도"] >= year_min) & (yearly["연도"] <= year_max)]
    )

    line = base.mark_line(point=True).encode(
        x=alt.X("연도:O", title="연도"),
        y=alt.Y("연도별 평균기온(℃):Q", title="연도별 평균기온 (℃)"),
        tooltip=[
            alt.Tooltip("연도:O", title="연도"),
            alt.Tooltip("연도별 평균기온(℃):Q", title="평균기온(℃)", format=".2f"),
        ],
    )

    rule = alt.Chart(pd.DataFrame({
        "y": [yearly["연도별 평균기온(℃)"].mean()]
    })).mark_rule(strokeDash=[4,4]).encode(
        y="y:Q"
    )

    return (line + rule).properties(height=420)

def main():
    st.title("🌤️ 서울 연도별 평균기온 추이")
    st.caption("데이터: 동일 폴더의 seoult.csv · 결측치는 시간 보간으로 채움")

    # 파일 존재 확인
    if not Path(CSV_NAME).exists():
        st.error(f"현재 폴더에 `{CSV_NAME}` 파일이 없습니다. 파일을 업로드한 뒤 다시 시도하세요.")
        st.stop()

    df = load_data(CSV_NAME)
    st.success(f"데이터 로드 완료: {len(df):,}행, {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")

    # 결측치 요약(처리 전)
    num_cols = ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
    before_na = df[num_cols].isna().sum().rename("결측개수").to_frame()
    with st.expander("결측치 현황 (처리 전)"):
        st.dataframe(before_na)

    # 보간 및 연도별 집계
    daily_imputed, yearly = impute_and_aggregate(df)

    # 결측치 요약(처리 후)
    after_na = daily_imputed[num_cols].isna().sum().rename("결측개수").to_frame()
    with st.expander("결측치 현황 (처리 후)"):
        st.dataframe(after_na)

    # 연도 선택 슬라이더
    year_min, year_max = int(yearly["연도"].min()), int(yearly["연도"].max())
    sel_min, sel_max = st.slider(
        "표시할 연도 범위",
        min_value=year_min,
        max_value=year_max,
        value=(max(year_min, year_max-50), year_max),
        step=1,
    )

    st.subheader("연도별 평균기온 변화")
    chart = make_chart(yearly, sel_min, sel_max)
    st.altair_chart(chart, use_container_width=True)

    with st.expander("연도별 평균기온 표 (다운로드 가능)"):
        st.dataframe(yearly)
        st.download_button(
            "연도별 평균기온 CSV 다운로드",
            data=yearly.to_csv(index=False).encode("utf-8"),
            file_name="yearly_mean_temperature_seoul.csv",
            mime="text/csv",
        )

    with st.expander("일자별(보간 후) 데이터 미리보기"):
        st.dataframe(daily_imputed.head(20))

    st.caption("보간 방식: 날짜 인덱스 기준 time interpolation → 가장자리 ffill/bfill → 필요 시 (최고+최저)/2 보정")

if __name__ == "__main__":
    main()
