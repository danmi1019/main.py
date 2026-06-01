import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import io

st.set_page_config(
    page_title="서울 기온 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🌡️ 서울 계절별 기온 변화 분석")
st.markdown("**1900년대 vs 2000년대** — 계절별 평균기온 차이 가설 검증")

# ── 데이터 로드 ──────────────────────────────────────────────
@st.cache_data
def process_data(raw_bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = df.columns.str.strip()
    df["날짜"] = df["날짜"].str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df["월"]   = df["날짜"].dt.month

    def get_season(m):
        if m in [3, 4, 5]:     return "봄"
        elif m in [6, 7, 8]:   return "여름"
        elif m in [9, 10, 11]: return "가을"
        else:                   return "겨울"

    df["계절"] = df["월"].apply(get_season)
    df["시대"]  = df["연도"].apply(
        lambda y: "1900년대" if 1900 <= y <= 1999 else
                  ("2000년대" if y >= 2000 else None)
    )
    return df[df["시대"].notna()].copy()

# ── 파일 업로드 UI ────────────────────────────────────────────
with st.sidebar:
    st.header("📂 데이터 업로드")
    uploaded = st.file_uploader(
        "기상청 CSV 파일을 업로드하세요",
        type=["csv"],
        help="기상청 일별 기온 데이터 (날짜, 지점, 평균기온, 최저기온, 최고기온)"
    )

if uploaded is None:
    st.info("👈 왼쪽 사이드바에서 CSV 파일을 업로드하면 분석이 시작됩니다.")
    st.markdown("""
    **필요한 파일 형식**
    - 기상청 일별 기온 관측 자료 (CSV)
    - 컬럼: `날짜`, `지점`, `평균기온(℃)`, `최저기온(℃)`, `최고기온(℃)`
    """)
    st.stop()

df = process_data(uploaded.read())

# ── 사이드바 설정 ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 분석 설정")
    selected_seasons = st.multiselect(
        "계절 선택",
        ["봄", "여름", "가을", "겨울"],
        default=["봄", "여름", "가을", "겨울"]
    )
    temp_col = st.radio(
        "기온 지표",
        ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
    )
    st.divider()
    st.caption(f"📅 데이터 기간: {df['날짜'].min().year} ~ {df['날짜'].max().year}")
    st.caption(f"📊 총 관측일: {len(df):,}일")

if not selected_seasons:
    st.warning("계절을 하나 이상 선택해주세요.")
    st.stop()

# ── 연간 계절별 평균 계산 ─────────────────────────────────────
df_f = df[df["계절"].isin(selected_seasons)]
yearly = (
    df_f.groupby(["연도", "계절", "시대"])[temp_col]
    .mean()
    .reset_index()
    .rename(columns={temp_col: "기온"})
)

SEASON_ORDER = [s for s in ["봄", "여름", "가을", "겨울"] if s in selected_seasons]
SEASON_COLOR = {"봄": "#4CAF50", "여름": "#F44336", "가을": "#FF9800", "겨울": "#2196F3"}
ERA_COLOR    = {"1900년대": "#5C7AEA", "2000년대": "#F76B6B"}

# ── KPI 카드 ─────────────────────────────────────────────────
st.markdown("### 📊 시대별 계절 평균기온 요약")
cols = st.columns(len(SEASON_ORDER))
for i, season in enumerate(SEASON_ORDER):
    s_data = yearly[yearly["계절"] == season]
    t1 = s_data[s_data["시대"] == "1900년대"]["기온"].mean()
    t2 = s_data[s_data["시대"] == "2000년대"]["기온"].mean()
    delta = t2 - t1
    cols[i].metric(
        label=f"{season}",
        value=f"{t2:.2f} °C",
        delta=f"{delta:+.2f} °C (2000s vs 1900s)",
        delta_color="inverse" if season == "겨울" else "normal"
    )

st.divider()

# ── 탭 ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 연도별 추이", "📦 분포 비교", "🔬 통계 검정", "🗓️ 월별 히트맵"]
)

# ────────────────────────────────────────────────────────────
# Tab 1: 연도별 추이
# ────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"계절별 {temp_col} 연도별 추이")

    n = len(SEASON_ORDER)
    ncols = 2 if n > 1 else 1
    nrows = (n + 1) // 2

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=SEASON_ORDER,
        shared_xaxes=False,
        vertical_spacing=0.12
    )

    for idx, season in enumerate(SEASON_ORDER):
        r = idx // 2 + 1
        c = idx % 2 + 1
        s = yearly[yearly["계절"] == season].sort_values("연도").copy()
        s["MA5"] = s["기온"].rolling(5, center=True).mean()

        # 산점도
        fig.add_trace(go.Scatter(
            x=s["연도"], y=s["기온"],
            mode="markers",
            marker=dict(color=SEASON_COLOR[season], size=4, opacity=0.35),
            name=season,
            legendgroup=season,
            showlegend=(idx == 0),
            hovertemplate="%{x}년: %{y:.1f}°C<extra></extra>"
        ), row=r, col=c)

        # 5년 이동평균
        fig.add_trace(go.Scatter(
            x=s["연도"], y=s["MA5"],
            mode="lines",
            line=dict(color=SEASON_COLOR[season], width=2.5),
            name=f"{season} 5yr MA",
            legendgroup=season,
            showlegend=False,
            hovertemplate="%{x}년 MA5: %{y:.1f}°C<extra></extra>"
        ), row=r, col=c)

        # 선형 추세선
        valid = s.dropna(subset=["기온"])
        z = np.polyfit(valid["연도"], valid["기온"], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=valid["연도"], y=p(valid["연도"]),
            mode="lines",
            line=dict(color="black", width=1.5, dash="dot"),
            name="추세선",
            legendgroup="trend",
            showlegend=(idx == 0),
            hovertemplate=f"추세: {z[0]:+.3f}°C/yr<extra></extra>"
        ), row=r, col=c)

        # 2000년대 음영
        fig.add_vrect(
            x0=2000, x1=int(s["연도"].max()),
            fillcolor=ERA_COLOR["2000년대"],
            opacity=0.07, layer="below", line_width=0,
            row=r, col=c
        )

    fig.update_layout(
        height=360 * nrows,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 음영 = 2000년대 | 점: 연간 평균 | 실선: 5년 이동평균 | 점선: 선형 추세")

# ────────────────────────────────────────────────────────────
# Tab 2: 분포 비교
# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("1900년대 vs 2000년대 기온 분포 (바이올린)")

    n = len(SEASON_ORDER)
    ncols = 2 if n > 1 else 1
    nrows = (n + 1) // 2

    fig_v = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=SEASON_ORDER,
        vertical_spacing=0.15
    )

    for idx, season in enumerate(SEASON_ORDER):
        r = idx // 2 + 1
        c = idx % 2 + 1
        s_data = yearly[yearly["계절"] == season]

        for era in ["1900년대", "2000년대"]:
            vals = s_data[s_data["시대"] == era]["기온"].dropna()
            fig_v.add_trace(go.Violin(
                y=vals,
                name=era,
                box_visible=True,
                meanline_visible=True,
                fillcolor=ERA_COLOR[era],
                line_color=ERA_COLOR[era],
                opacity=0.65,
                points="outliers",
                legendgroup=era,
                showlegend=(idx == 0),
                hovertemplate=f"{era} {season}<br>%{{y:.1f}}°C<extra></extra>"
            ), row=r, col=c)

    fig_v.update_layout(
        height=360 * nrows,
        template="plotly_white",
        violinmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_v.update_yaxes(title_text="기온 (°C)")
    st.plotly_chart(fig_v, use_container_width=True)

# ────────────────────────────────────────────────────────────
# Tab 3: 통계 검정
# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🔬 Welch's t-검정")
    st.markdown(
        "**귀무가설 H₀**: 1900년대와 2000년대의 계절 평균기온에 차이가 없다  \n"
        "**유의수준 α = 0.05**"
    )

    results = []
    for season in SEASON_ORDER:
        s = yearly[yearly["계절"] == season]
        g1 = s[s["시대"] == "1900년대"]["기온"].dropna()
        g2 = s[s["시대"] == "2000년대"]["기온"].dropna()
        if len(g1) < 3 or len(g2) < 3:
            continue
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        cohen_d = (g2.mean() - g1.mean()) / pooled_std if pooled_std > 0 else 0
        ci_low, ci_high = stats.t.interval(
            0.95, df=len(g1)+len(g2)-2,
            loc=g2.mean()-g1.mean(),
            scale=np.sqrt(g1.var()/len(g1) + g2.var()/len(g2))
        )
        results.append({
            "계절":         season,
            "1900s 평균(°C)": round(g1.mean(), 2),
            "2000s 평균(°C)": round(g2.mean(), 2),
            "차이 Δ(°C)":   round(g2.mean() - g1.mean(), 2),
            "95% CI":       f"[{ci_low:+.2f}, {ci_high:+.2f}]",
            "t 통계량":      round(t_stat, 3),
            "p-값":         round(p_val, 4),
            "Cohen's d":    round(cohen_d, 3),
            "결론":         "✅ 귀무가설 기각" if p_val < 0.05 else "❌ 기각 실패",
        })

    rdf = pd.DataFrame(results)

    def highlight_result(val):
        if val == "✅ 귀무가설 기각":
            return "color: #1a7c3e; font-weight: bold"
        elif val == "❌ 기각 실패":
            return "color: #c0392b"
        return ""

    styled = (
        rdf.style
           .applymap(highlight_result, subset=["결론"])
           .format({"p-값": "{:.4f}", "차이 Δ(°C)": "{:+.2f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # 효과 크기 막대 차트
    st.markdown("#### Cohen's d — 효과 크기")
    fig_d = go.Figure(go.Bar(
        x=[r["계절"] for r in results],
        y=[r["Cohen's d"] for r in results],
        marker_color=[SEASON_COLOR[r["계절"]] for r in results],
        text=[("{:+.3f}".format(r["Cohen's d"])) for r in results],
        textposition="outside"
    ))
    fig_d.add_hline(y=0.2,  line_dash="dot", line_color="gray",  annotation_text="소(0.2)")
    fig_d.add_hline(y=0.5,  line_dash="dot", line_color="orange", annotation_text="중(0.5)")
    fig_d.add_hline(y=0.8,  line_dash="dot", line_color="red",    annotation_text="대(0.8)")
    fig_d.update_layout(
        template="plotly_white", height=300,
        yaxis_title="Cohen's d", xaxis_title="계절",
        showlegend=False
    )
    st.plotly_chart(fig_d, use_container_width=True)

    st.info("""
    **해석 가이드**
    - **p < 0.05**: 통계적으로 유의미한 기온 차이 존재 → 귀무가설 기각
    - **Cohen's d**: 효과 크기 — 소(0.2) · 중(0.5) · 대(0.8)
    - **Welch's t-검정**: 두 집단의 분산이 달라도 적용 가능한 독립표본 t-검정
    """)

# ────────────────────────────────────────────────────────────
# Tab 4: 월별 히트맵
# ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("시대별 월 × 연도 기온 히트맵")

    col_a, col_b = st.columns(2)
    for col_ui, era in zip([col_a, col_b], ["1900년대", "2000년대"]):
        monthly = (
            df[df["시대"] == era]
            .groupby(["연도", "월"])[temp_col]
            .mean()
            .reset_index()
        )
        pivot = monthly.pivot_table(index="연도", columns="월", values=temp_col)
        pivot.columns = [f"{m}월" for m in pivot.columns]

        fig_h = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            title=f"{era}",
            labels=dict(color="°C", x="월", y="연도"),
            aspect="auto",
            zmin=df[temp_col].quantile(0.02),
            zmax=df[temp_col].quantile(0.98),
        )
        fig_h.update_layout(height=420, template="plotly_white",
                            coloraxis_colorbar=dict(title="°C"))
        col_ui.plotly_chart(fig_h, use_container_width=True)

    # 차이 히트맵
    st.markdown("#### 2000년대 − 1900년대 월별 평균 기온 차이")
    m1 = df[df["시대"]=="1900년대"].groupby("월")[temp_col].mean()
    m2 = df[df["시대"]=="2000년대"].groupby("월")[temp_col].mean()
    diff = (m2 - m1).reset_index()
    diff.columns = ["월", "차이(°C)"]
    diff["월명"] = diff["월"].map({
        1:"1월",2:"2월",3:"3월",4:"4월",5:"5월",6:"6월",
        7:"7월",8:"8월",9:"9월",10:"10월",11:"11월",12:"12월"
    })

    fig_diff = go.Figure(go.Bar(
        x=diff["월명"], y=diff["차이(°C)"],
        marker_color=[
            "#F44336" if v > 0 else "#2196F3" for v in diff["차이(°C)"]
        ],
        text=[f"{v:+.2f}°C" for v in diff["차이(°C)"]],
        textposition="outside"
    ))
    fig_diff.add_hline(y=0, line_color="black", line_width=1)
    fig_diff.update_layout(
        template="plotly_white", height=300,
        yaxis_title="기온 차이 (°C)", xaxis_title="월",
        showlegend=False
    )
    st.plotly_chart(fig_diff, use_container_width=True)

st.divider()
st.caption("📍 데이터 출처: 기상청 서울(지점 108) 일별 기온 관측 자료")
