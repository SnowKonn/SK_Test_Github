"""
Fama-French Style Size Factor (SMB) Performance
================================================
- 매년 6월 말 시가총액 기준 Small/Big 포트폴리오 구성
- 7월~익년 6월 보유, 시총 가중 수익률 계산
- SMB = Small - Big
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 데이터 로드 ──────────────────────────────────────────────
DATA_DIR = Path(
    "C:/Users/snofl/Desktop/Quant/# PARA/1_Projects/"
    "260213_investor_activity/Research_flow/Pre_research/rawdata_수급_parquet"
)

print("데이터 로드 중...")
price = pd.read_parquet(DATA_DIR / "price.parquet")
mcap_monthly = pd.read_parquet(DATA_DIR / "market_cap_monthly.parquet")

# ── 월간 수익률 계산 ────────────────────────────────────────
# 일간 → 월말 가격 추출 후 월간 수익률
price.index = pd.to_datetime(price.index)
mcap_monthly.index = pd.to_datetime(mcap_monthly.index)

price_monthly = price.resample("ME").last()
ret_monthly = price_monthly.pct_change()

# price와 mcap에 공통으로 있는 종목만 사용
common_stocks = price_monthly.columns.intersection(mcap_monthly.columns)
ret_monthly = ret_monthly[common_stocks]
mcap_monthly = mcap_monthly[common_stocks]

print(f"공통 종목 수: {len(common_stocks)}")
print(f"수익률 기간: {ret_monthly.index[0].strftime('%Y-%m')} ~ {ret_monthly.index[-1].strftime('%Y-%m')}")

# ── Fama-French 포트폴리오 구성 ─────────────────────────────
# 매년 6월 말 시총 기준, 7월~익년 6월 보유
def get_june_mcap(year):
    """해당 연도 6월 말 시가총액 추출"""
    june_dates = mcap_monthly.index[(mcap_monthly.index.year == year) & (mcap_monthly.index.month == 6)]
    if len(june_dates) == 0:
        return None
    return mcap_monthly.loc[june_dates[0]].dropna()


def compute_portfolio_returns(year):
    """year년 6월 기준 정렬 → year년 7월 ~ year+1년 6월 수익률"""
    june_mcap = get_june_mcap(year)
    if june_mcap is None or len(june_mcap) < 20:
        return None

    # 시총 중간값 기준 Small/Big 분류
    median_mcap = june_mcap.median()
    small_stocks = june_mcap[june_mcap <= median_mcap].index
    big_stocks = june_mcap[june_mcap > median_mcap].index

    # 보유 기간: 7월 ~ 익년 6월
    hold_start = pd.Timestamp(year, 7, 1)
    hold_end = pd.Timestamp(year + 1, 6, 30)
    hold_mask = (ret_monthly.index >= hold_start) & (ret_monthly.index <= hold_end)
    ret_period = ret_monthly.loc[hold_mask]

    if ret_period.empty:
        return None

    # 시총 가중 수익률 (매월 초 시총 = 이전 월말 시총)
    records = []
    for date in ret_period.index:
        # 해당 월의 시총 가중치: 직전 월말 mcap 사용
        prev_dates = mcap_monthly.index[mcap_monthly.index < date]
        if len(prev_dates) == 0:
            continue
        prev_mcap = mcap_monthly.loc[prev_dates[-1]]

        for label, stocks in [("Small", small_stocks), ("Big", big_stocks)]:
            valid = stocks.intersection(ret_period.columns)
            r = ret_period.loc[date, valid].dropna()
            w = prev_mcap[r.index].dropna()
            common = r.index.intersection(w.index)
            if len(common) < 5:
                continue
            r, w = r[common], w[common]
            w = w / w.sum()  # 정규화
            vw_ret = (r * w).sum()
            records.append({"date": date, "portfolio": label, "return": vw_ret, "n_stocks": len(common)})

    return pd.DataFrame(records)


# 전 기간 팩터 수익률 계산
print("\n포트폴리오 구성 중...")
all_results = []
for year in range(2004, 2026):
    result = compute_portfolio_returns(year)
    if result is not None:
        all_results.append(result)

df = pd.concat(all_results, ignore_index=True)

# 피벗: Small/Big 월간 수익률
pivot = df.pivot_table(index="date", columns="portfolio", values="return")
pivot["SMB"] = pivot["Small"] - pivot["Big"]
pivot = pivot.sort_index()

# ── 성과 지표 계산 ──────────────────────────────────────────
def performance_summary(series, name):
    """연환산 성과 지표"""
    n = len(series)
    ann_ret = (1 + series).prod() ** (12 / n) - 1
    ann_vol = series.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum_ret = (1 + series).prod() - 1
    rolling_max = (1 + series).cumprod().cummax()
    drawdown = (1 + series).cumprod() / rolling_max - 1
    max_dd = drawdown.min()
    win_rate = (series > 0).mean()

    return {
        "Portfolio": name,
        "Ann. Return": f"{ann_ret:.2%}",
        "Ann. Vol": f"{ann_vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Cum. Return": f"{cum_ret:.2%}",
        "Max DD": f"{max_dd:.2%}",
        "Win Rate": f"{win_rate:.1%}",
        "Months": n,
    }


print("\n" + "=" * 70)
print("  Fama-French Size Factor (SMB) Performance Summary")
print("=" * 70)

summary = []
for col in ["Small", "Big", "SMB"]:
    s = pivot[col].dropna()
    summary.append(performance_summary(s, col))

summary_df = pd.DataFrame(summary).set_index("Portfolio")
print(summary_df.to_string())

# ── 연도별 수익률 ───────────────────────────────────────────
print("\n" + "-" * 70)
print("  연도별 수익률")
print("-" * 70)

yearly = pivot.groupby(pivot.index.year).apply(lambda x: (1 + x).prod() - 1)
yearly.columns = ["Small", "Big", "SMB"]
yearly.index.name = "Year"
print(yearly.map(lambda x: f"{x:.2%}").to_string())

# ── 누적 수익률 저장 ────────────────────────────────────────
cumret = (1 + pivot).cumprod()
cumret.to_csv("size_factor_cumulative.csv")
pivot.to_csv("size_factor_monthly.csv")
print(f"\n결과 저장 완료: size_factor_cumulative.csv, size_factor_monthly.csv")

# ── 최근 5년 vs 전체 비교 ──────────────────────────────────
print("\n" + "-" * 70)
print("  기간별 비교")
print("-" * 70)

recent_5y = pivot[pivot.index >= "2020-01-01"]
first_half = pivot[pivot.index < "2015-01-01"]

for label, subset in [("전체 기간", pivot), ("~2014", first_half), ("2020~", recent_5y)]:
    smb = subset["SMB"].dropna()
    if len(smb) > 0:
        ann = (1 + smb).prod() ** (12 / len(smb)) - 1
        vol = smb.std() * np.sqrt(12)
        t_stat = smb.mean() / (smb.std() / np.sqrt(len(smb))) if smb.std() > 0 else 0
        print(f"  {label:12s} | Ann.Ret: {ann:+.2%} | Vol: {vol:.2%} | t-stat: {t_stat:.2f} | N={len(smb)}")
