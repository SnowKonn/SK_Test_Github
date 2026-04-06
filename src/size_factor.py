"""
Fama-French Style Size Factor (SMB) Performance
================================================
- 매년 6월 말 시가총액(FS0001) 기준 Small/Big 포트폴리오 구성
- 7월~익년 6월 보유, 시총 가중 수익률 계산
- SMB = Small - Big
- 데이터 소스: quant_db (factors, factor_values, stock_price)
"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pathlib import Path

load_dotenv()

OUT_DIR = Path(__file__).resolve().parent.parent / "out"
OUT_DIR.mkdir(exist_ok=True)

# ── DB 연결 ─────────────────────────────────────────────────
def get_engine():
    h, p = os.getenv("DB_HOST"), os.getenv("DB_PORT")
    u, pw = os.getenv("DB_USER"), os.getenv("DB_PASSWORD")
    db = os.getenv("DB_NAME")
    return create_engine(f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{db}")


# ── 데이터 로드 ─────────────────────────────────────────────
print("DB에서 데이터 로드 중...")
engine = get_engine()

# 시가총액 (FS0001) — 월말 기준
mcap = pd.read_sql(
    """
    SELECT date, stock_code, value_num AS mcap
    FROM factor_values
    WHERE factor_code = 'FS0001'
    ORDER BY date, stock_code
    """,
    engine,
    parse_dates=["date"],
)

# 월간 수익률 (view 활용)
ret = pd.read_sql(
    """
    SELECT rebal_date AS date, stock_code, monthly_return AS ret
    FROM view_monthly_stock_returns
    ORDER BY rebal_date, stock_code
    """,
    engine,
    parse_dates=["date"],
)

engine.dispose()

# 피벗
mcap_pivot = mcap.pivot(index="date", columns="stock_code", values="mcap")
ret_pivot = ret.pivot(index="date", columns="stock_code", values="ret")

common_stocks = mcap_pivot.columns.intersection(ret_pivot.columns)
mcap_pivot = mcap_pivot[common_stocks]
ret_pivot = ret_pivot[common_stocks]

print(f"종목 수: {len(common_stocks)}")
print(f"시총 기간: {mcap_pivot.index.min().strftime('%Y-%m')} ~ {mcap_pivot.index.max().strftime('%Y-%m')}")
print(f"수익률 기간: {ret_pivot.index.min().strftime('%Y-%m')} ~ {ret_pivot.index.max().strftime('%Y-%m')}")


# ── Fama-French 포트폴리오 구성 ─────────────────────────────
def get_june_mcap(year):
    """해당 연도 6월 말 시가총액 추출"""
    june = mcap_pivot.index[(mcap_pivot.index.year == year) & (mcap_pivot.index.month == 6)]
    if len(june) == 0:
        return None
    return mcap_pivot.loc[june[0]].dropna()


def compute_portfolio_returns(year):
    """year년 6월 기준 정렬 → year년 7월 ~ year+1년 6월 수익률"""
    june_mcap = get_june_mcap(year)
    if june_mcap is None or len(june_mcap) < 20:
        return None

    median_mcap = june_mcap.median()
    small_stocks = june_mcap[june_mcap <= median_mcap].index
    big_stocks = june_mcap[june_mcap > median_mcap].index

    hold_start = pd.Timestamp(year, 7, 1)
    hold_end = pd.Timestamp(year + 1, 6, 30)
    hold_mask = (ret_pivot.index >= hold_start) & (ret_pivot.index <= hold_end)
    ret_period = ret_pivot.loc[hold_mask]

    if ret_period.empty:
        return None

    records = []
    for date in ret_period.index:
        prev_dates = mcap_pivot.index[mcap_pivot.index < date]
        if len(prev_dates) == 0:
            continue
        prev_mcap = mcap_pivot.loc[prev_dates[-1]]

        for label, stocks in [("Small", small_stocks), ("Big", big_stocks)]:
            valid = stocks.intersection(ret_period.columns)
            r = ret_period.loc[date, valid].dropna()
            w = prev_mcap[r.index].dropna()
            common = r.index.intersection(w.index)
            if len(common) < 5:
                continue
            r, w = r[common], w[common]
            w = w / w.sum()
            vw_ret = (r * w).sum()
            records.append({
                "date": date, "portfolio": label,
                "return": vw_ret, "n_stocks": len(common),
            })

    return pd.DataFrame(records)


# ── 전 기간 계산 ────────────────────────────────────────────
print("\n포트폴리오 구성 중...")
years = range(mcap_pivot.index.min().year, mcap_pivot.index.max().year + 1)
all_results = [r for y in years if (r := compute_portfolio_returns(y)) is not None]

df = pd.concat(all_results, ignore_index=True)
pivot = df.pivot_table(index="date", columns="portfolio", values="return")
pivot["SMB"] = pivot["Small"] - pivot["Big"]
pivot = pivot.sort_index()


# ── 성과 지표 ────────────────────────────────────────────────
def performance_summary(series, name):
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

summary_df = pd.DataFrame(
    [performance_summary(pivot[c].dropna(), c) for c in ["Small", "Big", "SMB"]]
).set_index("Portfolio")
print(summary_df.to_string())

# ── 연도별 수익률 ───────────────────────────────────────────
print("\n" + "-" * 70)
print("  연도별 수익률")
print("-" * 70)

yearly = pivot.groupby(pivot.index.year).apply(lambda x: (1 + x).prod() - 1)
yearly.columns = ["Small", "Big", "SMB"]
yearly.index.name = "Year"
print(yearly.map(lambda x: f"{x:.2%}").to_string())

# ── 저장 ────────────────────────────────────────────────────
cumret = (1 + pivot).cumprod()
cumret.to_csv(OUT_DIR / "size_factor_cumulative.csv")
pivot.to_csv(OUT_DIR / "size_factor_monthly.csv")
print(f"\n결과 저장: {OUT_DIR / 'size_factor_cumulative.csv'}")
print(f"          {OUT_DIR / 'size_factor_monthly.csv'}")

# ── 기간별 비교 ─────────────────────────────────────────────
print("\n" + "-" * 70)
print("  기간별 비교")
print("-" * 70)

for label, subset in [
    ("전체 기간", pivot),
    ("~2014", pivot[pivot.index < "2015-01-01"]),
    ("2020~", pivot[pivot.index >= "2020-01-01"]),
]:
    smb = subset["SMB"].dropna()
    if len(smb) > 0:
        ann = (1 + smb).prod() ** (12 / len(smb)) - 1
        vol = smb.std() * np.sqrt(12)
        t_stat = smb.mean() / (smb.std() / np.sqrt(len(smb))) if smb.std() > 0 else 0
        print(f"  {label:12s} | Ann.Ret: {ann:+.2%} | Vol: {vol:.2%} | t-stat: {t_stat:.2f} | N={len(smb)}")
