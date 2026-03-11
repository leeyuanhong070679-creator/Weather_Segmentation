import os
import argparse
from typing import List, Tuple, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data(
    csv_path: str,
    date_col: str = "time",
    temp_col: str = "temperature_2m",   # 小时温度列
    rain_col: str = "rain",             # 小时降雨(每小时量)
    snow_col: str = "snowfall",         # 小时降雪(每小时量)
) -> pd.DataFrame:
    """
    小时数据 -> 日数据（用于复用原来的季节分段逻辑）
    输出列：
      time(日), temperature_2m_min, temperature_2m_mean, rain_sum, snowfall_sum
    """
    df = pd.read_csv(csv_path)

    # 解析时间
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # 数值列转数值
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    if rain_col in df.columns:
        df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce")
    if snow_col in df.columns:
        df[snow_col] = pd.to_numeric(df[snow_col], errors="coerce")

    # 设为时间索引，按天聚合
    df = df.set_index(date_col)

    agg = {
        temp_col: ["min", "mean"],
    }
    if rain_col in df.columns:
        agg[rain_col] = "sum"
    if snow_col in df.columns:
        agg[snow_col] = "sum"

    dfd = df.resample("D").agg(agg)

    # 展平多层列名，并对齐到你原脚本需要的名字
    dfd.columns = [
        "temperature_2m_min" if c == (temp_col, "min") else
        "temperature_2m_mean" if c == (temp_col, "mean") else
        "rain_sum" if c == (rain_col, "sum") else
        "snowfall_sum" if c == (snow_col, "sum") else
        f"{c[0]}_{c[1]}"
        for c in dfd.columns
    ]

    # 把 time 放回普通列
    dfd = dfd.reset_index().rename(columns={date_col: "time"})

    # 去掉温度缺失的天（否则 winter/split 会出问题）
    dfd = dfd.dropna(subset=["temperature_2m_min", "temperature_2m_mean"]).reset_index(drop=True)

    return dfd


def winter(
    df: pd.DataFrame,
    T0: float = 0.0,
    g: int = 10,
    min_len: int = 7,
    post_cold_min: int = 3,
    date_col: str = "time",
    tmin_col: str = "temperature_2m_min",
    write_cold: bool = True,
) -> Dict:
    """
    只负责冬季划分（基于 df 的最低温列）：
    1) cold[t] = 1 if tmin < T0 else 0
    2) 连续段构建：允许 cold=0 连续长度 <= g 不断开
    3) 过滤太短段：段长 < min_len 的剔除
    """
    tmin = df[tmin_col].tolist()

    cold = [1 if x < T0 else 0 for x in tmin]
    if write_cold:
        df = df.copy()
        df["cold"] = cold

    N = len(tmin)
    W_idx: List[Tuple[int, int]] = []

    in_seg = False
    l = -1
    last_cold_idx = -1

    warm_run = 0

    # 新增：候选结束相关状态
    pending_end = False       # 是否处于“候选结束(回暖>g)”状态
    r_candidate = -1          # 候选结束点（最后一个 cold=1 的位置）
    warm_gap = 0              # 当前候选回暖长度（>g 后的继续累计）
    post_cold_run = 0         # 回暖后重新进入 cold 的连续天数

    for i in range(N):
        if cold[i] == 1:
            if not in_seg:
                in_seg = True
                l = i
            last_cold_idx = i
            warm_run = 0
            
            if pending_end:
                # 回暖后重新变冷：统计连续冷天数
                post_cold_run += 1

                # 若回暖期长度 <= min_len 且冷回归足够长 -> 桥接成功，冬季不断开
                if warm_gap <= min_len and post_cold_run >= post_cold_min:
                    pending_end = False
                    r_candidate = -1
                    warm_gap = 0
                    post_cold_run = 0

        else:
            # cold=0
            if in_seg:
                warm_run += 1

                if not pending_end:
                    # 未进入候选结束
                    if warm_run > g:
                        pending_end = True
                        r_candidate = last_cold_idx
                        warm_gap = warm_run   # 已>g
                        post_cold_run = 0
                else:
                    # 已在候选结束：继续累计回暖长度
                    warm_gap += 1

                # 若回暖期长度超过 min_len -> 直接确认冬季结束
                if pending_end and warm_gap > min_len:
                    r = r_candidate
                    if r >= l and (r - l + 1) >= min_len:
                        W_idx.append((l, r))

                    # 重置所有状态（退出冬季段）
                    in_seg = False
                    l = -1
                    last_cold_idx = -1
                    warm_run = 0

                    pending_end = False
                    r_candidate = -1
                    warm_gap = 0
                    post_cold_run = 0

    # 收尾
    if in_seg:
        # 若候选结束未被桥接成功，则按候选结束点截断；否则用最后冷日
        if pending_end and r_candidate != -1:
            r = r_candidate
        else:
            r = last_cold_idx

        if r >= l and (r - l + 1) >= min_len:
            W_idx.append((l, r))

    dates = df[date_col].tolist()
    W_date = [(dates[l].date(), dates[r].date()) for (l, r) in W_idx]

    return {
        "df": df,
        "W_idx": W_idx,
        "W_date": W_date,
        "K_w": len(W_idx),
        "params": {
            "T0": T0,
            "g": g,
            "min_len": min_len,
            "post_cold_min": post_cold_min,
            "date_col": date_col,
            "tmin_col": tmin_col,
        },
    }


def non_winter(
    N: int,
    W_idx: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    冬季区间补集 -> 非冬季连续区间列表 [(p,q),...]
    """
    if N <= 0:
        return []
    if not W_idx:
        return [(0, N - 1)]

    W_sorted = sorted(W_idx)
    res: List[Tuple[int, int]] = []
    cur = 0

    for l, r in W_sorted:
        if l > cur:
            res.append((cur, l - 1))
        cur = max(cur, r + 1)

    if cur <= N - 1:
        res.append((cur, N - 1))

    return res


def split(
    tmean: List[float],
    p: int,
    q: int,
    min_seg_len: int = 15,
    smooth_win: int = 7,
    # --- tau2 的新策略参数 ---
    run_need: int = 10,      # 连续多少天都“低于夏季参考温度”
    delta: float = 6.5,     # 低于参考温度的裕度（°C），避免贴线抖动
    use_median: bool = True # 夏季参考温度用 median(更稳) 或 mean
) -> Tuple[Optional[Tuple[int, int]], Tuple[int, int], Optional[Tuple[int, int]]]:
    """
    对非冬季区间 [p,q] 做三段切分（保证最高温在夏季）
    返回：(spring_seg, summer_seg, autumn_seg)，均为全局索引闭区间

    tau1：峰值左侧最大上升斜率点
    tau2：优先取“首次出现连续 run_need 天都低于夏季参考温度(由 tau1->peak 得到) 的起点”
          找不到则回退到“最陡下降点”
    """
    y = np.array(tmean[p: q + 1], dtype=float)
    L = len(y)

    if L < 3 * min_seg_len:
        # 太短，不切：整体视为夏季
        return (None, (p, q), None)

    # =========================
    # 1) 平滑（移动平均）
    # =========================
    if smooth_win > 1:
        pad = smooth_win // 2
        ypad = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(smooth_win, dtype=float) / smooth_win
        y_sm = np.convolve(ypad, kernel, mode="valid")
    else:
        y_sm = y

    # =========================
    # 2) 峰值（强制夏季包含峰值）
    # =========================
    peak_local = int(np.argmax(y_sm))

    # 防止峰值过靠近边界导致无法切三段
    left_limit = min_seg_len
    right_limit = L - min_seg_len - 1
    peak_local = int(np.clip(peak_local, left_limit, right_limit))

    # 一阶差分：dy[i] = y_sm[i+1] - y_sm[i]
    dy = np.diff(y_sm)

    # =========================
    # 3) tau1
    # =========================
    base_r = min(L - 1, min_seg_len - 1)
    spring_base = y_sm[0: base_r + 1]
    spring_ref = float(np.median(spring_base)) if use_median else float(np.mean(spring_base))
    thr_up = spring_ref + delta

    search_l = min_seg_len
    search_r = peak_local - min_seg_len

    tau1_local = None
    run = 0
    run_start = None

    if search_r >= search_l:
        for t in range(search_l, search_r + 1):
            if y_sm[t] >= thr_up:
                if run == 0:
                    run_start = t
                run += 1
                if run >= run_need:
                    # summer 从 run_start 开始 => spring 结束在 run_start-1
                    tau1_local = run_start - 1
                    break
            else:
                run = 0
                run_start = None

    # 找不到就回退：最大上升斜率
    if tau1_local is None:
        left_start = min_seg_len - 1
        left_end = peak_local - min_seg_len
        if left_end <= left_start:
            tau1_local = peak_local - min_seg_len
        else:
            tau1_local = int(np.argmax(dy[left_start: left_end + 1]) + left_start)

    tau1_local = int(np.clip(tau1_local, min_seg_len - 1, peak_local - min_seg_len))

    # =========================
    # 4) tau2：峰值右侧“首次持续低于夏季参考温度”的起点（否则回退最陡下降）
    # =========================
    right_start = peak_local + min_seg_len
    right_end = L - min_seg_len - 1

    if right_start > right_end:
        tau2_local = peak_local + min_seg_len
    else:
        # ---- 4.1 用 [tau1+1, peak] 作为“确定夏季核心”算参考温度 ----
        core_l = tau1_local + 1
        core_r = peak_local
        if core_l > core_r:
            # 极端情况下兜底（理论上不会发生）
            core_l = max(0, peak_local - min_seg_len)
            core_r = peak_local

        summer_core = y_sm[core_l: core_r + 1]
        summer_ref = float(np.median(summer_core)) if use_median else float(np.mean(summer_core))

        # ---- 4.2 找首次连续 run_need 天都 <= summer_ref - delta 的起点 ----
        tau2_local = None
        run = 0
        run_start = None
        thr = summer_ref - delta

        # 注意：这里直接在 y_sm[t] 上判断“温度是否低于阈值”，而不是用 dy
        for t in range(right_start, right_end + 1):
            if y_sm[t] <= thr:
                if run == 0:
                    run_start = t
                run += 1
                if run >= run_need:
                    tau2_local = run_start
                    break
            else:
                run = 0
                run_start = None

        # ---- 4.3 如果找不到持续低温 onset，回退到“最陡下降点” ----
        if tau2_local is None:
            lo = right_start - 1  # dy 对应 i->i+1
            # 在 [lo, right_end-1] 上找最小 dy，得到切点位置 i+1
            tau2_local = int(np.argmin(dy[lo: right_end]) + lo + 1)

    # =========================
    # 5) 拼回全局索引
    # =========================
    tau1 = p + tau1_local
    tau2 = p + tau2_local

    spring = (p, tau1)
    summer = (tau1 + 1, tau2)
    autumn = (tau2 + 1, q)

    return spring, summer, autumn


def segmentation(
    df: pd.DataFrame,
    W_idx: List[Tuple[int, int]],
    tmean_col: str = "temperature_2m_mean",
    min_seg_len: int = 15,
    date_col: str = "time",
) -> Dict:
    """
    输出：
      - NW_idx: 非冬季区间
      - seasons: [{"spring":..., "summer":..., "autumn":...}, ...]
    """
    N = len(df)
    tmean = df[tmean_col].tolist()
    dates = df[date_col].tolist()

    NW_idx = non_winter(N, W_idx)
    NW_date = [(dates[l].date(), dates[r].date()) for (l, r) in NW_idx]

    seasons: List[Dict] = []

    spring_idx: List[Tuple[int, int]] = []
    summer_idx: List[Tuple[int, int]] = []
    autumn_idx: List[Tuple[int, int]] = []

    for p, q in NW_idx:
        spring, summer, autumn = split(tmean, p, q, min_seg_len=min_seg_len)

        seasons.append({"spring": spring, "summer": summer, "autumn": autumn})

        if spring is not None:
            spring_idx.append(spring)
        if summer is not None:
            summer_idx.append(summer)
        if autumn is not None:
            autumn_idx.append(autumn)

    spring_date = [(dates[l].date(), dates[r].date()) for (l, r) in spring_idx]
    summer_date = [(dates[l].date(), dates[r].date()) for (l, r) in summer_idx]
    autumn_date = [(dates[l].date(), dates[r].date()) for (l, r) in autumn_idx]

    return {
        "NW_idx": NW_idx,
        "NW_date": NW_date,
        "seasons": seasons,
        "spring_idx": spring_idx,
        "spring_date": spring_date,
        "summer_idx": summer_idx,
        "summer_date": summer_date,
        "autumn_idx": autumn_idx,
        "autumn_date": autumn_date,
    }


def graph(
    df: pd.DataFrame,
    W_idx: List[Tuple[int, int]],
    seasons: List[Dict],
    save_path: str,
    date_col: str = "time",
    tmin_col: str = "temperature_2m_min",
    tmean_col: str = "temperature_2m_mean",
    rain_col: str = "rain_sum",
    snow_col: str = "snowfall_sum",
    title: str = "Season Segmentation",
):
    # --- 列存在性检查（避免 KeyError）---
    need_cols = [date_col, tmin_col, tmean_col, rain_col, snow_col]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nCurrent columns: {list(df.columns)}")

    # --- 保存目录不存在则创建（Windows下常见）---
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    dates = df[date_col]
    tmin = df[tmin_col]
    tmean = df[tmean_col]
    rain = df[rain_col]
    snow = df[snow_col]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # ==================
    # 背景色：季节分段（先画春夏秋，再画冬季，让冬季更“压得住”）
    # ==================
    # 统一风格：背景尽量淡，不和线/柱的颜色冲突
    season_style = {
        "spring": ("#BFE7C3", 0.7),  # 淡绿
        "summer": ("#FFD1DC", 0.7),  # 淡红
        "autumn": ("#FFD2B8", 0.7),  # 淡橙
    }
    winter_style = ("#C7D7FF", 0.7)  # 淡蓝（冬季）

    # 春夏秋背景
    for s in seasons:
        for name in ("spring", "summer", "autumn"):
            seg = s.get(name)
            if seg is None:
                continue
            l, r = seg
            if l is None or r is None:
                continue
            c, a = season_style[name]
            ax1.axvspan(dates.iloc[l], dates.iloc[r], color=c, alpha=a, zorder=0)

    # 冬季背景（最后画，更明显）
    for l, r in W_idx:
        ax1.axvspan(dates.iloc[l], dates.iloc[r], color=winter_style[0], alpha=winter_style[1], zorder=0)

    # ==================
    # 左轴：温度（线）
    # ==================
    ax1.plot(dates, tmin, color="#6BAED6", alpha=0.70, linewidth=2.0, label="Min Temperature (°C)", zorder=3)
    ax1.plot(dates, tmean, color="#E31A1C", alpha=0.95, linewidth=2.2, label="Mean Temperature (°C)", zorder=4)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6, zorder=2)

    ax1.set_ylabel("Temperature (°C)")
    ax1.set_xlabel("Date")

    # ==================
    # 右轴：雨 + 雪（柱）
    # ==================
    ax2 = ax1.twinx()
    ax2.bar(dates, rain, width=1.0, color="#FDAE6B", alpha=0.55, label="Rain (mm)", zorder=1)
    ax2.bar(dates, snow, width=1.0, bottom=rain, color="#74C9E5", alpha=0.75, label="Snow (mm)", zorder=1)
    ax2.set_ylabel("Precipitation (mm)")

    # ==================
    # 图例：包含季节patch
    # ==================
    from matplotlib.patches import Patch

    season_patches = [
        Patch(facecolor=season_style["spring"][0], alpha=season_style["spring"][1], label="Spring"),
        Patch(facecolor=season_style["summer"][0], alpha=season_style["summer"][1], label="Summer"),
        Patch(facecolor=season_style["autumn"][0], alpha=season_style["autumn"][1], label="Autumn"),
        Patch(facecolor=winter_style[0], alpha=winter_style[1], label="Winter"),
    ]

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2 + season_patches,
        labels1 + labels2 + [p.get_label() for p in season_patches],
        loc="upper left",
        framealpha=0.95,
    )

    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def day_seg_to_hour_seg(
    df_hour: pd.DataFrame,
    day_start: pd.Timestamp,   # 例如 2024-09-03 00:00:00
    day_end: pd.Timestamp,     # 例如 2024-11-12 00:00:00（日表的time通常是当天0点）
    time_col: str = "time",
) -> Tuple[int, int]:
    """
    把 [day_start, day_end]（按天闭区间）映射到 df_hour 的行索引闭区间 [l_hour, r_hour]
    假设 df_hour 已按 time_col 升序排序，且 time_col 是 datetime。
    """
    # 取整到当天 00:00
    start_ts = pd.Timestamp(day_start).normalize()
    # 结束取到 day_end 当天的 23:00（小时数据）
    end_ts = pd.Timestamp(day_end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(hours=1)

    t = df_hour[time_col].values

    # searchsorted：左闭右开定位
    l = int(np.searchsorted(t, np.datetime64(start_ts), side="left"))
    r_excl = int(np.searchsorted(t, np.datetime64(end_ts), side="right"))
    r = r_excl - 1

    return l, r

def print_results(result: Dict, nw_result: Dict, df_hour: pd.DataFrame) -> None:
    # 取日表日期列表（Timestamp）
    dates_day = result["df"]["time"].tolist()

    def segs_to_hour(segs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """把日索引区间列表 [(l_day,r_day), ...] 映射成小时索引区间列表 [(l_hour,r_hour), ...]"""
        out: List[Tuple[int, int]] = []
        for (l_day, r_day) in segs:
            day_start = dates_day[l_day]
            day_end = dates_day[r_day]
            l_hour, r_hour = day_seg_to_hour_seg(df_hour, day_start, day_end, time_col="time")
            out.append((l_hour, r_hour))
        return out

    def print_hour_ranges(name: str, hour_segs: List[Tuple[int, int]]) -> None:
        print(f"{name}（小时索引）{name.lower()}_hour_idx = {hour_segs}")
        print(f"{name}（小时日期）{name.lower()}_hour_date =")
        for (l_hour, r_hour) in hour_segs:
            print("  ", df_hour.loc[l_hour, "time"], "->", df_hour.loc[r_hour, "time"])
        print()

    # --- Winter ---
    #print(f"冬季段数量 K_w = {result['K_w']}")
    #print(f"冬季段（索引）W_idx = {result['W_idx']}")
    #print("冬季段（日期）W_date =")
    #for a, b in result["W_date"]:
    #    print("  ", a, "->", b)

    W_hour_idx = segs_to_hour(result["W_idx"])
    print_hour_ranges("冬季段", W_hour_idx)

    # --- Non-winter ---
    #print(f"非冬季段（索引）NW_idx = {nw_result['NW_idx']}")
    #print("非冬季段（日期）NW_date =")
    #for a, b in nw_result["NW_date"]:
    #    print("  ", a, "->", b)

    NW_hour_idx = segs_to_hour(nw_result["NW_idx"])
    print_hour_ranges("非冬季段", NW_hour_idx)

    # --- Spring ---
    #print("春季段（索引）spring_idx =", nw_result["spring_idx"])
    #print("春季段（日期）spring_date =")
    #for a, b in nw_result["spring_date"]:
    #    print("  ", a, "->", b)

    spring_hour_idx = segs_to_hour(nw_result["spring_idx"])
    print_hour_ranges("春季段", spring_hour_idx)

    # --- Summer ---
    #print("夏季段（索引）summer_idx =", nw_result["summer_idx"])
    #print("夏季段（日期）summer_date =")
    #for a, b in nw_result["summer_date"]:
    #    print("  ", a, "->", b)

    summer_hour_idx = segs_to_hour(nw_result["summer_idx"])
    print_hour_ranges("夏季段", summer_hour_idx)

    # --- Autumn ---
    #print("秋季段（索引）autumn_idx =", nw_result["autumn_idx"])
    #print("秋季段（日期）autumn_date =")
    #for a, b in nw_result["autumn_date"]:
    #    print("  ", a, "->", b)

    autumn_hour_idx = segs_to_hour(nw_result["autumn_idx"])
    print_hour_ranges("秋季段", autumn_hour_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Winter segmentation for Dunhuang dataset")
    parser.add_argument("--path", type=str, default="dunhuang.csv", help="path to csv")
    parser.add_argument("--T0", type=float, default=0.0, help="cold threshold (°C)")
    parser.add_argument("--g", type=int, default=10, help="gap tolerance (days)")
    parser.add_argument("--min_len", type=int, default=15, help="minimum winter segment length (days)")
    parser.add_argument("--out", type=str, default="image/Season Segmentation.png")
    args = parser.parse_args()

    df_hour = pd.read_csv(args.path)
    df_hour["time"] = pd.to_datetime(df_hour["time"], errors="coerce")
    df_hour = df_hour.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    df = read_data(csv_path=args.path)

    result = winter(df=df, T0=args.T0, g=args.g, min_len=args.min_len/2)

    nw_result = segmentation(
        df=result["df"],
        W_idx=result["W_idx"],
        tmean_col="temperature_2m_mean",
        min_seg_len=args.min_len,
    )

    print_results(result, nw_result, df_hour)

    graph(
        df=result["df"],
        W_idx=result["W_idx"],
        seasons=nw_result["seasons"],
        save_path=args.out,
        rain_col="rain_sum",
        snow_col="snowfall_sum",
        tmin_col="temperature_2m_min",
        tmean_col="temperature_2m_mean",
        date_col="time",
    )