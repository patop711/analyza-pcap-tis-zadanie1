#!/usr/bin/env python3
"""
Analýza reálneho IP toku - TIS Zadanie 1

Autor  : Patrik  (6 písmen → mu = 6 p/s)
Použitie:
    python analyza_pcap.py <subor.pcap>          # plná analýza
    python analyza_pcap.py --only-task1          # iba Poissonova simulácia (bez pcap)
"""

import os
import sys
import argparse
import zipfile
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – bez GUI
import matplotlib.pyplot as plt
from scipy.stats import poisson

try:
    from scapy.all import rdpcap, TCP, UDP
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False

# ---------------------------------------------------------------------------
# Globálne nastavenia
# ---------------------------------------------------------------------------
NAME       = "Patrik"
MU         = float(len(NAME))   # 6 p/s  (počet písmen v mene)
N_SIM      = 10_000             # počet simulovaných hodnôt
CHOSEN_BIN = 1.0                # vybrané vzorkovanie [s]

OUTPUT_DIR = "output"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")   # .fig (pickle)
JPG_DIR    = os.path.join(OUTPUT_DIR, "jpg")        # .jpg
TXT_DIR    = os.path.join(OUTPUT_DIR, "data")       # .txt


# ---------------------------------------------------------------------------
# Pomocné funkcie – ukladanie
# ---------------------------------------------------------------------------

def _make_dirs():
    for d in (OUTPUT_DIR, FIG_DIR, JPG_DIR, TXT_DIR):
        os.makedirs(d, exist_ok=True)


def save_fig(fig, name: str):
    """Uloží figúru ako .fig (pickle) aj .jpg."""
    with open(os.path.join(FIG_DIR, f"{name}.fig"), "wb") as fh:
        pickle.dump(fig, fh)
    path_jpg = os.path.join(JPG_DIR, f"{name}.jpg")
    fig.savefig(path_jpg, dpi=150, bbox_inches="tight")
    print(f"  → {path_jpg}")


def save_txt(data: np.ndarray, name: str, header: str = ""):
    """Uloží pole do .txt (CSV-štýl, bodkočiarka ako oddeľovač)."""
    path = os.path.join(TXT_DIR, f"{name}.txt")
    np.savetxt(path, data, header=header, fmt="%.8g", delimiter=";")
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# TASK 1 – Simulácia Poissonovho toku
# ---------------------------------------------------------------------------

def task1_poisson_simulation():
    """
    4 obrázky:
      1. 10 000 náhodných čísel rovnomerného rozdelenia
      2. 10 000 náhodných čísel exponenciálneho rozdelenia
      3. Vzorkovaný Poissonov proces
      4. Porovnanie simulácie so teoretickým Poissonovým rozdelením
    """
    print("\n" + "=" * 60)
    print(f"TASK 1 – Simulácia Poissonovho toku  (λ = {MU:.0f} p/s, N = {N_SIM})")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # Obr. 1 – rovnomerné rozdelenie
    # ------------------------------------------------------------------
    uniform_vals = rng.uniform(0.0, 1.0, N_SIM)

    fig1, ax = plt.subplots(figsize=(11, 4))
    ax.plot(uniform_vals, ".", markersize=1, alpha=0.45, color="royalblue")
    ax.set_title(f"{N_SIM} náhodných čísel rovnomerného rozdelenia  U(0, 1)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Hodnota")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_fig(fig1, "task1_01_uniform")
    save_txt(uniform_vals.reshape(-1, 1), "task1_01_uniform",
             header="uniform_U(0,1)")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Obr. 2 – exponenciálne rozdelenie (medzery medzi príchodmi)
    # ------------------------------------------------------------------
    exp_gaps = rng.exponential(scale=1.0 / MU, size=N_SIM)

    fig2, ax = plt.subplots(figsize=(11, 4))
    ax.plot(exp_gaps, ".", markersize=1, alpha=0.45, color="darkorange")
    ax.set_title(
        f"{N_SIM} náhodných čísel exponenciálneho rozdelenia"
        f"  (λ = {MU:.0f} p/s,  stredná hodnota = {1/MU:.4f} s)"
    )
    ax.set_xlabel("Index")
    ax.set_ylabel("Medzera  [s]")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_fig(fig2, "task1_02_exponential")
    save_txt(exp_gaps.reshape(-1, 1), "task1_02_exponential",
             header="exponential_inter-arrival_time_s")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Vzorkujeme na 1-sekundové intervaly
    # ------------------------------------------------------------------
    arrival_times = np.cumsum(exp_gaps)          # kumulatívne časy príchodov
    t_end         = arrival_times[-1]
    t_bins        = np.arange(0.0, np.ceil(t_end) + 1.0, 1.0)
    counts_1s, _  = np.histogram(arrival_times, bins=t_bins)
    bin_centers   = 0.5 * (t_bins[:-1] + t_bins[1:])

    # ------------------------------------------------------------------
    # Obr. 3 – vzorkovaný Poissonov proces (priebeh + počty/s)
    # ------------------------------------------------------------------
    n_show = min(500, len(arrival_times))

    fig3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 7))

    # kumulatívny priebeh
    ax_top.step(arrival_times[:n_show],
                np.arange(1, n_show + 1),
                where="post", color="steelblue", linewidth=0.9)
    ax_top.set_title(f"Vzorkovaný Poissonov proces – kumulatívne príchody (prvých {n_show})")
    ax_top.set_xlabel("Čas  [s]")
    ax_top.set_ylabel("Kum. počet príchodov")
    ax_top.grid(True, alpha=0.25)

    # počty za 1 s
    show_s = min(200, len(counts_1s))
    ax_bot.bar(bin_centers[:show_s], counts_1s[:show_s],
               width=1.0, alpha=0.75, color="seagreen",
               edgecolor="darkgreen", linewidth=0.3)
    ax_bot.axhline(MU, color="red", linewidth=1.5, linestyle="--",
                   label=f"E[k] = λ = {MU:.0f}")
    ax_bot.set_title(f"Počet príchodov za 1 sekundu  (prvých {show_s} s)")
    ax_bot.set_xlabel("Čas  [s]")
    ax_bot.set_ylabel("Počet príchodov / s")
    ax_bot.legend()
    ax_bot.grid(True, alpha=0.25)

    fig3.suptitle("Simulácia Poissonovho toku – vzorkovanie na 1 s",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig3, "task1_03_poisson_sampled")
    save_txt(np.column_stack([bin_centers, counts_1s]),
             "task1_03_counts_per_second",
             header="time_start_s;counts_per_1s")
    plt.close(fig3)

    # ------------------------------------------------------------------
    # Obr. 4 – porovnanie simulácia vs. teoretické Poissonovo rozd.
    # ------------------------------------------------------------------
    k_max     = max(int(counts_1s.max()), int(MU) + 5)
    k_vals    = np.arange(0, k_max + 1)
    sim_cnt   = np.bincount(counts_1s, minlength=k_max + 1)[:k_max + 1]
    sim_prob  = sim_cnt / sim_cnt.sum()
    theo_prob = poisson.pmf(k_vals, mu=MU)

    fig4, ax = plt.subplots(figsize=(12, 5))
    w = 0.38
    ax.bar(k_vals - w / 2, sim_prob,  width=w, alpha=0.80,
           label="Simulácia",          color="steelblue",
           edgecolor="navy",      linewidth=0.4)
    ax.bar(k_vals + w / 2, theo_prob, width=w, alpha=0.80,
           label=f"Poisson(λ = {MU:.0f})", color="coral",
           edgecolor="darkred",   linewidth=0.4)
    ax.set_title(
        f"Porovnanie simulácie a teoretického Poissonovho rozdelenia"
        f"  (λ = {MU:.0f}, N = {N_SIM})"
    )
    ax.set_xlabel("k  (príchody za 1 s)")
    ax.set_ylabel("Pravdepodobnosť  P(X = k)")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    save_fig(fig4, "task1_04_poisson_comparison")
    save_txt(np.column_stack([k_vals, sim_prob, theo_prob]),
             "task1_04_poisson_comparison",
             header="k;sim_probability;theo_probability")
    plt.close(fig4)

    print(f"  Simulačný čas: {t_end:.1f} s  |  bins: {len(counts_1s)}")
    print("  Task 1 – hotovo.\n")


# ---------------------------------------------------------------------------
# TASK 2 & 3 – pomocné funkcie pre analýzu paketov
# ---------------------------------------------------------------------------

def _compute_gaps(times: np.ndarray) -> np.ndarray:
    """Vráti medzery (diff), odstráni záporné hodnoty."""
    gaps = np.diff(times)
    n_neg = int(np.sum(gaps < 0))
    if n_neg:
        print(f"    Odstraňujem {n_neg} záporných medzier")
    return gaps[gaps >= 0]


def _exp_fit_curve(gaps: np.ndarray, x: np.ndarray) -> tuple:
    """Vráti odhadovanú rýchlosť λ a zodpovedajúcu krivku PDF."""
    rate = 1.0 / np.mean(gaps) if np.mean(gaps) > 0 else 1.0
    return rate, rate * np.exp(-rate * x)


def _plot_gaps_3(gaps: np.ndarray, prefix: str, title: str):
    """
    3 obrázky pre medzery:
      _gaps01  – priebeh pred filtráciou a po filtrácii
      _gaps02  – histogram
      _gaps03  – kombinovaný (priebeh + histogram)
    """
    # --- 01 priebeh --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(gaps, ".", markersize=1.0, alpha=0.40, color="steelblue")
    axes[0].set_title("Priebeh medzier (bez záporných)")
    axes[0].set_xlabel("Index paketu")
    axes[0].set_ylabel("Medzera  [s]")
    axes[0].grid(True, alpha=0.25)

    n_bins = min(120, max(20, len(gaps) // 80))
    x_99   = np.percentile(gaps, 99) if len(gaps) > 0 else 1.0
    x_fit  = np.linspace(0, x_99, 300)
    rate, pdf_fit = _exp_fit_curve(gaps, x_fit)

    axes[1].hist(gaps, bins=n_bins, density=True, range=(0, x_99),
                 alpha=0.75, color="steelblue",
                 edgecolor="navy", linewidth=0.3)
    axes[1].plot(x_fit, pdf_fit, "r-", linewidth=2.0,
                 label=f"Exp fit  λ = {rate:.3f} p/s")
    axes[1].set_title("Histogram medzier")
    axes[1].set_xlabel("Medzera  [s]")
    axes[1].set_ylabel("Hustota pravdepodobnosti")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlim(left=0)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, f"{prefix}_gaps01_combined")
    plt.close(fig)

    # --- 02 histogram standalone -------------------------------------------
    fig2, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gaps, bins=n_bins, density=True, range=(0, x_99),
            alpha=0.75, color="steelblue",
            edgecolor="navy", linewidth=0.3)
    ax.plot(x_fit, pdf_fit, "r-", linewidth=2.0,
            label=f"Exp fit  λ = {rate:.3f} p/s")
    ax.set_title(f"{title} – histogram medzier")
    ax.set_xlabel("Medzera  [s]")
    ax.set_ylabel("Hustota pravdepodobnosti")
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_fig(fig2, f"{prefix}_gaps02_histogram")
    plt.close(fig2)

    # --- 03 priebeh standalone ----------------------------------------------
    fig3, ax = plt.subplots(figsize=(13, 4))
    ax.plot(gaps, ".", markersize=1.0, alpha=0.40, color="steelblue")
    ax.set_title(f"{title} – priebeh medzier medzi paketmi")
    ax.set_xlabel("Index paketu")
    ax.set_ylabel("Medzera  [s]")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_fig(fig3, f"{prefix}_gaps03_timeseries")
    plt.close(fig3)

    # uloženie dát
    save_txt(gaps.reshape(-1, 1), f"{prefix}_gaps",
             header="inter_packet_gap_s")


def _sample_flow(times: np.ndarray, bin_size: float) -> tuple:
    """Vzorkuje tok do bins; vracia (counts, bin_centers)."""
    t_start = times[0]
    t_end   = times[-1]
    bins    = np.arange(t_start, t_end + bin_size, bin_size)
    counts, _ = np.histogram(times, bins=bins)
    centers   = 0.5 * (bins[:-1] + bins[1:])
    return counts, centers


def _plot_flow_and_hist(counts: np.ndarray, centers: np.ndarray,
                        bin_size: float, prefix: str, title: str):
    """
    2 obrázky pre tok (jedno kombinované + dve samostatné):
      _flow  – priebeh toku
      _hist  – histogram + Poisson
    """
    mu_est    = counts.mean() if len(counts) > 0 else 0.0
    k_max     = max(int(counts.max()), int(mu_est) + 3, 1)
    k_vals    = np.arange(0, k_max + 1)
    sim_cnt   = np.bincount(counts, minlength=k_max + 1)[:k_max + 1]
    sim_prob  = sim_cnt / max(sim_cnt.sum(), 1)
    theo_prob = poisson.pmf(k_vals, mu=max(mu_est, 1e-6))

    # kombinovaný obrázok (flow + histogram vedľa seba)
    fig, (ax_f, ax_h) = plt.subplots(1, 2, figsize=(16, 5))

    ax_f.plot(centers, counts, "-", linewidth=0.8, color="steelblue")
    ax_f.fill_between(centers, counts, alpha=0.25, color="steelblue")
    ax_f.set_title(f"Tok paketov  (bin = {bin_size} s)")
    ax_f.set_xlabel("Čas  [s]")
    ax_f.set_ylabel(f"Pakety / {bin_size} s")
    ax_f.grid(True, alpha=0.25)

    w = 0.38
    ax_h.bar(k_vals - w / 2, sim_prob,  width=w, alpha=0.80,
             label="Data", color="steelblue",
             edgecolor="navy",    linewidth=0.4)
    ax_h.bar(k_vals + w / 2, theo_prob, width=w, alpha=0.80,
             label=f"Poisson(λ = {mu_est:.2f})", color="coral",
             edgecolor="darkred", linewidth=0.4)
    ax_h.set_title(f"Histogram vs. Poisson  (bin = {bin_size} s)")
    ax_h.set_xlabel(f"k  (pakety / {bin_size} s)")
    ax_h.set_ylabel("Pravdepodobnosť")
    ax_h.legend(fontsize=9)
    ax_h.grid(True, alpha=0.25, axis="y")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, f"{prefix}_bin{bin_size}_combined")
    plt.close(fig)

    # samostatné obrázky
    fig2, ax = plt.subplots(figsize=(13, 4))
    ax.plot(centers, counts, "-", linewidth=0.8, color="steelblue")
    ax.fill_between(centers, counts, alpha=0.25, color="steelblue")
    ax.set_title(f"{title} – tok paketov  (bin = {bin_size} s)")
    ax.set_xlabel("Čas  [s]")
    ax.set_ylabel(f"Pakety / {bin_size} s")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_fig(fig2, f"{prefix}_bin{bin_size}_flow")
    plt.close(fig2)

    fig3, ax = plt.subplots(figsize=(10, 5))
    ax.bar(k_vals - w / 2, sim_prob,  width=w, alpha=0.80,
           label="Data", color="steelblue",
           edgecolor="navy",    linewidth=0.4)
    ax.bar(k_vals + w / 2, theo_prob, width=w, alpha=0.80,
           label=f"Poisson(λ = {mu_est:.2f})", color="coral",
           edgecolor="darkred", linewidth=0.4)
    ax.set_title(f"{title} – histogram vs. Poisson  (bin = {bin_size} s)")
    ax.set_xlabel(f"k  (pakety / {bin_size} s)")
    ax.set_ylabel("Pravdepodobnosť")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    save_fig(fig3, f"{prefix}_bin{bin_size}_hist")
    plt.close(fig3)

    # dáta
    save_txt(np.column_stack([centers, counts]),
             f"{prefix}_bin{bin_size}_flow",
             header=f"time_s;packets_per_{bin_size}s")
    save_txt(np.column_stack([k_vals, sim_prob, theo_prob]),
             f"{prefix}_bin{bin_size}_hist",
             header=f"k;sim_prob;poisson_prob(mu={mu_est:.4f})")


# ---------------------------------------------------------------------------
# TASK 2 – Celý záznam
# ---------------------------------------------------------------------------

def task2_full_analysis(times: np.ndarray, protos: np.ndarray):
    """
    3 obrázky: medzery (priebeh, histogram, kombinovaný)
    3 × 2 = 6 obrázkov: 3 vzorkovania × (tok + histogram)
    3 × 2 = 6 obrázkov: TCP / UDP / zvyšok × (tok + histogram)
    """
    print("\n" + "=" * 60)
    print("TASK 2 – Analýza celého záznamu")
    print("=" * 60)

    # --- medzery (3 obrázky) ------------------------------------------------
    print("\n  Medzery – všetky pakety")
    gaps_all = _compute_gaps(times)
    _plot_gaps_3(gaps_all, "task2_all", "Všetky pakety")

    # --- 3 vzorkovania (6 obrázkov) -----------------------------------------
    bin_sizes = [0.01, 0.1, 1.0]
    print(f"\n  Tri vzorkovania: {bin_sizes} s")
    for bs in bin_sizes:
        print(f"    bin = {bs} s")
        counts, centers = _sample_flow(times, bs)
        _plot_flow_and_hist(counts, centers, bs,
                            f"task2_all_sampling",
                            f"Všetky pakety")

    # --- zvolené vzorkovanie → TCP / UDP / iné (6 obrázkov) ----------------
    print(f"\n  Zvolené vzorkovanie: {CHOSEN_BIN} s  →  TCP / UDP / iné")
    for proto_label in ("tcp", "udp", "other"):
        mask   = protos == proto_label
        t_prot = times[mask]
        if len(t_prot) < 2:
            print(f"    {proto_label.upper()}: príliš málo paketov, preskočím")
            continue
        name_human = {"tcp": "TCP pakety",
                      "udp": "UDP pakety",
                      "other": "Zvyšok (nie TCP/UDP)"}[proto_label]
        print(f"    {name_human}: {len(t_prot)} paketov")
        counts, centers = _sample_flow(t_prot, CHOSEN_BIN)
        _plot_flow_and_hist(counts, centers, CHOSEN_BIN,
                            f"task2_{proto_label}",
                            name_human)

    print("  Task 2 – hotovo.\n")


# ---------------------------------------------------------------------------
# TASK 3 – Zaujímavý a nudný úsek
# ---------------------------------------------------------------------------

def _find_segments(times: np.ndarray, window: float = 10.0):
    """
    Nájde najzaujímavejší (max aktivita) a najnudnejší (min aktivita > 0)
    úsek dĺžky `window` sekúnd.
    Vracia (mask_interesting, mask_boring, t_int_start, t_bor_start).
    """
    bins   = np.arange(times[0], times[-1] + window, window)
    counts, _ = np.histogram(times, bins=bins)
    starts = bins[:-1]

    idx_int = int(np.argmax(counts))
    nonzero = np.where(counts > 0)[0]
    idx_bor = int(nonzero[np.argmin(counts[nonzero])]) if len(nonzero) else 0

    t_int = starts[idx_int]
    t_bor = starts[idx_bor]

    print(f"    Najzaujímavejší: [{t_int:.1f} – {t_int + window:.1f}] s"
          f"  ({counts[idx_int]} pkt)")
    print(f"    Najnudnejší   : [{t_bor:.1f} – {t_bor + window:.1f}] s"
          f"  ({counts[idx_bor]} pkt)")

    mask_int = (times >= t_int) & (times < t_int + window)
    mask_bor = (times >= t_bor) & (times < t_bor + window)
    return mask_int, mask_bor, t_int, t_bor, window


def task3_segments(times: np.ndarray, protos: np.ndarray):
    """
    2 × 2 obrázky pre medzery (2 oblasti × 2 obrázky)
    2 × 4 × 2 obrázky (2 oblasti × 4 datasety × 2 obrázky)
    """
    print("\n" + "=" * 60)
    print("TASK 3 – Najzaujímavejší a najnudnejší úsek")
    print("=" * 60)

    # Hľadáme okno 10 s; ak by bola dĺžka záznamu kratšia, použijeme 1/10
    duration = times[-1] - times[0]
    window   = min(10.0, duration / 10.0)
    print(f"  Dĺžka záznamu: {duration:.1f} s  |  okno: {window:.1f} s")

    result = _find_segments(times, window)
    mask_int, mask_bor, t_int, t_bor, win = result

    segments = [
        ("interesting", mask_int, f"Najzaujímavejší úsek  (t ≈ {t_int:.0f} s)"),
        ("boring",      mask_bor, f"Najnudnejší úsek      (t ≈ {t_bor:.0f} s)"),
    ]

    # 2 × 2 kombinovaný prehľad medzier (jeden súhrnný obrázok)
    fig_overview, axes_ov = plt.subplots(2, 2, figsize=(16, 9))
    fig_overview.suptitle("Porovnanie medzier – zaujímavý vs. nudný úsek",
                          fontsize=13, fontweight="bold")

    for seg_idx, (seg_name, seg_mask, seg_title) in enumerate(segments):
        t_seg = times[seg_mask]
        p_seg = protos[seg_mask]

        if len(t_seg) < 2:
            print(f"  POZOR: {seg_name} – príliš málo paketov, preskočím")
            continue

        gaps_seg = _compute_gaps(t_seg)

        # ---- prehľadový panel ----
        ax_ts  = axes_ov[seg_idx, 0]
        ax_hst = axes_ov[seg_idx, 1]

        ax_ts.plot(gaps_seg, ".", markersize=1.5, alpha=0.50,
                   color="steelblue" if seg_name == "interesting" else "darkorange")
        ax_ts.set_title(f"{seg_title} – priebeh medzier")
        ax_ts.set_xlabel("Index")
        ax_ts.set_ylabel("Medzera  [s]")
        ax_ts.grid(True, alpha=0.25)

        x_99     = np.percentile(gaps_seg, 99) if len(gaps_seg) > 0 else 1.0
        n_bins_s = min(60, max(10, len(gaps_seg) // 10))
        x_fit    = np.linspace(0, x_99, 300)
        rate_s, pdf_s = _exp_fit_curve(gaps_seg, x_fit)

        ax_hst.hist(gaps_seg, bins=n_bins_s, density=True,
                    range=(0, x_99), alpha=0.75,
                    color="steelblue" if seg_name == "interesting" else "darkorange",
                    edgecolor="navy", linewidth=0.3)
        ax_hst.plot(x_fit, pdf_s, "r-", linewidth=1.8,
                    label=f"Exp  λ = {rate_s:.3f}")
        ax_hst.set_title(f"{seg_title} – histogram medzier")
        ax_hst.set_xlabel("Medzera  [s]")
        ax_hst.set_ylabel("Hustota")
        ax_hst.legend(fontsize=8)
        ax_hst.grid(True, alpha=0.25)
        ax_hst.set_xlim(left=0)

        # uloženie medzier úseku
        _plot_gaps_3(gaps_seg, f"task3_{seg_name}", seg_title)

        # ---- 4 datasety × 2 obrázky ----------------------------------------
        datasets = [
            ("all",   np.ones(len(t_seg), dtype=bool),
             f"{seg_title} – Všetky pakety"),
            ("tcp",   p_seg == "tcp",
             f"{seg_title} – TCP"),
            ("udp",   p_seg == "udp",
             f"{seg_title} – UDP"),
            ("other", p_seg == "other",
             f"{seg_title} – Zvyšok"),
        ]

        # súhrnná mriežka 4 × 2
        fig_grid, axes_g = plt.subplots(4, 2, figsize=(16, 22))
        fig_grid.suptitle(f"{seg_title} – tok paketov  (bin = {CHOSEN_BIN} s)",
                          fontsize=13, fontweight="bold")

        for ds_idx, (ds_name, ds_mask, ds_title) in enumerate(datasets):
            t_ds = t_seg[ds_mask]
            ax_flow = axes_g[ds_idx, 0]
            ax_hist = axes_g[ds_idx, 1]

            if len(t_ds) < 2:
                for ax_tmp in (ax_flow, ax_hist):
                    ax_tmp.text(0.5, 0.5, f"Nedostatok dát\n({ds_name})",
                                ha="center", va="center",
                                transform=ax_tmp.transAxes, fontsize=12)
                continue

            t_ds_norm = t_ds - t_ds[0]   # normalizácia na začiatok segmentu
            counts_g, centers_g = _sample_flow(t_ds_norm, CHOSEN_BIN)

            # tok
            ax_flow.plot(centers_g, counts_g, "-", linewidth=0.8,
                         color="steelblue")
            ax_flow.fill_between(centers_g, counts_g,
                                 alpha=0.25, color="steelblue")
            ax_flow.set_title(f"{ds_title} – tok")
            ax_flow.set_xlabel("Čas  [s]")
            ax_flow.set_ylabel(f"Pkt / {CHOSEN_BIN} s")
            ax_flow.grid(True, alpha=0.25)

            # histogram vs Poisson
            mu_ds    = counts_g.mean() if len(counts_g) > 0 else 0.0
            k_max_ds = max(int(counts_g.max()), int(mu_ds) + 3, 1)
            kv       = np.arange(0, k_max_ds + 1)
            sc       = np.bincount(counts_g, minlength=k_max_ds + 1)[:k_max_ds + 1]
            sp       = sc / max(sc.sum(), 1)
            tp       = poisson.pmf(kv, mu=max(mu_ds, 1e-6))

            w = 0.38
            ax_hist.bar(kv - w / 2, sp, width=w, alpha=0.80,
                        label="Data", color="steelblue",
                        edgecolor="navy", linewidth=0.4)
            ax_hist.bar(kv + w / 2, tp, width=w, alpha=0.80,
                        label=f"Poisson({mu_ds:.1f})", color="coral",
                        edgecolor="darkred", linewidth=0.4)
            ax_hist.set_title(f"{ds_title} – histogram")
            ax_hist.set_xlabel("k")
            ax_hist.set_ylabel("Pravd.")
            ax_hist.legend(fontsize=7)
            ax_hist.grid(True, alpha=0.25, axis="y")

            # ukladanie dát
            _plot_flow_and_hist(counts_g, centers_g, CHOSEN_BIN,
                                f"task3_{seg_name}_{ds_name}",
                                ds_title)

        plt.tight_layout()
        save_fig(fig_grid, f"task3_{seg_name}_grid4x2")
        plt.close(fig_grid)

    plt.tight_layout()
    save_fig(fig_overview, "task3_overview_gaps_2x2")
    plt.close(fig_overview)

    print("  Task 3 – hotovo.\n")


# ---------------------------------------------------------------------------
# Načítanie PCAP
# ---------------------------------------------------------------------------

def load_pcap(pcap_file: str):
    """
    Načíta .pcap súbor a vráti (times, protos).
    times  – numpy array časových pečiatok  [s]
    protos – numpy array reťazcov 'tcp' / 'udp' / 'other'
    """
    if not SCAPY_OK:
        print("Chyba: knižnica scapy nie je nainštalovaná.")
        print("       Spustite:  pip install scapy")
        sys.exit(1)

    print(f"\nNačítavam PCAP: {pcap_file}")
    packets = rdpcap(pcap_file)

    times_list  = []
    protos_list = []

    for pkt in packets:
        times_list.append(float(pkt.time))
        if pkt.haslayer(TCP):
            protos_list.append("tcp")
        elif pkt.haslayer(UDP):
            protos_list.append("udp")
        else:
            protos_list.append("other")

    times  = np.array(times_list,  dtype=float)
    protos = np.array(protos_list, dtype=str)

    # zoradiť podľa času
    order  = np.argsort(times)
    times  = times[order]
    protos = protos[order]

    n_tcp   = int(np.sum(protos == "tcp"))
    n_udp   = int(np.sum(protos == "udp"))
    n_other = int(np.sum(protos == "other"))
    print(f"  Celkový počet paketov : {len(times)}")
    print(f"  TCP / UDP / iné       : {n_tcp} / {n_udp} / {n_other}")
    print(f"  Trvanie záznamu       : {times[-1] - times[0]:.3f} s")

    return times, protos


# ---------------------------------------------------------------------------
# ZIP archív
# ---------------------------------------------------------------------------

def create_zip():
    zip_path = os.path.join(OUTPUT_DIR, "analyza_pcap_patrik.zip")
    print(f"\nVytvárám ZIP archív: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(OUTPUT_DIR):
            for fname in files:
                if fname.endswith(".zip"):
                    continue
                fpath   = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, OUTPUT_DIR)
                zf.write(fpath, arcname)
    print(f"  ZIP uložený: {zip_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            f"Analýza reálneho IP toku – TIS Zadanie 1  "
            f"(meno: {NAME}, λ = {MU:.0f} p/s)"
        )
    )
    parser.add_argument(
        "pcap_file", nargs="?", default=None,
        help="Vstupný .pcap súbor (voliteľný – bez neho beží iba Task 1)"
    )
    parser.add_argument(
        "--only-task1", action="store_true",
        help="Spustiť iba Task 1 (Poissonova simulácia)"
    )
    args = parser.parse_args()

    np.random.seed(42)
    _make_dirs()

    # Task 1 – vždy
    task1_poisson_simulation()

    if not args.only_task1:
        if args.pcap_file is None:
            print("\nPoznámka: nebol zadaný .pcap súbor – beží iba Task 1.")
            print("Použitie: python analyza_pcap.py <subor.pcap>")
        else:
            if not os.path.exists(args.pcap_file):
                print(f"Chyba: súbor '{args.pcap_file}' neexistuje.")
                sys.exit(1)
            times, protos = load_pcap(args.pcap_file)
            task2_full_analysis(times, protos)
            task3_segments(times, protos)

    create_zip()

    print("\n" + "=" * 60)
    print("Analýza dokončená!")
    print(f"  Figúry (.fig) : {FIG_DIR}/")
    print(f"  Obrázky (.jpg): {JPG_DIR}/")
    print(f"  Dáta (.txt)   : {TXT_DIR}/")
    print(f"  ZIP archív    : {OUTPUT_DIR}/analyza_pcap_patrik.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
