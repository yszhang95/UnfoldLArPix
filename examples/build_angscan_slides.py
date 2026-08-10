#!/usr/bin/env python3
"""Beamer slides for the muon/positron angle x burst scan.

Detailed run setup + solver hyperparameters; the closing frame reports only
OBSERVATIONS (measured numbers), no interpretation/conclusions.
"""
import json
from pathlib import Path
import numpy as np

OUT = Path("/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples/analysis_output/angscan")
NB = [1, 2, 4, 8, 16, 64]
ANG = ["00", "25", "50", "75"]
PARTS = [("mu", "muon"), ("pos", "positron")]


def load(tp, a, N):
    m = OUT / f"metrics_{tp}_a{a}_nb{N}.json"
    if not m.exists():
        return None
    d = json.loads(m.read_text())
    return d[list(d)[0]] if isinstance(list(d.values())[0], dict) else d


D = {(tp, a, N): load(tp, a, N) for tp, _ in PARTS for a in ANG for N in NB}


def g(tp, a, N, key):
    d = D.get((tp, a, N))
    return d.get(key) if d else None


def row(tp, a, key, over="nb", scale=1.0, fmt="%.3f"):
    """Format one series (over burst at fixed angle a, or over angle at fixed nb=a)."""
    if over == "nb":
        vals = [g(tp, a, N, key) for N in NB]
    else:  # over angle, a = fixed nburst
        vals = [g(tp, aa, a, key) for aa in ANG]
    return ", ".join((fmt % (v * scale)) if v is not None else "--" for v in vals)


def crop_png(base):
    """White-border-trimmed copy {base}_c.png; True if the source exists."""
    src = OUT / f"{base}.png"
    dst = OUT / f"{base}_c.png"
    if not src.exists():
        return False
    try:
        from PIL import Image, ImageChops
        im = Image.open(src).convert("RGB")
        bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
        (im.crop(bbox) if bbox else im).save(dst)
    except Exception:
        dst.write_bytes(src.read_bytes())
    return True


def trim(t):
    return crop_png(f"{t}_corr2d")


for _ev in ("mu_a00_nb4_event", "pos_a00_nb4_event",
            "anode_display", "kernel_cut_proof",
            "corr2d_evfid_pos", "corr2d_evfid_mu",
            "display_rawhits_pos_a50"):
    crop_png(_ev)


def fid_table_tex():
    """LaTeX rows for the baseline|acq_start=event integral table, read live
    from the fiducial-scan JSON (fid_table.py in the session scratchpad)."""
    import json as _json
    p = OUT.parent / "angscan_tacq" / "fid_table.json"
    if not p.exists():
        return None
    d = _json.loads(p.read_text())
    cuts = d["cuts"]
    head = ("config & " +
            " & ".join((r"no fid" if c == 0 else r"$\geq$%.0f cm" % c)
                       for c in cuts) + r" \\")
    lines = []
    for row in d["rows"]:
        cells = []
        for c in cuts:
            b = row["base"][str(c)][0]
            n = row["ev"][str(c)][0]
            cells.append(r"$%+.1f\,|\,%+.1f$" % (b, n))
        lines.append(row["tag"].replace("_", r"\_") + " & " +
                     " & ".join(cells) + r" \\")
    return (r"\begin{tabular}{lcccc}\toprule " + head + r" \midrule " +
            " ".join(lines) + r" \bottomrule\end{tabular}")


def rawhit_ledger_tex():
    """truth / recorded / reconstructed charge by depth band, live from JSON."""
    import json as _json
    p = OUT / "rawhit_ledger.json"
    if not p.exists():
        return None
    d = _json.loads(p.read_text())
    bands = ["<1", "1-2", "2-5", ">5"]
    rows = []
    for tag in ("pos_a00", "pos_a50", "pos_a75"):
        if tag not in d:
            continue
        tot = sum(d[tag][b][0] for b in bands)
        cells = []
        for b in bands:
            t, w, r = d[tag][b]
            # negligible truth in the band -> no meaningful ratio to quote
            cells.append("--" if t < 0.005 * tot
                         else "%.0f / %.0f / %.0f" % (t, w, r))
        rows.append(tag.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    head = ("depth band [cm] & " +
            " & ".join("$%s$" % b for b in bands) + r" \\")
    return (r"\begin{tabular}{lcccc}\toprule " + head + r" \midrule " +
            " ".join(rows) + r" \bottomrule\end{tabular}")


def grid(tags, perrow=3, maxh=0.40):
    imgs = [t for t in tags if trim(t)]
    if not imgs:
        return "(pending)"
    w = 0.99 / perrow - 0.01
    return " ".join(
        r"\includegraphics[width=%.3f\textwidth,height=%.3f\textheight,keepaspectratio]{%s_corr2d_c.png}"
        % (w, maxh, t) for t in imgs)


# observation series (live from metrics)
r_pos_b, r_mu_b = row("pos", "00", "pearson_r"), row("mu", "00", "pearson_r")
r_pos_a, r_mu_a = row("pos", 4, "pearson_r", "ang"), row("mu", 4, "pearson_r", "ang")
i_pos_b = row("pos", "00", "integral_pct", fmt="%+.1f")
i_pos_a, i_mu_a = row("pos", 4, "integral_pct", "ang", fmt="%+.1f"), row("mu", 4, "integral_pct", "ang", fmt="%+.1f")
sd_pos_a, sd_mu_a = row("pos", 4, "spec_dev", "ang", fmt="%.2f"), row("mu", 4, "spec_dev", "ang", fmt="%.2f")
sd_mu_b = row("mu", "00", "spec_dev", fmt="%.2f")
gh_all = [g(tp, a, N, "ghost_iso_frac") for tp, _ in PARTS for a in ANG for N in NB]
gh_max = max((v for v in gh_all if v is not None), default=float("nan")) * 100

tex = r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}\usecolortheme{seahorse}
\usepackage{graphicx}\usepackage{booktabs}
\graphicspath{{./}}
\setbeamertemplate{navigation symbols}{}
\setbeamerfont{block body}{size=\footnotesize}
\title[LArPix unfolding: burst $\times$ angle]{Burst-count and angle dependence of ZS LArPix charge unfolding}
\subtitle{through-going muons and positrons in ND-LAr TPC0}
\author{Yousen Zhang}\institute{Brookhaven National Laboratory}\date{\today}
\begin{document}
\frame{\titlepage}

\begin{frame}{Goal and scan}
\begin{itemize}
\item Unfold the ionization charge $q$ from the zero-suppressed (ZS) LArPix readout: solve $A\,q \simeq d$, where $A$ is the bin-integrated bipolar field-response operator sampled at the recorded latch windows and $d$ are the latched charges.
\item \textbf{Scan} (48 configs): $n_\mathrm{burst}\in\{1,2,4,8,16,64\}$ $\times$ anode angle $\{0,25,50,75\}^\circ$ $\times$ \{muon, positron\}.
\item Same detector response for both particles $\Rightarrow$ one common field response; only the primary and its angle change across the scan.
\end{itemize}
\end{frame}

\begin{frame}{Simulation setup (Geant4 $+$ tred)}
\footnotesize
\begin{itemize}
\item \textbf{Primaries}: $3$~GeV through-going muons (not stopped in LAr) and $3$~GeV positron showers; launched in \textbf{TPC0 only}; primary direction at $\{0,25,50,75\}^\circ$ to the anode ($\hat d=(-\sin\theta,0,\cos\theta)$).
\item \textbf{Geometry}: ND-LAr 2$\times$2, TPC0 drift length $30.3$~cm; drift velocity $1.60$~mm/$\mu$s; electron lifetime $20$~ms; pixel pitch $4.434$~mm.
\item \textbf{Readout} (tred \texttt{nd\_readout}): ZS threshold $5$~ke$^-$; \texttt{adc\_hold\_delay} $=30$~ticks ($1.5~\mu$s); \texttt{adc\_down\_time} $=24$~ticks ($1.2~\mu$s); \texttt{csa\_reset\_time} $=2$~ticks; one\_tick $=2$; time tick $0.05~\mu$s; memoryless CSA reset.
\item \textbf{Front-end noise}: uncorrelated $0.5$~ke$^-$, reset (kTC) $0.9$~ke$^-$, threshold dispersion $0.65$~ke$^-$; recombination fluctuations off.
\item \textbf{Bursts}: $n_\mathrm{burst}\in\{1,2,4,8,16,64\}$; $n_\mathrm{burst}=1$ is a single self-trigger latch.
\item \textbf{Field response}: $25\times25$-pixel \texttt{v2a\_full}, common to both particles.
\item \textbf{Fiducial note}: at large angle the muon track / positron shower is only partly contained in the $30$~cm drift; the truth used is the TPC0 effective charge only.
\end{itemize}
\end{frame}

\begin{frame}{Reconstruction and solver hyperparameters}
\footnotesize
\begin{itemize}
\item \textbf{Forward model}: $A = (\text{bin-integrated bipolar FR convolution})\circ(\text{latch-window sampling})$; solve $\min_{q\ge0}\tfrac12\|Aq-d\|^2 + \text{penalties}$ (FISTA), device CUDA/float32, seed $0$.
\item \textbf{Warm start}: FFT deconvolution with Gaussian filter $\sigma_\mathrm{time}=0.005$ (freq.), $\sigma_\mathrm{pixel}=0.2$, pad $12$~px.
\item \textbf{Measurement}: \texttt{split\_trigger}=true --- the first window is split at the trigger; the pre-trigger part carries the $5$~ke$^-$ threshold as a pseudo-measurement.
\item \textbf{Support (ROI)}: threshold the smoothed warm start at $\varepsilon=0.3$~ke$^-$, dilate $1$ voxel.
\item \textbf{Solver, dense regime ($n_\mathrm{burst}\ge2$)}: $150$ iterations; $\ell_1$ ladder $\alpha=[1.0,0.5,0.3]$, seed\_cut $0.5$, soft\_len $2.0$; quiet-hinge term $\beta=1.0$; centroid window $1$.
\item \textbf{Solver, self-trigger ($n_\mathrm{burst}=1$)}: $600$ iterations; same $\ell_1$ ladder; quiet $+$ censor (running-max) term $\beta=1.0$, margin $3.0$, $L_2$; centroid window $2$.
\item \textbf{Evaluation}: universal grid, Gaussian deposit, fitted per-voxel time offsets; truth-side smearing $\sigma_t\!\approx\!1.6~\mu$s, $\sigma_x\!\approx\!0.8$~pitch; all metrics on voxels with $q_\mathrm{reco}>0.5$~ke$^-$.
\item \textbf{First-window edge}: the scan below uses the legacy first integration window (from $-\infty$); the anode-proximity correction (\texttt{acq\_start}) is quantified on the dedicated frames at the end.
\end{itemize}
\end{frame}

\begin{frame}{Positron: 2D correlation across the burst scan ($0^\circ$)}
\begin{center}""" + grid(["pos_a00_nb%d" % N for N in NB], 3) + r"""\end{center}
\begin{center}\footnotesize $n_b = 1,2,4,8,16,64$ (left$\to$right, top$\to$bottom)\end{center}
\end{frame}

\begin{frame}{Muon: 2D correlation across the burst scan ($0^\circ$)}
\begin{center}""" + grid(["mu_a00_nb%d" % N for N in NB], 3) + r"""\end{center}
\begin{center}\footnotesize $n_b = 1,2,4,8,16,64$\end{center}
\end{frame}

\begin{frame}{Angle dependence at $n_b=4$}
\begin{center}\footnotesize positron $0/25/50/75^\circ$\end{center}
\begin{center}""" + grid(["pos_a%s_nb4" % a for a in ANG], 4, 0.33) + r"""\end{center}
\begin{center}\footnotesize muon $0/25/50/75^\circ$\end{center}
\begin{center}""" + grid(["mu_a%s_nb4" % a for a in ANG], 4, 0.33) + r"""\end{center}
\end{frame}

\begin{frame}{Event displays ($n_b=4$)}
\begin{center}\includegraphics[width=0.92\textwidth,height=0.40\textheight,keepaspectratio]{mu_a00_nb4_event_c.png}\end{center}
\begin{center}\includegraphics[width=0.92\textwidth,height=0.40\textheight,keepaspectratio]{pos_a00_nb4_event_c.png}\end{center}
\begin{center}\footnotesize muon track (top), positron shower (bottom); truth grey, reco coloured, isolated ghost green $\times$.\end{center}
\end{frame}

\begin{frame}{Observations (measured values only)}
\footnotesize
Pearson $r$ (reco vs.\ smeared truth), $n_b=1{\to}64$ at $0^\circ$:
\begin{itemize}\item positron: """ + r_pos_b + r""" \item muon: """ + r_mu_b + r"""\end{itemize}
Pearson $r$ at $n_b=4$, angle $0/25/50/75^\circ$:
\begin{itemize}\item positron: """ + r_pos_a + r""" \item muon: """ + r_mu_a + r"""\end{itemize}
Integral $(\Sigma q_\mathrm{reco}/\Sigma q_\mathrm{truth}-1)$ in \%:
\begin{itemize}
\item positron, $n_b=1{\to}64$ at $0^\circ$: """ + i_pos_b + r"""
\item positron, $n_b=4$, angle $0/25/50/75^\circ$: """ + i_pos_a + r"""
\item muon, $n_b=4$, angle $0/25/50/75^\circ$: """ + i_mu_a + r"""
\end{itemize}
Spectral deviation (spec\_dev) at $n_b=4$, angle $0/25/50/75^\circ$: positron """ + sd_pos_a + r"""; muon """ + sd_mu_a + r""". \; muon $0^\circ$, $n_b=1{\to}64$: """ + sd_mu_b + r""".
\par\smallskip
Isolated-ghost charge fraction: $\le """ + ("%.2f" % gh_max) + r"""\%$ across all 48 configs.
\end{frame}

\begin{frame}{Large-angle integral deficit: localizing the missing charge}
\begin{center}\includegraphics[width=0.98\textwidth,height=0.58\textheight,keepaspectratio]{anode_display_c.png}\end{center}
\footnotesize
\begin{itemize}
\item Positron $50^\circ$, truth vs.\ reco vs.\ difference in \emph{physical} coordinates (drift depth from the anode $\times$ pixel): \textbf{91\% of the missing charge lies within 5\,cm of the anode}.
\item A threshold-free, trigger-free fixed-interval readout of the same event shows the \emph{same} deficit ($-21\%$ vs.\ $-22\%$ ZS) $\Rightarrow$ not a ZS/trigger/solver property.
\end{itemize}
\end{frame}

\begin{frame}{Prompt induction near the anode}
\begin{center}\includegraphics[width=0.9\textwidth,height=0.52\textheight,keepaspectratio]{kernel_cut_proof_c.png}\end{center}
\footnotesize
\begin{itemize}
\item A deposit at drift distance $d$ promptly induces only $1-w(d)$ of its charge on the collection pixel (electron Ramo current over the actual drift path); the complement is locked in the static-ion image and released only on ion-drift timescales --- invisible to a $\mu$s readout.
\item Measured per-pixel captured/effq vs.\ collection time follows the response-kernel tail-cumulative prediction.
\item The operator's first latch window integrated from $-\infty$, crediting near-anode charge with kernel mass that is not in the data $\Rightarrow$ systematic under-recovery of the anode-side track end.
\end{itemize}
\end{frame}

\begin{frame}{Operator correction (\texttt{acq\_start}) and anode fiducial}
\footnotesize
Integral $(\Sigma q_\mathrm{reco}/\Sigma q_\mathrm{truth}-1)$ in \%, \textbf{first window from $-\infty$ $|$ from the event $t_0$} (\texttt{acq\_start: event}), vs.\ anode fiducial cut ($n_b=4$):
\begin{center}
""" + (fid_table_tex() or "(fiducial table pending)") + r"""
\end{center}
\begin{itemize}
\item First-window lower edge $=$ the event $t_0$ carried by the data (canonical \texttt{EventData.acq\_start}; channel-wise-ready; file-format translation confined to the loader).
\item Pearson $r$ is unchanged or improved everywhere (e.g.\ positron $75^\circ$: $0.957\to0.967$); $0^\circ$ rows are untouched (control).
\item With the corrected operator $+$ a 2\,cm anode fiducial, positron $25/50^\circ$ reach the $0^\circ$ level ($-1.8\%$ vs.\ $-2.4\%$).
\end{itemize}
\end{frame}


\begin{frame}{Raw hits in the display: was the charge recorded at all?}
\begin{center}\includegraphics[width=0.99\textwidth,height=0.46\textheight,keepaspectratio]{display_rawhits_pos_a50_c.png}\end{center}
\scriptsize
Raw hits $=$ the per-burst increments (cumulative latch columns differenced), each placed at its own latch window, so the recorded charge lives on the same physical axes as truth and reco. Charge in ke$^-$, \textbf{truth / recorded / reconstructed}:
\begin{center}\scriptsize
""" + (rawhit_ledger_tex() or "(ledger pending)") + r"""
\end{center}
\end{frame}

\begin{frame}{2D correlation, corrected operator: positron}
\begin{center}\includegraphics[width=0.72\textwidth,height=0.86\textheight,keepaspectratio]{corr2d_evfid_pos_c.png}\end{center}
\end{frame}

\begin{frame}{2D correlation, corrected operator: muon}
\begin{center}\includegraphics[width=0.72\textwidth,height=0.86\textheight,keepaspectratio]{corr2d_evfid_mu_c.png}\end{center}
\end{frame}

\end{document}
"""
(OUT / "slides_angscan.tex").write_text(tex)
print(f"wrote {OUT/'slides_angscan.tex'}")
