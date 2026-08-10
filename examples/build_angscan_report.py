#!/usr/bin/env python3
"""JINST-style single-column report for the muon/positron angle x burst scan.
Reads analysis_output/angscan/{metrics_*.json,*_stats.json,*_corr2d.png,
*_event.png,angscan_summary.png}. Emits report_angscan.tex."""
import json
from pathlib import Path
import numpy as np

OUT = Path("/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix/examples/analysis_output/angscan")
NB = [1, 2, 4, 8, 16, 64]
ANG = ["00", "25", "50", "75"]
PARTS = [("mu", "muon"), ("pos", "positron")]


def load(tp, ang, N):
    m = OUT / f"metrics_{tp}_a{ang}_nb{N}.json"
    s = OUT / f"{tp}_a{ang}_nb{N}_stats.json"
    if not m.exists():
        return None
    d = json.loads(m.read_text()); d = d[list(d)[0]] if isinstance(list(d.values())[0], dict) else d
    if s.exists():
        d.update(json.loads(s.read_text()))
    return d


D = {(tp, a, N): load(tp, a, N) for tp, _ in PARTS for a in ANG for N in NB}


def has(t):
    return (OUT / f"{t}_corr2d.png").exists()


def rrange(tp, a):
    vs = [(N, D[(tp, a, N)]["pearson_r"]) for N in NB if D.get((tp, a, N))]
    return vs


def table(tp, pname):
    rows = []
    for a in ANG:
        for N in NB:
            d = D.get((tp, a, N))
            if not d:
                continue
            rows.append(f"{int(a)} & {N} & {d['integral_pct']:+.2f} & {d['pearson_r']:.4f} & "
                        f"{d['slope']:.3f} & {100*d['ghost_frac']:.2f} & {d['ghost_iso_charge']:.2f} & "
                        f"{d['true_killed']:.0f} & {d.get('relrms_pointwise_hi',float('nan')):.1f} \\\\")
    return ("\\begin{table}[t]\\centering\\small\n"
            f"\\caption{{Reconstruction metrics for {pname} vs anode angle and burst count "
            "($q_\\mathrm{reco}>0.5$~ke$^-$). int.=integral bias; $Q^\\mathrm{iso}$=isolated-ghost "
            "charge; RMS=high-$q$ pointwise relative RMS.}\n"
            f"\\label{{tab:{tp}}}\n"
            "\\begin{tabular}{ccrrrrrrr}\n\\hline\n"
            "ang & $n_b$ & int.\\% & $r$ & slope & ghost\\% & $Q^\\mathrm{iso}$ & killed & RMS\\% \\\\\n"
            "\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}")


def montage(tags, cap, label, perrow=3):
    imgs = []
    for t in tags:
        if has(t):
            imgs.append(f"\\includegraphics[width={0.99/perrow:.3f}\\linewidth]{{{t}_corr2d.png}}")
    if not imgs:
        return ""
    return ("\\begin{figure}[t]\\centering\n" + "\n".join(imgs) + "\n"
            "\\caption{" + cap + "}\n\\label{" + label + "}\n\\end{figure}")


# key numbers for the findings
def rr(tp, a, N):
    d = D.get((tp, a, N)); return d["pearson_r"] if d else float("nan")


tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx,amsmath,amssymb,booktabs,siunitx,xcolor,caption}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue]{hyperref}
\graphicspath{{./}}
\captionsetup{font=small}
\title{\bfseries Burst-count and incidence-angle dependence of zero-suppressed\\
LArPix charge unfolding for muons and positrons in ND-LAr}
\author{Yousen Zhang\\ \small Brookhaven National Laboratory}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
\noindent We characterise the reconstruction (``unfolding'') of ionization charge
from zero-suppressed (ZS) LArPix readout as a function of the number of
charge-sensitive-amplifier bursts per trigger, $n_\mathrm{burst}\in\{1,2,4,8,16,64\}$,
and of the track/shower incidence angle to the anode plane
($0^\circ,25^\circ,50^\circ,75^\circ$), for two topologies: through-going muons
and $3$~GeV positron showers, simulated in TPC0 of the ND-LAr $2\times2$ geometry.
Particles are generated with Geant4, drifted and read out with the tred nd-readout
using the large $25\times25$-pixel field response, and unfolded with a constrained
FISTA solver. We report, for all 48 configurations, the truth--reco 2D charge
correlation and numeric metrics (integral bias, Pearson $r$, slope, ghost
fraction, high-charge resolution) under the physical $q_\mathrm{reco}>0.5$~ke$^-$
selection. Self-trigger ($n_b=1$) uses a censor-augmented sparse solve; denser
bursts use the dense configuration.
\end{abstract}

\section{Introduction}
ZS LArPix readout records, per trigger, a short burst of $n_\mathrm{burst}$
CSA samples; more bursts sample the drifted-charge waveform more finely. This
note maps how the unfolding quality depends on $n_\mathrm{burst}$ and on the
incidence angle to the anode, separately for a minimum-ionizing through-going
muon (a line topology, charge spread in drift time according to the angle) and
a $3$~GeV positron electromagnetic shower (a dense blob). The measurement solves
$A(d)\,q\simeq d$ for the ionization $q$; $A$ is the bin-integrated bipolar
field-response operator sampled at the latch windows.

\section{Simulation and analysis}
\paragraph{Geometry, angle, containment.}
Particles are shot into TPC0 (drift along $x$, anode the $y$--$z$ plane at
$x=3.069$~cm, drift length $30.3$~cm; $y\in[\pm62]$, $z\in[2.46,64.54]$~cm). The
incidence angle to the anode is $\theta=\arcsin|d_x|$, set by the direction
$d=(-\sin\theta,0,\cos\theta)$. Muons ($3$~GeV) are through-going (range
$\gg$ TPC0), entering from the cathode-side gap; at $\theta=0^\circ,25^\circ$ they
are fully contained in TPC0, while at $50^\circ,75^\circ$ the mostly-along-drift
track exits the anode into a neighbouring TPC and only the TPC0 segment is read
out (\texttt{tpc\_list}=[0]). Positrons ($3$~GeV) are contained at
$0^\circ,25^\circ$ ($\gtrsim97\%$ of the deposited energy in TPC0) but the shower
maximum lies beyond the $30$~cm drift at $50^\circ,75^\circ$, so only
$\sim\!18\%$ is contained --- an intrinsic limit of a $1$~m shower in a $30$~cm
drift, reported as a caveat. All analysis uses only the TPC0 charge, so
reco and truth are compared on the same (contained) charge.

\paragraph{Readout and field response.}
tred nd-readout, threshold $5$~ke$^-$, noise
$(\sigma_\mathrm{uncorr},\sigma_\mathrm{reset},\sigma_\mathrm{thr})=(0.5,0.9,0.65)$~ke$^-$,
hold delay $1.5~\mu$s. We use the $25\times25$-pixel \texttt{v2a\_full} field
response for \emph{both} particles: the response is a detector (drift/induction)
property, so a common FR gives a self-consistent, directly comparable muon--positron
scan; it is the canonical $25\times25$ non-shielded response, present on both
hosts, and its $30.4$~cm drift matches TPC0. (The historical muon ``nogrid''
response is not available as a drop-in $25\times25$ operator; using it would
require porting and is left as follow-up.)

\paragraph{Unfolding, filter, truth smearing, metrics.}
The warm start is a Gaussian-regularised FFT deconvolution with a
frequency-domain filter $g(f)=\exp(-\tfrac12 f^2/\sigma_f^2)$,
$\sigma_f=(0.2,0.2,0.005)$, i.e.\ real-space $\sigma_t=1.59~\mu$s,
$\sigma_\mathrm{pxl}=0.80$~pitch. The solve minimises $\lVert Aq-d\rVert^2$ with a
soft-seeded weighted-$\ell_1$ ladder ($\alpha=1.0,0.5,0.3$) and a quiet-hinge term;
the self-trigger ($n_b=1$) adds a running-max censor term and $600$ iterations.
For evaluation the truth is smeared with the same Gaussian
($\sigma_t=1.59~\mu$s, $\sigma_\mathrm{pxl}=0.80$~pitch) onto a universal grid;
all selections use the observable $q_\mathrm{reco}>0.5$~ke$^-$. Metrics: integral
bias $100(\Sigma q_\mathrm{reco}-\Sigma q_\mathrm{truth})/\Sigma q_\mathrm{truth}$;
Pearson $r$ and slope of reco vs.\ truth; ghost fraction (reco$>$cut \&
truth$<$cut); isolated-ghost charge; killed truth; and the high-charge
($>8$~ke$^-$) pointwise relative RMS of $(q_\mathrm{reco}-q_\mathrm{truth})/q_\mathrm{reco}$.

\section{Results}
Figure~\ref{fig:summary} summarises the scan: Pearson $r$, integral bias, ghost
fraction and high-charge RMS versus $n_\mathrm{burst}$, one curve per anode angle,
for muons (top) and positrons (bottom). Tables~\ref{tab:mu} and \ref{tab:pos}
give the full numbers. Figures~\ref{fig:corrpos}--\ref{fig:corrang} show
representative truth--reco 2D correlations, and Fig.~\ref{fig:ev} two event
displays.

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{angscan_summary.png}
\caption{Reconstruction metrics vs.\ burst count, per anode angle, for muons
(top row) and positrons (bottom row); $x$-axis is $n_\mathrm{burst}$ (log$_2$).}
\label{fig:summary}
\end{figure}

%%FINDINGS%%

%%TABLES%%

%%MONTAGES%%

\section{Conclusion}
Across topologies the burst scan behaves as expected --- more bursts sample the
drift waveform more finely and improve the correlation --- with a clear angle
dependence tied to how the charge distributes in drift time. The full 48-config
numbers (Tables~\ref{tab:mu},~\ref{tab:pos}) and 2D correlations are provided for
both particles. Caveats: large-angle muon tracks and positron showers are only
partially contained in the $30$~cm TPC0 drift and are analysed on their TPC0
charge; both particles use a common $25\times25$ \texttt{v2a} field response.

\end{document}
"""

# ---- findings (data-driven) ----
def fmt_r(tp, a):
    lo = rr(tp, a, 1); hi = rr(tp, a, 64)
    return lo, hi

fnd = []
for tp, pname in PARTS:
    for a in ["00"]:
        lo, hi = fmt_r(tp, a)
        if not (np.isnan(lo) or np.isnan(hi)):
            fnd.append(f"For {pname}s at $0^\\circ$, Pearson $r$ goes from "
                       f"{lo:.3f} ($n_b=1$) to {hi:.3f} ($n_b=64$).")
# angle effect at nb4
for tp, pname in PARTS:
    vs = [(int(a), rr(tp, a, 4)) for a in ANG if not np.isnan(rr(tp, a, 4))]
    if len(vs) >= 2:
        s = ", ".join(f"{ang}$^\\circ$:{r:.3f}" for ang, r in vs)
        fnd.append(f"At $n_b=4$, {pname} $r$ vs.\\ angle --- {s}.")
findings = ("\\paragraph{Findings.} " + " ".join(fnd)) if fnd else ""

tables = table("mu", "muon") + "\n\n" + table("pos", "positron")

montages = "\n\n".join(filter(None, [
    montage([f"pos_a00_nb{N}" for N in NB],
            "Positron, $0^\\circ$: truth--reco 2D correlation across the burst scan "
            "($n_b=1,2,4,8,16,64$, left$\\to$right, top$\\to$bottom).", "fig:corrpos"),
    montage([f"mu_a00_nb{N}" for N in NB],
            "Muon, $0^\\circ$: truth--reco 2D correlation across the burst scan.", "fig:corrmu"),
    montage([f"pos_a{a}_nb4" for a in ANG] + [f"mu_a{a}_nb4" for a in ANG],
            "Angle dependence at $n_b=4$: positron (top, $0/25/50/75^\\circ$) and "
            "muon (bottom).", "fig:corrang", perrow=4),
]))

evtags = [t for t in ["mu_a00_nb4", "pos_a00_nb4"] if (OUT / f"{t}_event.png").exists()]
evfig = ""
if evtags:
    imgs = "\n".join(f"\\includegraphics[width=\\linewidth]{{{t}_event.png}}" for t in evtags)
    evfig = ("\\begin{figure}[t]\\centering\n" + imgs +
             "\n\\caption{Event displays ($n_b=4$): muon track (top) and positron shower "
             "(bottom); truth grey, reco coloured by charge, isolated ghosts green $\\times$.}\n"
             "\\label{fig:ev}\n\\end{figure}")

tex = (tex.replace("%%FINDINGS%%", findings)
          .replace("%%TABLES%%", tables)
          .replace("%%MONTAGES%%", montages + "\n\n" + evfig))
(OUT / "report_angscan.tex").write_text(tex)
n = sum(1 for v in D.values() if v is not None)
print(f"wrote {OUT/'report_angscan.tex'}  ({n}/48 configs)")
