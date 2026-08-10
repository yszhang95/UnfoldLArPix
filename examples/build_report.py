#!/usr/bin/env python3
"""Assemble the reset-fix technical report (revtex4-2, JINST-like) from the
per-dataset metrics/stats JSON + figures produced by run_report_pipeline.sh.
Emits report.tex in the report dir; compile with pdflatex/latexmk."""
import json
import shutil
import sys
from pathlib import Path

REPO = Path("/home/yousen/Documents/NDLAr2x2/sp_deconv_tradition/UnfoldLArPix")
OUT = REPO / "examples/analysis_output/report_resetfix"
SCRATCH = Path("/tmp/claude-27625/-home-yousen-Documents-NDLAr2x2-sp-deconv-tradition-UnfoldLArPix/"
               "4da31237-6743-46ad-8782-581e1fb67e0f/scratchpad")

# tag -> (human label, nburst, FR, group)
META = {
    "nb4":            ("thres5k nburst4",              "4",   "v2a", "scan"),
    "nb16":           ("thres5k nburst16",             "16",  "v2a", "scan"),
    "nb64":           ("thres5k nburst64",             "64",  "v2a", "scan"),
    "nb1self":        ("thres5k self-trigger",         "1",   "v2a", "scan"),
    "nb4sh":          ("nburst4 shield",               "4",   "shield", "shield"),
    "nb4sh_r0":       ("nburst4 shield reset0",        "4",   "shield", "shield"),
    "nb8sh_r0":       ("nburst8 shield reset0",        "8",   "shield", "shield"),
    "nb1self_sh":     ("self-trigger shield",          "1",   "shield", "shield"),
    "nb1self_sh_r0":  ("self-trigger shield reset0",   "1",   "shield", "shield"),
}


def esc(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def load(tag):
    m = OUT / f"metrics_{tag}.json"
    s = OUT / f"{tag}_stats.json"
    if not m.exists():
        return None
    md = json.loads(m.read_text())
    md = md[list(md)[0]] if len(md) == 1 and isinstance(list(md.values())[0], dict) else md
    sd = json.loads(s.read_text()) if s.exists() else {}
    return {**md, **sd}


def main():
    tags = [t for t in META if (OUT / f"metrics_{t}.json").exists()]
    if not tags:
        print("no metrics found — run run_report_pipeline.sh first"); sys.exit(1)
    data = {t: load(t) for t in tags}

    figdir = OUT / "fig"; figdir.mkdir(exist_ok=True)
    for src in ["waveform_139_80_compare.png", "ghost_compare_resetfix.png",
                "neighborhood_reco.png"]:
        p = SCRATCH / src
        if p.exists():
            shutil.copy(p, figdir / src)

    def row(t):
        d = data[t]; lab, nb, fr, _ = META[t]
        return (f"{esc(lab)} & {nb} & {fr} & {d.get('integral_pct',0):+.2f} & "
                f"{d.get('pearson_r',0):.4f} & {d.get('slope',0):.3f} & "
                f"{100*d.get('ghost_frac',0):.2f} & {d.get('ghost_iso_charge',0):.2f} & "
                f"{d.get('true_killed',0):.0f} & "
                f"{d.get('relrms_pointwise_hi',float('nan')):.1f} & "
                f"{d.get('relrms_2x2x2_hi',float('nan')):.1f} \\\\")
    scan = [row(t) for t in tags if META[t][3] == "scan"]
    shield = [row(t) for t in tags if META[t][3] == "shield"]

    def figevent(t):
        lab = esc(META[t][0])
        return (r"\begin{figure}[t]\centering" "\n"
                rf"\includegraphics[width=\linewidth]{{{t}_event.png}}" "\n"
                rf"\caption{{Event display for \textbf{{{lab}}}: truth (grey), reconstructed "
                r"charge (colour, $1$--$10$~ke$^-$ capped, low overflow) and isolated ghosts "
                r"(green $\times$), in three projections.}" "\n"
                rf"\label{{fig:ev_{t}}}" "\n" r"\end{figure}")

    def montage(taglist, caption, label):
        imgs = "\n".join(rf"\includegraphics[width=0.32\linewidth]{{{t}_corr2d.png}}"
                         for t in taglist if t in data)
        return (r"\begin{figure*}[t]\centering" "\n" + imgs + "\n"
                rf"\caption{{{caption}}}" "\n" rf"\label{{{label}}}" "\n" r"\end{figure*}")

    events = "\n".join(figevent(t) for t in ("nb4", "nb1self") if t in data)

    tex = r"""\documentclass[aps,prd,reprint,nofootinbib,superscriptaddress]{revtex4-2}
\usepackage{graphicx,booktabs,amsmath,siunitx,xcolor}
\graphicspath{{./}{fig/}}
\begin{document}
\title{Correcting a readout reset-noise artifact in zero-suppressed\\
LArPix charge unfolding for ND-LAr, and its resolution limits}
\author{Yousen Zhang}
\affiliation{Brookhaven National Laboratory}
\date{\today}

\begin{abstract}
Zero-suppressed (ZS) charge readout in pixelated liquid-argon TPCs records, per
trigger, a short burst of charge-sensitive-amplifier (CSA) samples. We report a
reset-noise accumulation artifact in the readout simulation that faked sustained
late re-triggers on bright pixels, injected spurious late-time charge (a ``late
line'') into the deconvolution, and biased every reconstruction metric; we trace
it to an incremental CSA reset that never discarded previously added kTC
baselines and replace it with a memoryless reset. On positron particle-gun data
regenerated with the fixed readout and the large $25\times25$-pixel field
response, the late line vanishes and per-pixel charge returns to truth. We then
characterise the reconstruction across burst modes and the shielded response.
Two resolution limits are quantified: (i) a high-charge residual that is
dominantly a $\pm1$ time-bin misalignment; and (ii) a $+10$--$12\%$ integral
over-recovery specific to the shielded self-trigger, which we show is neither a
field-response nor a solver-prior effect but a coarse-time-bin
ill-conditioning of the operator that a half-bin operator removes. All voxel
selections use the physically available reconstructed charge
($q_\mathrm{reco}>0.5$~ke$^-$), never the (unknown) truth.
\end{abstract}
\maketitle

\section{Introduction}
Charge is read out per pixel as a sequence of \emph{bursts}: at threshold
crossing the CSA integrates for a fixed hold delay, latches, resets, and
re-arms. Reconstruction (``unfolding'') solves $A(d)\,q\simeq d$ for the
ionization $q$, where $A$ is the bin-integrated bipolar-response operator
sampled at the latch windows and $d$ is the vector of latch values. This report
(i) identifies and fixes a readout artifact that corrupted the measurement, and
(ii) documents the reconstruction and its resolution limits on the corrected
data. Section~\ref{sec:pipeline} defines the pipeline, operator, regularization
filter, truth smearing, and metrics precisely; Sec.~\ref{sec:fix} the artifact
and fix; Sec.~\ref{sec:results} the reconstruction; Sec.~\ref{sec:shield} the
shielded self-trigger over-recovery and its bin-resolution origin.

\section{Simulation and reconstruction pipeline}\label{sec:pipeline}

\subsection{Datasets}
$3$~GeV positron particle-gun events are simulated through the full chain
(Geant4 $\to$ drift/field-response $\to$ readout) with the large
$25\times25$-pixel field response: the un-shielded \texttt{v2a\_full} response,
and the shielded \texttt{v2a\_shield\_500V} variant (sharper, faster) for the
shield datasets. Readout noise is $\sigma_\mathrm{uncorr}=0.5$, kTC
$\sigma_\mathrm{reset}=0.9$, discriminator $\sigma_\mathrm{thr}=0.65$~ke$^-$;
trigger threshold $5$~ke$^-$; hold delay $B=30$ ticks $=1.5~\mu$s (fine tick
$0.05~\mu$s). Burst modes span $n_\mathrm{burst}=1$ (self-trigger),
$4,8,16,64$. All datasets were regenerated with the fixed readout
(Sec.~\ref{sec:fix}).

\subsection{Measurement operator $A(d)$}
$A = (\text{window sampling})\circ(\text{bin-integrated response convolution})$.
The field response is rebinned to bins of width $B$ (non-overlapping tick sums)
and convolved with the charge $q$ on the same $B$-bin grid; each latch window
$[t_\mathrm{lo},t_\mathrm{hi}]$ (the physical $B$-tick CSA integration) samples
the predicted current by overlap-fraction-weighted sums of the block bins. $A$
and $d$ are built from the \emph{raw} latches and are immutable through the
solve; the burst compensation feeds only the warm-start seed, never $A$ or $d$.

\subsection{Warm-start regularization filter}
The seed is a Gaussian-regularized FFT deconvolution of the compensated block.
The filter is a frequency-domain Gaussian applied per axis,
\[
  g(f) = \exp\!\left(-\tfrac{1}{2}\,f^2/\sigma_f^2\right),
\]
with $\sigma_f=(\sigma_\mathrm{pxl},\sigma_\mathrm{pxl},\sigma_\mathrm{time})
=(0.2,0.2,0.005)$ in cycles-per-bin (pixel, pixel, time). The equivalent
real-space resolution is
\[
  \sigma_t = \frac{t_\mathrm{tick}}{2\pi\,\sigma_\mathrm{time}}
  = \frac{0.05~\mu\mathrm{s}}{2\pi(0.005)} = 1.59~\mu\mathrm{s},\quad
  \sigma_\mathrm{pxl}^{\,\mathrm{sp}} = \frac{1}{2\pi(0.2)} = 0.80~\text{pitch}.
\]
A \emph{weaker} filter means a larger $\sigma_f$ (sharper real-space, less
smoothing). The seed provides $q_0$ and, thresholded, the support ROI.

\subsection{Constrained solve}
The solution minimises $\lVert A q - d\rVert^2$ plus objective terms with FISTA:
a soft-seeded weighted-$\ell_1$ homotopy ladder over $\alpha\in\{1.0,0.5,0.3\}$,
a quiet-hinge term penalising above-threshold charge on never-fired pixels, and
(sparse regime) a running-max censor term. Dense configs run $150$ iterations
per stage; the self-trigger (sparse) config runs $600$ and adds the censor.
Sub-bin positions are the local reconstruction centroid.

\subsection{Truth smearing and metrics}
For a matched-resolution comparison the truth ionization is deposited and
smeared with the \emph{same} Gaussian ($\sigma_t=1.59~\mu$s,
$\sigma_\mathrm{pxl}=0.80$~pitch) onto a universal grid. Metrics use the
physical selection $q_\mathrm{reco}>0.5$~ke$^-$: the integral bias
$100(\Sigma q_\mathrm{reco}-\Sigma q_\mathrm{truth})/\Sigma q_\mathrm{truth}$
(all voxels, no cut --- a pure charge-conservation number, independent of where
charge lands); Pearson $r$ and slope of reco vs.\ truth; ghost fraction
(reco$>$cut \& truth$<$cut) and isolated-ghost charge $Q^\mathrm{iso}$ (ghosts
$\ge2$ voxels from any truth); killed truth (truth$>$cut \& reco$<$cut); and the
relative RMS of $(q_\mathrm{reco}-q_\mathrm{truth})/q_\mathrm{reco}$ for
high-charge voxels ($>8$~ke$^-$), pointwise and over a $2\times2\times2$
neighborhood. The fast-ADC variant (hold $10$ vs.\ $30$ ticks) bins time three
times finer, is not matched by the adopted dense configuration
($-12\%$ integral, no high-charge voxels), and is excluded.

\section{The reset-noise artifact and its correction}\label{sec:fix}
The noisy accumulator was reset by subtracting the \emph{true} charge since the
last reset, $\texttt{Xacc}\mathrel{-}=\texttt{Xacc\_true}[t_\mathrm{reset}]$,
keeping $\texttt{Xacc}-\texttt{Xacc\_true}$ invariant, while a fresh kTC baseline
was \emph{added} each reset ($\texttt{Xacc}\mathrel{+}=b_k$). The subtraction
never removed the $b_k$, so they accumulated; on a bright pixel the piled-up
baseline stayed above threshold long after collection, faking late re-triggers
(pixel $(139,80)$: 18 latches recording $223$~ke$^-$ vs.\ a true $116.8$,
Fig.~\ref{fig:wave}) that unfold into the late line (Fig.~\ref{fig:ghost}). The
fix is a \emph{memoryless} reset: rebuild the accumulator directly from the
true one, $\texttt{Xacc}=\texttt{Xacc\_true}+\texttt{uncorr}+b_\mathrm{fresh}$,
so past baselines are forgotten and only one fresh kTC value is drawn per epoch.
After the fix, $(139,80)$ latches drop $18\to3$ and its recorded charge
$223\to115$~ke$^-$; the isolated-ghost count $22\to8$.

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{waveform_139_80_compare.png}
\caption{Reconstructed-charge time profile at the bright pixel $(139,80)$, old
(accumulating baseline) vs.\ new (memoryless reset). The old case trails
spurious late-time charge to $+350$~ticks ($83.5$~ke$^-$, $+26\%$); the new is a
localized peak matching the smeared truth ($65.6$ vs.\ $66.2$~ke$^-$).}
\label{fig:wave}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{ghost_compare_resetfix.png}
\caption{Deconvolution ghosts, old (top) vs.\ new (bottom), three projections.
The vertical line of isolated ghosts extending in time (old, to bin 257)
collapses into the truth body after the fix.}
\label{fig:ghost}
\end{figure}

""" + events + r"""

\section{Reconstruction results}\label{sec:results}
Table~\ref{tab:metrics} compares the reconstruction on the fixed data. The bulk
correlation is high and improves with burst count as more samples constrain the
waveform ($r: 0.981\to0.991\to0.994$ for $n_b=4,16,64$; Fig.~\ref{fig:corrscan});
isolated-ghost charge is small throughout. The high-charge relative RMS drops
under a $2\times2\times2$ sum, i.e.\ the residual high-$q$ scatter is dominantly
a $\pm1$ time-bin misalignment rather than a charge-fidelity loss
(Fig.~\ref{fig:mis}): a $\pm1$ \emph{time} sum collapses it far more than a
$3\times3$ \emph{spatial} sum. A residual $+5$--$8\%$ bias survives all sums and
is an Eddington-type selection effect of cutting on the noisy observable.

\begin{table*}[t]\centering
\caption{Reconstruction metrics on the fixed-readout positron datasets
($q_\mathrm{reco}>0.5$~ke$^-$). RMS columns: high-charge ($>8$~ke$^-$) relative
RMS of $(q_\mathrm{reco}-q_\mathrm{truth})/q_\mathrm{reco}$, pointwise and
$2^3$-summed.}
\label{tab:metrics}
\begin{tabular}{lccrrrrrrrr}
\toprule
dataset & $n_b$ & FR & int.\% & $r$ & slope & ghost\% & $Q^\mathrm{iso}$ & killed & RMS$_\mathrm{pt}$\% & RMS$_{2^3}$\% \\
\midrule
""" + "\n".join(scan) + r"""
\midrule
""" + ("\n".join(shield) if shield else "") + r"""
\bottomrule
\end{tabular}
\end{table*}

%%MONTAGES%%

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{neighborhood_reco.png}
\caption{High-charge residual under neighborhood summing (physical
$q_\mathrm{reco}$ selection, nb4). Summing over $\pm1$ time bin collapses the
scatter far more than a $3\times3$ spatial sum, locating the misalignment in
time; a $+5$--$8\%$ selection bias survives all sums.}
\label{fig:mis}
\end{figure}

\section{Shielded self-trigger over-recovery: a bin-resolution effect}\label{sec:shield}
The shielded self-trigger over-recovers the integral by $+10.5$--$12\%$
(Table~\ref{tab:metrics}), \emph{despite} near-zero isolated ghosts: the excess
sits on the truth voxels themselves (on-truth gain $q_\mathrm{reco}/q_\mathrm{truth}=1.13$),
so it is a total-charge (gain) effect, not a misplacement. It is specific to the
\emph{combination} shielded FR $\times$ single burst: shielded multi-burst
(nb4sh) has gain $1.03$ and un-shielded self-trigger $1.01$.

We isolate the cause (Table~\ref{tab:cond}). (i) The field response is correct:
the deconv FR is byte-identical to the generation FR, and forward-projecting the
truth gives $\Sigma(Aq_\mathrm{truth})\approx\Sigma d$ ($d/Aq=1.00$ multi-burst,
$1.045$ shield self-trigger). (ii) It is not the warm-start seed: seeding the
solve with the \emph{truth} converges to the same $+10.4\%$ as the warm seed.
(iii) It is not the censor: removing it gives $+11.6\%$. So the objective's
minimum itself lies at $+10.5\%$ --- the data term $\lVert Aq-d\rVert^2$ has a
charge-additive near-null direction. (iv) \textbf{Rebuilding $A(d)$ at half the
time-bin width ($B/2=15$ ticks) removes it}: from a truth seed the solve lands
at $-3.9\%$ instead of $+11.9\%$. Since halving the bin adds q degrees of freedom
against the same data $d$, pure information loss would worsen conditioning; that
it \emph{improves} shows the coarse $B$-bin was averaging the sharp shielded
response into a flat (near-null) direction that the finer, sharper kernel closes.

\begin{table}[t]\centering
\caption{Shielded self-trigger integral diagnostics. Total charge from a truth
($\Sigma=32426$) seed under the sparse solve, and forward-projection ratio.}
\label{tab:cond}
\begin{tabular}{lr}
\toprule
test & total (\% vs.\ truth) \\
\midrule
warm-start seed $\to$ solve            & $35848\ (+10.6\%)$ \\
\textbf{truth seed} $\to$ solve        & $35808\ (+10.4\%)$ \\
truth seed, censor off                 & $36085\ (+11.3\%)$ \\
forward-projection $d/(Aq_\mathrm{truth})$ & $1.045$ \\
\midrule
truth seed, $B$-bin solve              & $36286\ (+11.9\%)$ \\
\textbf{truth seed, $B/2$-bin solve}   & $\mathbf{31177\ (-3.9\%)}$ \\
\bottomrule
\end{tabular}
\end{table}

This points to two levers for the sharp shielded response, both reducing
smoothing where the response supports it: a finer ($B/2$) operator time bin, and
a \emph{weaker} regularization filter (larger $\sigma_f$). The high-$q$ time
misalignment of Sec.~\ref{sec:results} shares this coarse-time-bin origin, so a
$B/2$ operator is expected to improve both; the cost is a doubled time axis
(larger operator) and a mild under-recovery ($-3.9\%$) to be retuned.

\section{Conclusion}
An incremental CSA reset accumulated kTC baselines and faked late re-triggers,
producing a ghost line and biasing the unfolding; a memoryless reset removes it
at the source, and on regenerated data the late line vanishes, per-pixel charge
returns to truth, and the bulk reconstruction is preserved across burst modes
and the shielded response. The residual limits are a recoverable $\pm1$
time-bin misalignment and, for the shielded self-trigger, a $+10\%$ over-recovery
that is a coarse-time-bin ill-conditioning of $A(d)$ --- removed by a half-bin
operator --- not a field-response, seed, or censor effect.

\end{document}
"""
    montages = (
        montage(["nb4", "nb16", "nb64"],
                r"Truth--reco 2D correlation across the burst scan "
                r"($n_b=4,16,64$; $q_\mathrm{reco}>0.5$~ke$^-$): Pearson $r$ "
                r"improves $0.981\to0.991\to0.994$ as more bursts sample the "
                r"waveform. Axes annotate the reco cut and the truth smearing "
                r"($\sigma_t=1.59~\mu$s, $\sigma_\mathrm{pxl}=0.80$~pitch).",
                "fig:corrscan")
        + "\n" +
        montage(["nb1self", "nb4sh", "nb1self_sh"],
                r"2D correlation for the self-trigger (\texttt{nb1self}) and "
                r"shielded datasets. The shielded self-trigger (right) shows the "
                r"$\sim+10\%$ integral over-recovery of Sec.~\ref{sec:shield}.",
                "fig:corrshield"))
    tex = tex.replace("%%MONTAGES%%", montages)
    (OUT / "report.tex").write_text(tex)
    print(f"wrote {OUT/'report.tex'}  ({len(tags)} datasets: {', '.join(tags)})")


if __name__ == "__main__":
    main()
