"""Manim scenes for the accretion-disk explainer video.

Each scene plays its narration beat by beat: ``self.beat()`` starts the next
Kokoro audio clip, the animations for that beat are played (their run times
are fractions of the clip length, ``self.d``), and ``self.finish()`` waits
until the clip has ended.  All curves come from video/data/video_data.npz,
computed from the project's model and observational data by data_prep.py.

Render one scene with e.g.
    manim render --resolution 1920,1080 --frame_rate 30 video/scenes.py S2_Temperature
or everything with  python video/build_video.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import *  # noqa: F401,F403

HERE = Path(__file__).resolve().parent
DATA = np.load(HERE / "data" / "video_data.npz")
FACTS = json.loads((HERE / "data" / "facts.json").read_text())
DURATIONS = json.loads((HERE / "audio" / "durations.json").read_text())
BUILD = HERE / "build"

config.background_color = "#0d0f14"

# palette (3Blue1Brown-like on a dark background)
C_BLUE = "#58C4DD"
C_YELLOW = "#FFD866"
C_TEAL = "#5CD0B3"
C_RED = "#FC6255"
C_ORANGE = "#FF8C42"
C_VIOLET = "#B28CFF"
C_GREY = "#8A8F98"
C_TEXT = "#ECECEC"
HOT_RAMP = ["#7a1500", "#d23a00", "#ff7a1a", "#ffc266", "#fff1d6", "#dbe9ff"]


def heat_color(frac):
    """Colour for a normalised temperature 0 (cool) .. 1 (hottest)."""
    frac = float(np.clip(frac, 0, 1)) * (len(HOT_RAMP) - 1)
    i = min(int(frac), len(HOT_RAMP) - 2)
    return interpolate_color(ManimColor(HOT_RAMP[i]), ManimColor(HOT_RAMP[i + 1]), frac - i)


def tex(s, size=36, color=C_TEXT):
    return MathTex(s, font_size=size, color=color)


def txt(s, size=30, color=C_TEXT, **kw):
    return Text(s, font_size=size, color=color, **kw)


class CornerAxes(Axes):
    """Axes whose lines always sit at the lower-left corner of the data range.

    Manim draws the axes through the origin when it lies inside the range; here
    the ranges are shifted to start at zero and ``c2p`` undoes the shift.
    """

    def __init__(self, xr, yr, **kw):
        self.x0, self.y0 = xr[0], yr[0]
        super().__init__(x_range=[0, xr[1] - xr[0], 1], y_range=[0, yr[1] - yr[0], 1], **kw)

    def c2p(self, *coords):
        x, y = coords[0], coords[1]
        return super().c2p(x - self.x0, y - self.y0)


def log_axes(xr, yr, xl, yl, xticks=(), x_label=None, y_label=None, xtick_tex=None,
             y_label_buff=0.25):
    ax = CornerAxes(xr, yr, x_length=xl, y_length=yl, tips=False,
                    axis_config=dict(color=C_GREY, stroke_width=2, include_ticks=False))
    labels = VGroup()
    for v in xticks:
        lab = tex(xtick_tex(v) if xtick_tex else f"10^{{{v}}}", 24, C_GREY)
        lab.next_to(ax.c2p(v, yr[0]), DOWN, buff=0.15)
        tick = Line(ax.c2p(v, yr[0]), ax.c2p(v, yr[0]) + UP * 0.1, color=C_GREY, stroke_width=2)
        labels.add(lab, tick)
    ax.add(labels)
    if x_label:
        ax.add(tex(x_label, 30, C_GREY).next_to(ax.x_axis, DOWN, buff=0.6))
    if y_label:
        ax.add(tex(y_label, 30, C_GREY).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=y_label_buff))
    return ax


def data_curve(ax, x, y, color, sw=4, ylim=None, xlim=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ylim is not None:
        ok &= (y >= ylim[0]) & (y <= ylim[1])
    if xlim is not None:
        ok &= (x >= xlim[0]) & (x <= xlim[1])
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return VMobject()
    # keep the longest contiguous run
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    run = max(runs, key=len)
    pts = [ax.c2p(a, b) for a, b in zip(x[run], y[run])]
    return VMobject(stroke_color=color, stroke_width=sw).set_points_as_corners(pts)


def make_disk(r_in, r_out, n=28, tilt=0.32, hole=True, cool=0.0, hot=1.0, glow=True):
    """A glowing accretion disk seen at an angle (flattened annuli) with a black hole."""
    radii = np.geomspace(r_in, r_out, n + 1)
    g = VGroup()
    for k in range(n):
        frac = 1 - k / (n - 1)
        col = heat_color(cool + (hot - cool) * frac)
        a = Annulus(inner_radius=radii[k], outer_radius=radii[k + 1], fill_color=col,
                    fill_opacity=0.95 - 0.5 * k / n, stroke_width=0)
        g.add(a)
    disk = VGroup(g)
    if glow:
        for s, o in [(1.15, 0.08), (1.35, 0.05)]:
            disk.add(Circle(radius=r_out * s, stroke_width=0, fill_color=HOT_RAMP[2],
                            fill_opacity=o))
    disk.stretch(tilt, dim=1)
    if hole:
        bh = Circle(radius=r_in * 0.92, fill_color=BLACK, fill_opacity=1, stroke_color="#2a2d36",
                    stroke_width=2)
        ring = Circle(radius=r_in * 1.02, stroke_color="#ffe9c2", stroke_width=2.5,
                      stroke_opacity=0.8)
        disk.add(bh, ring)
    return disk


def stars(n=120, seed=3):
    rng = np.random.default_rng(seed)
    return VGroup(*[Dot([rng.uniform(-7.1, 7.1), rng.uniform(-4, 4), 0],
                        radius=rng.uniform(0.006, 0.02), color=WHITE,
                        fill_opacity=rng.uniform(0.2, 0.7)) for _ in range(n)])


def section_title(s):
    return txt(s, 34, C_YELLOW).to_corner(UL, buff=0.45)


class Narrated(Scene):
    """Scene that plays its narration clips beat by beat."""

    key = ""

    def setup(self):
        self.durs = DURATIONS[self.key]
        self.bi = -1
        self.starts = []

    @property
    def d(self):
        return self.durs[self.bi]

    def beat(self):
        self.bi += 1
        self.add_sound(str(HERE / "audio" / f"{self.key}_{self.bi}.wav"))
        self.t0 = self.renderer.time
        self.starts.append(self.t0)

    def left(self):
        return self.t0 + self.d - self.renderer.time

    def finish(self, pad=0.35):
        r = self.left() + pad
        if r > 1 / 30:
            self.wait(r)

    def save_beats(self):
        BUILD.mkdir(exist_ok=True)
        (BUILD / f"beats_{self.key}.json").write_text(json.dumps(
            dict(starts=self.starts, durations=self.durs, total=self.renderer.time)))


# ===========================================================================
class S1_Intro(Narrated):
    key = "S1_Intro"

    def construct(self):
        sky = stars()
        self.add(sky)
        # beat 0: black holes are dark ... yet bright
        self.beat()
        title = txt("Why do black holes glow?", 52).to_edge(UP, buff=0.6)
        bh = Circle(radius=0.75, fill_color=BLACK, fill_opacity=1, stroke_color="#30343f",
                    stroke_width=2)
        horizon = DashedVMobject(Circle(radius=0.95, color=C_GREY, stroke_width=2), num_dashes=40)
        lab = txt("event horizon", 24, C_GREY).next_to(horizon, DOWN, buff=0.25)
        self.play(Write(title), run_time=2)
        self.play(FadeIn(bh), Create(horizon), FadeIn(lab), run_time=2)
        # light that falls in never comes out
        photons = VGroup(*[Arrow(start=3.2 * np.array([np.cos(a), np.sin(a), 0]),
                                 end=1.0 * np.array([np.cos(a), np.sin(a), 0]),
                                 color=C_YELLOW, stroke_width=3, buff=0,
                                 max_tip_length_to_length_ratio=0.12)
                           for a in np.linspace(0.3, 2 * PI + 0.3, 8, endpoint=False)])
        self.play(LaggedStart(*[GrowArrow(p) for p in photons], lag_ratio=0.1), run_time=2)
        self.play(FadeOut(photons, scale=0.3), run_time=1.5)
        disk = make_disk(1.0, 3.6, n=30).move_to(bh)
        self.play(FadeIn(disk, scale=0.8), FadeOut(lab), FadeOut(horizon),
                  bh.animate.set_z_index(5), run_time=max(1.0, self.left() - 0.3))
        self.finish()

        # beat 1: light comes from gas spiralling in, in an accretion disk
        self.beat()
        lab2 = txt("accretion disk", 30, C_ORANGE).next_to(disk, DOWN, buff=0.5)
        paths, dots = [], []
        for k, a0 in enumerate(np.linspace(0, 2 * PI, 10, endpoint=False)):
            path = ParametricFunction(
                lambda t, a0=a0: np.array([(3.6 - 2.5 * t) * np.cos(a0 + 7 * t),
                                           0.32 * (3.6 - 2.5 * t) * np.sin(a0 + 7 * t), 0]),
                t_range=[0, 1], stroke_width=0)
            paths.append(path)
            dots.append(Dot(path.get_start(), radius=0.05, color=WHITE).set_z_index(4))
        self.play(FadeIn(lab2), *[FadeIn(dt) for dt in dots], run_time=1)
        self.play(*[MoveAlongPath(dt, p, rate_func=rate_functions.ease_in_quad)
                    for dt, p in zip(dots, paths)], run_time=max(2.0, self.left() - 1.2))
        self.play(*[FadeOut(dt) for dt in dots], run_time=0.6)
        self.finish(pad=0.2)

        # beat 2: small vs supermassive: X-rays vs ultraviolet
        self.beat()
        small = make_disk(0.25, 0.95, n=18, hot=1.0, cool=0.45).move_to(LEFT * 3.6 + DOWN * 0.3)
        big = make_disk(0.75, 2.6, n=26, hot=0.75, cool=0.0).move_to(RIGHT * 2.9 + DOWN * 0.3)
        self.play(FadeOut(lab2), FadeOut(title), ReplacementTransform(VGroup(disk, bh), big),
                  FadeIn(small), run_time=2.5)
        ls = VGroup(tex(r"\sim 10\,M_\odot", 40), txt("shines in X-rays", 28, C_BLUE)
                    ).arrange(DOWN, buff=0.2).next_to(small, DOWN, buff=0.6)
        lb = VGroup(tex(r"\sim 10^{8}\,M_\odot", 40), txt("shines in ultraviolet", 28, C_VIOLET)
                    ).arrange(DOWN, buff=0.2).next_to(big, DOWN, buff=0.4)
        self.play(FadeIn(ls, shift=UP * 0.2), run_time=1.5)
        self.wait(max(0.1, self.d * 0.25 - 4))
        self.play(FadeIn(lb, shift=UP * 0.2), run_time=1.5)
        q = txt("Same physics, different colours. Why?", 40, C_YELLOW).to_edge(UP, buff=0.7)
        self.wait(max(0.1, self.left() - 4.5))
        self.play(Write(q), run_time=2)
        self.finish(pad=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        self.save_beats()


# ===========================================================================
class S2_Temperature(Narrated):
    key = "S2_Temperature"

    def construct(self):
        # beat 0: thin flat sheet of gas in concentric rings
        self.beat()
        head = section_title("1.  A thin disk")
        radii = np.geomspace(0.45, 3.0, 10)
        rings = VGroup(*[Circle(radius=r, stroke_color=C_GREY, stroke_width=3) for r in radii])
        bh = Circle(radius=0.38, fill_color=BLACK, fill_opacity=1, stroke_color="#30343f")
        disk = VGroup(rings, bh)
        markers = VGroup(*[Dot(radius=0.05, color=C_TEXT).move_to(rr.point_at_angle(0.5 * k))
                           for k, rr in enumerate(rings)])
        for m, r in zip(markers, radii):
            m.add_updater(lambda mob, dt, r=r: mob.rotate(1.6 * (0.45 / r) ** 1.5 * dt,
                                                          about_point=ORIGIN))
        self.play(FadeIn(head), Create(rings, lag_ratio=0.1), FadeIn(bh), run_time=2.5)
        self.add(markers)
        self.play(FadeIn(markers), run_time=1)
        self.finish()

        # beat 1: friction -> heat -> radiated
        self.beat()
        T = DATA["T_T"]
        x = DATA["T_x"]
        tmax = T.max()
        heated = VGroup()
        for r in radii:
            frac = np.interp(r / 0.45 * 6, x, T) / tmax
            heated.add(Circle(radius=r, stroke_color=heat_color(frac), stroke_width=7))
        fr = txt("friction  →  heat", 30, C_ORANGE).to_edge(DOWN, buff=0.6)
        self.play(Transform(rings, heated), FadeIn(fr), run_time=0.45 * self.d)
        waves = VGroup(*[Arrow(rr.point_at_angle(a) * 1.0, rr.point_at_angle(a) * 1.35,
                               color=heat_color(0.6), stroke_width=3, buff=0,
                               max_tip_length_to_length_ratio=0.3)
                         for rr in (rings[2], rings[5], rings[8])
                         for a in np.linspace(0, 2 * PI, 6, endpoint=False)])
        self.play(LaggedStart(*[GrowArrow(w) for w in waves], lag_ratio=0.03),
                  run_time=0.3 * self.d)
        self.finish()

        # beat 2: energy balance equation
        self.beat()
        for m in markers:
            m.clear_updaters()
        small = VGroup(rings, bh, markers)
        self.play(FadeOut(waves), FadeOut(fr), small.animate.scale(0.42).to_corner(UR, buff=0.5),
                  run_time=1.5)
        eq = tex(r"\sigma T^4(r)", 46).copy()
        eq = MathTex(r"\sigma T^4(r)", "=", r"\frac{3GM\dot M}{8\pi r^3}",
                     r"\left[1-\sqrt{\frac{r_{\rm in}}{r}}\,\right]", font_size=46,
                     color=C_TEXT).next_to(head, DOWN, buff=0.5).set_x(-1.0)
        b1 = Brace(eq[0], DOWN, color=C_GREY)
        t1 = txt("radiated", 24, C_GREY).next_to(b1, DOWN, buff=0.1)
        b2 = Brace(eq[2:], DOWN, color=C_GREY)
        t2 = txt("released by gravity", 24, C_GREY).next_to(b2, DOWN, buff=0.1)
        self.play(Write(eq), run_time=0.45 * self.d)
        self.play(GrowFromCenter(b1), FadeIn(t1), GrowFromCenter(b2), FadeIn(t2),
                  run_time=1.5)
        self.finish()

        # beat 3: T ~ r^-3/4 far out; drops to zero at the inner edge
        self.beat()
        self.play(FadeOut(VGroup(b1, t1, b2, t2)), run_time=0.6)
        ax = log_axes([0.7, 5.1], [3.2, 7.2], 9.0, 3.6, xticks=range(1, 6),
                      x_label=r"r\ \ [GM/c^2]", y_label=r"T\ [{\rm K}]",
                      y_label_buff=0.75).to_edge(DOWN, buff=0.75)
        ylabs = VGroup(*[tex(f"10^{{{v}}}", 22, C_GREY).next_to(ax.c2p(0.7, v), LEFT, buff=0.12)
                         for v in (4, 5, 6, 7)])
        lx, lT = np.log10(x), np.log10(np.maximum(T, 1))
        curve = data_curve(ax, lx, lT, C_YELLOW, 5, ylim=(3.2, 7.2))
        power = DashedVMobject(data_curve(ax, lx, np.log10(DATA["T_pow"]), C_BLUE, 3,
                                          ylim=(3.2, 7.2)), num_dashes=60)
        self.play(Create(ax), FadeIn(ylabs), run_time=1.2)
        self.play(Create(curve), run_time=0.3 * self.d, rate_func=linear)
        plaw = tex(r"T\propto r^{-3/4}", 34, C_BLUE).move_to(ax.c2p(3.6, 5.4))
        self.play(Create(power), FadeIn(plaw), Indicate(eq[3], color=C_BLUE), run_time=2.5)
        isco = DashedLine(ax.c2p(np.log10(6), 3.2), ax.c2p(np.log10(6), 7.2), color=C_RED,
                          stroke_width=2.5)
        il = txt("innermost stable orbit", 22, C_RED).next_to(isco, RIGHT, buff=0.1).shift(DOWN * 1.1)
        self.play(Create(isco), FadeIn(il), run_time=1.5)
        self.finish()

        # beat 4: hottest at 49/36 r_in
        self.beat()
        rmax = FACTS["r_Tmax_rg"]
        dot = Dot(ax.c2p(np.log10(rmax), np.log10(FACTS["T_max"])), color=WHITE, radius=0.09)
        lab = tex(r"T_{\max}\ \text{at}\ r=\tfrac{49}{36}\,r_{\rm in}", 34, WHITE
                  ).next_to(dot, RIGHT, buff=0.3).shift(RIGHT * 1.2 + UP * 0.25)
        pointer = Line(lab.get_left() + LEFT * 0.05, dot.get_center(), color=WHITE, stroke_width=2,
                       buff=0.12)
        self.play(GrowFromCenter(dot), Write(lab), Create(pointer), run_time=2)
        self.play(Flash(dot, color=WHITE, flash_radius=0.4), run_time=1)
        self.finish(pad=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        self.save_beats()


# ===========================================================================
class S3_Spectrum(Narrated):
    key = "S3_Spectrum"

    def construct(self):
        nu = np.log10(DATA["sp_nu"])
        zone = np.log10(np.maximum(DATA["sp_zone_L"], 1e-300))
        total = np.log10(DATA["sp_total_L"])
        top = total.max()
        zone, total = zone - top, total - top
        yr = (-7.0, 1.6)
        ax = log_axes([12, 18.6], list(yr), 10.5, 4.6, xticks=range(12, 19),
                      x_label=r"\text{frequency } \nu\ [{\rm Hz}]", y_label=r"L_\nu"
                      ).to_edge(DOWN, buff=0.85).shift(RIGHT * 0.3)

        # beat 0: each ring glows like a blackbody
        self.beat()
        head = section_title("2.  From rings to a spectrum")
        planck = tex(r"B_\nu(T)=\frac{2h\nu^3}{c^2}\,\frac{1}{e^{h\nu/kT}-1}", 40
                     ).next_to(head, DOWN, buff=0.35).to_edge(RIGHT, buff=0.8)
        self.play(FadeIn(head), Write(planck), run_time=0.4 * self.d)
        self.play(Create(ax), run_time=1.5)
        self.finish()

        # beat 1: hot inner ring vs cool, large outer ring
        self.beat()
        Tz = DATA["sp_zone_T"]
        cols = [heat_color(np.log10(t / Tz.min()) / np.log10(Tz.max() / Tz.min())) for t in Tz]
        hot = data_curve(ax, nu, zone[0], cols[0], 4, ylim=yr)
        cold = data_curve(ax, nu, zone[-1], cols[-1], 4, ylim=yr)
        lh = txt("hot inner ring", 24, cols[0]).next_to(hot, UP, buff=0.1).shift(RIGHT * 1.3)
        lc = txt("cool outer ring: large area", 24, "#ff8a4c").next_to(cold, UP, buff=0.2)
        self.play(Create(hot), FadeIn(lh), run_time=0.4 * self.d)
        self.play(Create(cold), FadeIn(lc), run_time=0.4 * self.d)
        self.finish()

        # beat 2: add all rings -> broad spectrum
        self.beat()
        others = VGroup(*[data_curve(ax, nu, zone[k], cols[k], 3, ylim=yr) for k in range(1, 7)])
        tot = data_curve(ax, nu, total, WHITE, 6, ylim=yr)
        sig = tex(r"L_\nu = 2\pi\sum_k B_\nu(T_k)\,A_k", 40, WHITE
                  ).next_to(head, DOWN, buff=0.4).align_to(head, LEFT).shift(RIGHT * 0.3)
        self.play(FadeOut(lh), FadeOut(lc),
                  LaggedStart(*[Create(o) for o in others], lag_ratio=0.15), run_time=2.5)
        self.play(Create(tot), Write(sig), run_time=max(1.5, self.left() - 0.4))
        self.finish()

        # beat 3: the three regimes
        self.beat()
        fade = VGroup(hot, cold, others)
        self.play(fade.animate.set_stroke(opacity=0.25), run_time=1)

        def seg(x0, x1, slope, xa, color):
            ya = np.interp(xa, nu, total)
            xs = np.array([x0, x1])
            return Line(ax.c2p(x0, ya + slope * (x0 - xa)), ax.c2p(x1, ya + slope * (x1 - xa)),
                        color=color, stroke_width=4)
        s1 = DashedVMobject(seg(12.2, 13.6, 2.0, 12.9, C_BLUE), num_dashes=12)
        l1 = tex(r"\propto\nu^{2}", 34, C_BLUE).next_to(s1.get_center(), UL, buff=0.25)
        s2 = DashedVMobject(seg(14.6, 16.9, 1 / 3, 15.8, C_TEAL), num_dashes=16).shift(UP * 0.3)
        l2 = tex(r"\propto\nu^{1/3}", 34, C_TEAL).next_to(s2, LEFT, buff=0.15)
        l3 = tex(r"\propto e^{-h\nu/kT_{\max}}", 32, C_RED).move_to(ax.c2p(17.7, -3.2))
        l3.add_background_rectangle(color=config.background_color, opacity=0.85, buff=0.08)
        a3 = Arrow(l3.get_left(), ax.c2p(17.6, np.interp(17.6, nu, total)), color=C_RED,
                   stroke_width=3, buff=0.1)
        self.play(Create(s1), Write(l1), run_time=0.15 * self.d)
        self.wait(0.1 * self.d)
        self.play(FadeIn(l3), GrowArrow(a3), run_time=0.15 * self.d)
        self.wait(0.1 * self.d)
        self.play(Create(s2), Write(l2), run_time=0.15 * self.d)
        box = SurroundingRectangle(l2, color=C_TEAL, buff=0.1)
        self.play(Create(box), run_time=1)
        self.finish(pad=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        self.save_beats()


# ===========================================================================
class S4_Scaling(Narrated):
    key = "S4_Scaling"

    def construct(self):
        # beat 0: T depends on Mdot / M^2
        self.beat()
        head = section_title("3.  Mass and accretion rate")
        e1 = MathTex(r"T_{\max}", r"\propto", r"\left(\frac{\dot M}{M^2}\right)^{1/4}",
                     font_size=56).move_to(UP * 0.8)
        self.play(FadeIn(head), Write(e1), run_time=0.45 * self.d)
        self.play(Circumscribe(e1[2], color=C_YELLOW, fade_out=True), run_time=2)
        self.finish()

        # beat 1: same Eddington ratio -> Mdot ~ M -> T ~ M^-1/4
        self.beat()
        cond = MathTex(r"\text{same } L/L_{\rm Edd}:\quad \dot M\propto M", font_size=40,
                       color=C_BLUE).next_to(e1, DOWN, buff=0.7)
        self.play(FadeIn(cond, shift=UP * 0.2), run_time=0.25 * self.d)
        e2 = MathTex(r"T_{\max}", r"\propto", r"M^{-1/4}", font_size=56).move_to(e1)
        self.play(TransformMatchingTex(e1, e2), run_time=2.5)
        msg = txt("Bigger black holes have cooler disks", 34, C_YELLOW).next_to(cond, DOWN,
                                                                              buff=0.6)
        self.play(Write(msg), run_time=max(1.5, self.left() - 0.5))
        self.finish()

        # beat 2: dial up the mass
        self.beat()
        self.play(FadeOut(cond), FadeOut(msg), e2.animate.scale(0.7).to_corner(UR, buff=0.5),
                  run_time=1.2)
        lnu0 = np.log10(DATA["sc_nu"])
        lL0 = np.log10(DATA["sc_nuL"])
        yr = (33.5, 51.0)
        ax = log_axes([9.5, 19.5], list(yr), 11.0, 4.0, xticks=range(10, 20, 2),
                      x_label=r"\text{frequency } \nu\ [{\rm Hz}]", y_label=r"\nu L_\nu"
                      ).to_edge(DOWN, buff=0.8)
        bands = VGroup()
        for lo, hi, name, col in [(9.5, 11.5, "radio", "#3a3f4b"), (11.5, 14.6, "infrared", "#4b2f2f"),
                                  (14.6, 14.9, "", "#7a7a3a"), (14.9, 16.5, "ultraviolet", "#3f2f5a"),
                                  (16.5, 19.5, "X-rays", "#24425a")]:
            r = Rectangle(width=ax.c2p(hi, 0)[0] - ax.c2p(lo, 0)[0], height=0.32, stroke_width=0,
                          fill_color=col, fill_opacity=0.9)
            r.move_to([(ax.c2p(lo, 0)[0] + ax.c2p(hi, 0)[0]) / 2, ax.c2p(0, yr[1])[1] + 0.25, 0])
            bands.add(r)
            if name:
                bands.add(txt(name, 18, C_TEXT).move_to(r))
        bands.add(txt("optical", 16, C_TEXT).next_to(bands[4], UP, buff=0.05))
        logM = ValueTracker(1.0)

        def spec(lm, color=C_YELLOW, sw=5):
            return data_curve(ax, lnu0 - 0.25 * (lm - 1), lL0 + (lm - 1), color, sw,
                              ylim=yr, xlim=(9.5, 19.5))
        curve = always_redraw(lambda: spec(logM.get_value()))
        ipk = int(np.argmax(lL0))
        peak = always_redraw(lambda: Dot(ax.c2p(lnu0[ipk] - 0.25 * (logM.get_value() - 1),
                                                lL0[ipk] + logM.get_value() - 1),
                                         color=WHITE, radius=0.07))
        readout = always_redraw(lambda: MathTex(
            rf"M = 10^{{{logM.get_value():.1f}}}\,M_\odot", font_size=40, color=C_YELLOW
        ).next_to(head, DOWN, buff=0.3).align_to(head, LEFT))
        self.play(Create(ax), FadeIn(bands), run_time=1.5)
        self.add(curve, peak, readout)
        self.play(logM.animate.set_value(9.0), run_time=max(3.0, self.left() - 0.3),
                  rate_func=rate_functions.ease_in_out_sine)
        self.finish(pad=0.2)

        # beat 3: 10 Msun vs 1e8 Msun, factor 56
        self.beat()
        c10 = spec(1.0, C_BLUE, 5)
        c8 = spec(8.0, C_VIOLET, 5)
        self.play(FadeOut(curve), FadeOut(peak), FadeOut(readout),
                  FadeIn(c10), FadeIn(c8), run_time=1.5)
        p10 = ax.c2p(lnu0[ipk], lL0[ipk])
        p8 = ax.c2p(lnu0[ipk] - 1.75, lL0[ipk] + 7)
        d10, d8 = Dot(p10, color=C_BLUE), Dot(p8, color=C_VIOLET)
        l10 = VGroup(tex(r"10\,M_\odot", 32, C_BLUE),
                     tex(rf"\approx {FACTS['stellar_E_keV']:.1f}\ {{\rm keV}}", 30, C_BLUE),
                     txt("soft X-rays", 22, C_BLUE)).arrange(DOWN, buff=0.1).next_to(d10, UP, buff=0.25)
        l8 = VGroup(tex(r"10^{8}\,M_\odot", 32, C_VIOLET),
                    tex(rf"\approx {FACTS['smbh_lambda_A']:.0f}\ \text{{\AA}}", 30, C_VIOLET),
                    txt("far ultraviolet", 22, C_VIOLET)).arrange(DOWN, buff=0.1).next_to(d8, UP, buff=0.15)
        self.play(FadeIn(d10), FadeIn(l10), run_time=0.15 * self.d)
        self.play(FadeIn(d8), FadeIn(l8), run_time=0.15 * self.d)
        y_arrow = ax.c2p(0, 34.4)[1]
        arr = DoubleArrow([p8[0], y_arrow, 0], [p10[0], y_arrow, 0], color=C_YELLOW, buff=0,
                          stroke_width=4)
        fac = tex(rf"\times {FACTS['peak_ratio']:.0f} = \left(10^{{7}}\right)^{{1/4}}", 36,
                  C_YELLOW).move_to(ax.c2p(18.3, 46.0))
        g1 = DashedLine(p8, [p8[0], y_arrow, 0], color=C_VIOLET, stroke_width=2)
        g2 = DashedLine(p10, [p10[0], y_arrow, 0], color=C_BLUE, stroke_width=2)
        self.play(Create(g1), Create(g2), GrowFromCenter(arr), Write(fac), run_time=0.2 * self.d)
        self.finish()

        # beat 4: accretion rate: L ~ Mdot, peak ~ Mdot^1/4
        self.beat()
        self.play(*[FadeOut(m) for m in (c8, d8, l8, arr, fac, g1, g2, l10, d10, c10)], run_time=1)
        lam = ValueTracker(-2.0)          # log10(L/L_Edd); the base curve is at -1

        def specl(ll, color=C_TEAL, sw=5):
            return data_curve(ax, lnu0 + 0.25 * (ll + 1), lL0 + (ll + 1), color, sw,
                              ylim=yr, xlim=(9.5, 19.5))
        ghost_lo = specl(-2.0, C_GREY, 2.5)
        cur = always_redraw(lambda: specl(lam.get_value()))
        rd = always_redraw(lambda: MathTex(rf"L/L_{{\rm Edd}} = {10 ** lam.get_value():.2f}",
                                           font_size=40, color=C_TEAL
                                           ).next_to(head, DOWN, buff=0.3).align_to(head, LEFT))
        rule = tex(r"L\propto\dot M,\qquad \nu_{\rm peak}\propto\dot M^{1/4}", 40, C_TEXT
                   ).move_to(e2).to_edge(RIGHT, buff=0.5)
        self.add(ghost_lo, cur, rd)
        self.play(FadeOut(e2), FadeIn(rule), run_time=1)
        self.play(lam.animate.set_value(0.0), run_time=max(2.5, 0.55 * self.d),
                  rate_func=rate_functions.ease_in_out_sine)
        pk_lo = ax.c2p(lnu0[ipk] - 0.25, lL0[ipk] - 1)
        pk_hi = ax.c2p(lnu0[ipk] + 0.25, lL0[ipk] + 1)
        v = Arrow(pk_lo, [pk_lo[0], pk_hi[1], 0], color=C_YELLOW, buff=0.05, stroke_width=4)
        lv = tex(r"\times 100", 32, C_YELLOW).next_to(v, LEFT, buff=0.12)
        lv.add_background_rectangle(color=config.background_color, opacity=0.85, buff=0.05)
        h = Arrow([pk_lo[0], pk_hi[1], 0], pk_hi, color=C_YELLOW, buff=0.02, stroke_width=4,
                  max_tip_length_to_length_ratio=0.5)
        lh = tex(r"\times 3", 30, C_YELLOW).next_to(h, UP, buff=0.1)
        self.play(GrowArrow(v), FadeIn(lv), GrowArrow(h), FadeIn(lh),
                  run_time=max(1.0, self.left() - 0.2))
        self.finish(pad=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        self.save_beats()


# ===========================================================================
class S5_Observer(Narrated):
    key = "S5_Observer"

    def construct(self):
        # beat 0: inclination and distance
        self.beat()
        head = section_title("4.  What a telescope sees")
        inc = ValueTracker(0.0)
        dist = ValueTracker(1.0)
        center = LEFT * 3.2 + DOWN * 0.4

        def disk_now():
            c = max(np.cos(np.radians(inc.get_value())), 0.06)
            dd = make_disk(0.45, 2.4, n=22, tilt=c, glow=False)
            return dd.scale(1 / dist.get_value()).move_to(center)
        disk = always_redraw(disk_now)
        eye = VGroup(Ellipse(width=0.9, height=0.5, color=C_TEXT, stroke_width=3),
                     Dot(radius=0.12, color=C_TEXT)).move_to(RIGHT * 5.6 + DOWN * 0.4)
        el = txt("observer", 22, C_GREY).next_to(eye, DOWN, buff=0.2)
        ray = DashedLine(center + RIGHT * 2.3, eye.get_left() + LEFT * 0.2, color=C_GREY,
                         stroke_width=2)
        F = MathTex(r"F_\nu", "=", r"{\cos i", r"\over", r"D^2}", r"\sum_k B_\nu(T_k)\,A_k",
                    font_size=44).to_edge(UP, buff=1.3).shift(RIGHT * 1.6)
        ci = always_redraw(lambda: MathTex(
            rf"i = {inc.get_value():.0f}^\circ,\quad \cos i = {np.cos(np.radians(inc.get_value())):.2f}",
            font_size=34, color=C_BLUE).next_to(F, DOWN, buff=0.45))
        self.play(FadeIn(head), FadeIn(disk), FadeIn(eye), FadeIn(el), Create(ray),
                  Write(F), run_time=2)
        self.add(ci)
        self.play(Indicate(F[2], color=C_BLUE), run_time=1)
        self.play(inc.animate.set_value(70), run_time=0.35 * self.d)
        self.play(Indicate(F[4], color=C_YELLOW), run_time=1)
        dl = always_redraw(lambda: MathTex(rf"D \times {dist.get_value():.1f}", font_size=34,
                                           color=C_YELLOW).next_to(ci, DOWN, buff=0.3))
        self.add(dl)
        self.play(dist.animate.set_value(2.5), run_time=max(1.5, self.left() - 0.3))
        self.finish()

        # beat 1: same shape, scaled down; L_iso = 2 cos i L
        self.beat()
        lnu0 = np.log10(DATA["sc_nu"])
        lL0 = np.log10(DATA["sc_nuL"])
        lL0 = lL0 - lL0.max()
        ax = log_axes([14, 19], [-4, 0.6], 5.4, 2.6, xticks=(15, 17, 19),
                      y_label=r"\nu F_\nu").move_to(RIGHT * 2.0 + DOWN * 1.6)
        face = data_curve(ax, lnu0, lL0, C_GREY, 3, ylim=(-4, 0.6), xlim=(14, 19))
        tilted = data_curve(ax, lnu0, lL0 + np.log10(np.cos(np.radians(70))) - 0.4, C_YELLOW, 4,
                            ylim=(-4, 0.6), xlim=(14, 19))
        lf = txt("face-on, nearby", 20, C_GREY).move_to(ax.c2p(14.2, 0.35), aligned_edge=LEFT)
        lt = txt("tilted, farther: same shape", 20, C_YELLOW).next_to(lf, DOWN, buff=0.1
                                                                     ).align_to(lf, LEFT)
        self.remove(ci, dl)
        self.play(FadeOut(ray), FadeOut(eye), FadeOut(el), FadeOut(VGroup(ci.copy(), dl.copy())),
                  Create(ax), Create(face), FadeIn(lf), run_time=0.25 * self.d)
        self.play(TransformFromCopy(face, tilted), FadeIn(lt), run_time=0.2 * self.d)
        liso = tex(r"L_{\rm iso}=4\pi D^2 F = 2\cos i\;L", 40, C_TEAL).next_to(F, DOWN, buff=0.5)
        self.play(Write(liso), run_time=max(1.5, self.left() - 1.2))
        self.play(Circumscribe(liso, color=C_TEAL, fade_out=True), run_time=1)
        self.finish(pad=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        self.save_beats()


# ===========================================================================
class S6_Numerics(Narrated):
    key = "S6_Numerics"

    def construct(self):
        # beat 0: the pipeline
        self.beat()
        head = section_title("5.  Computing it")

        def box(t1, t2, col):
            g = VGroup(txt(t1, 26, col), tex(t2, 34)).arrange(DOWN, buff=0.2)
            r = RoundedRectangle(width=3.6, height=1.7, corner_radius=0.2, color=col,
                                 stroke_width=3).surround(g, buff=0.3)
            return VGroup(r, g)
        b1 = box("agnpy  SSDisk", r"T(r)", C_BLUE)
        b2 = box("astropy  BlackBody", r"B_\nu(T)", C_TEAL)
        b3 = box("2000 log-spaced rings", r"L_\nu=2\pi\sum_k B_\nu A_k", C_YELLOW)
        row = VGroup(b1, b2, b3).arrange(RIGHT, buff=0.8).move_to(UP * 0.6)
        arrows = VGroup(Arrow(b1.get_right(), b2.get_left(), color=C_GREY, buff=0.1),
                        Arrow(b2.get_right(), b3.get_left(), color=C_GREY, buff=0.1))
        self.play(FadeIn(head), run_time=0.8)
        for b, a in ((b1, None), (b2, arrows[0]), (b3, arrows[1])):
            anims = [FadeIn(b, shift=UP * 0.2)] + ([GrowArrow(a)] if a else [])
            self.play(*anims, run_time=0.22 * self.d)
        self.finish()

        # beat 1: convergence
        self.beat()
        self.play(VGroup(row, arrows).animate.scale(0.7).to_edge(UP, buff=1.2), run_time=1)
        N, err = np.log10(DATA["cv_N"]), np.log10(DATA["cv_err"])
        ax = log_axes([0.5, 3.9], [-6.5, 0.5], 6.5, 3.4, xticks=range(1, 4),
                      x_label=r"\text{number of rings } N",
                      y_label=r"\text{error in } L").to_edge(DOWN, buff=0.8).shift(LEFT * 1.5)
        pts = VGroup(*[Dot(ax.c2p(a, b), radius=0.06, color=C_YELLOW) for a, b in zip(N, err)])
        ref = DashedLine(ax.c2p(0.6, err[0] + 2 * (N[0] - 0.6)),
                         ax.c2p(3.8, err[0] - 2 * (3.8 - N[0])), color=C_BLUE, stroke_width=3)
        lr = tex(r"\propto N^{-2}", 34, C_BLUE).next_to(interpolate(ref.get_start(), ref.get_end(), 0.55), UR, buff=0.3)
        res = tex(rf"N=2000:\ \ \text{{error}}\approx {FACTS['L_err_2000'] * 1e6:.0f}\times10^{{-6}}",
                  34, C_TEXT).next_to(ax, RIGHT, buff=0.6)
        self.play(Create(ax), run_time=1)
        self.play(LaggedStart(*[GrowFromCenter(p) for p in pts], lag_ratio=0.08),
                  run_time=0.3 * self.d)
        self.play(Create(ref), FadeIn(lr), FadeIn(res), run_time=max(1.0, self.left() - 0.2))
        self.finish(pad=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        self.save_beats()


# ===========================================================================
class S7_Observations(Narrated):
    key = "S7_Observations"

    def construct(self):
        # beat 0: SDSS composite spectrum
        self.beat()
        head = section_title("6.  Testing against real data")
        lam, lfl = np.log10(DATA["cp_lam"]), np.log10(DATA["cp_lflam"])
        mlam, mlfl = np.log10(DATA["cp_mlam"]), np.log10(DATA["cp_mlflam"])
        off = lfl.max()
        lfl, mlfl = lfl - off, mlfl - off
        ax = log_axes([3.0, 3.95], [-1.15, 0.12], 10.5, 4.4, xticks=[3.0, 3.3, 3.6, 3.9],
                      x_label=r"\text{rest wavelength}\ [\text{\AA}]",
                      y_label=r"\lambda f_\lambda",
                      xtick_tex=lambda v: f"{10 ** v:.0f}").to_edge(DOWN, buff=0.85)
        comp = data_curve(ax, lam, lfl, C_BLUE, 2.5, ylim=(-1.15, 0.12))
        src = txt("SDSS composite of 2204 quasars (Vanden Berk et al. 2001)", 22, C_BLUE
                  ).next_to(head, DOWN, buff=0.3).align_to(head, LEFT)
        self.play(FadeIn(head), Create(ax), run_time=1.5)
        self.play(Create(comp), FadeIn(src), run_time=max(2.0, self.left() - 0.3),
                  rate_func=linear)
        self.finish()

        # beat 1: thin disk for a typical quasar: right shape, too blue
        self.beat()
        mod = data_curve(ax, mlam, mlfl, C_ORANGE, 5, ylim=(-1.15, 0.12))
        lm = tex(rf"\text{{thin disk}},\ M=10^{{{FACTS['quasar_logM']:.1f}}}M_\odot,\ "
                 rf"L/L_{{\rm Edd}}={10 ** FACTS['quasar_logEdd']:.2f}", 30, C_ORANGE
                 ).next_to(src, DOWN, buff=0.2).align_to(src, LEFT)
        self.play(Create(mod), FadeIn(lm), run_time=0.4 * self.d)
        bump = txt("big blue bump", 24, C_YELLOW).move_to(ax.c2p(3.1, -0.05))
        red = Arrow(ax.c2p(3.62, -0.9), ax.c2p(3.62, -0.58), color=C_RED, buff=0, stroke_width=4)
        lr = txt("data redder than the disk", 24, C_RED).next_to(red, LEFT, buff=0.15)
        self.play(FadeIn(bump), run_time=1)
        self.play(GrowArrow(red), FadeIn(lr), run_time=max(1.0, self.left() - 0.3))
        self.finish()

        # beat 2: 1e5 quasars: slope histograms
        self.beat()
        self.play(*[FadeOut(m) for m in (ax, comp, mod, lm, bump, red, lr, src)], run_time=1)
        bins = DATA["hist_bins"]
        mids = 0.5 * (bins[1:] + bins[:-1])
        panels = VGroup()
        for line, name in (("CIV", "C IV  (far UV)"), ("MGII", "Mg II  (near UV)"),
                           ("HB", "Hβ  (optical)")):
            ho, hm = DATA[f"hist_obs_{line}"], DATA[f"hist_mod_{line}"]
            mx = max(ho.max(), hm.max())
            a = CornerAxes([-3.5, 1.0], [0, 1.05], x_length=3.8, y_length=2.6,
                           tips=False, axis_config=dict(color=C_GREY, include_ticks=False))
            bars = VGroup(*[Rectangle(width=a.c2p(bins[i + 1], 0)[0] - a.c2p(bins[i], 0)[0],
                                      height=max(a.c2p(0, ho[i] / mx)[1] - a.c2p(0, 0)[1], 1e-3),
                                      stroke_width=0, fill_color=C_BLUE, fill_opacity=0.75
                                      ).move_to(a.c2p(mids[i], 0), aligned_edge=DOWN)
                            for i in range(len(ho))])
            step_pts = []
            for i in range(len(hm)):
                step_pts += [a.c2p(bins[i], hm[i] / mx), a.c2p(bins[i + 1], hm[i] / mx)]
            step = VMobject(stroke_color=C_YELLOW, stroke_width=3).set_points_as_corners(step_pts)
            title = txt(name, 22, C_TEXT).next_to(a, UP, buff=0.15)
            pct = txt(f"{FACTS[f'{line}_redder'] * 100:.0f}% redder than model", 20,
                      C_RED if FACTS[f"{line}_redder"] > 0.6 else C_TEAL).next_to(a, DOWN, buff=0.2)
            xl = tex(r"\alpha_\lambda", 24, C_GREY).next_to(a.x_axis, RIGHT, buff=0.05)
            panels.add(VGroup(a, bars, step, title, pct, xl))
        panels.arrange(RIGHT, buff=0.55).move_to(DOWN * 0.3)
        key = VGroup(Square(0.22, fill_color=C_BLUE, fill_opacity=0.75, stroke_width=0),
                     txt("observed slopes (Shen et al. 2011)", 22, C_BLUE),
                     Line(ORIGIN, RIGHT * 0.4, color=C_YELLOW, stroke_width=3),
                     txt("thin-disk prediction for each quasar", 22, C_YELLOW)
                     ).arrange(RIGHT, buff=0.2).next_to(head, DOWN, buff=0.35).align_to(head, LEFT)
        self.play(FadeIn(key), *[Create(p[0]) for p in panels], *[FadeIn(p[3]) for p in panels],
                  *[FadeIn(p[5]) for p in panels], run_time=1.5)
        self.play(*[LaggedStart(*[GrowFromEdge(b, DOWN) for b in p[1]], lag_ratio=0.02)
                    for p in panels], run_time=0.2 * self.d)
        self.play(*[Create(p[2]) for p in panels], run_time=0.2 * self.d)
        self.play(FadeIn(panels[0][4]), run_time=1)
        self.wait(max(0.1, 0.12 * self.d))
        self.play(FadeIn(panels[1][4]), FadeIn(panels[2][4]), run_time=1)
        self.finish()

        # beat 3: LMC X-3 X-ray spectrum
        self.beat()
        self.play(FadeOut(panels), FadeOut(key), run_time=1)
        E, R, Er = np.log10(DATA["lx_E"]), DATA["lx_rate"], DATA["lx_err"]
        yr = (-1.7, 2.0)
        ax2 = log_axes([0.45, 1.33], list(yr), 7.0, 4.4, xticks=[np.log10(3), np.log10(5),
                                                                np.log10(10), np.log10(20)],
                       x_label=r"\text{energy}\ [{\rm keV}]",
                       y_label=r"\text{counts s}^{-1}\,\text{keV}^{-1}",
                       xtick_tex=lambda v: f"{10 ** v:.0f}").to_edge(DOWN, buff=0.8).shift(LEFT * 2.2)
        pts = VGroup()
        for e, r, s in zip(E, R, Er):
            if r <= 0:
                continue
            lo, hi = np.log10(max(r - s, 10 ** yr[0])), np.log10(r + s)
            if np.log10(r) < yr[0]:
                continue
            pts.add(VGroup(Line(ax2.c2p(e, lo), ax2.c2p(e, min(hi, yr[1])), color=C_TEXT,
                                stroke_width=2),
                           Dot(ax2.c2p(e, np.log10(r)), radius=0.04, color=C_TEXT)))
        info = VGroup(txt("LMC X-3", 32, C_TEXT),
                      txt(f"RXTE, {FACTS['lx_date']}", 22, C_GREY),
                      tex(r"M=6.98\,M_\odot,\ i=69^\circ,\ D=48\,{\rm kpc}", 30, C_GREY),
                      txt("(Orosz et al. 2014)", 20, C_GREY)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_edge(RIGHT, buff=0.5).to_edge(UP, buff=1.2)
        self.play(Create(ax2), FadeIn(info), run_time=0.3 * self.d)
        self.play(LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.03),
                  run_time=max(1.5, self.left() - 0.3))
        self.finish()

        # beat 4: fits: f_col = 1 fails, 1.7 poor, free 1.83 fits; population 1.81
        self.beat()

        def mcurve(y, col, sw):
            return data_curve(ax2, E, np.log10(np.maximum(y, 1e-6)), col, sw, ylim=yr)
        cbb = mcurve(DATA["lx_bb"], C_RED, 3.5)
        c17 = mcurve(DATA["lx_fixed"], C_ORANGE, 3.5)
        cfr = mcurve(DATA["lx_free"], C_TEAL, 5)
        legend = VGroup(
            tex(rf"\text{{pure blackbody }} (f_{{\rm col}}=1):\ \chi^2={FACTS['lx_chi_bb']}", 26, C_RED),
            tex(rf"f_{{\rm col}}=1.7\ \text{{fixed}}:\ \chi^2={FACTS['lx_chi_fixed']}", 26, C_ORANGE),
            tex(rf"f_{{\rm col}}={FACTS['lx_fcol']:.2f}\ \text{{fitted}}:\ \chi^2={FACTS['lx_chi_free']}", 26, C_TEAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(info, DOWN, buff=0.35).align_to(info, LEFT)
        self.play(Create(cbb), FadeIn(legend[0]), run_time=0.12 * self.d)
        self.play(Create(c17), FadeIn(legend[1]), run_time=0.12 * self.d)
        self.play(Create(cfr), FadeIn(legend[2]), FadeOut(cbb), c17.animate.set_stroke(opacity=0.35),
                  run_time=0.14 * self.d)
        hard = tex(r"\text{each ring: } f_{\rm col}^{-4}\,B_\nu(f_{\rm col}T)", 34, C_YELLOW
                   ).next_to(legend, DOWN, buff=0.3).align_to(legend, LEFT)
        self.play(Write(hard), run_time=0.12 * self.d)
        nl = NumberLine(x_range=[1.5, 2.1, 0.1], length=4.6, color=C_GREY, include_numbers=True,
                        font_size=20, decimal_number_config=dict(num_decimal_places=1)
                        ).next_to(hard, DOWN, buff=0.75).align_to(hard, LEFT).shift(RIGHT * 0.3)
        band = Rectangle(width=nl.n2p(1.8)[0] - nl.n2p(1.6)[0], height=0.5, stroke_width=0,
                         fill_color=C_GREY, fill_opacity=0.3).move_to(
            [(nl.n2p(1.6)[0] + nl.n2p(1.8)[0]) / 2, nl.get_center()[1] + 0.25, 0])
        rng = np.random.default_rng(1)
        fdots = VGroup(*[Dot([nl.n2p(f)[0], nl.get_center()[1] + 0.15 + rng.uniform(0, 0.25), 0],
                             radius=0.035, color=C_TEAL) for f in DATA["lx_fcol"]])
        lbl = tex(rf"48\text{{ spectra: }} f_{{\rm col}}={FACTS['lx_fcol_median']:.2f}\pm0.09", 28,
                  C_TEAL).next_to(nl, DOWN, buff=0.45)
        theory = txt("theory ≈ 1.7", 18, C_GREY).next_to(band, UP, buff=0.08)
        self.play(Create(nl), FadeIn(band), FadeIn(theory), run_time=1)
        self.play(LaggedStart(*[FadeIn(dt) for dt in fdots], lag_ratio=0.03), Write(lbl),
                  run_time=max(1.0, self.left() - 0.2))
        self.finish(pad=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        self.save_beats()


# ===========================================================================
class S8_Conclusion(Narrated):
    key = "S8_Conclusion"

    def construct(self):
        # beat 0: recap
        self.beat()
        eqs = VGroup(
            tex(r"\sigma T^4=\frac{3GM\dot M}{8\pi r^3}\Big[1-\sqrt{r_{\rm in}/r}\Big]", 40),
            tex(r"L_\nu=2\pi\sum_k B_\nu(T_k)\,A_k", 40),
            tex(r"T_{\max}\propto\left(\dot M/M^2\right)^{1/4}", 40, C_YELLOW),
        ).arrange(DOWN, buff=0.85).to_edge(LEFT, buff=0.8).shift(DOWN * 0.1)
        labels = VGroup(txt("energy balance", 22, C_GREY), txt("blackbody rings", 22, C_GREY),
                        txt("fourth-root scaling", 22, C_GREY))
        for l, e in zip(labels, eqs):
            l.next_to(e, UP, buff=0.08).align_to(e, LEFT)
        small = make_disk(0.18, 0.7, n=14, hot=1.0, cool=0.45).move_to(RIGHT * 2.6 + UP * 1.2)
        big = make_disk(0.45, 1.6, n=20, hot=0.75, cool=0.0).move_to(RIGHT * 4.6 + DOWN * 1.3)
        ls = txt("X-rays", 24, C_BLUE).next_to(small, DOWN, buff=0.25)
        lb = txt("ultraviolet", 24, C_VIOLET).next_to(big, DOWN, buff=0.25)
        for e, l in zip(eqs, labels):
            self.play(FadeIn(l), Write(e), run_time=0.2 * self.d)
        self.play(FadeIn(small), FadeIn(ls), FadeIn(big), FadeIn(lb),
                  run_time=max(1.0, self.left() - 0.3))
        self.finish()

        # beat 1: what is missing; where it fails is informative
        self.beat()
        self.play(*[FadeOut(m) for m in (eqs, labels, small, big, ls, lb)], run_time=1)
        miss = VGroup(*[txt("•  " + s, 30, C_TEXT) for s in
                        ("general relativity and black hole spin",
                         "the hot X-ray corona",
                         "dust, host galaxies, extra emission in quasars")]
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(UP * 0.6)
        head = txt("Not in the model", 34, C_YELLOW).next_to(miss, UP, buff=0.5).align_to(miss, LEFT)
        self.play(FadeIn(head), LaggedStart(*[FadeIn(m, shift=RIGHT * 0.2) for m in miss],
                                            lag_ratio=0.35), run_time=0.4 * self.d)
        moral = txt("Where the model fails tells us what else is going on.", 32, C_TEAL
                    ).next_to(miss, DOWN, buff=0.8)
        self.play(Write(moral), run_time=max(1.5, self.left() - 0.5))
        self.finish(pad=1.0)
        end = VGroup(txt("Thermal emission from thin accretion disks", 36, C_TEXT),
                     txt("Shakura–Sunyaev disk · agnpy + astropy · SDSS · RXTE", 24, C_GREY),
                     txt("Narration: Kokoro TTS  ·  Animation: Manim", 22, C_GREY)
                     ).arrange(DOWN, buff=0.3)
        self.play(FadeOut(VGroup(head, miss, moral)), run_time=0.8)
        self.play(FadeIn(end), run_time=1.2)
        self.wait(2.5)
        self.play(FadeOut(end), run_time=1)
        self.save_beats()
