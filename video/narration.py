"""Narration for the video, split into scenes and beats.

Each beat is one sentence group: Kokoro speaks it as one audio clip, and the
Manim scene plays the matching animations while it is spoken.

``text`` is what appears in the subtitles. ``say`` (optional) is a phonetic
spelling for the TTS engine where the written form would be mispronounced.
"""

SCENES = {
    "S1_Intro": [
        dict(text="Black holes are, by definition, dark. Nothing that crosses the event "
                  "horizon ever comes back out. And yet the regions around black holes are "
                  "among the brightest places in the universe."),
        dict(text="The light doesn't come from the black hole itself. It comes from gas "
                  "spiraling inward, in a flattened, glowing structure called an accretion disk."),
        dict(text="Here is the puzzle we'll unpack. Disks around small black holes, a few times "
                  "the mass of the Sun, shine mostly in X-rays. Disks around supermassive black "
                  "holes, a hundred million times heavier, shine mostly in ultraviolet light. "
                  "Same physics, different colors. Why?"),
    ],
    "S2_Temperature": [
        dict(text="Let's build the simplest model that could possibly work. Picture the disk as "
                  "a thin, flat sheet of gas, orbiting the black hole in concentric rings."),
        dict(text="As gas drifts inward, friction between neighboring rings turns gravitational "
                  "energy into heat, and each ring radiates that heat away from its surface."),
        dict(text="Balancing the energy released at each radius against the energy radiated "
                  "gives the local temperature of the disk."),
        dict(text="Far from the black hole, the bracket is close to one, so the temperature "
                  "falls off like r to the minus three quarters. But the disk ends at the "
                  "innermost stable orbit, and there the temperature drops to zero."),
        dict(text="So the hottest gas sits just outside the inner edge, at forty-nine "
                  "thirty-sixths of the inner radius."),
    ],
    "S3_Spectrum": [
        dict(text="How does a disk like this look spectrally? The key assumption is that the "
                  "disk is optically thick, so each ring glows like a blackbody at its own "
                  "temperature."),
        dict(text="A hot inner ring peaks at high frequency. A cool outer ring peaks at low "
                  "frequency, but it has a much larger area."),
        dict(text="Add them all up, ring by ring, and you get something no single blackbody can "
                  "produce: a broad, stretched-out spectrum."),
        dict(text="At low frequencies every ring is in its Rayleigh-Jeans tail, so the spectrum "
                  "rises like nu squared. At high frequencies it is cut off exponentially by the "
                  "hottest gas. And in between, the overlapping rings combine into a gentle power "
                  "law, nu to the one third: the signature of a multi-temperature disk.",
             say="At low frequencies every ring is in its Rayleigh-Jeans tail, so the spectrum "
                 "rises like new squared. At high frequencies it is cut off exponentially by the "
                 "hottest gas. And in between, the overlapping rings combine into a gentle power "
                 "law, new to the one third: the signature of a multi-temperature disk."),
    ],
    "S4_Scaling": [
        dict(text="Here's where it gets interesting. The temperature scale of the disk depends "
                  "on just one combination: the accretion rate divided by the mass squared, to "
                  "the one quarter power."),
        dict(text="Compare black holes feeding at the same fraction of their Eddington limit, "
                  "and the accretion rate grows in proportion to the mass. So the temperature "
                  "goes like mass to the minus one quarter. Bigger black holes have cooler disks."),
        dict(text="Watch the spectrum as we dial up the mass. The disk becomes vastly more "
                  "luminous, but its peak slides steadily toward lower frequencies."),
        dict(text="A ten solar mass black hole peaks near one kilo-electron-volt, in soft "
                  "X-rays. A hundred million solar mass black hole peaks near seven hundred and "
                  "forty angstroms, in the far ultraviolet. The ratio is ten million to the one "
                  "quarter: a factor of fifty-six. Same equations, different colors."),
        dict(text="Changing the accretion rate does something different. The luminosity rises "
                  "in direct proportion, but the peak barely moves: a hundredfold increase "
                  "shifts it by only about a factor of three."),
    ],
    "S5_Observer": [
        dict(text="What does a telescope actually see? Tilt the disk away from us, and its "
                  "projected area shrinks by the cosine of the inclination. Move it farther "
                  "away, and the flux falls as one over the distance squared."),
        dict(text="The observer sees the same spectral shape, just scaled down. Assume the disk "
                  "shines equally in all directions, and you misjudge its luminosity by a "
                  "factor of two cosine i."),
    ],
    "S6_Numerics": [
        dict(text="To compute all of this, we take the temperature profile from the published "
                  "agnpy package, use astropy's blackbody function, and sum two thousand "
                  "logarithmically spaced rings.",
             say="To compute all of this, we take the temperature profile from the published "
                 "A G N pie package, use astro-pie's blackbody function, and sum two thousand "
                 "logarithmically spaced rings."),
        dict(text="The total luminosity matches the exact formula to a few parts in a million, "
                  "and the error shrinks as one over the number of rings squared."),
    ],
    "S7_Observations": [
        dict(text="The real test, of course, is nature. First, quasars: this is the average "
                  "spectrum of more than two thousand of them, from the Sloan Digital Sky Survey."),
        dict(text="A thin disk for a typical quasar captures the overall shape, the big blue "
                  "bump rising into the far ultraviolet. But the real spectrum is redder, "
                  "especially toward the optical."),
        dict(text="Go further, with a hundred thousand individual quasars. For each one, predict "
                  "the continuum slope from its own mass and accretion rate. In the far "
                  "ultraviolet, the model matches on average. In the optical, almost nine out of "
                  "ten quasars are redder than a bare disk. Something else is in the mix: dust, "
                  "the host galaxy, extra emission."),
        dict(text="Now a stellar-mass black hole: LMC X-3, in the Large Magellanic Cloud. Its "
                  "mass, distance and inclination are measured independently. We fold the model "
                  "through the X-ray detector's response and fit forty-eight spectra from the "
                  "RXTE satellite.",
             say="Now a stellar-mass black hole: L M C X three, in the Large Magellanic Cloud. "
                 "Its mass, distance and inclination are measured independently. We fold the "
                 "model through the X-ray detector's response and fit forty-eight spectra from "
                 "the R X T E satellite."),
        dict(text="It fits beautifully, but only if each ring's light is hardened: it looks like "
                  "a blackbody about one point eight times hotter, then diluted. That is exactly "
                  "what electron scattering in a disk atmosphere is predicted to do. A pure "
                  "blackbody is ruled out completely."),
    ],
    "S8_Conclusion": [
        dict(text="So a handful of ideas, energy balance, blackbody rings and a fourth-root "
                  "scaling, explain why the same physics paints small black holes in X-rays "
                  "and giant ones in ultraviolet light."),
        dict(text="The model leaves things out: general relativity, black hole spin, the hot "
                  "corona, and the messy surroundings of real quasars. But that is what makes it "
                  "useful. Where it fails tells us exactly what else is going on."),
    ],
}

SCENE_ORDER = list(SCENES)
