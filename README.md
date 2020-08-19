# MEME.jl
**M**etamaterial **E**lectro**M**agnetic **E**mulator

MEME directly simulates the time evolution of Maxwell's equations, and other equations of motion for materials like (but not limited to!):

 - Harmonic oscillation of a [Lorentzian](https://demonstrations.wolfram.com/DrudeLorentzModelForDispersionInDielectrics/) dielectric (effective for insulators)
 - Flow of current in a Drude Metal

MEME is built on many other amazing packages in the Julia ecosystem. It's incredible speed is due to [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). Graphical analysis is easy with [Makie.jl](https://github.com/JuliaPlots/Makie.jl). All quantities, even those used in GPU kernels, have units attached thanks to [Unitful.jl](https://github.com/PainterQubits/Unitful.jl). This means you can input physical quantities in the units of your choice and make measurements that automatically have useful units. It has been indispensable for development because it ensures all formulae are consistent (it would error for `1m + 1s`).

This is an active, but early stage project. If you're interested in simulating electromagnetics a few orders of magnitude faster than on a CPU, stay tuned.

# Nonlinearity
Rather than assuming any forms for nonlinear effects, MEME directly simulates their emergence. For insulators, [anharmonic potentials can be used to produce second harmonic responses (SHG)](https://en.wikipedia.org/wiki/Second-harmonic_generation). For metals, simulation of current flow using the cold plasma equations can accurately predict SHG ([Liu et al](https://www.mdpi.com/2304-6732/2/2/459)).

Currently the behavior has been implemented and validated only for insulators. For metals, a linear response is implemented and validated. There is not yet a working solution for the continuity equations of current. As such, charge may not be conserved using the linear model for asymmetric structures.

# Examples
Here are three simple simulations which show this is going in the right direction.
1. `examples/thin_lens.jl` Light being focused by a lens. [Video link](https://vimeo.com/449371033).
2. `examples/oscillator_1d.jl` A 1D simulation of a Gaussian pulse hitting an insulator with an anharmonic binding potential for charges. This spectrogram shows SHG. ![enter image description here](https://raw.githubusercontent.com/xaellison/MEME.jl/master/images/stfft.png)


3. `examples/oscillator_1d.jl` 1D simulation of monochromatic light hitting a vacuum/metal interface. This shows the expected dispersion relationship for a Drude metal.![enter image description here](https://raw.githubusercontent.com/xaellison/MEME.jl/master/images/drude_index.png)

# Usage

I'm excited to share the functional core and some peripheral tools, but it's far from something I can recommend using right now. That said, Atom and the Juno IDE allow for an interactive usage which makes it feasible, but more of a "notebook" programming experience.
