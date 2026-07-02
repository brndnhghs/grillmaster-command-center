# Method Audit Report

**195 methods scanned** across all category packages.

Sorted by gap severity (confirmed missing × 3 + inferred missing).

| ID | Name | File | Class | Current Outputs | Signals | Missing |
|---|---|---|---|---|---|---|
| `166` | Parametric Oscillator Lattice | `simulations/oscillon_resonance.py` | simulation | image, luminance | SCALAR, FIELD? | **SCALAR:damping, SCALAR:epsilon, SCALAR:peak_amplitude, SCALAR:resonance_energy, FIELD?** |
| `167` | Spectral Ocean Synthesis | `simulations/ocean_spectral.py` | simulation | image, luminance | SCALAR | **SCALAR:peak_freq, SCALAR:phillips_alpha, SCALAR:significant_height, SCALAR:wind_speed** |
| `130` | Particle Painter | `simulations/particle_painter.py` | simulation | image, luminance | PARTICLES?, FIELD? | **FIELD?** |
| `14` | Geometric Abstraction | `codegen/geometric_abstraction.py` | generator | image, luminance | PARTICLES?, FIELD? | **PARTICLES?** |
| `144` | Faraday Wave Patterns | `simulations/faraday_waves.py` | simulation | image, luminance | FIELD? | **FIELD?** |
| `148` | GPE Quantum Vortex Turbulence | `simulations/gpe_vortex.py` | simulation | image, luminance | FIELD? | **FIELD?** |
| `15` | Typography | `codegen/typography.py` | generator | image, luminance | PARTICLES?, FIELD? | **PARTICLES?** |
| `150` | FPU Chain Lattice | `simulations/chua_lattice.py` | simulation | image, luminance | FIELD? | **FIELD?** |
| `158` | Darcy-Bénard Porous Convection | `simulations/darcy_benard.py` | simulation | image, luminance | FIELD? | **FIELD?** |
| `159` | Phase Separation + Darcy Advection | `simulations/ac_darcy.py` | simulation | image, luminance | FIELD? | **FIELD?** |
| `16` | Flow Field | `codegen/flow_field.py` | generator | image, luminance | PARTICLES?, FIELD? | **FIELD?** |
| `161` | Spectral Tapestry | `simulations/spectral_tapestry.py` | simulation | image, luminance | FIELD? | **FIELD?** |
| `163` | Fractional Laplacian Reaction-Diffusion | `simulations/fractional_rd.py` | simulation | image, luminance | FIELD? | **FIELD?** |
| `169` | Turing Morphogenesis | `simulations/turing_morphogenesis.py` | simulation | image, luminance | FIELD? | **FIELD?** |
| `170` | Phase Field Crystal | `simulations/pfc.py` | simulation | image, luminance | FIELD? | **FIELD?** |
| `20` | Particle System | `simulations/particles.py` | filter | image, luminance, particles | PARTICLES, FIELD?, FILTER | **FIELD?** |
| `37` | Collage | `codegen/collage.py` | generator | image, luminance | PARTICLES? | **PARTICLES?** |
| `01` | ASCII Art | `codegen/ascii_art.nd-bak-4cb20def.py` | filter | image, luminance | SCALAR, FILTER | **—** |
| `01` | ASCII Art | `codegen/ascii_art.nd-bak-830dc061.py` | filter | image, luminance | SCALAR, FILTER | **—** |
| `01` | ASCII Art | `codegen/ascii_art.nd-bak-beeb879f.py` | filter | image, luminance | FILTER | **—** |
| `01` | ASCII Art | `codegen/ascii_art.py` | generator | image, luminance | — | **—** |
| `02` | Quasicrystal | `patterns/quasicrystal.py` | generator | image, field | FIELD | **—** |
| `03` | Moiré | `patterns/moire.py` | generator | image, luminance | — | **—** |
| `04` | Worley Noise | `patterns/worley_noise.py` | generator | image, luminance | PARTICLES? | **—** |
| `05` | Procedural Noise | `patterns/noise.py` | generator | image, luminance | — | **—** |
| `06` | Wallpaper Group | `patterns/wallpaper.py` | generator | image, luminance | — | **—** |
| `07` | Truchet Tiles | `patterns/truchet.py` | generator | image, luminance | — | **—** |
| `08` | Phyllotaxis | `patterns/phyllotaxis.py` | generator | image, luminance | FIELD? | **—** |
| `09` | QR Code | `codegen/qr_code.py` | generator | image, luminance | — | **—** |
| `10` | Color Palette | `codegen/color_palette.py` | generator | image, luminance, palette | — | **—** |
| `100` | Wave Equation | `simulations/wave_equation.py` | simulation | image, luminance, amplitude, field | FIELD, SCALAR, PARTICLES? | **—** |
| `101` | Viscous Fingering | `simulations/viscous_fingering.py` | simulation | image, field | FIELD | **—** |
| `102` | Swarmalators | `simulations/swarmalators.py` | simulation | image, particles | PARTICLES | **—** |
| `103` | Chaotic Pendulums | `simulations/chaotic_pendulums.py` | simulation | image, field | FIELD, PARTICLES? | **—** |
| `104` | Spherical Harmonics | `math_art/spherical_harmonics.py` | generator | image, luminance | FIELD? | **—** |
| `105` | Morph Grid | `patterns/rotating_snakes.py` | filter | image, luminance | FIELD?, FILTER | **—** |
| `106` | Dielectric Breakdown | `simulations/dielectric_breakdown.py` | simulation | image, luminance, field | FIELD | **—** |
| `107` | Magnetic Pendulum | `simulations/magnetic_pendulum.py` | simulation | image, field | FIELD, PARTICLES? | **—** |
| `108` | 4D Hypercube | `math_art/polytope_4d.py` | generator | image, luminance | PARTICLES? | **—** |
| `109` | Rayleigh-Taylor Instability | `simulations/rayleigh_taylor.py` | filter | image, luminance | FILTER | **—** |
| `11` | Gradient | `codegen/gradient.nd-bak-904b8f5c.py` | generator | image, luminance | — | **—** |
| `11` | Gradient | `codegen/gradient.py` | generator | image, luminance | — | **—** |
| `110` | Sheared Rayleigh-Taylor | `simulations/sheared_rayleigh_taylor.py` | filter | image, luminance | FILTER | **—** |
| `111` | Multi-Layer RT | `simulations/multilayer_rt.py` | filter | image, luminance | FILTER | **—** |
| `112` | Kelvin-Helmholtz Instability | `simulations/kelvin_helmholtz.py` | filter | image, field | FIELD, FILTER | **—** |
| `113` | N-Body Gravity | `simulations/nbody_gravity.py` | simulation | image, luminance, field | FIELD | **—** |
| `114` | Spring-Mass Network | `simulations/spring_mass_network.py` | simulation | image, luminance | PARTICLES? | **—** |
| `116` | Point Vortex Dynamics | `simulations/point_vortex.py` | simulation | image, luminance, field | FIELD | **—** |
| `117` | Refractive Caustics | `simulations/refractive_caustics.py` | simulation | image, luminance | FIELD? | **—** |
| `118` | Lotka-Volterra RD | `simulations/lotka_volterra.py` | simulation | image, luminance, field | FIELD | **—** |
| `119` | LV Turing Regime | `simulations/lv_turing.py` | filter | image, luminance, field | FIELD, FILTER | **—** |
| `12` | Kaleidoscope | `codegen/kaleidoscope.py` | generator | image, luminance | — | **—** |
| `120` | LV 3-Species Food Web | `simulations/lv_3species.py` | filter | image, luminance, field | FIELD, FILTER | **—** |
| `121` | LV Anisotropic Diffusion | `simulations/lv_anisotropic.py` | filter | image, luminance, field | FIELD, FILTER | **—** |
| `122` | Dendritic Solidification | `simulations/dendritic_solidification.py` | simulation | image, luminance, field | FIELD | **—** |
| `123` | Animated LIC Flow | `simulations/lic_flow.py` | simulation | image, luminance | — | **—** |
| `124` | Nonlinear Schrödinger Equation | `simulations/nlse.py` | simulation | image, field | FIELD | **—** |
| `125` | Chladni Eigenmode Morphing | `simulations/chladni.py` | simulation | image, luminance, field | FIELD | **—** |
| `126` | Complex Ginzburg-Landau | `simulations/complex_ginzburg_landau.py` | simulation | image, luminance | FIELD? | **—** |
| `127` | Kuramoto-Sivashinsky | `simulations/kuramoto_sivashinsky.py` | simulation | image, luminance | — | **—** |
| `128` | Swift-Hohenberg Pattern Formation | `simulations/active_brownian.py` | simulation | image, luminance, field | FIELD | **—** |
| `129` | Cellular Potts Model | `simulations/cellular_potts.py` | simulation | image, luminance, field | FIELD | **—** |
| `13` | Dithering | `filters/dither.py` | filter | image, luminance | FILTER | **—** |
| `131` | Burridge-Knopoff Earthquakes | `simulations/burridge_knopoff.py` | simulation | image, luminance | — | **—** |
| `132` | Shallow Water Waves | `simulations/shallow_water.py` | simulation | image, luminance | PARTICLES? | **—** |
| `133` | FitzHugh-Nagumo Excitable Media | `simulations/fitzhugh_nagumo.py` | simulation | image, field | FIELD | **—** |
| `134` | Nonlocal Aggregation (Chemotaxis) | `simulations/chemotaxis.py` | simulation | image, luminance | FIELD? | **—** |
| `135` | KPZ Surface Growth / Erosion | `simulations/kpz_surface_growth.py` | simulation | image, luminance | — | **—** |
| `136` | Elastic Coiling Instability | `simulations/elastic_coiling.py` | simulation | image, luminance, field | FIELD, PARTICLES? | **—** |
| `137` | Image Blend | `compositing/blend.py` | composite | image, luminance | — | **—** |
| `138` | Scalar Math | `compositing/math_merge.py` | composite | value | SCALAR | **—** |
| `139` | Field Combine | `compositing/field_combine.py` | composite | field | FIELD | **—** |
| `140` | Particle Merge | `compositing/particle_merge.py` | composite | particles | PARTICLES | **—** |
| `141` | Apply Mask | `compositing/apply_mask.py` | composite | image, luminance, mask | MASK, FILTER | **—** |
| `142` | Coupled Map Lattice | `simulations/coupled_map_lattice.py` | simulation | image, luminance | FIELD? | **—** |
| `143` | Bacterial Colony (v2) | `simulations/bacterial_colony_v2.py` | simulation | image, luminance | — | **—** |
| `145` | Dynamic Fracture Network | `simulations/dynamic_fracture.py` | simulation | image, luminance | FIELD? | **—** |
| `146` | AC + PM Diffusion | `simulations/cahn_hilliard.py` | simulation | image, luminance | — | **—** |
| `146` | Sand Dune Migration | `simulations/sand_dune_migration.py` | simulation | image, luminance | — | **—** |
| `147` | Viscoelastic Buckling Lattice | `simulations/buckling_lattice.py` | simulation | image, luminance | PARTICLES?, FIELD? | **—** |
| `149` | Ferrofluid Rosensweig Instability | `simulations/ferrofluid.py` | simulation | image, luminance | — | **—** |
| `151` | 4D Polytope | `simulations/tesseract.py` | simulation | image, luminance | — | **—** |
| `152` | Magnetic Reconnection | `simulations/magnetic_reconnection.py` | simulation | image, luminance | PARTICLES?, FIELD? | **—** |
| `153` | Spatial Prisoner's Dilemma | `simulations/spatial_pd.py` | simulation | image, luminance | — | **—** |
| `154` | Continuous Spatial PD (Replicator Dynamics) | `simulations/spd_replicator.py` | simulation | image, luminance | FIELD? | **—** |
| `155` | Gray-Scott Reaction-Diffusion | `simulations/gray_scott.py` | simulation | image, field | FIELD | **—** |
| `156` | Hydraulic Erosion / River Network | `simulations/hydraulic_erosion.py` | simulation | image, luminance, field, max_erosion, total_sediment, drainage_density | FIELD, SCALAR | **—** |
| `157` | Swift-Hohenberg Pattern Formation | `simulations/rayleigh_benard.py` | simulation | image, luminance | — | **—** |
| `160` | Bacterial Colony Morphogenesis | `simulations/bacterial_colony.py` | simulation | image, luminance | — | **—** |
| `162` | Rössler Oscillator Array | `simulations/roessler_array.py` | simulation | image, luminance | — | **—** |
| `164` | Moiré Patterns | `simulations/brusselator.py` | simulation | image, luminance | — | **—** |
| `168` | PM Anisotropic RD | `simulations/pm_anisotropic_rd.py` | simulation | image, luminance | — | **—** |
| `17` | Glitch Art | `filters/glitch.py` | filter | image, luminance | PARTICLES?, FILTER | **—** |
| `18` | Cellular Automata | `codegen/simulations.py` | generator | image, luminance | — | **—** |
| `19` | L-System | `fractals/lsystem.py` | fractal | image, luminance | — | **—** |
| `21` | SD1.5 (diffusers) | `ml_models.py` | generator | image, luminance | — | **—** |
| `22` | ffmpeg Frame | `cli_tools.nd-bak-728734db.py` | filter | image, luminance | FILTER | **—** |
| `22` | ffmpeg Frame | `cli_tools.py` | generator | image, luminance | — | **—** |
| `23` | ImageMagick | `cli_tools.nd-bak-728734db.py` | filter | image, luminance | FILTER | **—** |
| `23` | ImageMagick | `cli_tools.py` | generator | image, luminance | — | **—** |
| `24` | pyfiglet | `cli_tools.nd-bak-728734db.py` | generator | image, luminance | — | **—** |
| `24` | pyfiglet | `cli_tools.py` | generator | image, luminance | — | **—** |
| `25` | boxes | `cli_tools.nd-bak-728734db.py` | generator | image, luminance | — | **—** |
| `26` | cowsay | `cli_tools.nd-bak-728734db.py` | generator | image, luminance | — | **—** |
| `27` | qrencode | `cli_tools.nd-bak-728734db.py` | generator | image, luminance | — | **—** |
| `27` | qrencode | `cli_tools.py` | generator | image, luminance | — | **—** |
| `28` | ComfyUI | `ml_models.py` | generator | image, luminance | — | **—** |
| `29` | Voronoi Tiles | `codegen/voronoi_tiles.py` | generator | image, luminance | — | **—** |
| `30` | SVG Vector | `codegen/svg_vector.py` | generator | image, luminance | — | **—** |
| `31` | Plasma Fractal | `fractals/plasma.py` | fractal | image, luminance | — | **—** |
| `32` | Reaction Diffusion | `simulations/reaction_diffusion.py` | filter | image, luminance, field | FIELD, PARTICLES?, FILTER | **—** |
| `33` | Fractal Explorer | `fractals/fractal.py` | fractal | image, luminance | — | **—** |
| `34` | Boids Flocking | `simulations/boids.py` | filter | image, luminance, spread, particles | PARTICLES, SCALAR, FIELD?, FILTER | **—** |
| `35` | Flow Field | `simulations/flowfield.py` | filter | image, luminance, field | FIELD, PARTICLES?, FILTER | **—** |
| `36` | DLA | `simulations/dla.py` | simulation | image, field | FIELD, PARTICLES? | **—** |
| `38` | Dataviz | `math_art/dataviz.py` | generator | image, luminance | — | **—** |
| `39` | Posterize | `codegen/posterize.py` | generator | image, luminance | — | **—** |
| `40` | Pixel Sort | `filters/pixelsort.py` | filter | image, luminance | FILTER | **—** |
| `41` | Oil Paint | `filters/oil_paint.py` | filter | image, luminance | FILTER | **—** |
| `42` | Fake HDR | `filters/hdr.py` | filter | image, luminance | FILTER | **—** |
| `43` | Density Heatmap | `math_art/density_heatmap.py` | filter | image, field | FIELD, FILTER | **—** |
| `44` | img2txt | `cli_tools.nd-bak-728734db.py` | filter | image, luminance | PARTICLES?, FILTER | **—** |
| `44` | img2txt | `cli_tools.py` | generator | image, luminance | — | **—** |
| `45` | Graphviz | `cli_tools.nd-bak-728734db.py` | generator | image, luminance | — | **—** |
| `45` | Graphviz | `cli_tools.py` | generator | image, luminance | — | **—** |
| `46` | ImageMagick Plasma | `cli_tools.nd-bak-728734db.py` | generator | image, luminance | — | **—** |
| `46` | ImageMagick Plasma | `cli_tools.py` | generator | image, luminance | — | **—** |
| `47` | Chafa | `cli_tools.nd-bak-728734db.py` | filter | image, luminance | FILTER | **—** |
| `47` | Chafa | `cli_tools.py` | generator | image, luminance | — | **—** |
| `48` | FFT Art | `math_art/fft_art.py` | filter | image, luminance | FILTER | **—** |
| `49` | Buddhabrot | `fractals/buddhabrot.py` | fractal | image, field | FIELD, PARTICLES? | **—** |
| `50` | Barnsley Fern | `fractals/barnsley_fern.py` | fractal | image, field, particles | PARTICLES, FIELD | **—** |
| `51` | Burning Ship | `fractals/burning_ship.py` | fractal | image, luminance | — | **—** |
| `52` | Newton Fractal | `fractals/newton_fractal.py` | fractal | image, luminance | — | **—** |
| `53` | Metaballs | `simulations/metaballs.py` | filter | image, field | FIELD, PARTICLES?, FILTER | **—** |
| `54` | Ulam Spiral | `math_art/ulam_spiral.py` | filter | image, field | FIELD, PARTICLES?, FILTER | **—** |
| `55` | Sandpile | `simulations/sandpile.py` | simulation | image, field, particles | PARTICLES, FIELD | **—** |
| `56` | Maze | `math_art/maze.py` | generator | image, field | FIELD | **—** |
| `57` | Slit Scan | `filters/slitscan.py` | filter | image, luminance | FIELD?, FILTER | **—** |
| `58` | Cellular Automata (Variants) | `simulations_cellular.py` | simulation | image, luminance | — | **—** |
| `59` | Data Bending | `filters/data_bending.py` | filter | image, luminance | PARTICLES?, FILTER | **—** |
| `62` | Chaotic Map | `math_art/chaotic_map.py` | generator | image, field | FIELD | **—** |
| `63` | Cross Stitch | `filters/cross_stitch.py` | filter | image, luminance | FILTER | **—** |
| `64` | Edge Halftone | `filters/edge_halftone.py` | filter | image, luminance | FILTER | **—** |
| `65` | Waveform | `math_art/waveform.py` | generator | image, luminance | — | **—** |
| `66` | Julia Set | `fractals/julia_set.py` | fractal | image, luminance | PARTICLES? | **—** |
| `67` | Sierpinski Carpet | `fractals/sierpinski.py` | filter | image, field | FIELD, FILTER | **—** |
| `69` | Lyapunov Fractal | `fractals/lyapunov.py` | fractal | image, luminance | — | **—** |
| `70` | Fractal Flame | `fractals/fractal_flame.py` | fractal | image, field | FIELD | **—** |
| `71` | Chaos Game | `fractals/chaos_game.py` | fractal | image, field, particles | PARTICLES, FIELD | **—** |
| `72` | Pythagorean Tree | `fractals/pythagorean_tree.py` | fractal | image, luminance | — | **—** |
| `73` | Low Poly | `math_art/lowpoly.py` | filter | image, luminance | FIELD?, FILTER | **—** |
| `74` | Swirl Displacement | `filters/swirl.py` | filter | image, luminance | FILTER | **—** |
| `76` | Binary Counter | `math_art/binary_counter.py` | generator | image, luminance | — | **—** |
| `77` | False Color IR | `codegen/false_color_ir.py` | filter | image, luminance | FILTER | **—** |
| `78` | Circle Packing | `math_art/circle_packing.py` | generator | image, luminance | — | **—** |
| `79` | Random Walk | `simulations/random_walk.py` | filter | image, particles | PARTICLES, FIELD?, FILTER | **—** |
| `80` | Pixel Mosaic | `filters/pixel_mosaic.py` | filter | image, luminance | FILTER | **—** |
| `81` | Fourier Circles | `math_art/fourier_circles.py` | generator | image, luminance | PARTICLES?, FIELD? | **—** |
| `82` | GPU Procedural Shaders | `gpu_shaders.py` | generator | image, luminance | — | **—** |
| `83` | p5.js Sketch | `p5_sketches.py` | generator | image, luminance | — | **—** |
| `83` | Langton's Ant | `simulations/langtons_ant.py` | simulation | image, luminance, particles, field | PARTICLES, FIELD | **—** |
| `84` | Quantum Wave Interference | `simulations/quantum_interference.py` | simulation | image, field | FIELD | **—** |
| `85` | Strange Attractors (Chaos Density) | `math_art/strange_attractors.py` | generator | image, luminance | — | **—** |
| `86` | Physarum Slime Mold | `simulations/physarum.py` | simulation | image, luminance, field, particles | PARTICLES, FIELD | **—** |
| `87` | Cyclic CA | `simulations/cyclic_ca.py` | simulation | image, luminance | — | **—** |
| `88` | Particle Life | `simulations/particle_life.py` | simulation | image, luminance, particles | PARTICLES | **—** |
| `89` | Kuramoto Sync | `simulations/kuramoto.py` | simulation | image, luminance, r | SCALAR, PARTICLES?, FIELD? | **—** |
| `90` | Lenia | `simulations/lenia.py` | simulation | image, luminance | — | **—** |
| `91` | BZ Oregonator | `simulations/bz_oregonator.py` | simulation | image, luminance | — | **—** |
| `92` | Lattice Gas | `simulations/lattice_gas.py` | simulation | image, luminance | PARTICLES?, FIELD? | **—** |
| `93` | Ising Model | `simulations/ising.py` | simulation | image, luminance, magnetization, field | FIELD, SCALAR | **—** |
| `94` | Stadium Billiards | `simulations/stadium_billiards.py` | simulation | image, field | FIELD, PARTICLES? | **—** |
| `95` | Coupled Logistic | `simulations/coupled_logistic.py` | simulation | image, luminance | FIELD? | **—** |
| `96` | Forest Fire | `simulations/forest_fire.py` | simulation | image, luminance | — | **—** |
| `97` | Lloyd's Algorithm | `simulations/lloyds_algorithm.py` | simulation | image, luminance | PARTICLES? | **—** |
| `98` | Smoothed Particle Hydrodynamics | `simulations/sph.py` | simulation | image, luminance | FIELD? | **—** |
| `99` | Active Nematic Liquid Crystals | `simulations/active_nematic.py` | simulation | image, luminance | — | **—** |
| `__age_heat__` | AgeHeat | `channels.py` | generator | value, r, g, b | — | **—** |
| `__beats__` | Beats | `channels.py` | generator | beat, bar, trigger | — | **—** |
| `__blend__` | Blend | `channels.py` | generator | value | — | **—** |
| `__burst__` | Burst | `channels.py` | generator | value, active | — | **—** |
| `__counter__` | Counter | `channels.py` | generator | value, phase | — | **—** |
| `__envelope__` | Envelope | `channels.py` | generator | value | — | **—** |
| `__image_to_mask__` | Image to Mask | `compositing/image_to_mask.py` | composite | mask, luminance | SCALAR, FILTER | **—** |
| `__lfo__` | LFO | `channels.py` | generator | value, bipolar | — | **—** |
| `__logic__` | Logic | `channels.py` | generator | value | — | **—** |
| `__math__` | Math | `channels.py` | generator | value | — | **—** |
| `__noise1d__` | Noise1D | `channels.py` | generator | value | PARTICLES? | **—** |
| `__noise__` | Noise | `compositing/noise_node.py` | composite | field, image, luminance, mask, particles, amplitude | FIELD? | **—** |
| `__ramp__` | Ramp | `channels.py` | generator | value, phase | — | **—** |
| `__strobe__` | Strobe | `channels.py` | generator | value, trigger | — | **—** |
| `__test__` | Test Node | `compositing/test_node.py` | composite | image, luminance, field, particles, mask, test_scalar | — | **—** |
| `__timeline__` | Timeline | `system/timeline_node.py` | generator | t, phase, speed, beat, segment | SCALAR | **—** |
| `__transform__` | Transform | `filters/transform.py` | filter | image, luminance, field | FIELD?, FILTER | **—** |

## Legend

- **PARTICLES** — `write_particles()` called but `particles` not in `outputs=` (confirmed)
- **FIELD** — `write_field()` called but `field` not in `outputs=` (confirmed)
- **MASK** — `write_mask()` called but `mask` not in `outputs=` (confirmed)
- **SCALAR:key** — `write_scalars(key=…)` called but key not in `outputs=` (confirmed)
- **PARTICLES?** — particle-like variable names found in assignments (inferred)
- **FIELD?** — field/grid/trail-like variable names found in assignments (inferred)
- **—** — no gaps detected
