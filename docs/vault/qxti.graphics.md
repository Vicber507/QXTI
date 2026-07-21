---
tags: [package, graphics]
updated: 2026-07-21
---

# 📦 qxti.graphics — ploteo

> Solo dibuja desde datos en disco. Convierte a eV/Å aquí (nunca antes).
> [[Home]] · [[Architecture Map]]

**Depende de:** [[qxti.core]], [[qxti.data]]
**Usado por:** el usuario (`python qxti/graphics/graphics.py <cfg>`) y los runners.

## Archivos

| Archivo | Rol |
| --- | --- |
| [graphics.py](../../qxti/graphics/graphics.py) | **Despachador**: auto-detecta familias con datos y plotea |
| [plot_hamiltonian.py](../../qxti/graphics/plot_hamiltonian.py) | Bandas, superficies de energía, campo de velocidad |
| [plot_harmonics.py](../../qxti/graphics/plot_harmonics.py) | Espectros HHG: total, componentes, **intra/inter**, **RCP/LCP** |
| [plot_response.py](../../qxti/graphics/plot_response.py) | Poblaciones/coherencias (heatmaps, animaciones kx-ky) |
| [plot_susceptibility_tensor.py](../../qxti/graphics/plot_susceptibility_tensor.py) | Tensores σ/χ: base cartesiana + **helicidad** |
| [plot_dos.py](../../qxti/graphics/plot_dos.py) | g(E), PDOS, A(k,E), plano E₀ |
| plot_bands.py | ⚠️ **stub vacío** (delegado a plot_hamiltonian) |

## graphics.py (despachador)

`plot_all_graphics_from_saved_data(cfg)` intenta todas las familias y **salta** las que no tienen
datos (sin error). Estandariza rutas con `with_standard_output_dirs()` y lee `.npz` de
`<output_dir>/data/`. Sin flag ⇒ plotea todo lo que exista; con `-cmd/-xtp/-ldos` ⇒ una familia.

## Descomposiciones (dónde viven)

- **RCP/LCP** (`plot_harmonics.py`): `J_R=(J_x−iJ_y)/√2`, `J_L=(J_x+iJ_y)/√2`. Estándar.
- **Intra/inter**: usa las claves `current_*_intraband/interband` del dataset (solo si están).
- **Helicidad** (`plot_susceptibility_tensor.py`): rota el tensor con `e_± = (x±iy)/√2`.

Contexto físico y convenciones en [[Concept - Inter-Intra Decomposition]].

⛔ **Invariantes**
- No calcular física aquí (regla de oro 4). Si necesitas un observable nuevo, créalo en
  [[qxti.data]]/[[qxti.response]] y aquí solo lo lees.
- Lee claves por nombre → si cambian en [[qxti.data|io.py]], actualiza ambos lados.

➕ **Extender:** nuevo plot → [[Playbook - Add an LDOS Method or Plot]].

---

Relacionado: [[qxti.data]] · [[Concept - Inter-Intra Decomposition]]
