---
tags: [concept, units]
updated: 2026-07-21
---

# 🧠 Concept — Unidades atómicas

> Regla de oro 1. Todo el núcleo trabaja en **Hartree/Bohr**; la conversión a eV/Å ocurre
> **solo** al plotear. [[Home]] · [[Architecture Map]]

## La regla

- Energías en **Hartree**, momentos k en **1/Bohr**, tiempos/frecuencias en a.u.
- El `.cfg` está en a.u. (`omega`, `E0`, `coherence_time`, `fermi_level`, …).
- El **wrapper del modelo** (`models/<x>.py`, función `H`) convierte de eV/Å (donde suele estar
  escrita la física) a a.u. antes de devolver H(k).
- [[qxti.graphics]] convierte a eV (`AU_TO_EV = 27.211386245988`) y a 1/Å para los ejes.

⛔ **No** metas conversiones de unidad en `core`/`response`/`analytics`. Si un número parece 27×
o 1/27× de lo esperado, casi siempre es una conversión colada donde no debe.

## Dónde viven las constantes hoy

`qxti/utils/constants.py` está **vacío** (stub). Las conversiones están dispersas en los modelos y
en las gráficas. Si algún día se centralizan, `constants.py` es el sitio natural — sin romper nada,
porque hoy nadie importa de ahí. Ver [[qxti.utils]].

## Intensidad del láser

`I = E0² · 3.5e16 W/cm²` (`INTENSITY_AU_TO_W_CM2` en [[qxti.physics|laser.py]]) — única salida en
unidades "de laboratorio", para reportar intensidad de pico.

---

Relacionado: [[Models and Inputs]] · [[qxti.utils]] · [[qxti.graphics]]
