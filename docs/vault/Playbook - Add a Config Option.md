---
tags: [playbook, howto, config]
updated: 2026-07-21
---

# 🛠️ Playbook — Añadir una opción de config

> Meta: un campo nuevo en el `.cfg` que llegue tipado hasta el motor. [[Home]] · [[qxti.core]]

## Pasos (en `qxti/core/config.py`)

1. **Añade el campo** a la dataclass correcta (`CMDConfig`, `XTPConfig`, `LDOSConfig`,
   `KGridConfig`, …) con **tipo y valor por defecto** que preserve el comportamiento actual.
2. **Parseo:** en `_parse_<sección>_section()`, lee la clave con el helper adecuado
   (`_parse_scalar`, `_parse_csv`, `_parse_int_tuple`, `_parse_float_list`, …). Claves con puntos
   (`a.b.c`) van a dicts anidados automáticamente.
3. **Validación:** si es un enum (p.ej. un método), normaliza a minúsculas y lanza `ValueError` si
   no está en el set permitido (sigue el patrón de `response_method`).
4. **Propágalo al motor:** pásalo desde el *builder* correspondiente en [[qxti.core|simulation]] /
   `susceptibility_scan` / `ldos_scan` hacia [[qxti.response|CMD/XTP]] o [[qxti.analytics]].
5. **Documenta** el campo con un comentario en el `.cfg` de referencia (los `.cfg` llevan comentarios
   inline con `#`).

## Checklist de invariantes

- [ ] **Default retrocompatible** (un `.cfg` viejo sigue corriendo igual).
- [ ] `with_standard_output_dirs()` sigue siendo **idempotente** (si añades una ruta, respeta la
      explícita del usuario y solo reescribe la default).
- [ ] Nada de unidades no-a.u. en el núcleo → [[Concept - Atomic Units]].
- [ ] Si el campo afecta RAM/paralelismo, respeta `reserve_gb`/`n_workers` →
      [[Concept - Memory and Parallelism]].

## Ejemplo mental

`reserve_gb` y `n_workers` se añadieron así: campo en `CMDConfig`/`LDOSConfig` → parseo scalar →
pasado a los motores (mesh/CMD) → usado por [[qxti.utils|memory.py]].

## Prueba

`pytest tests/test_config_and_simulation.py` (parseo + defaults). Añade un caso si el campo tiene
lógica no trivial.

---

Relacionado: [[qxti.core]] · [[Playbook - Add an Observable or Order]] · [[Playbook - Invariants Not to Break]]
