# Paso 1 — Auditoría de identidad molecular (6-nov-2026)
Script: paso1_identidad.py · salidas: paso1_hpa_direccional_identidad.csv, paso1_cptac_duplicados*.{txt,csv}

- HPA LUAD (TCGA) con dirección: 1.507 filas = 207 validated + 1.300 potential (1.507 símbolos únicos, 1.507 Ensembl únicos).
- Panel por símbolo (intersección de 3 matrices): 11.484. HPA dentro: 1.080. Fuera: 427 (346 símbolo en CPTAC, 409 en TCGA, **0** en GSE68465 por símbolo exacto).
  - 26 recuperables partiendo los probes GPL96 «A /// B»; ~311 no están en GPL96 por símbolo (renombrados o sin sonda): MRE11 vs MRE11A, KARS1 vs KARS, BMAL2 vs ARNTL2, UFD1 vs UFD1L… → hace falta tabla de alias versionada.
- MKKS: la llamada direccional es de ENSG00000285508; las matrices usan ENSG00000125863 (HPA: unprognostic). Identidad NO coincide → excluir (1.079). No es de los 91 reproducidos.
- **Hallazgo nuevo:** `data/processed/cptac_luad_fpkm.csv.gz` conserva la PRIMERA fila de cada símbolo duplicado (399 símbolos con >1 fila en el crudo). En 96 la fila conservada tiene <1/5 de la expresión de la mejor (retrocopias/pseudogenes); 15 de ellos están en el panel de 11.484. Afecta a 3 genes HPA del corpus: PSMA2, PSMA1, MRPS17 (ninguno entre los 91; p CPTAC 0,20/0,74/0,52).
- Efecto probable sobre los resultados principales: pequeño (15/11.484 genes), pero debe corregirse y recalcularse antes de congelar el instrumento.

## 10-nov — Matriz CPTAC corregida (v2)
`paso1b_cptac_v2.py` → `data/processed/cptac_luad_fpkm_v2.csv.gz` (sha256 53f672aeb61d1db464c823fa3d06ac9d09b7fcb3b92c8b2d61597009d152ee78). Original v1 intacto. Criterio: ante símbolo duplicado, fila de mayor expresión media.
- Cambian 162 filas en total, 19 dentro del panel de 11.484.
- p log-rank CPTAC v1→v2 (genes con cambio relevante): MRPS17 0,52→0,0094 (HPA unfavorable, GSE68465 p=0,0054: candidato a 92.º reproducido, comprobar dirección); TUBB3 0,92→0,017; PSMA2 0,20→0,030; CLN3 0,028→0,62 y RPL41 0,028→0,83 (dejan de ser nominales).
- Consecuencia: los 91, el panel de 121 y los 949 nominales pueden variar en unidades. Hay que re-ejecutar fases 2–4 con v2 antes de congelar. v1 sigue en uso en los resultados actuales.
- TCGA (`fetch_tcga_gdc.py`, protein_coding): 24 símbolos duplicados, casi todos genes de la región pseudoautosómica X/Y (ACTL10, CD99, CRLF2, CSF2RA, IL3RA…); se conserva la primera fila. Impacto menor, documentado como limitación; no se recalcula.
- `src/fetch_cptac_luad.py` corregido en el repo privado (se queda con la fila de mayor expresión). Matriz v1 conservada para trazabilidad.
