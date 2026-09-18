# DATASHEET — The Biomarker Graveyard (Fase 1)

Tipo tumoral: **adenocarcinoma de pulmón (LUAD)**. Se pivotó desde mama (BRCA), el plan
original, porque CPTAC-BRCA solo tiene 2/105 eventos de muerte (1.9%) en su ventana de
seguimiento disponible — insuficiente para cribar nada de forma honesta. LUAD tiene tasas de
eventos reales y comparables en las tres cohortes (ver tabla). Los datos de BRCA ya descargados
(CPTAC-BRCA, TCGA-BRCA n=300, GSE21653) se conservan en `data/raw/` y `data/processed/` por si se
quiere retomar mama como segundo tipo tumoral más adelante, pero no forman parte del pipeline
principal de este estudio.

## Cohortes

| Rol | Fuente | n | Eventos (muerte) | % eventos | Features | Plataforma |
|---|---|---|---|---|---|---|
| Descubrimiento | CPTAC-LUAD (washu RNA-seq) | 106 | 24 | 22.6% | 58.387 genes | RNA-seq FPKM |
| Réplica 1 | TCGA-LUAD vía GDC API (cohorte completa) | 508 | 183 | 36.0% | 19.938 genes (protein-coding) | RNA-seq TPM (STAR-Counts) |
| Réplica 2 | GSE68465 (Shedden et al. 2008) | 442 | 236 | 53.4% | 13.515 genes | Microarray Affymetrix HG-U133A |

**Genes presentes en las tres cohortes simultáneamente: 11.484** — ese es el universo real de
biomarcadores candidatos auditables end-to-end (descubrimiento + 2 réplicas) en la Fase 2.

**Endpoint**: supervivencia global (overall survival). `os_time` en días (CPTAC, TCGA) o meses
(GSE68465 — unidad distinta, documentado en `src/data.py`, no mezclar sin convertir).
`os_event`: 1 = fallecido, 0 = censurado (vivo o perdido de seguimiento al final del estudio).

**Sin missing data** en ninguna de las tres matrices de expresión ni en las columnas de outcome:
verificado (0 NaN de 6.18M / 5.86M / 5.97M celdas respectivamente). Esto es por diseño, no
casualidad — cada cohorte se filtró a las muestras con expresión Y outcome completos antes de
guardarse en `data/processed/`; las muestras con outcome faltante simplemente no entraron.

## Independencia de las cohortes (verificado, no asumido)

- **CPTAC-LUAD vs TCGA-LUAD**: los `case_id` de CPTAC-LUAD tienen formato `C3L-xxxxx`/`C3N-xxxxx`
  (convención estándar de recolección prospectiva CPTAC3), NO barcodes TCGA (formato
  `TCGA-XX-XXXX`). Verificado por test automático (`test_cptac_and_tcga_patients_are_disjoint` en
  `tests/test_data_integrity.py`): los conjuntos de IDs son disjuntos y ningún `case_id` de CPTAC
  empieza por "TCGA-". Son cohortes de pacientes reclutados de forma independiente (CPTAC3 es una
  recolección prospectiva nueva, distinta del pilotaje CPTAC2 que sí reanalizó tejido TCGA en 2016
  para mama/ovario). LUAD es CPTAC3 puro.
- **GSE68465 vs las otras dos**: plataforma de microarray completamente distinta (Affymetrix
  HG-U133A) frente a RNA-seq — no hay solapamiento técnico posible, y es un estudio
  multi-institucional (Michigan, Moffitt, Memorial Sloan-Kettering, Dana-Farber) publicado en
  2008, anterior a la recolección de CPTAC3/TCGA-LUAD moderna.

## Sesgos conocidos a declarar en el paper

- **Sesgo de sitio/batch**: cada cohorte mezcla varios centros de recolección con protocolos de
  extracción de ARN y secuenciación propios (documentado explícitamente en el título original de
  GSE68465: "a multi-site, blinded validation study"). No se ha hecho corrección de batch
  explícita (ComBat u otra) porque cada estadístico de señal nula se calcula DENTRO de una sola
  cohorte a la vez — el diseño no depende de fusionar las tres matrices de expresión en una sola,
  solo de comparar si el mismo gen (por símbolo HGNC) es significativo en cada cohorte por
  separado. La comparabilidad entre cohortes es a nivel de "¿sobrevive aquí también?", no a nivel
  de valores de expresión crudos.
- **Escalas de expresión no comparables entre plataformas**: FPKM (CPTAC) vs TPM (TCGA) vs
  intensidad de microarray (GSE68465) no son la misma unidad. Se aplica `log2(x+1)` a cada una por
  separado (transformación fija, sin fuga de información) pero NUNCA se comparan valores absolutos
  entre cohortes, solo direcciones de asociación/significancia.
- **Endpoint heterogéneo en granularidad temporal**: días (CPTAC, TCGA) vs meses (GSE68465).
  Cualquier análisis que combine tiempos debe convertir explícitamente; los loaders NO lo hacen
  automáticamente para evitar errores silenciosos de unidad.
- **Censura administrativa vs por pérdida de seguimiento**: no se distingue en las fuentes
  originales; se trata todo censurado como censura no informativa (supuesto estándar de Kaplan-Meier
  /Cox, a declarar como limitación).
- **CPTAC-LUAD es una cohorte pequeña y de seguimiento corto** (máx. ~601 días en el caso de BRCA
  que se descartó; para LUAD el seguimiento es mayor pero aun así más corto que TCGA/GSE68465, que
  llevan más de una década de seguimiento acumulado). Esto limita el poder estadístico del
  screening de descubrimiento — se declarará explícitamente en el paper, no se disimulará.

## Corpus de biomarcadores "claimados" (HPA Pathology Atlas)

`load_hpa_prognostic_corpus()` en `src/data.py` da forma larga a las columnas
`Cancer prognostics - <tipo> (TCGA|validation)` de `proteinatlas.tsv`. Para LUAD en concreto
(`hpa_lung_adenocarcinoma_claims()`, cohorte TCGA de HPA): **14.734 genes evaluados**, de los
cuales HPA llama "potential prognostic" o "validated prognostic" (favorable o desfavorable) a
**1.507** (847 unfavorable potencial + 453 favorable potencial + 202 unfavorable validado + 5
favorable validado) — el resto, 13.227, son "unprognostic" según su propio umbral p<0.001 sin
corrección por comparaciones múltiples. Esa cifra de 1.507 es el punto de partida natural para
comparar contra el recuento propio de "hits naive" que se calculará en la Fase 2 sobre la cohorte
de descubrimiento CPTAC-LUAD.

## Reproducibilidad

- `sha256sum` de todos los ficheros crudos en `data/raw/CHECKSUMS.sha256`.
- `src/fetch_cptac_luad.py`, `src/fetch_tcga_gdc.py --project TCGA-LUAD`,
  `src/fetch_gse68465.py` regeneran `data/raw/` + `data/processed/` desde cero. La muestra de
  TCGA-LUAD usa `random.seed(42)` sobre la lista ordenada de `case_id` — reproducible mientras el
  conjunto de casos disponibles en GDC no cambie (declarar esto como limitación menor si se
  re-ejecuta meses después: GDC añade casos nuevos con el tiempo).
- `src/data.py` son los loaders "rápidos" que leen directamente `data/processed/` sin tocar la red.
