# ACCESS.md — Vías de acceso a datos verificadas (Fase 0)

Fecha de verificación: 2026-08-11. Todo lo marcado como VERIFICADO fue efectivamente descargado
y cargado en este entorno (Ryzen 9 7900X, WSL2/Ubuntu, conda env `biomarker`, Python 3.11.15).
Nada de lo que sigue es una promesa ni una suposición de documentación oficial: cada afirmación
tiene un comando o una descarga real detrás, ejecutados hoy.

## Resumen ejecutivo — qué se puede descargar YA sin pedir permiso

| Fuente | Estado | Rol en el estudio |
|---|---|---|
| CPTAC (paquete `cptac`) | ✅ VERIFICADO, sin credenciales | Cohorte de descubrimiento (multi-cáncer, expresión + clínico con supervivencia real) |
| GEO — GSE68465 (LUAD) | ✅ VERIFICADO a nivel de campo | Cohorte de réplica 1 |
| GEO — GSE39582 (CRC), GSE21653 (mama) | ⚠️ Identificadas, NO verificadas a nivel de campo | Candidatas a réplica 2/3, pendiente de confirmar en Fase 1 |
| TCGA vía GDC REST API (no vía `cptac`) | ✅ VERIFICADO, acceso abierto sin token | Réplica adicional / ampliación de cohortes |
| Corpus de biomarcadores "claimados" — HPA Pathology Atlas | ✅ VERIFICADO, descarga masiva real | Sustituye a una revisión sistemática limpia (no existe una única con effect sizes bulk-descargables) |
| `nullcal` (motor de calibración de señal nula) | ❌ **NO ENCONTRADO** | Bloqueante para la Fase 2 — ver sección dedicada abajo |

---

## 1. CPTAC — paquete `cptac` (PayneLab)

**Instalación:** `pip install cptac` → instala limpio, versión `1.5.14`, Python 3.11.

**Corrección importante frente a la propuesta original:** el paquete `cptac` da acceso a los
datos de **CPTAC** (proteogenómica), NO a TCGA en general. Solo incluye muestras TCGA de forma
incidental donde CPTAC reanalizó las mismas muestras (28 OV + 14 BRCA, según metadatos internos
del paquete). Si se quiere TCGA como tal, hay que usar la GDC API (sección 3).

**Verificado hoy:**
- `cptac.list_datasets()` lista 196 combinaciones cáncer×fuente×tipo de dato sin login. Cánceres
  disponibles: `brca, ccrcc, coad, gbm, hnscc, lscc, luad, ov, pdac, ucec` (+ `all_cancers` para
  clínico pan-cáncer).
- Descarga real sin credenciales de:
  - `mssm-all_cancers-clinical-clinical_Pan-cancer.May2022.tsv.gz` → **1933 pacientes, 125
    columnas**. Contiene columnas de supervivencia genuinas y utilizables: `Overall survival, days`,
    `Survival status (1, dead; 0, alive)`, `Recurrence-free survival, days`,
    `Recurrence status (1, yes; 0, no)`, y `tumor_code` (11 tipos tumorales, ej. BR=134, LUAD=229,
    CCRCC=261...).
  - `broad-brca-transcriptomics-BRCA.rsem_transcripts_tpm.txt.gz` → matriz de expresión RSEM TPM,
    121 muestras tumorales BRCA.
- La descarga masiva de **todo** un tipo tumoral (`cptac.download_cancer('brca')`, que trae
  proteómica, fosfoproteómica, CNV, miRNA...) es lenta (se cortó a los 180s sin terminar); para
  este estudio no hace falta multiómica completa, basta con transcriptómica + clínico por fuente
  específica, que sí se descarga rápido (~25MB, unos segundos).

**Rol propuesto:** cohorte de DESCUBRIMIENTO. Multi-cáncer real, con outcome de supervivencia
verificado, sin necesidad de armonizar manualmente el endpoint.

## 2. GEO — vía descarga directa de `series_matrix` (FTP) + `GEOparse`

**Instalación:** `pip install GEOparse` → instala limpio, versión `2.0.4`.

**Verificado hoy — GSE68465** (Shedden et al., adenocarcinoma de pulmón):
- `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68465/matrix/GSE68465_series_matrix.txt.gz`
  → HTTP 200, 36.6MB, descargado y descomprimido.
- Campos `!Sample_characteristics_ch1` confirmados por inspección directa del archivo (no de la
  página web, del archivo real): `disease_state`, `Sex`, `age`, `race`, **`vital_status`**,
  `disease_stage`, `first_progression_or_relapse`, `months_to_first_progression`,
  **`months_to_last_contact_or_death`**, `smoking_history`, `histologic_grade`. Es decir, tiene
  supervivencia global Y libre de progresión utilizables directamente.
- n=462 muestras según la página de GEO (no recontado fila a fila todavía; se hará en Fase 1).

**Identificadas pero NO verificadas a nivel de campo** (solo confirmado que el accession existe y
el tamaño de cohorte reportado en el paper original — falta bajar el `series_matrix` y comprobar
`characteristics_ch1` como se hizo con GSE68465):
- GSE39582 — colorrectal (CIT/Ligue), n≈585.
- GSE21653 — mama (Institut Paoli-Calmettes), n=266, endpoint DFS a 5 años.

**Rol propuesto:** cohortes de RÉPLICA independientes. GSE68465 ya listo para usar; las otras dos
se confirman en Fase 1 antes de comprometerlas al pipeline.

## 3. TCGA — vía GDC REST API (`api.gdc.cancer.gov`), NO vía `cptac`

**Verificado hoy:** query real a
`https://api.gdc.cancer.gov/files?filters=...cases.project.project_id=TCGA-BRCA...data_type=Gene Expression Quantification...access=open`
devuelve archivos reales (`access: "open"`, formato TSV, RNA-Seq STAR gene counts) **sin ningún
token ni autenticación**. Los datos "open access" (expresión procesada, nivel 3) no requieren
Data Use Agreement; solo lo requerirían las secuencias crudas/controladas (BAM/VCF), que este
estudio no necesita.

No existe un equivalente Python maduro y mantenido de `TCGAbiolinks` (paquete de R). La vía
recomendada y verificada es construir un loader fino sobre la API REST del GDC con `requests` en
`src/data.py` durante la Fase 1.

**Rol propuesto:** ampliación de cohortes de descubrimiento/réplica más allá de lo que cubre CPTAC,
específicamente para tipos tumorales donde interese más profundidad muestral TCGA.

## 4. Corpus de biomarcadores CLAIMADOS en la literatura

**No existe una única revisión sistemática limpia con effect sizes bulk-descargables** que cubra
biomarcadores pronósticos de cáncer de forma amplia (se investigaron PrognoScan, GENT2,
PROGgeneV2: los tres son de consulta gen-a-gen vía web, sin exportación masiva de toda la base).

**Alternativa verificada y adoptada** (contemplada explícitamente por el encargo si no hay base
limpia): **Human Protein Atlas — Pathology Atlas**, bulk file
`https://www.proteinatlas.org/download/proteinatlas.tsv.zip` → HTTP 200, 7.46MB comprimido, 43MB
descomprimido, descargado y verificado hoy.

- **20.162 genes** × columnas `Cancer prognostics - <tipo tumoral> (TCGA)` para **20 tipos
  tumorales** de TCGA, y columnas `(validation)` adicionales para 13 de esos tipos con una cohorte
  de validación independiente.
- Cada celda es un veredicto + p-value log-rank real, ej.:
  `"potential prognostic favorable (5.30e-6)"` o `"unprognostic (1.74e-1)"`.
- Metodología documentada por HPA en `proteinatlas.org/humanproteome/pathology`: umbral **p<0.001**
  en Kaplan-Meier (correlación expresión mRNA–supervivencia) para llamar "prognostic".
- Procede del paper de referencia Uhlén et al. 2017, *Science* ("A pathology atlas of the human
  cancer transcriptome").

**Limitación honesta a declarar en el paper:** esto NO es una colección de miles de papers
individuales que cada uno reclama "su" biomarcador — es un único cribado sistemático pan-cáncer ya
publicado, con su propio umbral de significancia. Auditarlo con `nullcal` significa preguntar:
*de los genes que ESTE cribado sistemático (equivalente en tamaño y ambición a la mayoría de la
literatura de biomarcadores agregada) llama "prognostic" a p<0.001 sin corrección por
comparaciones múltiples, ¿cuántos sobreviven permutación + FDR y cuántos replican en cohortes
independientes (CPTAC, GEO)?* Es la alternativa que el propio encargo contempla ("reanalizar un
panel amplio de features y comparar la tasa de hits naive vs calibrada"), y es honesta siempre que
el paper la describa así explícitamente y no la presente como "auditoría de 20.000 papers
publicados".

**Rol propuesto:** define el universo de features candidatas + su "hit naive" ya pre-calculado por
HPA (screening naive sustituto), a re-testar con permutación/FDR en CPTAC (descubrimiento) y a
replicar en GEO/GDC (réplica).

## 5. `nullcal` — RESUELTO

En la primera pasada de la Fase 0, `nullcal_pkg` no existía en ningún sitio del sistema pese a que
el encargo decía que estaba adjunto. Se recibió después como `nullcal_pkg.tar.gz`, se extrajo a
`./nullcal_pkg`, se auditó el código fuente entero (sin red, sin ejecución dinámica, estadística
estándar bien implementada) y se instaló con `pip install -e ./nullcal_pkg`.

Verificado en esta máquina (no solo confiado al reporte de quien lo construyó):
- `pytest nullcal_pkg/tests/` → 7/7 tests pasan.
- `python nullcal_pkg/examples/biomarker_graveyard_demo.py` → sobre datos simulados: naive p<0.05
  declara 21 biomarcadores (FDR=52.4%), calibración de señal nula deja 9 (FDR=0.0%) → 43% de
  supervivencia. Cifra de demo, no del estudio real.

Detalle completo de la API y del resultado de verificación en `PROGRESS.md`.
