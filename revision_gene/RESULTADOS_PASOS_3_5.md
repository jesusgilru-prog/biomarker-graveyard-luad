# Resultados pasos 3–5 (10-nov-2026) — protocolo congelado ejecutado con matriz CPTAC v2
Scripts y salidas en `analisis/` (paso3_*, paso4_*, paso5_*). Corpus evaluable: **1.067** = 150 *validated* + 917 *potential* (1.080 − MKKS − 12 genes del metagén). Cierra el desacuerdo 5 del acta: el denominador correcto es 157/923 (Astra), no 925.

## Paso 3 — contraste principal (estratificación por |ρ| con el metagén de 13 genes)
Reproducen en CPTAC y GSE68465 (p<0,05 y dirección HPA): **92/1.067**. (Con matriz v1 eran 91/1.080; MRPS17 entra con v2.)
| Quintil |ρ| | rango | n | reproducen | tasa | IC95 % | nulo (perm. etiquetas) |
|---|---|---|---|---|---|---|
| 1 | 0,00–0,13 | 214 | 1 | 0,5 % | 0,05–2,2 % | 0,06 % (p_emp 0,11) |
| 2 | 0,13–0,24 | 213 | 10 | 4,7 % | 2,4–8,2 % | 0,06 % (p 0,001) |
| 3 | 0,24–0,34 | 213 | 10 | 4,7 % | 2,4–8,2 % | 0,06 % |
| 4 | 0,34–0,50 | 213 | 24 | 11,3 % | 7,6–16,0 % | 0,07 % |
| 5 | 0,50–0,94 | 214 | 47 | 22,0 % | 16,8–27,9 % | 0,07 % |
Cochran–Armitage sobre deciles: z=8,35, p=6,8e-17. **Regla del protocolo → resultado A («eje»)**: tendencia monótona y quintil inferior indistinguible del nulo (el único gen es KCNH2). Matiz: los quintiles 2–3 (|ρ| 0,13–0,34) reproducen 4,7 % ≫ nulo, así que el acoplamiento al eje no lo explica todo; se redacta «indistinguible», no «equivalente».
Extra: *validated* de HPA reproducen 15,3 % vs 7,5 % de *potential* (Fisher p=0,004), pero dentro de cada quintil de |ρ| las tasas son parecidas (quintil 5: 24,6 % vs 20,9 %): la etiqueta de validación de HPA apenas añade una vez controlado el acoplamiento (descriptivo).

## Paso 4 — magnitudes (M0 gen+edad+estadio vs M1 +metagén, mismos pacientes: CPTAC 106, GSE68465 439)
Sobre los 92 reproducidos, log-HR orientado por HPA: mediana M0 = 0,279; atenuación relativa **28 %** (IC95 % bootstrap de pacientes 22–32 %, B=100). Por quintil de |ρ|: 16 %, 34 %, 49 %, 57 %, 67 %.
**Nulo emparejado** (60 módulos de 13 genes coexpresados con |ρ| mediana similar): atenuación media 20 % (rango 2,5–97,5 %: −1 a 47 %). **El 28 % observado cae dentro del nulo → la atenuación no identifica el programa de proliferación frente a otros módulos igual de acoplados** (cláusula de refutación del acta; hay que escribirlo). Limitaciones: B=100 y 60 nulos (piloto, no 1.000); CPTAC con 24 eventos; los módulos nulos son «otros programas», no controles de ortogonalidad.

## Paso 5 — conjunción parcial (BH sobre max(p_CPTAC, p_GSE) unilateral orientado)
**0/1.067** con BH y con BY (mejores: PFKP q=0,057, CTSH q=0,092). Marginales: 186 en CPTAC, 335 en GSE68465, 92 en ambas. Coincide con Astra (0 supervivientes). Titular honesto: ningún gen individual se reproduce con control de error de replicación.

## Veredicto provisional (según PROTOCOLO_CONGELADO.md)
Resultado A con salvedades: la reproducibilidad externa de las llamadas pronósticas de HPA está fuertemente graduada por el acoplamiento a proliferación (0,5 % → 22 %), ningún gen pasa conjunción parcial, y la atenuación multivariable no distingue el eje de un módulo coexpresado cualquiera. No es «inconcluyente» (C) en el sentido del protocolo, pero la parte multivariable sí lo es. Pendiente: recálculo fases 2–7 con v2 (en marcha, run_v2/).

---
## Addendum 10-nov tarde — tras la revisión cruzada de Opus (`revision_cruzada/opus.md`; Astra sin cuota hasta 11-nov 9:16)
- **Bug corregido (paso3):** la identidad Ensembl de CPTAC usaba solo 29 muestras para elegir la fila de mayor expresión; con todas las columnas el corpus (1.067) y los 92 no cambian.
- **Paso 4 rehecho (paso4b):** estimador = mediana por gen de 1−β̄_M1/β̄_M0 (92 reproducidos); bootstrap de pacientes con los mismos índices para todos los genes (B=200). Resultado: **21 % (IC95 −13 a 46 %)**. Módulos nulos SIN acoplamiento con el metagén (|ρ| con el metagén 0,21): atenuación 2 % (−12 a 16 %), con |ρ| con los reproducidos 0,20 (no emparejados con 0,50 del metagén). Comparadores a posteriori: módulo alveolar 27 %, linfocitario 3 %.
- **Controles (paso6):** |ρ| solo en TCGA → gradiente 4/8/14/24/42 (z=7,1). Terciles: 5/22/65 de 356 (Q1 = 1,4 %, por encima del nulo → la regla A es frágil). Módulo alveolar (ρ con metagén −0,50/−0,65): gradiente igual de fuerte (1/3/7/22/58; z=10,3) → eje dominante proliferación–diferenciación. Módulo linfocitario: tendencia inversa débil (z=−2,8). Genes con llamada HPA reproducen más que los sin llamada a igual acoplamiento: acoplamiento solo predice 37,4 de los 92 (41 %).
- Los 12 genes del metagén con llamada HPA: 12/12 con la dirección HPA en ambas cohortes, 0/12 con p<0,05 en ambas (metagén: p=0,155 CPTAC, 0,004 GSE) → el criterio está limitado por potencia.
- **Etapa C (10⁶ permutaciones) re-ejecutada en v2:** mejor q_BH=0,069 (LDHA y CHCHD3); 18 genes constantes (antes 22); cociente empírico/analítico CPTAC 0,993.
- Figuras 0 y 1 regeneradas (v2: 950 y 122). Título cambiado a «…dominant proliferation–differentiation expression axis».
