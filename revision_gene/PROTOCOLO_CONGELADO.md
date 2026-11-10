# Protocolo congelado — reencuadre para Gene (redactado 10-nov-2026, ANTES de ejecutar el paso 3)
Retrospectivo: no es prerregistro. Fija instrumento, contraste y regla de decisión antes de ver el resultado.

## Datos
- CPTAC-LUAD con matriz **v2** (fila de mayor expresión por símbolo duplicado; sha256 53f672aeb61d1db464c823fa3d06ac9d09b7fcb3b92c8b2d61597009d152ee78), TCGA-LUAD (n=508), GSE68465 (n=442).
- Corpus de evaluación: llamadas direccionales de HPA LUAD (TCGA) cuyo **Ensembl** coincide con el de la matriz (excluidos MKKS y los de identidad no resoluble), evaluadas SOLO en CPTAC y GSE68465 (TCGA es circular). Denominador = el que salga de la auditoría, no 1.080 por decreto.

## Instrumento
- Metagén de 13 genes: MKI67, TOP2A, PCNA, CCNB1, AURKA, TYMS, BIRC5, CDK1, MCM2, CCNA2, KIF20A, PLK1, TPX2. Media de z-scores dentro de cohorte.
- Intersección con el panel de 121 = 0 (verificar). Los genes del metagén presentes en el corpus se excluyen del corpus antes de contar.

## Contraste principal (paso 3)
- ρ de Spearman gen–metagén dentro de cada cohorte; |ρ| combinado (media de las dos cohortes de evaluación).
- Quintiles de |ρ|. Por quintil: fracción de genes con p<0,05 y dirección igual a la de HPA en CPTAC **y** GSE68465.
- Nulo: permutación de etiquetas de supervivencia (preserva correlación génica), por estrato.
- Tendencia: Cochran–Armitage sobre deciles de |ρ|, bilateral, α=0,05.

## Regla de decisión (titular según desenlace)
- A «Eje»: tendencia monótona significativa Y quintil inferior indistinguible del nulo (IC95 % de la tasa incluye la esperada por azar) → titular: la reproducibilidad externa de las llamadas de HPA se explica en gran parte por acoplamiento al eje de proliferación.
- B «Señal ortogonal»: quintil inferior con tasa por encima del azar (IC95 % la excluye) → titular: existe señal reproducible desacoplada del eje; se cuantifica el exceso.
- C «Inconcluyente»: curva plana por encima del azar en todos los estratos, o IC demasiado anchos para distinguir A de B → **se detiene el envío a Gene** y el resultado se reporta como limitación/negativo; no se completa el portal.
- Resultado negativo en A no demuestra equivalencia al azar (objeción de Astra): se redacta como «indistinguible», nunca «equivalente».

## Inferencia por gen
- Conjunción parcial: BH sobre max(p_CPTAC, p_GSE) unilateral orientado por HPA; BY como conservador. «Cero detectados» se reporta como titular junto a los marginales, no como ausencia de señal.

## Fuera del cuerpo principal
- Benchmark ML y foundation models → suplemento. Folds 89–188× fuera de abstract y highlights. §3.4 (permutaciones) se refina o se retira.
