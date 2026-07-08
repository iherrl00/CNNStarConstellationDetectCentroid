# Proyecto Isabella: CNN Star Tracker - Progreso

## Fase 1 v1

- Pipeline de datos, entrenamiento y evaluación completado inicialmente
- Precisión del 91% lograda
- RMSE de 0.22 subpíxeles, superando la baseline de 0.35
- Infraestructura de datos resiliente con create_offline_crops.py
- PyTorch optimizado con autocast, GradScaler y BCEWithLogitsLoss

## El problema que encontre

- OTSU Thresholding no distinguía estrellas dentro de nebulosas
- Estrellas en fondos brillantes quedaban marcadas como fondo
- Nuestro algoritmo base etiquetó estrellas reales como fondo
- Red fue penalizada por detectar correctamente
- Recall colapsó al 35%
- Modelo aprendió a ignorar el 65% de las estrellas

## Acción tomada

- Hard Reset ejecutado
- 60GB de datos corruptos eliminados
- Modelos fallidos borrados
- Scripts de entrenamiento mejorados
- Optimizaciones preservadas

## Fase 1 v2 

- Enfoque corregido: calidad del algoritmo base y separación de responsabilidades
- generate_stellarium_gt.py reescrito con filtro Top-Hat 11x11
- Umbral binario estricto nivel 8
- Toda estrella se aisla de la nebulosa sin fusionarse
- Red neuronal solo hace extracción morfológica
- Filtrado por brillo delegado a Fase 2 (Astrometry)
- Dataset generado: 6,000 imágenes, 24,000 archivos .npy en training_data
- Modelo guardado en src/training/saved_models/MobileUNet_2_40.pt

## Evaluación Fase 1 v2

- eval_v2_final.py creado con extracción de contornos
- Trilateración de distancias descartada
- 600 imágenes de validación analizadas
- 636,580 estrellas reales procesadas

## Resultados finales  

- Precisión 99.08%
- Sensibilidad 85.77%
- F1-Score 91.95%
- Error posicional 0.6498 píxeles

## Conclusión Fase 1

Modelo neuronal estabilizado, agnóstico al FOV gracias a recortes 480x480, maneja desbalance de clases con BCE pos_weight, y superó defecto de calidad de datos. Extracción geométrica directa validada para continuar.

## Fase 2: Integración Astrometry 

- Consolida el proceso desde la imagen hasta la resolución de actitud.
- Inferencia por mosaicos de 480x480 con solapamiento de 64 px.
- Filtro fotométrico Top 40 para reducir candidatos y acelerar el Plate Solver.
- Generación de Clean Sky en formato FITS a partir de detecciones IA, con PSF simulado para compatibilidad astrométrica.
- Automatización de Astrometry.net para extraer RA, DEC, escala de píxel e identificación de constelaciones.

## Resultados Fase 2

- Validación final sobre Casiopea (FOV 08): 669 estrellas detectadas y el Plate Solver resolvió con éxito usando la lista Top 40 destilada en 20.6 segundos.
- Coincidencia del 100% si montamos la imagen completa en formato .png a la API de Astrometry.net.

## Problema de parseo en `stellarium_solver.py`

- La imagen de Osa Mayor y otras tomadas por una camara de celular, etc. no resolvía de forma consistente con los límites de escala iniciales.
- No me di cuenta de esto por que solo habia probado con la imagen de Casiopea, que es un campo estelar denso y con muchas estrellas brillantes.
- Se amplió el rango de búsqueda de Astrometry y se dejó un fallback sin escala predefinida.
- Con ese ajuste, el solver volvió a resolver el frame y a devolver RA, DEC y escala de píxel correctos.

## Ajuste de interfaz y visualización

- Se dejó la GUI modular en PyQt6 para inferencia, video y envío a Astrometry.
- Se integró WebView embebida para resultados anotados con zoom y pan.
- Se reemplazó el JSON crudo por un panel visual de calibración y objetos.
- La API key y los parámetros operativos quedaron integrados en la interfaz.

## Fase 3 Telescopio