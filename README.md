# CNN Star Constellation Detect Centroid - Proyecto Isabella

Sistema de determinación de actitud mediante CNN para CubeSats, robusto ante ruido y variaciones de FOV.

## Interfaz

python tests/inference_astrometry_gui.py

## Estructura del proyecto

```text
isabella_developments/
├── dataset6k/               # Imágenes originales de entrada (Stellarium)
├── src/
│   ├── data_generation/     # Pipeline de Ground Truth
│   │   └── generate_stellarium_gt.py
│   ├── solver/              # Integración Astrometría
│   │   ├── VideosConstelaciones
│   │   └── stellarium_solver.py 
│   └── training/            # Núcleo de entrenamiento y evaluación
│       ├── conventional_centroiding/ # Método
│       ├── neural_net/      # Arquitecturas (MobileUNet, etc.)
│       ├── saved_models/    # Modelo resultante (MobileUNet_2_40.pt)
│       ├── data_load.py     # DataLoader con Tiling e Invarianza FOV  
│       ├── eval_v2_final.py # Métricas de éxito (Precision/Recall)
│       ├── loss_func.py     # Función de pérdida (Masked MSE)
│       ├── RUN_TRAINING.bat # Entrenamiento automatizado
│       └── training_stepLR.py # Script de entrenamiento principal
├── tests/
│   ├── inference_astrometry_gui.py   # Aplicación
│   └── astrometry_ui/
├── .gitignore               # Configurado para ignorar datos, runs/ y 
├── README.md                # Guía 
├── SETUP.md                 # Guía de instalaciòn
└── requirements.txt         # Dependencias
```

## Ejecución

### Generación de datos
```bash
python src/data_generation/generate_stellarium_gt.py
```

### Entrenamiento
```bash
.\src\training\RUN_TRAINING.bat
```

### Inferencia y Resolución 

Pipeline: detección IA, filtro fotométrico, FITS y Astrometry API.

```bash
python src/solver/stellarium_solver.py
```

### Evaluación
```bash
python src/training/evaluation.py
```

## Resultados

### Evaluación del modelo

Se ejecuto sobre 600 imágenes (636,580 estrellas):

- Precision: 99.08%
- Recall: 85.77%
- F1-Score: 91.95%
- Error posicional: 0.65 píxeles (RMS)

### Solución de Astrometry

Calculada para la imagen `Cassiopeia_apr_h0_fov08_atm001.png`:

- El modelo detectó 669 estrellas.
- Status: SOLVED
- RA: 10.1127°
- DEC: 56.5362°
- ESCALA: 37.74 arcsec/pixel
- Constelaciones: NGC 281, Cassiopeia (Cas), NGC 7789, Caph (β Cas).