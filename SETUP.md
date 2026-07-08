# CNN Star Tracker — Guía de instalación local

## Requisitos previos

- Visual Studio Code
- Conexión a internet

---

## Instalación

Abre una **Terminal** o **PowerShell** y ejecuta estos comandos uno a uno:

### 1. Crear entorno virtual

```bash
conda create -n startracker python=3.10 -y
conda activate startracker
```

### 2. Instalar PyTorch (CPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Instalar el resto de dependencias

```bash
pip install numpy opencv-python matplotlib scipy tensorboard pandas thop tqdm requests PyQt6 PyQt6-WebEngine astropy
```

---

## Ejecución

Desde la carpeta raíz del proyecto "CNNStarConstellationDetectCentroid", con el entorno activado:

```bash
conda activate startracker
python tests/inference_astrometry_gui.py
```

---

## Uso de la interfaz

### Inferencia sobre vídeo
1. En **Modelo → Ruta**, el modelo `MobileUNet_2_40.pt` se carga automáticamente. Pulsar **Cargar**.
2. En **Vídeo**, pulsar **Examinar** y seleccionar un `.MP4` de cielo estrellado (Ej. Empezar con Osa Mayor/Menor).
3. Pulsar **Iniciar** — el panel derecho muestra las estrellas detectadas en verde en tiempo real.
Nota_1: Puedes ir viendo los logs al final de la página.

### Resolver con Astrometry
> Requiere una API key gratuita de [nova.astrometry.net](https://nova.astrometry.net)

1. (OPCIONAL) Introducir tu API key en el campo **API Key**.
2. Tras procesar el vídeo (log: Video finalizado), pulsar **Astrometry frame** para resolver el frame actual.
3. Los resultados aparecen en el panel inferior: RA, Dec, escala, constelaciones detectadas.
Nota_2: Navega por todas las pestañas y botones.

### Apuntar telescopio
1. Una vez resuelta la imagen, ir a la pestaña **Objetos**.
2. Hacer clic en una constelación de la lista (Ej. Osa mayor/menor).
3. El panel **Apuntar Telescopio** muestra las coordenadas RA/Dec calculadas desde las anotaciones reales de la imagen.
4. Usar los botones **Copiar RA**, **Copiar Dec** o **Copiar RA + Dec** para enviarlas al software del telescopio.
