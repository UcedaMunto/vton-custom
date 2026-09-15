# Propuestas para comercializar FASHN VTON v1.5 sustituyendo `fashn-human-parser`

**Fecha de revisión:** 15 de septiembre de 2026  
**Objetivo:** utilizar FASHN VTON v1.5 en un servicio comercial de prueba virtual de prendas, eliminando o sustituyendo el componente `fashn-human-parser`, cuya licencia heredada de NVIDIA SegFormer restringe el uso a investigación/evaluación no comercial.

> **Aviso:** este documento es una propuesta técnica y de cumplimiento de licencias, no asesoría legal. Antes de un lanzamiento comercial conviene hacer una revisión final de licencias, avisos y procedencia de pesos/datasets.

---

## 1. Situación actual

FASHN VTON v1.5 está publicado bajo **Apache License 2.0** y el repositorio oficial declara como componentes de terceros:

- DWPose — Apache 2.0.
- YOLOX — Apache 2.0.
- `fashn-human-parser` — licencia heredada de NVIDIA SegFormer.

El problema está en `fashn-human-parser`: su repositorio indica que hereda la licencia de SegFormer, y la licencia oficial de SegFormer limita el software y sus derivados a uso **no comercial**, definido como investigación o evaluación.

Además, aunque FASHN VTON v1.5 se presenta como un modelo *maskless* y el parámetro `segmentation_free=True` está activado por defecto, la implementación oficial actual todavía inicializa `FashnHumanParser` y ejecuta:

```python
person_seg_pred = self.hp_model.predict(person_image_np)
garment_seg_pred = self.hp_model.predict(garment_image_np)
```

Por lo tanto, **no basta con activar `segmentation_free=True` para eliminar la dependencia comercialmente problemática**.

### Fuentes oficiales

- FASHN VTON v1.5:  
  https://github.com/fashn-AI/fashn-vton-1.5

- Pipeline oficial:  
  https://github.com/fashn-AI/fashn-vton-1.5/blob/main/src/fashn_vton/pipeline.py

- FASHN Human Parser:  
  https://github.com/fashn-AI/fashn-human-parser

- Licencia NVIDIA SegFormer:  
  https://github.com/NVlabs/SegFormer/blob/master/LICENSE

---

# 2. Arquitectura objetivo

La arquitectura comercial debería quedar aproximadamente así:

```text
Cliente
   │
   ▼
Frontend / E-commerce
   │
   ▼
API propia de Virtual Try-On
   │
   ├── Validación de imagen
   ├── Preprocesamiento
   ├── DWPose
   │
   ├── Segmentación comercialmente compatible
   │      ├── Sin parser
   │      ├── SAM 2
   │      ├── Grounding DINO + SAM 2
   │      └── Parser propio
   │
   ▼
FASHN VTON v1.5
   │
   ▼
Imagen generada
```

El objetivo es conservar:

```text
FASHN VTON v1.5
model.safetensors
DWPose
YOLOX
```

y eliminar:

```text
fashn-human-parser
SegFormer restringido
```

---

# 3. Propuesta 1 — Fork completamente `parser-free`

## Recomendación

**Es la primera opción que probaría.**

FASHN describe v1.5 como un modelo *maskless* y recomienda `segmentation_free=True`. En el código actual, sin embargo, el parser todavía se ejecuta antes de que el flujo decida no aplicar la máscara.

La propuesta consiste en crear un fork en el cual:

```text
segmentation_free=True
+
garment_photo_type="flat-lay"
```

evite completamente cargar y ejecutar `FashnHumanParser`.

## Flujo propuesto

```text
Foto de persona
      │
      ├── DWPose
      │
      └── Imagen original
              │
              ▼
        FASHN VTON
              ▲
              │
      Imagen de prenda
        tipo flat-lay
```

Para fotografías `flat-lay`, la prenda ya debería estar aislada o fotografiada sobre un fondo controlado.

## Cambios principales

Eliminar del `pyproject.toml`:

```python
"fashn-human-parser>=0.1.1",
```

Eliminar del pipeline:

```python
from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE, FashnHumanParser
```

y:

```python
self._setup_hp_model()
```

Crear un flujo similar a:

```python
if segmentation_free:
    ca_image = person_image_np.copy()
else:
    ca_image = commercial_parser.create_agnostic_image(...)

if garment_photo_type == "flat-lay":
    garment_image_processed = garment_image_np.copy()
else:
    garment_image_processed = commercial_parser.extract_garment(...)
```

Para el primer MVP comercial se puede restringir la entrada de prendas a:

```text
✔ Prendas flat-lay
✔ Fondo blanco
✔ Fondo neutro
✔ Fondo transparente
✔ Producto fotografiado sin modelo
```

## Ventajas

- Elimina por completo la dependencia problemática.
- Menor consumo de VRAM y GPU.
- Menor latencia.
- Menos modelos cargados en memoria.
- Menor complejidad operativa.
- Aprovecha el funcionamiento *segmentation-free* previsto por FASHN.
- Muy apropiado para un catálogo de e-commerce controlado.

## Desventajas

- Debe validarse que la calidad no disminuya en los casos reales.
- Las fotografías de prendas usadas por modelos pueden requerir segmentación.
- Puede requerir estandarizar las fotografías del catálogo.

## Esfuerzo estimado

**Bajo: aproximadamente 1–4 días de desarrollo y pruebas iniciales.**

## Nivel de recomendación

**★★★★★ — Primera prueba recomendada.**

---

# 4. Propuesta 2 — Sustituir Human Parser por SAM 2

## Descripción

Meta publica **SAM 2** bajo Apache License 2.0. El repositorio oficial indica expresamente que los checkpoints, el código de entrenamiento y el código de demostración se distribuyen bajo Apache 2.0.

Fuente:

https://github.com/facebookresearch/sam2

Licencia:

https://github.com/facebookresearch/sam2/blob/main/LICENSE

La idea sería utilizar SAM 2 para generar las máscaras necesarias sin utilizar SegFormer.

## Arquitectura

```text
Persona
   │
   ├── DWPose
   │      │
   │      └── keypoints / bounding boxes
   │
   ▼
SAM 2
   │
   ├── máscara torso
   ├── máscara piernas
   └── máscara cuerpo/prenda
          │
          ▼
Adaptador FASHN
          │
          ▼
FASHN VTON
```

## Punto importante

SAM 2 no es directamente un *human parser* con las mismas 18 etiquetas de FASHN.

Por ello se implementaría un adaptador que genere solamente las regiones que realmente necesita el VTON.

Ejemplo:

```text
tops
  → región torso/brazos

bottoms
  → región cadera/piernas

one-pieces
  → región torso + cadera + piernas
```

DWPose ya conoce posiciones aproximadas de:

- hombros;
- codos;
- muñecas;
- cadera;
- rodillas;
- tobillos.

Estos puntos pueden convertirse en prompts para SAM 2.

## Ventajas

- Apache 2.0.
- Código y checkpoints explícitamente licenciados.
- Segmentación muy flexible.
- No requiere recrear exactamente un parser semántico de 18 clases.
- Buena base para futuras funciones de edición de ropa.

## Desventajas

- Hay que desarrollar el adaptador.
- Más cómputo que la solución `parser-free`.
- Puede requerir heurísticas con DWPose.
- Es necesario hacer pruebas de precisión.

## Esfuerzo estimado

**Medio: aproximadamente 1–2 semanas para un prototipo sólido.**

## Nivel de recomendación

**★★★★☆ — Excelente segunda opción.**

---

# 5. Propuesta 3 — Grounding DINO + SAM 2

Esta opción es especialmente útil cuando la prenda de entrada está siendo utilizada por otra persona.

Ejemplo:

```text
Foto de modelo usando camisa
             │
             ▼
       Grounding DINO
             │
        "shirt"
             │
             ▼
          bbox
             │
             ▼
           SAM 2
             │
             ▼
     máscara de camisa
             │
             ▼
        FASHN VTON
```

## Componentes

### Grounding DINO

Repositorio oficial:

https://github.com/IDEA-Research/GroundingDINO

Licencia del repositorio:

**Apache License 2.0**

### SAM 2

Repositorio oficial:

https://github.com/facebookresearch/sam2

Licencia de código y checkpoints:

**Apache License 2.0**

## Ejemplos de prompts

```text
"shirt"
"t-shirt"
"jacket"
"dress"
"pants"
"skirt"
"shorts"
```

Grounding DINO determina la ubicación de la prenda y SAM 2 genera la máscara precisa.

## Ventajas

- Funciona mejor con fotografías de prendas sobre personas.
- No necesita construir un parser humano completo.
- Muy flexible.
- Permite incorporar nuevas categorías de prendas.
- Componentes principales con licencia Apache 2.0.

## Desventajas

- Dos modelos adicionales.
- Mayor VRAM.
- Mayor latencia.
- Mayor complejidad de despliegue.
- Hay que auditar los pesos/checkpoints exactos que se utilicen, no solamente la licencia del código.

## Esfuerzo estimado

**Medio: aproximadamente 1–3 semanas.**

## Nivel de recomendación

**★★★★☆ — Recomendado si se aceptarán prendas usadas por modelos.**

---

# 6. Propuesta 4 — Parser humano propio con Detectron2

Esta es la opción de mayor independencia tecnológica.

Meta publica Detectron2 bajo Apache License 2.0:

https://github.com/facebookresearch/detectron2

La empresa podría construir su propio:

```text
CommercialHumanParser
```

utilizando una arquitectura soportada por Detectron2 y entrenarla con datos propios, sintéticos o con datasets cuya licencia permita explícitamente el uso comercial.

## Etiquetas sugeridas

No necesariamente hay que copiar todas las clases originales de FASHN.

Se puede crear algo más pequeño:

```text
0 background
1 head
2 hair
3 torso
4 left_arm
5 right_arm
6 upper_clothing
7 lower_clothing
8 dress
9 left_leg
10 right_leg
11 shoes
```

Después se crea una tabla de conversión:

```python
COMMERCIAL_TO_FASHN = {
    "upper_clothing": ...,
    "lower_clothing": ...,
    "dress": ...,
}
```

## Dataset

Para minimizar riesgos legales:

```text
Datos propios
+
Fotos licenciadas
+
Personas sintéticas
+
Prendas del propio catálogo
+
Anotaciones propias
```

Evitaría entrenar el modelo comercial directamente con un dataset cuya licencia limite el uso a investigación.

## Ventajas

- Control completo.
- IP propia.
- Sin dependencia directa de un proveedor de parser.
- Se puede optimizar específicamente para el catálogo de la empresa.
- Puede convertirse en tecnología propia reutilizable.

## Desventajas

- Necesita dataset.
- Requiere entrenamiento.
- Requiere anotaciones.
- Mayor inversión inicial.
- Hay que auditar cualquier backbone, checkpoint inicial y dataset utilizados.

## Esfuerzo estimado

**Alto: aproximadamente 4–10 semanas para una primera versión productiva**, dependiendo del dataset y nivel de precisión esperado.

## Nivel de recomendación

**★★★★★ a largo plazo.**

---

# 7. Propuesta 5 — Arquitectura híbrida

La solución que considero más equilibrada para producción es utilizar diferentes estrategias según la entrada.

```text
                     ┌── Flat-lay ──────────────► Parser-free
                     │
Imagen de prenda ────┤
                     │
                     └── Prenda sobre modelo ──► Grounding DINO + SAM 2
                                                   │
                                                   ▼
                                              máscara prenda
                                                   │
                                                   ▼
Persona ─────────► DWPose ─────────────────────► FASHN VTON
```

Además:

```text
segmentation_free=True
```

para la persona siempre que las pruebas demuestren que mantiene la calidad requerida.

## Resultado

En la mayoría de solicitudes:

```text
Persona
+
Flat-lay
+
DWPose
+
FASHN VTON
```

Sin parser.

Solamente los casos difíciles utilizarían:

```text
Grounding DINO
+
SAM 2
```

Esto reduce:

- latencia;
- uso de GPU;
- VRAM;
- costo eléctrico;
- complejidad.

---

# 8. Comparativa

| Alternativa | Uso comercial | Complejidad | GPU adicional | Calidad esperada | Recomendación |
|---|---:|---:|---:|---:|---:|
| Parser-free + flat-lay | Sí | Muy baja | Ninguna | Alta si catálogo controlado | ★★★★★ |
| SAM 2 | Sí | Media | Media | Alta | ★★★★☆ |
| Grounding DINO + SAM 2 | Sí | Media/Alta | Alta | Muy alta para prenda sobre modelo | ★★★★☆ |
| Parser propio con Detectron2 | Sí* | Alta | Media | Potencialmente muy alta | ★★★★★ largo plazo |
| SegFormer actual | No, salvo licencia comercial específica | Baja | Media | Alta | No usar por defecto |

\* Siempre que se utilicen código, pesos de inicialización y datos de entrenamiento con licencias compatibles con el uso comercial.

---

# 9. Recomendación para nuestro proyecto

Propongo desarrollar en tres etapas.

## Etapa 1 — MVP comercial

Modificar FASHN VTON:

```text
FASHN VTON 1.5
   +
DWPose
   +
segmentation_free
   +
prendas flat-lay
   -
fashn-human-parser
```

Objetivo:

**comprobar si podemos obtener calidad suficiente sin ningún Human Parser.**

Esta sería la alternativa más barata y simple.

---

## Etapa 2 — Soporte para casos difíciles

Agregar:

```text
SAM 2
```

para aislar prendas o regiones cuando sea necesario.

Si necesitamos recibir fotografías de prendas usadas por modelos:

```text
Grounding DINO
      ↓
SAM 2
      ↓
FASHN VTON
```

---

## Etapa 3 — Tecnología propia

Crear:

```text
company-human-parser
```

entrenado con:

```text
dataset propio
+
dataset sintético
+
prendas reales del catálogo
```

Esto permite controlar:

- calidad;
- categorías;
- licencia;
- rendimiento;
- optimización;
- propiedad intelectual.

---

# 10. Fork comercial sugerido

Una estructura posible:

```text
commercial-vton/
│
├── src/
│   ├── vton/
│   │   ├── pipeline.py
│   │   ├── tryon_mmdit.py
│   │   ├── dwpose.py
│   │   └── preprocessing/
│   │
│   ├── segmentation/
│   │   ├── base.py
│   │   ├── none.py
│   │   ├── sam2.py
│   │   ├── grounded_sam2.py
│   │   └── custom_parser.py
│   │
│   └── api/
│       └── main.py
│
├── weights/
│   ├── fashn/
│   ├── dwpose/
│   ├── sam2/
│   └── grounding_dino/
│
├── licenses/
│   ├── FASHN_APACHE_2.txt
│   ├── SAM2_APACHE_2.txt
│   ├── GROUNDING_DINO_APACHE_2.txt
│   └── THIRD_PARTY_NOTICES.md
│
└── THIRD_PARTY_NOTICES.md
```

---

# 11. Interfaz común de segmentación

Conviene evitar que FASHN dependa directamente de un modelo concreto.

Ejemplo:

```python
from abc import ABC, abstractmethod

class SegmentationProvider(ABC):

    @abstractmethod
    def extract_person_region(self, image, category):
        pass

    @abstractmethod
    def extract_garment(self, image, category):
        pass
```

Implementaciones:

```python
NoSegmentationProvider
SAM2SegmentationProvider
GroundedSAM2Provider
CustomHumanParserProvider
```

Entonces el pipeline puede recibir:

```python
pipeline = TryOnPipeline(
    weights_dir="./weights",
    segmentation_provider=NoSegmentationProvider()
)
```

o:

```python
pipeline = TryOnPipeline(
    weights_dir="./weights",
    segmentation_provider=GroundedSAM2Provider()
)
```

Esto evita quedar atados nuevamente a una sola implementación.

---

# 12. Modificación conceptual del pipeline

Actualmente:

```python
self._setup_tryon_model()
self._setup_pose_model()
self._setup_hp_model()
```

Propuesta:

```python
self._setup_tryon_model()
self._setup_pose_model()

if segmentation_provider is not None:
    self.segmentation_provider = segmentation_provider
```

Durante inferencia:

```python
if segmentation_free:
    ca_image = person_image_np.copy()
else:
    ca_image = self.segmentation_provider.extract_person_region(
        person_image_np,
        category
    )
```

Para la prenda:

```python
if garment_photo_type == "flat-lay":
    garment_image_processed = garment_image_np
else:
    garment_image_processed = self.segmentation_provider.extract_garment(
        garment_image_np,
        category
    )
```

Con esto:

```text
fashn-human-parser
```

desaparece completamente de la instalación.

---

# 13. Dependencias que habría que eliminar

En `pyproject.toml`:

```diff
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "safetensors>=0.3.0",
    "huggingface_hub>=0.20.0",
    "pillow>=9.0.0",
    "numpy>=1.21.0",
    "opencv-python>=4.5.0",
    "tqdm>=4.65.0",
    "einops>=0.6.0",
    "onnxruntime-gpu>=1.14.0",
    "matplotlib>=3.5.0",
-   "fashn-human-parser>=0.1.1",
]
```

Además se debe verificar que no quede:

```text
fashn-human-parser
SegFormer
NVIDIA SegFormer weights
```

ni en:

```text
Docker image
pip cache
Hugging Face cache
build artifacts
production image
```

---

# 14. Pruebas mínimas antes de producción

Crear un benchmark de al menos:

```text
100 tops
100 bottoms
100 dresses / one-pieces
```

con variedad de:

- personas;
- tonos de piel;
- poses;
- fondos;
- tallas;
- prendas holgadas;
- prendas ajustadas;
- brazos cruzados;
- cabello largo;
- oclusiones.

Comparar:

```text
A = pipeline original de evaluación

B = parser-free

C = SAM 2

D = Grounding DINO + SAM 2
```

Medir:

- calidad visual;
- conservación de rostro;
- conservación de manos;
- fidelidad del estampado;
- fidelidad del color;
- deformación de logos;
- tiempo de inferencia;
- VRAM;
- RAM;
- tasa de fallos.

---

# 15. Checklist de licencias para producción

Antes del lanzamiento:

- [ ] Confirmar licencia de FASHN VTON v1.5.
- [ ] Guardar copia de Apache 2.0.
- [ ] Mantener los avisos de copyright correspondientes.
- [ ] Crear `THIRD_PARTY_NOTICES.md`.
- [ ] Eliminar `fashn-human-parser`.
- [ ] Eliminar pesos derivados de SegFormer.
- [ ] Verificar DWPose.
- [ ] Verificar YOLOX.
- [ ] Verificar SAM 2 si se utiliza.
- [ ] Verificar Grounding DINO si se utiliza.
- [ ] Verificar cada checkpoint concreto descargado.
- [ ] Verificar la licencia de los datasets si se hace fine-tuning.
- [ ] Mantener registro de la URL, versión, commit y licencia de cada dependencia.
- [ ] Hacer revisión legal final antes del lanzamiento público.

---

# 16. Matriz de decisión final

## Si las prendas del catálogo son `flat-lay`

Usar:

```text
FASHN VTON
+
DWPose
+
Parser-free
```

**Esta es mi recomendación principal.**

---

## Si las prendas pueden aparecer sobre modelos

Usar:

```text
FASHN VTON
+
DWPose
+
Grounding DINO
+
SAM 2
```

---

## Si queremos independencia tecnológica a largo plazo

Desarrollar:

```text
FASHN VTON
+
DWPose
+
Parser propio
```

entrenado con datasets cuya procedencia y licencia sean controladas por la empresa.

---

# 17. Ruta recomendada

```text
                     HOY
                      │
                      ▼
        Fork de FASHN VTON v1.5
                      │
                      ▼
         Eliminar fashn-human-parser
                      │
                      ▼
        segmentation_free=True
                      │
            ┌─────────┴──────────┐
            │                    │
            ▼                    ▼
        Flat-lay            Model-worn
            │                    │
            ▼                    ▼
      Sin segmentar      Grounding DINO
                                 │
                                 ▼
                               SAM 2
                                 │
            └─────────┬──────────┘
                      ▼
                FASHN VTON
                      │
                      ▼
              Servicio comercial
```

---

# 18. Conclusión

La estrategia recomendada es **no reemplazar inmediatamente `fashn-human-parser` por otro Human Parser complejo**, sino aprovechar primero la naturaleza *segmentation-free* de FASHN VTON v1.5 y eliminar completamente esa dependencia para el flujo de prendas `flat-lay`.

La prioridad sería:

1. **Fork parser-free de FASHN VTON v1.5.**
2. **SAM 2 para casos que requieran máscaras.**
3. **Grounding DINO + SAM 2 para prendas usadas por modelos.**
4. **Parser propio como solución estratégica de largo plazo.**

Esto permitiría conservar la parte más valiosa de FASHN VTON v1.5 y sus pesos, pero construir un pipeline comercial en el que las dependencias críticas tengan licencias permisivas y auditables.

---

## Referencias

1. FASHN VTON v1.5  
   https://github.com/fashn-AI/fashn-vton-1.5

2. FASHN VTON v1.5 — Pipeline  
   https://github.com/fashn-AI/fashn-vton-1.5/blob/main/src/fashn_vton/pipeline.py

3. FASHN Human Parser  
   https://github.com/fashn-AI/fashn-human-parser

4. NVIDIA SegFormer — License  
   https://github.com/NVlabs/SegFormer/blob/master/LICENSE

5. SAM 2  
   https://github.com/facebookresearch/sam2

6. SAM 2 — License  
   https://github.com/facebookresearch/sam2/blob/main/LICENSE

7. Grounding DINO  
   https://github.com/IDEA-Research/GroundingDINO

8. Detectron2  
   https://github.com/facebookresearch/detectron2
