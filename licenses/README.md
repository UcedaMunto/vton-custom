# `licenses/` — inventario y verificación de terceros

Directorio de cumplimiento del fork comercial (equivalente al de IDM-CUSTOM):

| Archivo | Qué es |
|---|---|
| [`manifest.json`](manifest.json) | Inventario machine-readable: artefactos (fuente, licencia, sha256, ruta local) y `forbidden_artifacts` (lo que nunca debe aparecer) |
| [`verify_hashes.py`](verify_hashes.py) | Verificador CLI (solo stdlib): hashes + búsqueda de artefactos prohibidos |
| [`Apache-2.0.txt`](Apache-2.0.txt) | Texto íntegro de Apache-2.0 (la licencia del proyecto y de los pesos de FASHN) |
| [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) | Aviso público de componentes de terceros |

## Uso

```bash
# Verificación normal (hashes de los pesos + scan de prohibidos): 0 = ok, 2 = prohibido presente
python licenses/verify_hashes.py

# Ver otra raíz (p. ej. una imagen de producción con los pesos en /models)
python licenses/verify_hashes.py --root /models

# Capturar los sha256 de los archivos presentes (para actualizar el manifiesto)
python licenses/verify_hashes.py --capture

# Salida JSON (para CI)
python licenses/verify_hashes.py --json
```

Estados: `OK` (hash coincide) · `MISMATCH` (¡bloqueante!) · `MISSING` (peso no
descargado) · `UNVERIFIED` (hash aún no registrado) · `NO_PATH` (librería, no
checkpoint).

## Método

El sha256 se calcula **localmente sobre el archivo real** y se registra en el
manifiesto. Cuando el artefacto viene de HuggingFace, eso permite contrastarlo
con el `oid` LFS que publica la API (`https://huggingface.co/api/models/<repo>/tree/main/<ruta>`);
una coincidencia byte a byte demuestra la procedencia, independientemente de
dónde se descargara.

## Reglas

1. Todo artefacto nuevo (checkpoint, dataset, modelo de un proveedor) entra
   primero al manifiesto **con su hash** y su licencia, y luego al pipeline.
2. Ningún artefacto de `forbidden_artifacts` puede estar presente en el árbol ni
   instalado en el entorno: el gate `scripts/verify_commercial_readiness.py`
   falla si aparece.
3. Al cambiar de versión de una librería (torch, onnxruntime, gradio…), revisar
   su licencia y anotar la versión usada.
