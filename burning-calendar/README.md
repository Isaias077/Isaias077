# Calendario de actividad de quema

Aplicación web para el reto de NASA Space Apps de armonización de MODIS y VIIRS.
Une los hotspots de los dos sensores en una sola serie mensual, de 2000 a 2025,
y la muestra como un calendario año × mes para un área de interés.

## Qué permite hacer

- **Elegir el área** dibujando un rectángulo en el mapa o escribiendo sus
  coordenadas, y un rango de fechas.
- **Ver el calendario** de hotspots por mes y por año, por cantidad o por
  anomalía frente a la mediana del mismo mes en los demás años.
- **Comparar un año** con el perfil estacional habitual (mediana y percentiles
  25 a 75).
- **Ver el mapa de densidad** de hotspots del rango elegido. Un clic en un mes
  del calendario lo muestra en el mapa.
- **Entender la armonización**: MODIS frente a VIIRS en los años en que
  conviven, con el ajuste y su error de validación.
- **Descargar** los hotspots del área y el rango (CSV), el calendario
  armonizado (CSV) y la imagen del calendario (SVG).

## Cómo correrla

```bash
pip install -r requirements.txt
python -m server.ingest --country Paraguay     # baja ~500 MB de FIRMS, unos minutos
uvicorn server.app:app --port 8000
```

Después abre http://localhost:8000. El nombre del país va en inglés, como lo
escribe FIRMS (`Argentina`, `Brazil`, `Bolivia`…). Se pueden cargar varios
países; aparecen en el selector. Si la descarga se corta, vuelve a correr el
mismo comando: retoma donde quedó. Los tests corren con `python -m pytest -q`.

## Datos

FIRMS publica sin clave un CSV por país y por año:

| Sensor | Satélites | Período | Resolución |
|---|---|---|---|
| MODIS | Terra y Aqua | noviembre de 2000 a 2025 | 1 km |
| VIIRS | S-NPP | enero de 2012 a 2025 | 375 m |

Solo se cuentan fuegos de vegetación (tipo 0) con confianza nominal o alta.
Quedan afuera volcanes, fuentes fijas como antorchas de gas y detecciones
dudosas.

## Cómo se armoniza

La serie final está en **hotspots equivalentes VIIRS**, para que el sensor de
origen deje de importar:

1. **Terra solo → Terra + Aqua.** Antes de julio de 2002 solo volaba Terra. Sus
   conteos se escalan con el cociente (Terra + Aqua) / Terra de cada mes del
   año, medido cuando volaban los dos.
2. **MODIS → VIIRS.** En los años en que conviven se ajusta, para el área
   elegida, `log(1 + VIIRS) = a + b · log(1 + MODIS)`. No es un factor fijo:
   VIIRS ve fuegos chicos que MODIS pierde, así que el cociente cambia con el
   nivel de actividad.
3. **Serie final.** Desde febrero de 2012 es VIIRS observado. Antes, MODIS
   convertido, con una banda de incertidumbre del 90 % tomada de los residuos.

Si el área tiene menos de 18 meses con fuego en el período común, se usa el
ajuste de todo el país y la aplicación lo avisa.

### Qué tan bien funciona

La validación deja un año afuera, ajusta con los demás y predice el VIIRS de ese
año solo con MODIS. En Paraguay (2,9 millones de hotspots):

| Área | Sin armonizar | Armonizado, por mes | Armonizado, por año |
|---|---|---|---|
| Todo el país | ×3,2 | ×1,07 | ×1,03 |
| Recuadro de 50 km en el Chaco | ×2,6 | ×1,40 | ×1,10 |

Los valores son el error típico como factor: ×1,07 quiere decir que el conteo
predicho difiere del observado en un 7 % en un mes típico.

## Límites

- **Termina en diciembre de 2025.** Los archivos por país se publican por año
  completo. Para el año en curso hace falta la API de FIRMS con una MAP_KEY, que
  todavía no está integrada.
- **Es una armonización estadística de conteos.** Supone que la relación entre
  MODIS y VIIRS de 2012–2025 vale también para 2000–2011. No corrige nubes ni
  cambios de órbita.
- **En áreas chicas el error crece**, como muestra la tabla.
- **VIIRS es solo S-NPP.** NOAA-20 y NOAA-21 no se usan, para mantener un único
  sensor de referencia.
- **El mapa muestra detecciones crudas**, no armonizadas: VIIRS si todo el
  rango cae desde 2012, MODIS si no.
- **La interfaz está solo en español** y el área es un rectángulo, no un
  polígono libre.
- **La tendencia por década es una recta sobre los totales anuales.** Sirve de
  orientación, no como prueba estadística.

El filtro de asimilación de `../modis-viirs-enkf` armoniza la potencia radiativa
a escala de un incendio; esta aplicación no lo usa.

Leaflet 1.9.4 está incluido en `web/vendor/leaflet` (licencia BSD de 2 cláusulas).
