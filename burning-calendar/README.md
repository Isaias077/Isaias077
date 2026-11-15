# Calendario de actividad de quema

Aplicación web para el reto de NASA Space Apps de armonización de MODIS y VIIRS.
Une los hotspots de los dos sensores en una sola serie mensual, desde 2000 hasta
la semana en curso, y la muestra como un calendario año × mes para un área de
interés.

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
python -m server.ingest --country Paraguay            # 2000 a 2025, sin clave (~500 MB)
export FIRMS_MAP_KEY=tu_clave                         # gratuita en firms.modaps.eosdis.nasa.gov/api/map_key
python -m server.ingest --country Paraguay --recent   # año en curso, por la API
uvicorn server.app:app --port 8000
```

Para mantener el año en curso al día, repetí el paso `--recent` una vez por día
(por ejemplo con cron). Solo vuelve a pedir la última semana; el resto queda en
`data/raw/api/`. La clave se lee de la variable de entorno y nunca se guarda en
archivos ni aparece en los mensajes de error.

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

El año en curso llega por la API de área de FIRMS. Usa el archivo estándar (SP)
mientras existe y el de tiempo casi real (NRT) después; FIRMS informa hasta qué
día llega cada uno. Dos diferencias con el archivo por país:

- **La API devuelve un rectángulo, no el país.** Se recorta con el contorno de
  Natural Earth 1:50m ensanchado 2 km, que conserva el 99,96 % del archivo de
  Paraguay. Sin el recorte, el 39 % de los hotspots de una muestra eran de
  Argentina, Bolivia o Brasil.
- **Las fuentes NRT no traen la columna `type`.** Un hotspot NRT en el mismo
  lugar (celda de 1 km) donde el archivo registró antorchas o volcanes se trata
  como fuente fija; el resto, como fuego de vegetación.

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
   Si VIIRS se corta antes que MODIS (como S-NPP NRT, que hoy llega solo al 2
   de noviembre de 2026), el mes que VIIRS no cubre entero también se arma con
   MODIS convertido.

### El mes en curso

Un mes incompleto no se compara con meses enteros: los hotspots del 1 al 15 de
noviembre se comparan con la mediana del 1 al 15 de noviembre de los demás
años, todos en la misma escala armonizada. En el calendario el mes en curso
aparece rayado y no entra en la climatología.

Con los datos de Paraguay bajados el 15 de noviembre de 2026: del 1 al 15 de
noviembre hubo 3.542 hotspots armonizados, contra una mediana de 4.626 en los
mismos días de 2000 a 2025 (×0,77, percentil 35). En el perfil estacional ese
punto aparece hueco y se ve bajo porque se dibuja contra la mediana del mes
entero; el número comparable es el de la tarjeta y el del calendario.

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

- **El año en curso depende de la API y de su clave.** Sin `--recent`, la serie
  termina en diciembre del año pasado.
- **Los datos NRT se reprocesan.** FIRMS reemplaza los NRT por los SP unos meses
  después; volver a correr `--recent` baja la versión SP cuando aparece. Los
  últimos meses pueden cambiar un poco.
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
