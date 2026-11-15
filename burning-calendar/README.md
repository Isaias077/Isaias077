# Calendario de actividad de quema

Aplicación web para el reto de NASA Space Apps de armonización de MODIS y VIIRS.
Une los hotspots de los dos sensores en una sola serie mensual, desde 2000 hasta
la semana en curso, y la muestra como un calendario año × mes para **cualquier
país, provincia o rectángulo** del mundo. No hay que descargar nada antes: los
datos se bajan de NASA FIRMS la primera vez que se consulta un área.

## Qué permite hacer

- **Elegir el área** de tres formas: escribiendo un país (252 países en
  español), eligiendo una de sus provincias o regiones (4596 en total), o
  haciendo clic en el mapa para tomar la provincia de ese punto. También se
  puede dibujar un rectángulo, que puede cruzar fronteras.
- **Ver el calendario** de hotspots por mes y por año, por cantidad o por
  anomalía frente a la mediana del mismo mes en los demás años.
- **Comparar el año en curso** con el perfil estacional habitual. El mes en
  curso se compara con los mismos días de los demás años.
- **Ver el mapa de densidad** de un rango de fechas. Un clic en un mes del
  calendario lo muestra en el mapa.
- **Entender la armonización**: MODIS frente a VIIRS en los años en que
  conviven, con el ajuste y su error de validación.
- **Descargar** los hotspots del área y el rango (CSV), el calendario
  armonizado (CSV) y la imagen del calendario (SVG).
- **Compartir el enlace**: el área elegida queda en la dirección
  (`#pais=PRY`, `#provincia=...`, `#bbox=...`).

## Cómo correrla

```bash
pip install -r requirements.txt
uvicorn server.app:app --port 8000
```

Después abre http://localhost:8000 y elige un área. Los tests corren con
`python -m pytest -q` (usan un FIRMS simulado, sin red).

Para dejar países listos de antemano, por ejemplo antes de una demo:

```bash
python -m server.data PRY ARG BRA
```

## Cómo llegan los datos

La primera vez que un área necesita un país, el servidor lo baja en segundo
plano y la página muestra el progreso:

1. **Busca el país en FIRMS.** Los archivos se llaman por el nombre del país en
   inglés, que no siempre coincide con Natural Earth (`Russian_Federation`,
   `Czech_Republic`). Prueba el nombre conocido y lo recuerda.
2. **Baja los archivos anuales sin clave**: MODIS desde noviembre de 2000 y
   VIIRS S-NPP desde 2012, del año más nuevo al más viejo y alternando
   sensores. Así la armonización, que necesita los dos, puede empezar con los
   primeros archivos, y el calendario se completa mientras llegan los demás.
   Los meses que todavía faltan aparecen rayados como pendientes, nunca como
   cero.
3. **Baja el año en curso por la API de área**, con la MAP_KEY del proyecto.
   Usa el archivo estándar (SP) mientras existe y el de tiempo casi real (NRT)
   después. La API devuelve un rectángulo, así que se recorta con el contorno
   del país ensanchado 2 km. Las fuentes NRT no traen la columna `type`: un
   hotspot NRT en el mismo lugar (celda de 1 km) donde el archivo registró
   antorchas o volcanes se trata como fuente fija.
4. **Resume cada archivo** en conteos por celda de 0,05° (unos 5 km), día y
   satélite. Una provincia o un rectángulo se calcula sumando sus celdas, sin
   releer millones de puntos: cualquier consulta tarda menos de un segundo una
   vez que los datos están.

Todo queda en `data/paises/<ISO3>/`: las consultas siguientes son
instantáneas. El año en curso se vuelve a pedir si tiene más de 6 horas, y solo
se rehace la última semana.

Tiempos y espacio medidos en esta sesión, con una conexión que FIRMS cortaba
seguido:

| País | Archivos anuales | Espacio en disco |
|---|---|---|
| Uruguay | unos minutos | 1,4 MB |
| Paraguay | unos minutos | 36 MB |
| Argentina | unos minutos | 54 MB |

El año en curso por la API son unos 130 pedidos por país; con esa conexión
tardaba entre 10 y 25 minutos. Un país grande como Brasil baja varios GB de
CSV la primera vez.

La MAP_KEY de FIRMS está escrita en `server/firms.py` (`DEFAULT_MAP_KEY`). Para
usar otra, definí la variable de entorno `FIRMS_MAP_KEY`, que tiene prioridad.
Cada clave admite 5000 pedidos cada 10 minutos, compartidos entre todos los
que la usen.

## Cómo se armoniza

La serie final está en **hotspots equivalentes VIIRS**, para que el sensor de
origen deje de importar:

1. **Terra solo → Terra + Aqua.** Antes de julio de 2002 solo volaba Terra. Sus
   conteos se escalan con el cociente (Terra + Aqua) / Terra de cada mes del
   año, medido cuando volaban los dos.
2. **MODIS → VIIRS.** En los meses con los dos sensores se ajusta, para el
   área elegida, `log(1 + VIIRS) = a + b · log(1 + MODIS)`. No es un factor
   fijo: VIIRS ve fuegos chicos que MODIS pierde, así que el cociente cambia
   con el nivel de actividad.
3. **Serie final.** Donde hay VIIRS se usa VIIRS observado; donde no, MODIS
   convertido, con una banda de incertidumbre del 90 %. Un mes que VIIRS no
   cubre entero (S-NPP NRT hoy llega solo al 2 de noviembre de 2026) también se
   arma con MODIS convertido.

Si el área tiene menos de 18 meses con fuego o menos de 1000 hotspots VIIRS en
el período común, el ajuste propio sería ruido: se usa el de los países que la
contienen, y la página lo avisa.

### Qué tan bien funciona

La validación deja un año afuera, ajusta con los demás y predice el VIIRS de ese
año solo con MODIS. El valor es el error típico como factor: ×1,07 quiere decir
que el conteo predicho difiere del observado en un 7 % en un mes típico.

| Área | Sin armonizar | Armonizado, por mes | Armonizado, por año |
|---|---|---|---|
| Paraguay | ×3,3 | ×1,07 | ×1,03 |
| Uruguay | ×3,4 | ×1,27 | ×1,20 |
| Rectángulo de 1° en el Chaco paraguayo | ×2,9 | ×1,30 | ×1,13 |

### El mes en curso

Un mes incompleto no se compara con meses enteros: los hotspots del 1 al 15 de
noviembre se comparan con la mediana del 1 al 15 de noviembre de los demás
años, todos en la misma escala armonizada. En el calendario el mes en curso
aparece rayado y no entra en la climatología. En Paraguay, al 15 de noviembre
de 2026: 3.572 hotspots contra una mediana de 4.626 en los mismos días (×0,77,
percentil 35).

## Límites

- **La primera consulta de un país tarda.** Unos minutos para un país mediano
  y bastante más para uno grande. Mientras tanto el calendario muestra lo que
  ya llegó.
- **Un rectángulo espera a todos los países que toca.** Los meses quedan
  pendientes hasta que llegan los datos de todos; si no, el total del área
  estaría incompleto sin que se note. Se admiten hasta 6 países y 400 grados
  cuadrados.
- **Las áreas se recortan por celdas de 5 km.** Una celda cuenta para la
  provincia donde cae su centro. Las provincias de Uruguay suman el total del
  país con menos de 3 % de diferencia; en provincias muy chicas el error es
  mayor.
- **Algunos países pueden no encontrarse en FIRMS** si su nombre de archivo no
  coincide con ninguna variante conocida. Para esos países solo hay año en
  curso, y la página lo avisa. Se corrige agregando el nombre en
  `FIRMS_ALIASES` (`server/regions.py`).
- **Es una armonización estadística de conteos.** Supone que la relación entre
  MODIS y VIIRS de 2012 en adelante vale también para 2000–2011. No corrige
  nubes ni cambios de órbita.
- **VIIRS es solo S-NPP.** NOAA-20 y NOAA-21 no se usan, para mantener un único
  sensor de referencia.
- **Los datos NRT se reprocesan.** FIRMS los reemplaza por la versión SP unos
  meses después; los últimos meses pueden cambiar un poco.
- **El mapa muestra detecciones crudas** por celda, no armonizadas: VIIRS si
  todo el rango cae en su cobertura, MODIS si no.
- **La interfaz está solo en español.**

## Archivos

| Archivo | Qué hace |
|---|---|
| `server/app.py` | API y servidor web |
| `server/data.py` | Descargas a demanda, resúmenes por celda y consultas por área |
| `server/firms.py` | Acceso a FIRMS: archivos anuales, API de área y lectura de CSV |
| `server/regions.py` | Países, provincias, rectángulos y grilla de celdas |
| `server/harmonize.py` | Armonización MODIS → VIIRS, validación y anomalías |
| `server/geo/regions.json.gz` | Límites simplificados; se regenera con `tools/build_regions.py` |
| `web/` | Interfaz: mapa, calendario y gráficos |

Límites de países y provincias: Natural Earth 1:10m (dominio público), tomados
del paquete `geojson-places` 1.0.8 (licencia ISC). Leaflet 1.9.4 está incluido
en `web/vendor/leaflet` (licencia BSD de 2 cláusulas).
