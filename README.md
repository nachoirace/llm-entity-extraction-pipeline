# LLM Entity Extraction Pipeline

Pipeline LCEL que recibe un texto técnico sin procesar (un log de error, una
descripción de arquitectura, un reporte de incidente) y devuelve un objeto
**validado con Pydantic** con las tecnologías mencionadas, el nivel de
criticidad y un resumen técnico. Funciona con **OpenAI** o **Anthropic** y
reintenta automáticamente si el modelo devuelve un JSON mal formado, incompleto
o que no pasa la validación.

```python
from chain import process_text

resultado = await process_text("FastAPI devuelve 503: pool de PostgreSQL agotado...")
print(resultado.datos if resultado.ok else resultado.error)
```

## Ejemplo de salida

Entrada (`main.py`, ejemplo "Log de error"):

```
2026-11-08 03:12:44 ERROR [api-gateway] FastAPI worker timeout after 30s on POST /orders.
psycopg.OperationalError: connection pool exhausted (max=20) for PostgreSQL 16. Redis cache
hit rate dropped to 12% after the last deploy. 503 Service Unavailable returned to 38% of requests.
```

Salida (objeto `ExtraccionTecnica` serializado; el texto exacto varía según el modelo):

```json
{
  "tecnologias": ["FastAPI", "PostgreSQL", "Redis"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API en FastAPI devuelve 503 al 38% de los requests porque se agotó el pool de conexiones a PostgreSQL; además cayó el hit rate de la caché en Redis tras el último deploy."
}
```

## Estructura

```
├── schemas.py        # Pydantic: ExtraccionTecnica (salida del LLM), ResultadoExtraccion y LLMConfig
├── chain.py          # cliente ChatOpenAI/ChatAnthropic, ChatPromptTemplate, cadena LCEL,
│                     # validación, reintentos y process_text()
├── main.py           # mini-script de prueba asíncrono con 3 textos de ejemplo
├── tests/            # tests sin red (APIs simuladas)
├── .env.example
└── requirements.txt
```

## Instalación

Requiere **Python 3.12+**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # y completá tu API key
```

## Variables de entorno

| Variable | Obligatoria | Default | Descripción |
|---|---|---|---|
| `OPENAI_API_KEY` | si usás OpenAI | – | API key de OpenAI |
| `ANTHROPIC_API_KEY` | si usás Anthropic | – | API key de Anthropic |
| `LLM_PROVIDER` | no | `openai` | `openai` o `anthropic` |
| `OPENAI_MODEL` | no | `gpt-4.1-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | no | `claude-haiku-5-5` | Modelo de Anthropic |
| `LLM_TEMPERATURE` | no | `0` | Entre 0 y 2. Sólo la aplica OpenAI |
| `LLM_MAX_TOKENS` | no | `1024` | Máximo de tokens de salida |
| `LLM_TIMEOUT` | no | `60` | Timeout por request, en segundos |
| `LLM_MAX_RETRIES` | no | `2` | Reintentos extra (2 → hasta 3 intentos) |

Los valores se validan con Pydantic al arrancar (por ejemplo, `LLM_TEMPERATURE=5`
se rechaza antes de llamar a la API).

## Ejecutar el script de prueba

`main.py` procesa tres textos: un log de error, una descripción de arquitectura
y un texto ambiguo (prueba de estrés: no nombra tecnologías ni deja clara la
gravedad).

```bash
python main.py                          # proveedor de LLM_PROVIDER
python main.py --provider anthropic     # forzar un proveedor
python main.py --texto "Kafka perdió mensajes al reiniciar el broker"
```

Los logs muestran cada paso de la validación y los reintentos:

```
INFO    Procesando texto (281 caracteres)
INFO    Llamando al modelo (intento 1)
WARNING Validación fallida en el intento 1: tecnologias: Value error, la lista de tecnologías no puede estar vacía
INFO    Llamando al modelo (intento 2)
INFO    Validación OK: 3 tecnologías, criticidad=alta
INFO    Texto procesado en 2140 ms
```

## Tests

No usan red ni API keys: simulan las respuestas HTTP de OpenAI y Anthropic y
cubren el esquema, el prompt, la configuración, JSON mal formado, respuestas
cortadas por tokens, datos inválidos, rate limit, errores 5xx y API key inválida.

```bash
pip install -r requirements-dev.txt
pytest
```

## Cómo funciona

### 1. Contrato (`schemas.py`)

`ExtraccionTecnica` define la salida: `tecnologias` (lista de strings),
`nivel_de_criticidad` (enum `baja` / `media` / `alta`) y `resumen_tecnico`.
Además de los tipos, tiene validadores de **sentido común**: la lista de
tecnologías no puede estar vacía (se limpian espacios y duplicados) y el
resumen tiene que tener contenido. Estas reglas viven en Python y no en el JSON
Schema, así el esquema que recibe el proveedor queda simple y compatible con el
modo estricto de OpenAI.

### 2. Prompt modular (`chain.py`)

Un `ChatPromptTemplate` con un mensaje `system` y uno `human`. Las
instrucciones de formato (qué cuenta como criticidad alta, media o baja) son
una variable del template, `{instrucciones_formato}`, que se completa con
`.partial()`. Así se pueden cambiar sin tocar el resto, y el único input de la
cadena es `{texto}`. No hay f-strings armando el prompt.

### 3. Cadena LCEL

```python
cadena = (
    prompt
    | model.with_structured_output(ExtraccionTecnica, include_raw=True)
    | RunnableLambda(validar_salida)
)
cadena = cadena.with_retry(...)
```

- `with_structured_output` hace que el modelo devuelva directamente un
  `ExtraccionTecnica` (OpenAI usa JSON Schema; Anthropic, tool calling).
- Con `include_raw=True` también llega el mensaje crudo del modelo, que trae el
  `finish_reason`.
- `validar_salida` es un paso más de la cadena (un `RunnableLambda`) que revisa
  el resultado **antes** de devolverlo.

### 4. Resiliencia

`validar_salida` lanza una excepción en tres casos, y `.with_retry()` vuelve a
llamar al modelo:

1. **Respuesta incompleta:** `finish_reason` es `length` (OpenAI) o
   `max_tokens` (Anthropic), o sea que el modelo se quedó sin tokens y el JSON
   quedó cortado. Se detecta antes de intentar transformarlo.
2. **JSON mal formado:** no se pudo parsear.
3. **Datos inválidos:** el JSON es válido pero no pasa Pydantic (por ejemplo,
   una criticidad que no existe o una lista de tecnologías vacía).

También se reintentan los errores transitorios de la API (conexión, timeout,
rate limit 429 y 5xx), con espera exponencial entre intentos. Los errores
permanentes (API key inválida, request mal armado) no se reintentan.

Los clientes se crean con `max_retries=0` para que haya una sola política de
reintentos, la de la cadena (igual que en el Módulo 1).

### 5. `process_text()`

Ejecuta la cadena con `await cadena.ainvoke(...)` y devuelve un
`ResultadoExtraccion`: con `ok=True` y los datos, o con `ok=False`, el tipo de
error (`validacion`, `respuesta_incompleta`, `rate_limit`, `conexion`,
`autenticacion`, ...) y el mensaje. No lanza excepciones del modelo hacia
afuera, así un texto que falla no tumba al resto.

Un callback de LangChain (`LogIntentos`) loguea cada llamada al modelo y cada
error, así se puede seguir el proceso de validación y los reintentos.

## Relación con el Módulo 1

`chain.py` configura el cliente (`ChatOpenAI` o `ChatAnthropic`) reutilizando la
lógica del cliente asíncrono del Módulo 1: el
proveedor se elige con `LLM_PROVIDER`, los parámetros se validan con Pydantic,
se usan sólo métodos asíncronos (`ainvoke`) y los errores se devuelven
controlados en lugar de romper el programa. La diferencia es que ahora el
modelo es un Runnable de LangChain y se compone con `|`.
