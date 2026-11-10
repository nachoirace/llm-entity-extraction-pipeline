"""Pipeline LCEL de extracción de entidades técnicas.

Cadena:

    (prompt | model.with_structured_output(ExtraccionTecnica) | validar_salida).with_retry(...)

- `model`: `ChatOpenAI` o `ChatAnthropic`, según `LLM_PROVIDER` (lógica del Módulo 1).
- `prompt`: `ChatPromptTemplate` con el texto de entrada y las instrucciones de formato.
- `with_structured_output`: el modelo devuelve un `ExtraccionTecnica` ya validado.
  Con `include_raw=True` también recibimos el mensaje crudo, para poder mirar
  el `finish_reason` y detectar respuestas cortadas por falta de tokens.
- `validar_salida`: convierte "JSON mal formado / incompleto / inválido" en una
  excepción, para que `.with_retry()` pueda reintentar.
- `.with_retry()`: reintenta ante esos errores y ante fallos transitorios de la API.
"""

from __future__ import annotations

import logging
import os
import time
from functools import cache
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import ValidationError

from schemas import ExtraccionTecnica, LLMConfig, ResultadoExtraccion

logger = logging.getLogger("pipeline")

# --- Modelo (reutiliza la lógica del Módulo 1) -------------------------------
#
# Igual que `AsyncLLMManager`: el proveedor se elige con `LLM_PROVIDER` y los
# parámetros se validan con Pydantic antes de llamar a la API. La diferencia es
# que ahora el modelo es un chat model de LangChain, que se compone con `|`.

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-5-5",
}

API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ConfigError(Exception):
    """Configuración inválida (falta la API key, temperatura fuera de rango, etc.)."""


def config_from_env(provider: str | None = None) -> LLMConfig:
    """Arma un `LLMConfig` a partir de las variables de entorno (ver `.env.example`)."""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ConfigError(f"LLM_PROVIDER inválido: {provider!r} (opciones: openai, anthropic)")

    raw = {
        "provider": provider,
        "model": os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider],
        "temperature": os.getenv("LLM_TEMPERATURE"),
        "max_tokens": os.getenv("LLM_MAX_TOKENS"),
        "timeout": os.getenv("LLM_TIMEOUT"),
        "max_retries": os.getenv("LLM_MAX_RETRIES"),
    }
    try:
        return LLMConfig(**{k: v for k, v in raw.items() if v not in (None, "")})
    except ValidationError as exc:
        raise ConfigError(f"Configuración inválida:\n{exc}") from exc


def crear_modelo(config: LLMConfig, api_key: str | None = None) -> BaseChatModel:
    """Configura el cliente `ChatOpenAI` o `ChatAnthropic` según el proveedor.

    `max_retries=0` en el cliente: los reintentos los maneja `.with_retry()` en
    la cadena, así hay una sola política (igual que en el Módulo 1).
    """
    api_key = api_key or os.getenv(API_KEY_ENV[config.provider])
    if not api_key:
        raise ConfigError(f"Falta la variable de entorno {API_KEY_ENV[config.provider]}")

    if config.provider == "openai":
        return ChatOpenAI(
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=0,
        )

    # Sin temperature: la API actual de Anthropic no acepta parámetros de sampling
    # en los modelos nuevos (mismo criterio que en el Módulo 1).
    return ChatAnthropic(
        model=config.model,
        api_key=api_key,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=0,
    )


# --- Prompt ------------------------------------------------------------------

SYSTEM_PROMPT = """\
Sos un ingeniero de software senior. Tu tarea es analizar textos técnicos \
(logs de error, descripciones de arquitectura, reportes de incidentes) y \
extraer la información clave.

{instrucciones_formato}"""

INSTRUCCIONES_FORMATO = """\
Devolvé los datos con este formato:
- tecnologias: tecnologías, lenguajes, frameworks, bases de datos o servicios \
mencionados, con su nombre propio (ej. "FastAPI", "PostgreSQL"). Sin repetir. \
Si el texto no nombra ninguna de forma explícita, inferí la más probable.
- nivel_de_criticidad:
  - "alta": caída del servicio, pérdida o corrupción de datos, riesgo de \
seguridad o impacto directo en usuarios.
  - "media": degradación de rendimiento, errores intermitentes o algo que \
puede escalar si no se atiende.
  - "baja": información, mejoras o problemas menores sin impacto en usuarios.
- resumen_tecnico: una o dos oraciones en español que expliquen la situación \
y, si lo hay, el problema principal."""


def crear_prompt(instrucciones_formato: str = INSTRUCCIONES_FORMATO) -> ChatPromptTemplate:
    """Prompt modular: las instrucciones de formato son una variable más del template.

    Se completan con `.partial()`, así se pueden cambiar sin tocar el resto del
    prompt, y el único input que queda abierto es `{texto}`.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Texto a analizar:\n\n{texto}"),
        ]
    ).partial(instrucciones_formato=instrucciones_formato)


# --- Validación de la salida -------------------------------------------------


class RespuestaIncompletaError(OutputParserException):
    """El modelo cortó la respuesta (se quedó sin tokens) y el JSON quedó incompleto."""


# OpenAI informa "length" y Anthropic "max_tokens" cuando se acaban los tokens.
FINISH_REASONS_INCOMPLETOS = {"length", "max_tokens"}

# Con OpenAI, `with_structured_output` usa el modo JSON Schema y el propio SDK
# parsea la respuesta: si viene cortada lanza LengthFinishReasonError, y si el
# JSON está roto, un ValidationError. Esos casos no llegan a `validar_salida`,
# así que también los tratamos como salida inválida y se reintentan.
ERRORES_DE_SALIDA_OPENAI = (LengthFinishReasonError, ValidationError)


def _finish_reason(raw: AIMessage | None) -> str | None:
    if raw is None:
        return None
    meta = raw.response_metadata or {}
    return meta.get("finish_reason") or meta.get("stop_reason")


def validar_salida(salida: dict[str, Any]) -> ExtraccionTecnica:
    """Revisa lo que devolvió `with_structured_output(..., include_raw=True)`.

    Si la respuesta está cortada, el JSON no se pudo parsear o no pasó la
    validación de Pydantic, lanza una excepción para que `.with_retry()` reintente.
    """
    raw: AIMessage | None = salida.get("raw")
    finish_reason = _finish_reason(raw)

    # 1. Antes de validar: ¿el modelo terminó la respuesta o la cortó por tokens?
    if finish_reason in FINISH_REASONS_INCOMPLETOS:
        logger.warning("Respuesta incompleta (finish_reason=%s): se quedó sin tokens", finish_reason)
        raise RespuestaIncompletaError(
            f"Respuesta incompleta: el modelo se quedó sin tokens (finish_reason={finish_reason})"
        )

    # 2. ¿Se pudo parsear y validar contra el esquema?
    error = salida.get("parsing_error")
    if error is not None:
        logger.warning("Validación fallida: %s", _resumir_error(error))
        if isinstance(error, OutputParserException | ValidationError):
            raise error
        raise OutputParserException(str(error)) from error

    datos = salida.get("parsed")
    if datos is None:
        logger.warning("Validación fallida: el modelo no devolvió datos estructurados")
        raise OutputParserException("El modelo no devolvió datos estructurados")

    logger.info(
        "Validación OK: %d tecnologías, criticidad=%s",
        len(datos.tecnologias),
        datos.nivel_de_criticidad,
    )
    return datos


def _resumir_error(error: BaseException) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'objeto'}: {e['msg']}" for e in error.errors()
        )
    return str(error).splitlines()[0][:200]


# --- Cadena ------------------------------------------------------------------

# Qué se reintenta: salidas inválidas (JSON mal formado, incompleto o que no pasa
# la validación) y errores transitorios de la API. Un error de autenticación o de
# request mal armado no se reintenta: fallaría igual.
ERRORES_REINTENTABLES: tuple[type[BaseException], ...] = (
    OutputParserException,  # incluye RespuestaIncompletaError
    ValidationError,
    LengthFinishReasonError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelRateLimitError,
    ModelAPIError,  # 5xx y "overloaded"
)


def crear_cadena(
    model: BaseChatModel, max_retries: int = 2, esperar_entre_intentos: bool = True
) -> Runnable[dict, ExtraccionTecnica]:
    """Arma la cadena LCEL: prompt | modelo con salida estructurada | validación, con reintentos.

    `esperar_entre_intentos=False` sólo se usa en los tests, para no esperar el backoff.
    """
    prompt = crear_prompt()
    # include_raw=True: además del objeto, recibimos el mensaje crudo con el finish_reason.
    cadena = (
        prompt
        | model.with_structured_output(ExtraccionTecnica, include_raw=True)
        | RunnableLambda(validar_salida, name="validar_salida")
    )
    return cadena.with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES,
        # Espera creciente con algo de azar entre intentos (1s, 2s, 4s...).
        wait_exponential_jitter=esperar_entre_intentos,
        stop_after_attempt=max_retries + 1,
    )


@cache
def cadena_por_defecto() -> Runnable[dict, ExtraccionTecnica]:
    """Cadena armada desde el `.env`. Se crea una sola vez y se reutiliza."""
    config = config_from_env()
    logger.info("Usando %s (%s)", config.provider, config.model)
    return crear_cadena(crear_modelo(config), max_retries=config.max_retries)


# --- Logs de intentos --------------------------------------------------------


class LogIntentos(AsyncCallbackHandler):
    """Callback de LangChain que loguea cada llamada al modelo y sus errores.

    Se crea uno por cada `process_text`, así que contar las llamadas al modelo
    es contar los intentos.
    """

    def __init__(self) -> None:
        self.intentos = 0

    async def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.intentos += 1
        logger.info("Llamando al modelo (intento %d)", self.intentos)

    async def on_llm_error(self, error: BaseException, **kwargs) -> None:
        if isinstance(error, LengthFinishReasonError):
            logger.warning(
                "Respuesta incompleta en el intento %d (finish_reason=length): se quedó sin tokens",
                self.intentos,
            )
        elif isinstance(error, ERRORES_DE_SALIDA_OPENAI):
            # El SDK de OpenAI valida el JSON por su cuenta y falla antes que validar_salida.
            logger.warning("Validación fallida en el intento %d: %s", self.intentos, _resumir_error(error))
        else:
            logger.warning(
                "Error de la API en el intento %d: %s: %s",
                self.intentos,
                type(error).__name__,
                _resumir_error(error),
            )


# --- Punto de entrada --------------------------------------------------------


async def process_text(
    text: str, cadena: Runnable[dict, ExtraccionTecnica] | None = None
) -> ResultadoExtraccion:
    """Extrae las entidades técnicas de `text` ejecutando la cadena con `.ainvoke()`.

    Nunca lanza errores del modelo: si después de los reintentos sigue fallando,
    devuelve un `ResultadoExtraccion` con `ok=False` y el motivo.
    """
    inicio = time.perf_counter()
    if not text or not text.strip():
        return _error(inicio, "El texto está vacío", "input_invalido")

    cadena = cadena or cadena_por_defecto()
    logger.info("Procesando texto (%d caracteres)", len(text))
    try:
        datos = await cadena.ainvoke({"texto": text}, config={"callbacks": [LogIntentos()]})
    except (RespuestaIncompletaError, LengthFinishReasonError):
        return _error(
            inicio, "Respuesta incompleta: el modelo se quedó sin tokens", "respuesta_incompleta"
        )
    except (OutputParserException, ValidationError) as exc:
        return _error(inicio, f"Salida inválida tras los reintentos: {_resumir_error(exc)}", "validacion")
    except ModelError as exc:
        return _error(inicio, _resumir_error(exc), _tipo_error_api(exc))

    latencia = _ms(inicio)
    logger.info("Texto procesado en %.0f ms", latencia)
    return ResultadoExtraccion(ok=True, datos=datos, latencia_ms=latencia)


def _tipo_error_api(exc: ModelError) -> str:
    tipos = {
        ModelAuthenticationError: "autenticacion",
        ModelRateLimitError: "rate_limit",
        ModelTimeoutError: "timeout",
        ModelConnectionError: "conexion",
        ModelAPIError: "servidor",
    }
    for clase, nombre in tipos.items():
        if isinstance(exc, clase):
            return nombre
    return "api"


def _error(inicio: float, mensaje: str, tipo: str) -> ResultadoExtraccion:
    logger.error("No se pudo procesar el texto [%s]: %s", tipo, mensaje)
    return ResultadoExtraccion(ok=False, error=mensaje, tipo_error=tipo, latencia_ms=_ms(inicio))


def _ms(inicio: float) -> float:
    return round((time.perf_counter() - inicio) * 1000, 1)
