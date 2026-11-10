"""Tests sin red: las APIs de OpenAI y Anthropic se simulan con `httpx.MockTransport`.

Así se prueba la cadena real (LangChain + SDK oficial + parseo + validación +
reintentos) sin gastar tokens ni necesitar API keys.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import anthropic
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

try:  # Las versiones nuevas de los SDKs usan httpx2; las anteriores, httpx.
    import httpx2 as httpx
except ImportError:
    import httpx

from chain import ConfigError, config_from_env, crear_cadena, crear_modelo, crear_prompt, process_text
from schemas import ExtraccionTecnica, LLMConfig, NivelCriticidad

TEXTO = "FastAPI devuelve 503: el pool de conexiones de PostgreSQL está agotado."

VALIDO = {
    "tecnologias": ["FastAPI", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "La API devuelve 503 porque se agotó el pool de conexiones a PostgreSQL.",
}

Handler = Callable[[httpx.Request], httpx.Response]


# --- Helpers para simular las APIs -------------------------------------------


def secuencia(*respuestas: httpx.Response) -> tuple[Handler, list[httpx.Request]]:
    """Handler que devuelve las respuestas en orden y guarda cada request recibido."""
    requests: list[httpx.Request] = []
    pendientes = list(respuestas)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return pendientes.pop(0)

    return handler, requests


def openai_ok(content: str, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        },
    )


def openai_error(status: int, message: str = "error") -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "error"}})


def anthropic_ok(datos: dict, stop_reason: str = "tool_use") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "ExtraccionTecnica", "input": datos}
            ],
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 50, "output_tokens": 30},
        },
    )


def cadena_openai(handler: Handler):
    model = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key="sk-test",
        temperature=0,
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return crear_cadena(model, max_retries=2, esperar_entre_intentos=False)


def cadena_anthropic(handler: Handler):
    model = ChatAnthropic(model="claude-haiku-5-5", api_key="sk-ant-test", max_tokens=512, max_retries=0)
    # ChatAnthropic no acepta un http client propio: le cambiamos el cliente del SDK.
    model.__dict__["_async_client"] = anthropic.AsyncAnthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return crear_cadena(model, max_retries=2, esperar_entre_intentos=False)


# --- Esquema -----------------------------------------------------------------


def test_esquema_valido_limpia_y_deduplica_tecnologias():
    datos = ExtraccionTecnica(
        tecnologias=[" Redis ", "redis", "PostgreSQL", ""],
        nivel_de_criticidad="media",
        resumen_tecnico="  Latencia alta en la caché de Redis.  ",
    )
    assert datos.tecnologias == ["Redis", "PostgreSQL"]
    assert datos.nivel_de_criticidad is NivelCriticidad.MEDIA
    assert datos.resumen_tecnico == "Latencia alta en la caché de Redis."


@pytest.mark.parametrize(
    "cambio",
    [
        {"tecnologias": []},
        {"tecnologias": ["  "]},
        {"nivel_de_criticidad": "urgente"},
        {"resumen_tecnico": "corto"},
    ],
)
def test_esquema_rechaza_datos_invalidos(cambio):
    with pytest.raises(ValidationError):
        ExtraccionTecnica(**{**VALIDO, **cambio})


def test_el_json_schema_no_incluye_restricciones_de_los_validadores():
    # Las restricciones viven en Python, no en el schema que se manda al proveedor.
    schema = json.dumps(ExtraccionTecnica.model_json_schema())
    assert "minItems" not in schema and "minLength" not in schema


# --- Prompt y configuración --------------------------------------------------


def test_prompt_solo_espera_el_texto_y_trae_las_instrucciones_de_formato():
    prompt = crear_prompt()
    assert prompt.input_variables == ["texto"]
    mensajes = prompt.format_messages(texto="hola")
    assert "nivel_de_criticidad" in mensajes[0].content
    assert mensajes[1].content.endswith("hola")


def test_config_valida_rangos(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "5")
    with pytest.raises(ConfigError, match="temperature"):
        config_from_env("openai")


def test_config_elige_proveedor_y_modelo(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    config = config_from_env()
    assert (config.provider, config.model) == ("anthropic", "claude-test")


def test_falta_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        crear_modelo(LLMConfig(provider="openai", model="gpt-4.1-mini"))


# --- Pipeline con OpenAI -----------------------------------------------------


async def test_extraccion_valida():
    handler, requests = secuencia(openai_ok(json.dumps(VALIDO)))
    resultado = await process_text(TEXTO, cadena_openai(handler))

    assert resultado.ok
    assert resultado.datos == ExtraccionTecnica(**VALIDO)
    body = json.loads(requests[0].content)
    assert body["response_format"]["type"] == "json_schema"  # with_structured_output
    assert TEXTO in body["messages"][1]["content"]


async def test_reintenta_si_el_json_esta_mal_formado(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    handler, requests = secuencia(
        openai_ok('{"tecnologias": ["FastAPI", '),  # JSON roto
        openai_ok(json.dumps(VALIDO)),
    )
    resultado = await process_text(TEXTO, cadena_openai(handler))

    assert resultado.ok
    assert len(requests) == 2
    assert "Validación fallida" in caplog.text
    assert "intento 2" in caplog.text


async def test_reintenta_si_la_respuesta_se_corta_por_tokens(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    handler, requests = secuencia(
        openai_ok('{"tecnologias": ["FastAPI"], "nivel_de', finish_reason="length"),
        openai_ok(json.dumps(VALIDO)),
    )
    resultado = await process_text(TEXTO, cadena_openai(handler))

    assert resultado.ok
    assert len(requests) == 2
    assert "Respuesta incompleta" in caplog.text


async def test_reintenta_si_no_pasa_la_validacion_y_devuelve_error_controlado():
    invalido = json.dumps({**VALIDO, "tecnologias": []})
    handler, requests = secuencia(*(openai_ok(invalido) for _ in range(3)))
    resultado = await process_text(TEXTO, cadena_openai(handler))

    assert not resultado.ok
    assert resultado.tipo_error == "validacion"
    assert "tecnologias" in resultado.error
    assert len(requests) == 3  # 1 intento + 2 reintentos


async def test_reintenta_errores_transitorios_de_la_api():
    handler, requests = secuencia(openai_error(503, "overloaded"), openai_ok(json.dumps(VALIDO)))
    resultado = await process_text(TEXTO, cadena_openai(handler))

    assert resultado.ok
    assert len(requests) == 2


async def test_rate_limit_persistente_devuelve_error_controlado():
    handler, requests = secuencia(*(openai_error(429, "Rate limit reached") for _ in range(3)))
    resultado = await process_text(TEXTO, cadena_openai(handler))

    assert not resultado.ok
    assert resultado.tipo_error == "rate_limit"
    assert len(requests) == 3


async def test_api_key_invalida_no_se_reintenta():
    handler, requests = secuencia(openai_error(401, "Incorrect API key provided"))
    resultado = await process_text(TEXTO, cadena_openai(handler))

    assert not resultado.ok
    assert resultado.tipo_error == "autenticacion"
    assert len(requests) == 1


async def test_texto_vacio():
    handler, requests = secuencia()
    resultado = await process_text("   ", cadena_openai(handler))

    assert not resultado.ok
    assert resultado.tipo_error == "input_invalido"
    assert requests == []


# --- Pipeline con Anthropic --------------------------------------------------


async def test_anthropic_extraccion_valida():
    handler, requests = secuencia(anthropic_ok(VALIDO))
    resultado = await process_text(TEXTO, cadena_anthropic(handler))

    assert resultado.ok
    assert resultado.datos == ExtraccionTecnica(**VALIDO)
    body = json.loads(requests[0].content)
    assert body["tools"][0]["name"] == "ExtraccionTecnica"  # with_structured_output usa tool calling
    assert "temperature" not in body


async def test_anthropic_reintenta_si_no_pasa_la_validacion():
    handler, requests = secuencia(
        anthropic_ok({**VALIDO, "nivel_de_criticidad": "urgentísimo"}),
        anthropic_ok(VALIDO),
    )
    resultado = await process_text(TEXTO, cadena_anthropic(handler))

    assert resultado.ok
    assert len(requests) == 2


async def test_anthropic_reintenta_si_se_corta_por_max_tokens():
    handler, requests = secuencia(
        anthropic_ok({"tecnologias": ["FastAPI"]}, stop_reason="max_tokens"),
        anthropic_ok(VALIDO),
    )
    resultado = await process_text(TEXTO, cadena_anthropic(handler))

    assert resultado.ok
    assert len(requests) == 2
