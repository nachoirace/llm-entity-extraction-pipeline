"""Modelos Pydantic del pipeline: la salida del LLM, el resultado de `process_text`
y la configuración del modelo.

`ExtraccionTecnica` es el esquema que se le pasa a `.with_structured_output()`.
Las restricciones de "sentido común" (lista no vacía, sin duplicados, resumen
con contenido) van en validadores de Pydantic y no en el JSON Schema: así el
esquema que recibe el proveedor queda simple y compatible con el modo estricto
de OpenAI, y la validación se hace igual en Python.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (log, descripción de arquitectura, etc.)."""

    model_config = ConfigDict(frozen=True)

    tecnologias: list[str] = Field(
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos o servicios mencionados "
            "en el texto, con su nombre propio (ej. 'FastAPI', 'PostgreSQL')."
        )
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Impacto del problema o de la situación descrita: baja, media o alta."
    )
    resumen_tecnico: str = Field(
        description="Resumen técnico en una o dos oraciones, en español."
    )

    @field_validator("tecnologias")
    @classmethod
    def tecnologias_no_vacias(cls, value: list[str]) -> list[str]:
        # Limpia espacios, descarta strings vacíos y duplicados (sin importar mayúsculas).
        limpias: list[str] = []
        vistas: set[str] = set()
        for tech in value:
            tech = tech.strip()
            if tech and tech.lower() not in vistas:
                vistas.add(tech.lower())
                limpias.append(tech)
        if not limpias:
            raise ValueError("la lista de tecnologías no puede estar vacía")
        return limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def resumen_con_contenido(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 15:
            raise ValueError("el resumen técnico es demasiado corto (mínimo 15 caracteres)")
        return value


class ResultadoExtraccion(BaseModel):
    """Lo que devuelve `process_text`: los datos validados o un error controlado.

    Igual que en el Módulo 1, el pipeline no lanza excepciones hacia afuera:
    si algo falla después de los reintentos, `ok` es False y `error` explica por qué.
    """

    ok: bool
    datos: ExtraccionTecnica | None = None
    error: str | None = None
    tipo_error: str | None = None
    latencia_ms: float


class LLMConfig(BaseModel):
    """Parámetros del modelo, validados al arrancar (igual que `ModelConfig` en el Módulo 1)."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["openai", "anthropic"]
    model: str = Field(min_length=1)
    # Para extracción conviene temperatura 0: queremos la respuesta más predecible.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=128_000)
    timeout: float = Field(default=60.0, gt=0, description="Segundos por request.")
    max_retries: int = Field(
        default=2, ge=0, le=10, description="Reintentos extra (2 → hasta 3 intentos)."
    )
