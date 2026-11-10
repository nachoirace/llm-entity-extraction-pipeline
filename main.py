"""Mini-script de prueba: corre el pipeline sobre tres textos de ejemplo.

    python main.py                          # usa LLM_PROVIDER del .env
    python main.py --provider anthropic
    python main.py --texto "Kafka perdió mensajes al reiniciar el broker..."
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from chain import ConfigError, config_from_env, crear_cadena, crear_modelo, process_text
from schemas import ResultadoExtraccion

EJEMPLOS: dict[str, str] = {
    "Log de error": (
        "2026-11-08 03:12:44 ERROR [api-gateway] FastAPI worker timeout after 30s on "
        "POST /orders. psycopg.OperationalError: connection pool exhausted (max=20) "
        "for PostgreSQL 16. Redis cache hit rate dropped to 12% after the last deploy. "
        "503 Service Unavailable returned to 38% of requests."
    ),
    "Arquitectura": (
        "El frontend en React consume una API GraphQL escrita en Node.js. Los eventos "
        "de pedidos se publican en Kafka y un worker en Go los procesa y guarda en "
        "MongoDB. Todo corre en Kubernetes sobre AWS. Estamos evaluando agregar "
        "OpenTelemetry porque hoy cuesta rastrear errores entre servicios."
    ),
    # Prueba de estrés: texto vago, sin tecnologías explícitas ni gravedad clara.
    "Texto ambiguo": "Desde ayer el sistema anda raro, a veces tarda y otras veces no. Nadie sabe bien por qué.",
}


def mostrar(titulo: str, resultado: ResultadoExtraccion) -> None:
    print(f"\n── {titulo} ── ({resultado.latencia_ms:.0f} ms)")
    if resultado.ok and resultado.datos:
        print(resultado.datos.model_dump_json(indent=2))
    else:
        print(f"✗ Error controlado [{resultado.tipo_error}]: {resultado.error}")


async def main(provider: str | None, texto: str | None) -> int:
    try:
        config = config_from_env(provider)
        cadena = crear_cadena(crear_modelo(config), max_retries=config.max_retries)
    except ConfigError as exc:
        print(f"✗ {exc}\nRevisá tu .env (ver .env.example).")
        return 1

    print(f"Proveedor: {config.provider} · modelo: {config.model}")
    textos = {"Texto propio": texto} if texto else EJEMPLOS

    # Uno por uno, para que los logs de cada texto se lean en orden.
    for titulo, texto_actual in textos.items():
        print(f"\n▶ {titulo}")
        resultado = await process_text(texto_actual, cadena)
        mostrar(titulo, resultado)
    return 0


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=os.getenv("LLM_PROVIDER"))
    parser.add_argument("--texto", help="Procesa este texto en lugar de los ejemplos.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for ruidoso in ("httpx", "httpx2", "openai", "anthropic", "langchain_openai", "langchain_anthropic"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    # LangChain loguea su propio traceback cuando una respuesta viene cortada;
    # nuestro pipeline ya lo informa en una línea y lo reintenta.
    logging.getLogger("langchain_core.output_parsers").setLevel(logging.CRITICAL)

    sys.exit(asyncio.run(main(args.provider, args.texto)))
