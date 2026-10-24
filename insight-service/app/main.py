"""Central factory: uvicorn app.main:create_app --factory --workers 1.

The legacy app.main:app retains its original health/validation-only contract.
The factory composes all existing capabilities without initializing model weights.
"""

from app.api.base import app, health, validate_analysis


def create_app():
    """Build the complete API with each method/path registered exactly once."""
    from fastapi import FastAPI
    from app.api.base import router as base_router
    from app.api.sentiment import router as sentiment_router
    from app.api.batch import router as batch_router
    from app.api.aggregation import router as aggregation_router, weighted_router
    from app.api.signals import router as signals_router
    from app.helper.validation_errors import install_validation_error_handler

    application = FastAPI(title=app.title, version=app.version)
    install_validation_error_handler(application)
    for router in (base_router, sentiment_router, batch_router, aggregation_router,
                   weighted_router, signals_router):
        application.include_router(router)
    return application
