"""
Main FastAPI application module for the RAIDO Data Preprocessing API.
Routes are organized into logical groups based on functionality and data type (images vs time series).
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn

from api.routes.images import (
    missing_data,
    visualization,
    outliers,
    dim_reduction,
    noise,
    enrichment,
)

from api.routes.time_series import (
    timeseries
)

tags_metadata = [
    {
        "name": "API Endpoints",
        "description": "List all available endpoints in the API.",
    },
    {
        "name": "Images Missing Data",
        "description": "Operations related to detecting, fixing and visualizing missing data in images.",
    },
    {
        "name": "Images Visualization",
        "description": "Operations related to generic image visualization.",
    },
    {
        "name": "Images Outliers",
        "description": "Operations related to detecting and visualizing outliers in images.",
    },
    {
        "name": "Images Dimensionality Reduction",
        "description": "Operations related to dimensionality reduction in images.",
    },
    {
        "name": "Images Noise Removal",
        "description": "Operations related to noise removal in images.",
    },
    {
        "name": "Images Enrichment",
        "description": "Operations related to image enrichment.",
    },
]

app = FastAPI(title="RAIDO Data Preprocessing API", openapi_tags=tags_metadata)

app.include_router(missing_data.router)
app.include_router(visualization.router)
app.include_router(outliers.router)
app.include_router(dim_reduction.router)
app.include_router(noise.router)
app.include_router(enrichment.router)
app.include_router(timeseries.router)


@app.get("/endpoints", tags=["API Endpoints"])
def list_endpoints():
    try:
        endpoints = []
        for route in app.routes:
            if hasattr(route, "methods"):
                for method in route.methods:
                    if method in ["GET", "POST", "PUT", "DELETE"]:
                        endpoints.append({
                            "path": route.path,
                            "method": method,
                            "name": route.name
                        })
        return JSONResponse(status_code=200, content=jsonable_encoder(endpoints))
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
