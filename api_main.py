from fastapi import FastAPI
import uvicorn

from api.routes.images import (
    missing_data,
    visualization,
    outliers,
    dim_reduction,
    noise,
    enrichment,
    balancing,
)


tags_metadata = [
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
    {
        "name": "Image Balancing",
        "description": "Operations related to image balancing.",
    },
]

app = FastAPI(title="RAIDO Data Preprocessing API", openapi_tags=tags_metadata)

app.include_router(missing_data.router)
app.include_router(visualization.router)
app.include_router(outliers.router)
app.include_router(dim_reduction.router)
app.include_router(noise.router)
app.include_router(enrichment.router)
app.include_router(balancing.router)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8008)


if __name__ == "__main__":
    main()
