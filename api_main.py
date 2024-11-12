from fastapi import FastAPI
import uvicorn

from api.routes.images import (
    missing_data,
    visualization,
    outliers,
    dim_reduction,
    noise,
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
        "description": "Operations related to detecting and visualizingoutliers in images.",
    },
    {
        "name": "Images Dimensionality Reduction",
        "description": "Operations related to dimensionality reduction in images.",
    },
    {
        "name": "Images Noise Removal",
        "description": "Operations related to noise removal in images.",
    },
]

app = FastAPI(title="RAIDO Data Preprocessing API", openapi_tags=tags_metadata)

app.include_router(missing_data.router)
app.include_router(visualization.router)
app.include_router(outliers.router)
app.include_router(dim_reduction.router)
app.include_router(noise.router)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8008)


if __name__ == "__main__":
    main()


# TODO: Update src image outliers and dim reduction
# -> Add visualization functions in the respective src .py files
# -> Update functions to be self-contained similarly to how it is done in the APIs (match input/output between src and API functions)
# -> Update API routes to match the new functions