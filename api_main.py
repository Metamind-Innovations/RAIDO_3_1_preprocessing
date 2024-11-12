from fastapi import FastAPI
import uvicorn

from api.routes.images import missing_data, visualization, outliers, dim_reduction


tags_metadata = [
    {
        "name": "Images Missing Data",
        "description": "Operations related to detecting, fixing and visualizing missing data in images.",
    },
    {
        "name": "Images Visualization",
        "description": "Operations related to visualizing images.",
    },
    {
        "name": "Images Outliers",
        "description": "Operations related to detecting outliers in images.",
    },
    {
        "name": "Images Dimensionality Reduction",
        "description": "Operations related to dimensionality reduction in images.",
    },
]

app = FastAPI(title="RAIDO Data Preprocessing API", openapi_tags=tags_metadata)

app.include_router(missing_data.router)
app.include_router(visualization.router)
app.include_router(outliers.router)
app.include_router(dim_reduction.router)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8008)


if __name__ == "__main__":
    main()
