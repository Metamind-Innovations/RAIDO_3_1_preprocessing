import io
import requests
from pathlib import Path
from PIL import Image


def image_to_bytes(image, format):
    """Convert PIL Image to bytes."""
    img_byte_arr = io.BytesIO()
    # Convert 'jpg' to 'jpeg' for PIL compatibility
    format = "jpeg" if format.lower() == "jpg" else format
    image.save(img_byte_arr, format=format.upper())
    img_byte_arr.seek(0)
    return img_byte_arr


def test_multi_file_endpoint(
    base_url: str = "http://localhost:8000",
    test_input_dir: Path = Path("test_input"),
    test_output_dir: Path = Path("test_output"),
):
    test_output_dir.mkdir(exist_ok=True)

    # Get all image files with common image extensions
    test_image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        test_image_files.extend(test_input_dir.glob(f"*{ext}"))

    # List to store images and their extensions
    test_images = []
    for img_path in test_image_files:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        test_images.append((img, img_path.suffix))

    # Convert images to bytes, preserving original format
    test_files = [
        (
            "images",
            (f"image_{i}{ext}", image_to_bytes(img, ext[1:]), f"image/{ext[1:]}"),
        )
        for i, (img, ext) in enumerate(test_images)
    ]

    # Change according to the endpoint tested
    params = {"n_components": 3, "image_size": 360}

    # ----------- TEST ONE ENDPOINT AT A TIME -----------

    # TEST PCA WITH MULTIPLE IMAGES
    print("Testing PCA with multiple images...")
    endpoint = "/images/dim_reduction/multiple_images_pca"

    response = requests.post(f"{base_url}{endpoint}", files=test_files, params=params)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    output_path = test_output_dir / "multiple_images_pca_results.json"
    with open(output_path, "wb") as f:
        f.write(response.content)

    print("Test PCA with multiple images complete")

    # # TEST RECONSTRUCTED IMAGES WITH MULTIPLE IMAGES
    # print("Testing reconstructed images with multiple images...")
    # endpoint = "/images/dim_reduction/visualize_multiple_reconstructed_images"

    # response = requests.post(f"{base_url}{endpoint}", files=test_files, params=params)
    # assert response.status_code == 200
    # assert response.headers["content-type"] == "image/png"

    # output_path = test_output_dir / "multiple_images_reconstructed_images.png"
    # with open(output_path, "wb") as f:
    #     f.write(response.content)

    # print("Test reconstructed images with multiple images complete")

    # # TEST EIGEN IMAGES WITH MULTIPLE IMAGES
    # print("Testing eigen images with multiple images...")
    # endpoint = "/images/dim_reduction/visualize_multiple_pca_images"

    # response = requests.post(f"{base_url}{endpoint}", files=test_files, params=params)
    # assert response.status_code == 200
    # assert response.headers["content-type"] == "image/png"

    # output_path = test_output_dir / "multiple_images_pca_images.png"
    # with open(output_path, "wb") as f:
    #     f.write(response.content)

    # print("Test eigen images with multiple images complete")

    # Clean up open files
    for _, (_, file_obj, _) in test_files:
        file_obj.close()


def main():
    base_url = "http://localhost:8000"
    test_input_dir = Path("test_input")
    test_output_dir = Path("test_output")

    test_multi_file_endpoint(base_url, test_input_dir, test_output_dir)


if __name__ == "__main__":
    main()
