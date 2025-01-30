import json

from src.images.utils import image_path2id
from src.images.invalid_pixel_detection import detect_missing_data, detect_pixel_level_outliers
from src.images.invalid_pixel_handling import impute_invalid_pixels, interpolate_invalid_pixels
from src.images.image_outliers import detect_image_level_outliers, remove_image_outliers
from src.images.noise import detect_noise, denoise_non_local_means


def main():
    img_json = json.load(open("img_pipeline_info.json"))

    print("\nInitial img_json:")
    print(json.dumps(img_json, indent=4))

    img_json = image_path2id(img_json)
    print("\nAfter image_path2id:")
    print(json.dumps(img_json, indent=4))

    img_json = detect_missing_data(img_json)
    print("\nAfter detect_missing_data:")
    print(json.dumps(img_json, indent=4))

    img_json = detect_pixel_level_outliers(img_json)
    print("\nAfter detect_outlier_pixels:")
    for key, value in img_json.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} entries")
        elif isinstance(value, list):
            print(f"{key}: {len(value)} items")
    
    print("\nOutlier coords structure:")
    for img_id, coords in img_json["outlier_coords"].items():
        print(f"{img_id}: {len(coords)} outlier pixels")
        if coords:  # Only print example if there are outliers
            print(f"Example coordinate format: {coords[0]}")

    # img_json = impute_invalid_pixels(img_json)
    # # img_json = interpolate_invalid_pixels(img_json)
    # img_json = detect_image_level_outliers(img_json)
    # img_json = remove_image_outliers(img_json)

    # print("\nFinal img_json:")
    # print(json.dumps(img_json, indent=4))

    img_json = detect_noise(img_json)
    print("\nFinal img_json:")
    print(json.dumps(img_json, indent=4))

    img_json = denoise_non_local_means(img_json)


if __name__ == "__main__":
    main()
