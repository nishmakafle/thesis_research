import cv2
import numpy as np
from PIL import Image


def ela(image_path, quality=50):
    """
    Performs Error Level Analysis on an image.

    Args:
      image_path: Path to the image file.
      quality: Quality level for recompression (lower values increase sensitivity).

    Returns:
      The ELA image.
    """

    # Read the image
    img = cv2.imread(image_path)

    # Recompress the image at a lower quality
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil.save("temp.jpg", quality=quality)
    img_recompressed = cv2.imread("temp.jpg")

    # Calculate the difference between the original and recompressed images
    diff = cv2.absdiff(img, img_recompressed)

    # Convert the difference image to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Enhance the contrast of the difference image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    diff_clahe = clahe.apply(diff_gray)

    return diff_clahe


def main():
    # Example usage:
    image_path = "/Users/logpoint/Documents/SoftwareProjects/Thesis/CASIA2/Tp/Tp_D_CNN_M_B_nat00056_nat00099_11105.jpg"
    ela_image = ela(image_path)

    # Display the ELA image
    cv2.imshow("ELA Image", ela_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
