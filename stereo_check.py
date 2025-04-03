import cv2
import numpy as np
def load_image_as_array(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = image_rgb.astype(np.float32) / 255.0

    return image_array

#resize the larger image to match the smaller one
def resize_to_smaller_dimension(left_image, right_image):
    if left_image.shape[0] > right_image.shape[0] or left_image.shape[1] > right_image.shape[1]:
        resized_left = cv2.resize(left_image, (right_image.shape[1], right_image.shape[0]))
        return resized_left, right_image
    else:
        resized_right = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))
        return left_image, resized_right

left_image_path = 'inputs_output_cGAN/inputs_output_cGAN/test_image_1.png'
disparity_map_path = 'inputs_output_cGAN/inputs_output_cGAN/disparity_map_1.npy'
right_image_path = 'inputs_output_cGAN/inputs_output_cGAN/generated_right_view.jpg'

try:
    left_image_array = load_image_as_array(left_image_path)
    right_image_array = load_image_as_array(right_image_path)

    left_image_resized, right_image_resized = resize_to_smaller_dimension(left_image_array, right_image_array)

    pixel_difference = np.abs(left_image_resized - right_image_resized)
    print("Pixel Difference:")
    print(pixel_difference)
    average_difference = np.mean(pixel_difference)
    print(f"\nAverage Pixel Difference: {average_difference}")

except Exception as e:
    print(f"An error occurred: {e}")
