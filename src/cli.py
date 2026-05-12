import sys
from unittest import case
import os
from preprocessing.image_io import load_image
from preprocessing.otsus import otsu_threshold
from preprocessing.otsus import otsu_binarize
# from preprocessing.rotation import deskew_image
# from preprocessing.rotation import rotate_image
import segmentation.hough as hough
import segmentation.rotation as rotation
from PIL import Image

# approximate angles to rotate by for OCR, based on Fibonacci sequence and common angles
candidate_angles = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 90]

def main():
    
    command = input("enter a command (type 'help' for available commands): ")
    check = 0

    while (1):
        if command.lower() == "help":

            print("available commands:")
            print("$help - show this message")
            print("$exit - exit the program")
            print("$load <image_file_path> - load an image file")
            print("$ocr - perform OCR on the loaded image")

        elif command.lower() == "exit":
            print("exiting...")
            break

        elif command.lower() == "load":
            # load the image file specified by the user
            image_path = input("enter img_name.type: ").strip()
            if not image_path:
                print("no path entered.")
                check = 0
                continue
            if image_path[0] == '"' and image_path[-1] == '"':
                image_path = image_path[1:-1]

            # load the image as a grayscale image (2D numpy array)
            grayscale_image = load_image(image_path)
            if grayscale_image is not None:
                print(f"loading image from {image_path}...")
                check = 1
                # image is now loaded as a 2D numpy array of grayscale pixel values 0-255
            else:
                print("failed to load image.")
                check = 0
                continue

            # apply Otsu's method to find optimal threshold
            threshold = otsu_threshold(grayscale_image)
            print(f"optimal threshold found: {threshold}")

            # binarize the image using the optimal threshold
            binary_image = otsu_binarize(grayscale_image, threshold)
            print("binarization complete.")
            # # displays the binarized image for testing
            # img = Image.fromarray(binary_image)
            # img.show()

            # perform houghs
            lines_img = hough.run_hough(binary_image)
            img = Image.fromarray(lines_img)
            img.show()



        elif command.lower() == "ocr":
            if check == 1:
                print("performing OCR on the loaded image...")
            else:
                print("no image loaded.")

        else:
            print(f"unknown command: {command}")

        command = input("enter command: ")

    

if __name__ == "__main__": # type: ignore
    main() # type: ignore