import sys
import os

from PIL import Image
import numpy as np

from preprocessing.image_io import load_image
from preprocessing.otsus import otsu_threshold
from preprocessing.otsus import otsu_binarize
from segmentation.hough import run_hough
from segmentation.rotation import rotate_image


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

            image_path = input("enter img_name.type: ").strip()
            if not image_path:
                print("no path entered.")
                check = 0
                continue
            if image_path[0] == '"' and image_path[-1] == '"':
                image_path = image_path[1:-1]

            grayscale_image = load_image(image_path)
            if grayscale_image is not None:
                check = 1
                print(f"loading image from {image_path}...")
            else:
                print("failed to load image.")
                check = 0
                continue

            threshold = otsu_threshold(grayscale_image)
            binary_image = otsu_binarize(grayscale_image, threshold)
            rotation_angle = 1
            while (rotation_angle != 0):
                hough_result = run_hough(binary_image)
                rotation_angle = hough_result[0]
                lines_array = hough_result[1]
                ## for troubleshooting the difference between the angle and lines
                # img = Image.fromarray(lines_array)
                # img.show()
                rotated_image = rotate_image(binary_image, rotation_angle)
                # img = Image.fromarray(rotated_image)
                # img.show()
                print(f"Estimated rotation angle: {rotation_angle}")
                binary_image = rotated_image
            
            

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