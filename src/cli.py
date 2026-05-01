import sys
from unittest import case
import os
from image_io import load_image
from otsus import otsu_threshold
from otsus import otsu_binarize
from PIL import Image

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
            image_path = input("enter img_name.type: ")
            if image_path[0] == '"' and image_path[-1] == '"':
                image_path = image_path[1:-1]
            grayscale_image = load_image(image_path)
            if grayscale_image is not None:
                print(f"loading image from {image_path}...")
                check = 1
                # image is now loaded as a 2D numpy array of grayscale pixel values 0-255
                # for row in grayscale_image:
                #     for pixel in row:
                #         print(f"{pixel:3}", end=" ")
                #     print()
            else:
                print("failed to load image.")
                check = 0
                continue

            # apply Otsu's method to find optimal threshold
            threshold = otsu_threshold(grayscale_image)
            print(f"optimal threshold found: {threshold}")
            # binarize the image using the optimal threshold
            binary_image = otsu_binarize(grayscale_image, threshold)
            # displays the binarized image for testing
            # img = Image.fromarray(binary_image)
            # img.show()
            print("binarization complete.")


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