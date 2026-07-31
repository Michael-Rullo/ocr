import  sys
import  os

import  argparse
import  numpy   as      np

from    PIL     import  Image

from Preprocessing  import ImageIO
from Preprocessing  import Otsu
from Segmentation   import Hough
from Segmentation   import Rotation
from Segmentation   import ProjectionProfiles

## Sets the argparser's arguments
def SetParserArgs(parser):
    ##
    parser.add_argument(
        "path",
        "-p",
        type=str,
        default=None,
        help="The absolute path to the image"
    )

## Main running logic
def main():
    ## Parse the user's passed image path
    argumentParser = argparse.ArgumentParser(description="Basic command-line level OCR")
    SetParserArgs(argumentParser)
    args = argumentParser.parse_args()

    imagePath = args.path.strip()
    if not imagePath:
        print("no path entered.")
        return 1
    
    if imagePath[0] == '"' and imagePath[-1] == '"':
        imagePath = imagePath[1:-1]

    ## Load image and turn it from RGB to grayscale
    greyscaleImage = ImageIO.LoadImage(imagePath)
    if greyscaleImage is not None:
        print(f"loading image from {imagePath}...")
    else:
        print("failed to load image.")
        return 1

    ## Threshold the image and binarize it
    threshold = Otsu.OtsuThreshold(greyscaleImage)
    binarizedImage = Otsu.OtsuBinarize(greyscaleImage, threshold)

    ## Rotate the image based on Hough Lines
    rotationAngle = 1
    while (rotationAngle != 0):
        houghResult = Hough.RunHough(binarizedImage)
        rotationAngle = houghResult[0]
        houghLines = houghResult[1]
        rotatedImage = rotatedImage(binarizedImage, rotationAngle)
        print(f"Estimated rotation angle: {rotationAngle}")
    binarizedImage = rotatedImage

    ## Get the projection profiles for 
    projected_profile = ProjectionProfiles.RunProfiles(binarizedImage)
    img = Image.fromarray(projected_profile)
    img.show()

if __name__ == "__main__": # type: ignore
    main() # type: ignore