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
from Recognition    import Recognize
from model          import Predict

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "characterCnn.npz")

## Sets the argparser's arguments
def SetParserArgs(parser):
    ##
    parser.add_argument(
        "-path",
        "-p",
        type=str,
        default=None,
        help="The absolute path to the image"
    )
    parser.add_argument(
        "-model",
        "-m",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained CNN weights"
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

    Image.fromarray(greyscaleImage).show()

    ## Threshold the image and binarize it
    threshold = Otsu.OtsuThreshold(greyscaleImage)
    binarizedImage = Otsu.OtsuBinarize(greyscaleImage, threshold)

    ## Rotate the image based on Hough Lines
    rotationAngle = 1
    while (rotationAngle != 0):
        houghResult = Hough.RunHough(binarizedImage)
        rotationAngle = houghResult[0]
        houghLines = houghResult[1]
        rotatedImage = Rotation.RotateImage(binarizedImage, rotationAngle)
        print(f"Estimated rotation angle: {rotationAngle}")
        binarizedImage = rotatedImage
        ## for show
        Image.fromarray(houghLines).show()
        
    ## Get the projection profiles for 
    boundingCoordinates = ProjectionProfiles.RunProfiles(binarizedImage)

    ## Recognize each segmented character with the CNN
    if not os.path.exists(args.model):
        print(f"no trained model at {args.model}, run training/Train.py first.")
        return 1

    predictor = Predict.CharacterPredictor(args.model)
    text = Recognize.RecognizeText(binarizedImage, boundingCoordinates, predictor)
    print("recognized text:")
    print(text)

if __name__ == "__main__": # type: ignore
    main() # type: ignore