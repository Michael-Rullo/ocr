# Jigsaw OCR
**Done by: Michael Rullo** *(if anyone sees this — including myself)*

### My Experience Thus Far:
5/1
    The CLI is proving to be messier than i had hoped, wanted to use switch-case instead of the if blocks but whatever i will fix this in a future update
    So far we can load an image from its absolute path and get it into a grayscale array (0 - 255) and i have started reading about Otsu binarization. Otsu wikipedia was absolutely too many words, watched a youtube video which made it seem like absolute child's play. Pretty cool, and its implemented meaning we can move onto phase 3.

5/2
    I have done some research for the rotation utility and deskewing, unsure if i will implement today as 
    have been busy. I am attempting to implement the image rotation, but get a lot of lines and noise on the rotation. I am guessing that the fix is a k-nearest neighbor after the rotation. as it turns out i was wrong. I was doing calculations wrong and stretching the image rather than rotating it. The fix is simple but i should also inverse-map form the rotated image to the original to select pixels rather than normal imaging becuase this eliminates the forward mapping(which i thought about but wasnt sure how to implement the fix). Cool to learn once again. I will attempt to fix this tomorrow.

5/3
    I got the rotation utility down by getting the inverse rotation matrix. had to dust off the cobwebs on some linear algebra. The rotation utility is decent but the line variance calculator isnt getting the result that i would hope for so i will have to fix tmr.

5/5
    Working on houghs at work, i feel like my head is spinning it is so complex

5/7
    Segmentation is hell and this is only rotation holy fuh

5/11
    Segmentation hell, trying to fix houghs and make the code cleaner

5/30
    Finally getting back to this project. I will have to clean up the houghs code again its looking very messy. The fix was two feedback loops, one for the number of lines and the second that recomputes houghs until no rotation is needed. This solves the issue of an initial bad rotation or two. Now onto line segmentation.