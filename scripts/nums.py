# for making random data for a test set of 1000 digits, to be used for testing the OCR system on random data
import random as rand

for i in range(0, 1000):
    print(rand.randint(0, 9), end="")