import cv2
import numpy as np
import matplotlib.pyplot as plt

def cartoonify_image_lut(image_path):
    # Step 1: Read the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (600, 600))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Step 2: Create a Look-Up Table (LUT) for posterization
    def create_lut():
        # Creating a LUT that reduces the color levels
        levels = 4  # Adjust this for more or fewer color levels
        lut = np.zeros(256, dtype=np.uint8)
        step = 256 // levels
        for i in range(256):
            lut[i] = (i // step) * step + step // 2
        return lut
    
    lut = create_lut()
    
    # Step 3: Apply the LUT to each channel of the image
    img_lut = cv2.LUT(img, lut)
    
    # Step 4: Convert to grayscale and apply median blur for edge detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    
    # Step 5: Detect edges using adaptive thresholding
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, blockSize=3, C=2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)

    # Step 6: Combine the edges and LUT-transformed image
    img_cartoon = cv2.bitwise_and(img_lut, img_edge)
    
    # Save and display the result
    fig1=plt.figure(figsize=(10,7))
    fig1.add_subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original")
    fig1.add_subplot(1,2,2)
    plt.imshow(img_cartoon)
    plt.axis('off')
    plt.title("median blur")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
cartoonify_image_lut("C:/Users/yazhi/Desktop/CV_Builder_Series/projects/Obama.png")
