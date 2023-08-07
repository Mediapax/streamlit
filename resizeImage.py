# resizeImage

def resizeImage(image, newWidth):
        width, height = image.size
        newHeight = int((newWidth / width) * height)
        resizedImage = image.resize((newWidth, newHeight))
        return resizedImage

def loadImage(image,width) :
    from PIL import Image
    image = Image.open(image)
    resizedImage = resizeImage(image, width)
    return resizedImage