from PIL import Image
import glob as g


def crop_image():

    list_image = g.glob('data_set_png/train/avancado/*.png')
    list_image1 = g.glob('data_set_png/train/normal/*.png')

    cont = 0

    for i, value in enumerate(list_image):
        im = Image.open(value)
        im1 = im.resize((720, 470), Image.NEAREST)
        half_the_width = im1.size[0] / 2
        half_the_height = im1.size[1] / 2
        im2 = im1.crop(
            (
                half_the_width - 230,
                half_the_height - 230,
                half_the_width + 230,
                half_the_height + 230,
            )
        )
        im2 = im2.convert('L')
        im2.save('/home/deusimar/Pictures/crop/crop-train/avancado/image%d.png' % cont)

        cont = cont + 1

        im = Image.open(list_image1[i])
        im1 = im.resize((720, 470), Image.NEAREST)
        half_the_width = im1.size[0] / 2
        half_the_height = im1.size[1] / 2
        im2 = im1.crop(
            (
                half_the_width - 230,
                half_the_height - 230,
                half_the_width + 230,
                half_the_height + 230,
            )
        )
        im2 = im2.convert('L')
        im2.save('/home/deusimar/Pictures/crop/crop-train/normal/image%d.png' % cont)

        cont = cont + 1

        if i == 5250:
            break


def conference():

    import cv2
    list_image = g.glob('crop/test/*.png')

    for i, value in enumerate(list_image):
        img = cv2.imread(value)
        print(img.shape[0], 'x', img.shape[1])


if __name__ == '__main__':
    crop_image()
    # conference()