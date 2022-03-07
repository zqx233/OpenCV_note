import cv2 as cv
import numpy as np


def read_demo():
    image = cv.imread("D:/Code/1.jpg")  # 读取图片，对于OpenCV来说，RGB通道的图片读取到的色彩空间为BGR通道
    gary = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转换为灰度
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # 转换为hsv
    hsv = cv.cvtColor(image, cv.COLOR_HSV2BGR)  # hsv转换为BGR，H通道范围为0-180，其余为0-255
    cv.imshow("gary", gary)
    cv.imshow("hsv", hsv)
    cv.waitKey(0)  # 等待按键，否则窗口闪一下就结束运行了
    cv.destroyAllWindows()  # 关闭所有窗口


def mat_demo():
    image = cv.imread("D:/Code/1.jpg")  # (666, 999, 3)
    print(image.shape)  # 输出图片的高，宽，通道 h, w, c
    roi = image[60:200, 50:100, :]  # 将图片一部分区域设置为roi区域
    blank = np.zeros_like(image)  # 创建一个和图片一样大小的空白图片
    blank[60:200, 50:100, :] = image[60:200, 60:110, :]  # 将图片的一部分拷贝到空白图片中
    # 复制图像
    # blank = np.copy(image)
    # blank = image
    cv.imshow("blank", blank)
    cv.imshow("image", image)
    cv.imshow("roi", roi)
    cv.waitKey(0)
    cv.destroyAllWindows()


def pixel_demo():
    image = cv.imread("D:/Code/1.jpg")  # (666, 999, 3)
    cv.imshow("image", image)
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            b, g, r = image[row, col]
            image[row, col] = (255 - b, 255 - g, 255 - r)
    cv.imshow("result", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    pixel_demo()
