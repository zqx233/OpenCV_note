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


def math_demo():
    image = cv.imread("D:/Code/1.jpg")  # (666, 999, 3)
    cv.imshow("input", image)
    blank = np.zeros_like(image)
    blank[:, :] = (10, 10, 10)  # 为blank图像像素的通道赋值
    cv.imshow("blank", blank)
    result = cv.add(image, blank)  # 相加
    result = cv.subtract(image, blank)  # 相减
    result = cv.multiply(image, blank)  # 相乘
    result = cv.divide(image, blank)  # 相除
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def print_value(a):
    print(a)


def adjust_lightness_demo():
    image = cv.imread("D:/Code/1.jpg")  # (666, 999, 3)
    cv.namedWindow("input",
                   cv.WINDOW_AUTOSIZE)  # 创建窗口 (窗口名，窗口大小自适应图片大小并且不能更改） WINDOW_NORMAL 用户可以改变这个窗口大小，WINDOW_OPENGL 窗口创建的时候会支持OpenGL
    cv.createTrackbar("lightness", "input", 0, 255, print_value)  # 创建滚动条 （滚动条名字，滚动条属于的窗口名字，滚动条取值范围，回调函数）
    cv.imshow("input", image)
    blank = np.zeros_like(image)
    while True:
        pos = cv.getTrackbarPos("lightness", "input")  # 获取当前滚动条的值
        blank[:, :] = (pos, pos, pos)
        result = cv.add(image, blank)
        cv.imshow("result", result)
        c = cv.waitKey(1)  # 等待用户触发事件时间
        if c == 27:  # 如果按下ESC（ASCII码为27）
            cv.destroyAllWindows()


def adjust_contrast_demo():
    image = cv.imread("D:/Code/1.jpg")  # (666, 999, 3)
    cv.namedWindow("input",cv.WINDOW_AUTOSIZE)  # 创建窗口 (窗口名，窗口大小自适应图片大小并且不能更改）
    cv.createTrackbar("lightness", "input", 0, 255, print_value)  # 创建滚动条 （滚动条名字，滚动条属于的窗口名字，滚动条取值范围，回调函数）
    cv.createTrackbar("contrast", "input", 0, 255, print_value)  # 创建滚动条 （滚动条名字，滚动条属于的窗口名字，滚动条取值范围，回调函数）
    cv.imshow("input", image)
    blank = np.zeros_like(image)
    while True:
        light = cv.getTrackbarPos("lightness", "input")  # 获取当前滚动条的值
        contrast = cv.getTrackbarPos("contrast", "input")/100   # 获取当前滚动条的值
        print("light:", light, ",contrast:", contrast)
        result = cv.addWeighted(image, contrast, blank, 0, light)  # 对两张图片进行不同权重合并
        cv.imshow("result", result)
        c = cv.waitKey(1)  # 等待用户触发事件时间
        if c == 27:  # 如果按下ESC（ASCII码为27）
            cv.destroyAllWindows()


if __name__ == "__main__":
    adjust_contrast_demo()
