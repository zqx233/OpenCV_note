import cv2 as cv
import numpy as np


def read_demo():
    image = cv.imread("src/1.jpg")  # 读取图片，对于OpenCV来说，RGB通道的图片读取到的色彩空间为BGR通道
    gary = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转换为灰度
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # 转换为hsv
    hsv = cv.cvtColor(image, cv.COLOR_HSV2BGR)  # hsv转换为BGR，H通道范围为0-180，其余为0-255
    cv.imshow("gary", gary)
    cv.imshow("hsv", hsv)
    cv.waitKey(0)  # 等待按键，否则窗口闪一下就结束运行了
    cv.destroyAllWindows()  # 关闭所有窗口


def mat_demo():
    image = cv.imread("src/1.jpg")  # (666, 999, 3)
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
    image = cv.imread("src/1.jpg")  # (666, 999, 3)
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
    image = cv.imread("src/1.jpg")  # (666, 999, 3)
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
    image = cv.imread("src/1.jpg")  # (666, 999, 3)
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
    image = cv.imread("src/1.jpg")  # (666, 999, 3)
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)  # 创建窗口 (窗口名，窗口大小自适应图片大小并且不能更改）
    cv.createTrackbar("lightness", "input", 0, 255, print_value)  # 创建滚动条 （滚动条名字，滚动条属于的窗口名字，滚动条取值范围，回调函数）
    cv.createTrackbar("contrast", "input", 0, 255, print_value)  # 创建滚动条 （滚动条名字，滚动条属于的窗口名字，滚动条取值范围，回调函数）
    cv.imshow("input", image)
    blank = np.zeros_like(image)
    while True:
        light = cv.getTrackbarPos("lightness", "input")  # 获取当前滚动条的值
        contrast = cv.getTrackbarPos("contrast", "input") / 100  # 获取当前滚动条的值
        print("light:", light, ",contrast:", contrast)
        result = cv.addWeighted(image, contrast, blank, 0, light)  # 对两张图片进行不同权重合并
        cv.imshow("result", result)
        c = cv.waitKey(1)  # 等待用户触发事件时间
        if c == 27:  # 如果按下ESC（ASCII码为27）
            cv.destroyAllWindows()


def keys_demo():
    image = cv.imread("src/1.jpg")  # (666, 999, 3)
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)  # 创建窗口 (窗口名，窗口大小自适应图片大小并且不能更改）
    cv.imshow("input", image)
    while True:
        c = cv.waitKey(1)  # 等待用户触发事件时间
        if c == 49:  # 按键1
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            cv.imshow("result", gray)
        if c == 50:  # 按键2
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            cv.imshow("result", hsv)
        if c == 27:  # 如果按下ESC（ASCII码为27）
            cv.destroyAllWindows()


def color_table_demo():
    colormap = [
        cv.COLORMAP_AUTUMN,
        cv.COLORMAP_BONE,
        cv.COLORMAP_JET,
        cv.COLORMAP_WINTER,
        cv.COLORMAP_RAINBOW,
        cv.COLORMAP_OCEAN,
        cv.COLORMAP_SUMMER,
        cv.COLORMAP_SPRING,
        cv.COLORMAP_COOL,
        cv.COLORMAP_PINK,
        cv.COLORMAP_HOT,
        cv.COLORMAP_PARULA,
        cv.COLORMAP_MAGMA,
        cv.COLORMAP_INFERNO,
        cv.COLORMAP_PLASMA,
        cv.COLORMAP_VIRIDIS,
        cv.COLORMAP_CIVIDIS,
        cv.COLORMAP_TWILIGHT,
        cv.COLORMAP_TWILIGHT_SHIFTED]
    image = cv.imread("src/1.jpg")  # (666, 999, 3)
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)  # 创建窗口 (窗口名，窗口大小自适应图片大小并且不能更改）
    cv.imshow("input", image)
    index = 0
    while True:
        dst = cv.applyColorMap(image, colormap[index % 19])
        index += 1
        cv.imshow("result", dst)
        c = cv.waitKey(2000)  # 等待用户触发事件时间
        if c == 27:  # 如果按下ESC（ASCII码为27）
            break
    cv.destroyAllWindows()


def bitwise_demo():
    b1 = np.zeros((400, 400, 3), dtype=np.uint8)
    b1[:, :] = (255, 0, 255)
    b2 = np.zeros((400, 400, 3), dtype=np.uint8)
    b2[:, :] = (0, 255, 255)
    cv.imshow("b1", b1)
    cv.imshow("b2", b2)
    dst1 = cv.bitwise_and(b1, b2)  # 与运算
    dst2 = cv.bitwise_or(b1, b2)  # 或运算
    cv.imshow("bitwise_and", dst1)
    cv.imshow("bitwise_or", dst2)
    cv.waitKey(0)


def channels_split_demo():
    b1 = cv.imread("src/1.jpg")
    print(b1.shape)
    cv.imshow("input", b1)
    cv.imshow("b1", b1[:, :, 2])
    mv = cv.split(b1)
    mv[0][:, :] = 255  # 将第一通道的值都改为255
    result = cv.merge(mv)  # 将第一通道合并到图像中
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def color_space_demo():
    b1 = cv.imread("src/1.jpg")
    print(b1.shape)
    cv.imshow("input", b1)
    hsv = cv.cvtColor(b1, cv.COLOR_BGR2HSV)
    print(hsv)
    cv.imshow("hsv", hsv)
    mask = cv.inRange(hsv, (0, 30, 46), (25, 255, 255))  # 将图像中颜色在第二个参数和第三个参数范围内的颜色设为0，其余为1
    # cv.bitwise_not(mask, mask)  # 自身取反
    result = cv.bitwise_and(b1, b1, mask=mask)  # mask：指定操作范围
    cv.imshow("mask", mask)
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def pixel_stat_demo():
    # b1 = cv.imread("src/1.jpg")
    b1 = np.zeros((512, 512, 3), dtype=np.uint8)
    print(b1.shape)
    cv.imshow("input", b1)
    means, dev = cv.meanStdDev(b1)  # 均值，方差
    print("means:", means, "dev:", dev)
    cv.waitKey(0)
    cv.destroyAllWindows()


def drawing_demo():
    b1 = np.zeros((512, 512, 3), dtype=np.uint8)
    cv.rectangle(b1, (50, 50), (400, 400), (0, 0, 255), 2, 8, 0)
    cv.circle(b1, (225, 225), 175, (255, 0, 0), -1, 8, 0)
    cv.line(b1, (50, 50), (400, 400), (0, 255, 0), 2, 8, 0)
    # b1[:,:] = 0
    cv.putText(b1, "This is a text", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, 8)
    cv.imshow("result", b1)
    cv.waitKey(0)
    cv.destroyAllWindows()


def random_color_demo():
    b1 = np.zeros((512, 512, 3), dtype=np.uint8)
    while True:
        xx = np.random.randint(0, 512, 2, dtype=np.int)
        yy = np.random.randint(0, 512, 2, dtype=np.int)
        bgr = np.random.randint(0, 255, 3, dtype=np.int32)
        cv.line(b1, (xx[0], yy[0]), (xx[1], yy[1]), (np.int(bgr[0]), np.int(bgr[1]), np.int(bgr[2])), 1, 8, 0)
        cv.imshow("result", b1)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()


def polyline_drawing_demo():
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    pts = np.array([[100, 100], [300, 100], [450, 200], [320, 450], [80, 400]])
    # cv.fillPoly(canvas, [pts], (255, 0, 255), 8, 0)
    # cv.polylines(canvas, [pts], True, (0, 0, 255), 2, 8, 0)
    cv.drawContours(canvas, [pts], 1, (255, 0, 0), -1)  # 可实现前两个函数功能，最后一个参数为-1时填充，正数时不填充
    cv.imshow("polyline", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()


b1 = cv.imread("D:/Code/py/OpenCV/src/1.jpg")
img = np.copy(b1)
x1 = -1
x2 = -1
y1 = -1
y2 = -1


def mouse_drawing(event, x, y, flags, param):
    global x1, x2, y1, y2
    if event == cv.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
    if event == cv.EVENT_MOUSEMOVE:
        if x1 < 0 or y1 < 0:
            return
        x2 = x
        y2 = y
        b1[:, :, :] = img[:, :, :]
        cv.rectangle(b1, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)
    if event == cv.EVENT_LBUTTONUP:
        if x1 < 0 or y1 < 0:
            return
        x2 = x
        y2 = y
        b1[:, :, :] = img[:, :, :]
        cv.rectangle(b1, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)
        x1 = -1
        x2 = -1
        y1 = -1
        y2 = -1


def mouse_demo():
    cv.namedWindow("mouse_demo", cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("mouse_demo", mouse_drawing)
    while True:
        cv.imshow("mouse_demo", b1)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()


def norm_demo():
    image = cv.imread("D:/Code/py/OpenCV/src/1.jpg")
    cv.namedWindow("norm_demo", cv.WINDOW_AUTOSIZE)
    result = np.zeros_like(np.float32(image))  # 将图片转化为浮点型
    cv.normalize(np.float32(image), result, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)  # 第二种方法将图片转换为浮点型
    cv.imshow("norm_demo", np.uint8(result * 255))  # 将图片数据类型还原
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    norm_demo()
