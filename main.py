import cv2
# import numpy as np
# import PIL
# from PIL import Image
import train


image = cv2.imread("cccd10.jpg")
image = cv2.resize(image, (1280, 960))

# resize image smaller
# scale_percent = 50
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dsize = (width, height)
# image = cv2.resize(image, dsize)
# print(image.shape[1], image.shape[0])


# đổi dang ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ảnh nhị phân
thresh_img = cv2.threshold(
    gray, 135, 255, cv2.THRESH_BINARY)[1]

# làm mờ
gray = cv2.medianBlur(gray, 5)


# bọc viền
edges = cv2.Canny(gray, 50, 180)


# Hiển thị ảnh kết quả
cv2.imshow('Gray', gray)
cv2.imshow('ed', edges)
cv2.imshow('thresh_img', thresh_img)

# Tìm các đường viền
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Tìm hình chữ nhật bao quanh đối tượng
i = 0
arr_ID = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if ((400 < y < 470) and 950 > x > 500 and h > 30):
        if (w > 60):
            arr_ID.append([x, y, w//2, h])
            cv2.rectangle(image, (x, y), (x+w//2, y+h), (0, 255, 0), 2)
            arr_ID.append([x+w//2, y, w//2, h])
            cv2.rectangle(image, (x+w//2, y), (x+w, y+h), (0, 255, 0), 2)
            continue
        arr_ID.append([x, y, w, h])
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.rectangle(image, (500, 300), (1200, 470), (0, 255, 0), 2)


# sắp xếp thứ tự ảnh lần lượt từ trái qua phải
z = 0
arr_ID.sort(key=lambda x: x[0])
for i in arr_ID:
    x, y, w, h = i
    cropped_image = thresh_img[y:y+h, x:x+w]
    cropped_image1 = cv2.resize(cropped_image, (20, 20))
    cv2.imwrite("img/"+str(z)+".jpg", cropped_image1)
    z += 1


cv2.imshow('Bounding Rectangle', image)
train.main()

# end
cv2.waitKey(0)
cv2.destroyAllWindows()
