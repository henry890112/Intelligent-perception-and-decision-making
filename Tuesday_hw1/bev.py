import cv2
import numpy as np
from math import tan

points = []
new_pixels = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)

        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        ### TODO ###

        ##先求出focal length
        fov_theta = 90
        f = (512/2)*(1/tan(np.radians(fov_theta/2)))
        
        ############Test fail##############
        # theta = np.radians(theta)  #theta原本的已經從下方main中匯入
        # c, s = np.cos(theta), np.sin(theta)
        # T = np.array([[1,0,0,0],
        #             [0,c,-s,0], 
        #             [0,s,c,1.5],
        #             [0,0,0,1]])
        # #print(T) 
        
        # I = np.array([[1,0,0,0],
        #             [0,1,0,0], 
        #             [0,0,1,0]])
        # #print(I)

        # K = np.array([[f,0,256],
        #             [0,f,256],         
        #             [0,0,1]])
        # #print(K)

        # K_inverse = np.linalg.inv(K)

        # ####input 2d pixel
        # for i in range(len(points)):

        #     u = points[i][0]
        #     v = points[i][1]
        #     Z = 1.5
        #     A = np.array([[u/Z,v/Z,1.5]])
        #     A = A.transpose()
        #     #print(A)
            
        #     BEV_true = K_inverse @ A
        #     BEV_true = np.append(BEV_true, 1)  #增加一個數字1
        #     BEV_true = np.expand_dims(BEV_true, axis = 0)  #增加一個維度，才能轉置
        #     BEV_true = BEV_true.transpose()
        #     #print(BEV_true)

        #     fov_points = K@I@T@BEV_true

        #     new_pixels.append([int((fov_points[0][0]/Z)+256), int((fov_points[1][0]/Z)+256)])

        theta = np.radians(theta)  #theta原本的已經從下方main中匯入
        c, s = np.cos(theta), np.sin(theta)
        T = [[1,0,0,0],
            [0,c,-s,0], 
            [0,s,c,-1.5],
            [0,0,0,1]]

        for i in range(len(points)):
            ####input 2d pixel
            u = points[i][0]-256
            v = points[i][1]-256
            Z = 1.5
            A = [[u/f*Z,v/f*Z,Z]]
            A = np.append(A, 1)  #增加一個數字1
            A = np.expand_dims(A, axis = 0)  #增加一個維度，才能轉置
            A = A.transpose()

            new_XYZ = T@A
            new_pixels.append([int(-(new_XYZ[0][0]/new_XYZ[2][0]*f)+256), int(new_XYZ[1][0]/new_XYZ[2][0]*f+256)])

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)
    #print(points)
        
    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

    return points

if __name__ == "__main__":

    roll_ang = 90

    front_rgb = "./front/front_view_path1.png"
    top_rgb = "./top/top_view_path1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("pick point", points)
    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=roll_ang)
    print("projection point:", new_pixels)
    projection.show_image(new_pixels)

print("Done for projection from bev to front image")