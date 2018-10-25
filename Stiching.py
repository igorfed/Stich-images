
import argparse
import cv2
import os.path as path
import os
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import time
class Stich(object):
    def __init__(self, imageName, imageOutput, newSize, DetectorType):
        self.imageName = imageName
        self.imageOutput = imageOutput
        self.DetectorType = DetectorType
        self.newSize = newSize
        self.VideoCap()
        self.StichImage()

    def get_cmap(self,n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    def Figure(self, WinName, image):
        cv2.namedWindow(WinName, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WinName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(WinName, image)

    def warpTwoImages(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2_ = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2_), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        result = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin))
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
        return result

    def VideoCap(self):
        self.image0 = cv2.imread(self.imageName[0])
        self.image1 = cv2.imread(self.imageName[1])
        (H, W,C ) = self.image0.shape
        w, h = int(W / self.newSize), int(H / self.newSize)
        self.image0 = cv2.resize(self.image0, (w, h))
        self.image1 = cv2.resize(self.image1, (w, h))

        print("-----------------------")

    def detectAndDescribe(self, image, text):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #orb = cv2.ORB_create(MAX_FEATURES)
        kps, features = self.descriptor.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        print('Features in '+ text + ': ', str(kps.shape))
        return (kps, features)

    def matchKeypoints(self, kps0, kps1, features0, features1):
        matches = self.matcher.knnMatch(features0, features1, 2)
        good = []
        matchesMask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))
                matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)
        if len(good) > 2:
            pts0 = np.float32([kps0[i] for (_, i) in good])
            pts1 = np.float32([kps1[i] for (i, _) in good])
            (H, status) = cv2.findHomography(pts0, pts1, cv2.RANSAC, self.reprojectThres)
            print ('Match is done')
            return (good, H, status, draw_params)
        print ('Matche is faild')
        return None

    def drawMatches(self, kps0, kps1, matches,status):
        # initialize the output visualization image
        (hA, wA) = self.image0.shape[:2]
        (hB, wB) = self.image1.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = self.image0
        vis[0:hB, wA:] = self.image1

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s ==1:
                # draw the match
                ptA = (int(kps0[queryIdx][0]), int(kps0[queryIdx][1]))
                ptB = (int(kps1[trainIdx][0]) + wA, int(kps1[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (randint(0, 255), randint(0, 255), randint(0, 255)), 2)
        print ('drawMatches is done')
        return vis

    def StichImage(self):
        if self.DetectorType == 'SIFT':
            self.descriptor = cv2.xfeatures2d.SIFT_create()
            index_params = dict(algorithm=0, trees=5)
            search_params = dict(checks=100)
            self.ratio = 0.75 # 0.75
            self.reprojectThres = 4.0
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        elif self.DetectorType == 'SURF':
            self.descriptor = cv2.xfeatures2d.SURF_create()
            index_params = dict(algorithm=0, trees=5)
            search_params = dict(checks=50)
            self.ratio = 0.75 # 0.75
            self.reprojectThres = 4.0
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        print('Detector Type: ' + self.DetectorType)

        (kps0, features0) = self.detectAndDescribe(self.image0, 'Left')
        (kps1, features1) = self.detectAndDescribe(self.image1, 'Right')

        M = self.matchKeypoints(kps0, kps1, features0, features1)

        if M is None:
            return None
        (matches, H, status, draw_params) = M
        print("Homography", H)
        result = self.warpTwoImages(self.image1, self.image0, H)
        #matchesMask = [[0, 0] for i in range(len(matches))]
        vis = self.drawMatches(kps0, kps1, matches, status)
        self.Figure("match", vis)
        self.Figure("Result", result)
        self.WriteImage(result, kps0,kps1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def WriteImage(self, image, kps0, kps1):
        print (self.imageOutput)
        S = 'Features in the Left: '+ str(kps0.shape)
        cv2.putText(image, self.DetectorType, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4,8)
        cv2.putText(image, S, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, 8)
        S = 'Features in the Left: '+ str(kps1.shape)
        cv2.putText(image, S, (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, 8)
        cv2.imwrite(self.imageOutput, image)
def imshow(figName, image0, image1, label):
    fig = plt.figure(figName)
    ax0, ax1  = fig.add_subplot(121), fig.add_subplot(122)
    ax0.imshow(image0)
    ax1.imshow(image1)
    ax0.minorticks_on()
    ax0.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
    ax0.grid(which='minor', linestyle=':', linewidth='0.5', color='green')
    ax1.minorticks_on()
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='green')




def main():
    print ("opencv version", cv2.__version__)
    #inputPath = path.abspath(path.join(__file__, "Source"))
    inputPath = os.getcwd() + '/Source'
    #imageInput = [inputPath + "/" + "20181021_112308.jpg", inputPath + "/" +  "20181021_112312.jpg"]
    imageInput = [inputPath + "/" + "4.jpg", inputPath + "/" + "3.jpg"]
    #imageInput = [inputPath + "/" + "1_.jpg", inputPath + "/" + "2_.jpg"]
    #imageInput = [inputPath + "/" + "20180622_194428.jpg", inputPath + "/" + "20180622_194426.jpg"]

    dirName = 'output'
    try:
        os.mkdir(dirName)
        print("Directory: ", dirName, " Created ")
    except FileExistsError:
        print("Directory: ", dirName, " Already exists")
    outPath = path.abspath(path.join(__file__, dirName))
    RS = 2
    s = "SURF"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    outPath = os.getcwd() + '/Output'
    imageOutput = outPath + "/" + "out" + timestr +".jpg"
    parser = argparse.ArgumentParser(description="Do something.")
    parser.add_argument("-L", "--imageL", required=False, type=str, default= imageInput[0] ,help='path to the left camera' )
    parser.add_argument("-R", "--imageR", required=False, type=str, default= imageInput[1] ,help='path to the right camera' )
    parser.add_argument("-O", "--imageOut", required=False, type=str, default= imageOutput ,help='file with a points name' )
    parser.add_argument("-RS", "--r", required=False, type=int, default=RS, help='resize')
    parser.add_argument("-s", "--s", required=False, type=str, default=s, help='SIFT or SURF')

    args = parser.parse_args()
    return args