import cv2
import numpy as np
image = cv2.imread('marker_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, _ = detector.detectMarkers(gray)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5,))
marker_length = 0.05
if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    obj_points = np.array([
        [-marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ], dtype=np.float32)

    for i in range(len(ids)):
        img_points = corners[i][0].astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

        if success:
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            print(f"Marker ID: {ids[i][0]}")
            print("Rotation Vector:", rvec.ravel())
            print("Translation Vector:", tvec.ravel())
else:
    print("No markers detected.")
cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()