from scipy.spatial import distance
import re
import cv2
import numpy as np
import os
import io
import base64
import requests
import json



def greet(name):
    """This function prints the string saying Hello to the name passed as argument.

    Args:
      name (str): The name of the person.

    Returns:
      None
    """
    print(f'Hello, {name}!')


def get_dist(pt1, pt2):
    """This function returns the distance between two points provided as argument.

    Args:
      pt1 (tuple): The tuple containing x, y coordinates of the first point.
      pt2 (tuple): The tuple containing x, y coordiantes of the second point.

    Returns:
      float: The distance between two points.
    """
    dist = distance.cdist((pt1,), (pt2,), 'euclidean')
    return dist[0][0]


class Rectangle:
    """This class contains the methods and attributes to get the information
    like area, perimeter etc from a rectangle which was created using the 
    height and width arguments passed to the constructor.

    Attributes:
      height (float): The height of the rectangle.
      width (float): The width of the rectangle.
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def area(self):
        """This function returns the area of the rectangle.

        Returns:
          float: The area of the rectangle.
        """
        return self.height*self.width

    def perimeter(self):
        """This function returns the perimeter of the rectangle.

        Returns:
          float: The perimeter of the rectangle.
        """
        return 2*(self.height+self.width)



# def get_arrow_coords(img):
#     _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
#     labels, stats = cv2.connectedComponentsWithStats(img, 8)[1:3]

#     # for label in np.unique(labels)[1:]:
#     arrow = labels == np.unique(labels)[1:][0]
#     indices = np.transpose(np.nonzero(arrow))  # y,x
#     dist = distance.cdist(indices, indices, "euclidean")

#     far_points_index = np.unravel_index(np.argmax(dist), dist.shape)  # y,x

#     far_point_1 = indices[far_points_index[0], :]  # y,x
#     far_point_2 = indices[far_points_index[1], :]  # y,x

#     ### Slope
#     arrow_slope = (far_point_2[0] - far_point_1[0]) / (far_point_2[1] - far_point_1[1])
#     arrow_angle = math.degrees(math.atan(arrow_slope))

#     ### Length
#     arrow_length = distance.cdist(
#         far_point_1.reshape(1, 2),
#         far_point_2.reshape(1, 2),
#         "euclidean",
#     )[0][0]

#     ### Thickness
#     x = np.linspace(far_point_1[1], far_point_2[1], 20)
#     y = np.linspace(far_point_1[0], far_point_2[0], 20)
#     line = np.array([[yy, xx] for yy, xx in zip(y, x)])

#     x1, y1 = tuple(line[-1][::-1].astype(int))
#     x2, y2 = tuple(line[0][::-1].astype(int))

#     return x1, y1, x2, y2


def sort_by_num(list_of_strings):
    """This function sorts the list_of_strings based on the number.
    
    Args:
      list_of_strings (list): The list of strings each having a number as substring.
    
    Returns:
      list: The list of sorted strings based on number.
    """
    def comp(string):
        return int(re.findall('\d+', string)[0])
    
    return sorted(list_of_strings, key=comp)


def extract_red(image, save_to):
    """This function applies the upper and lower ranges to the image
    to extract the red color from it and saved the image with extracted
    text in the path specified.

    Args:
      image (numpy.ndarray): The image in the form of numpy array.
      save_to (str): The path where the output image will be saved.

    Returns:
      None
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    e = cv2.inRange(
        hsv, np.array([170, 50, 50]), np.array([180, 155, 150])
    )  ##Qualitive for newRed
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])
    lower_mask = cv2.inRange(hsv, lower1, upper1)
    upper_mask = cv2.inRange(hsv, lower2, upper2)
    full_mask = lower_mask + upper_mask + e
    rmask = full_mask
    _, ir = cv2.threshold(rmask, 100, 220, cv2.THRESH_BINARY_INV)
    cv2.imwrite(save_to, ir)


class Point:
    """This is a class to create a point object.

    Attributes:
      pt (tuple): The tuple containing x and y coordinate of the point.
    """
    def __init__(self, pt):
        self.x, self.y = pt


class Points:
    """This is a class for algebric operations on multiple points.
    
    Attributes:
      pts (list): The list of points. Each point is a tuple containing x and y coordinate.
    """
    def __init__(self, pts):
        self.pts = pts
        
    def are_collinear(self):
        """This function checks if all the points lie on a same line or not.
        
        Returns:
          bool: The boolean value to determine whether points are collinear or not.
        """
        pt1, pt2 = self.pts[:2]
        pts_rest = self.pts[2:]
        x1, y1 = pt1
        x2, y2 = pt2
        for pt in pts_rest:
            x3, y3 = pt
            if y3*(x2-x1) != (y2-y1)*(x3-x1)+y1*(x2-x1):
                return False
        return True
    
    def to_list(self):
        """This function returns the list of the points where each point is a tuple
        containing x and y coordinate.
        
        Returns:
          list: The list of the points where each point is a tuple
          containing x and y coordinate.
        """
        return self.pts
    
    def to_obj(self):
        """This function returns the list of the points where each point is an object
        of the Point class.
        
        Returns:
          list: The list of the points where each point is an object
          of the Point class.
        """
        return [Point(pt) for pt in self.pts]


class Rectangles:
    """This is a class which contains methods to compare two rectangles.

    Attributes:
      l1 (Point object): The Point object that contains coordinates of top-left corner
        of the first rectangle.
      r1 (Point object): The Point object that contains coordinates of bottom-right corner
        of the first rectangle.
      l2 (Point object): The Point object that contains coordinates of top-left corner 
        of the second rectangle.
      r2 (Point object): The Point object that contains coordiantes of bottom-right corner
        of the second rectangle.
    """
    def __init__(self, l1, r1, l2, r2):
        self.l1 = l1
        self.r1 = r1
        self.l2 = l2
        self.r2 = r2

    # Returns true if two rectangles(l1, r1)
    # and (l2, r2) overlap
    def doOverlap(self):
        """This function checks whether two rectangles overlap or not.

        Returns:
          bool: The boolean value to determine whether rectangles overlap or not.
        """
        # To check if either rectangle is actually a line
        # For example : l1 ={-1,0} r1={1,1} l2={0,-1} r2={0,1}

        if (
            self.l1.x == self.r1.x
            or self.l1.y == self.r1.y
            or self.l2.x == self.r2.x
            or self.l2.y == self.r2.y
        ):
            # the line cannot have positive overlap
            return False

        # If one rectangle is on left side of other
        if self.l1.x >= self.r2.x or self.l2.x >= self.r1.x:
            return False

        # If one rectangle is above other
        if self.r1.y >= self.l2.y or self.r2.y >= self.l1.y:
            return False

        return True

    def common_area(self):
        """This function returns the common area shared by the two rectangles.

        Returns:
          float: The common area shared by the two rectangles.
        """
        if not self.doOverlap():
            return 0
        # find max distance of left side of rectangles from origin on x-axis
        top_left_x_of_common_rectangle = max(self.l1.x, self.l2.x)
        # find min distance of top side of rectangles from origin on y-axis
        top_left_y_of_common_rectangle = min(self.l1.y, self.l2.y)
        # find min distance of right side of rectangles from origin on x-axis
        bottom_right_x_of_common_rectangle = min(self.r1.x, self.r2.x)
        # find max distance of bottom side of rectangles from origin on y-axis
        bottom_right_y_of_common_rectangle = max(self.r1.y, self.r2.y)

        width_of_common_rectangle = (
            bottom_right_x_of_common_rectangle - top_left_x_of_common_rectangle
        )
        height_of_common_rectangle = (
            top_left_y_of_common_rectangle - bottom_right_y_of_common_rectangle
        )

        area_of_common_rectangle = (
            width_of_common_rectangle * height_of_common_rectangle
        )

        return area_of_common_rectangle
    
    def doCompleteOverlap(self):
        """This function check whether the two rectangles completely overlap
        or not.

        Returns:
          bool: The boolean value to determine whether the two rectangles completely
          overlap or not.
        """
        # To check if either rectangle is actually a line
        # For example : l1 ={-1,0} r1={1,1} l2={0,-1} r2={0,1}

        if (
            self.l1.x == self.r1.x
            or self.l1.y == self.r1.y
            or self.l2.x == self.r2.x
            or self.l2.y == self.r2.y
        ):
            # the line cannot have positive overlap
            return False

#         # If one rectangle is on left side of other
#         if self.l1.x >= self.r2.x or self.l2.x >= self.r1.x:
#             return False

#         # If one rectangle is above other
#         if self.r1.y >= self.l2.y or self.r2.y >= self.l1.y:
#             return False
        if self.l1.x < self.l2.x and self.l1.y > self.l2.y and self.r1.x > self.r2.x and self.r1.y < self.r2.y:
            # second rectangel completely lies inside the first one
            return True

        return False


def list_files(dir_path="."):
    """This function returns the list of files inside the given directory path.
    
    Args:
      dir_path (str): The path of the directory for listing the files from.
      
    Returns:
      filter: The filter object containing all the files in the given directory path.
    """
    return filter(os.path.isfile, os.listdir(dir_path))


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def vis_vis(response):
    """This function visualizes the response returned by Vision API.
    Just pass the response after converting into dictionary and
    see the magic.

    Args:
      response (dict): A reponse dictionary returned by Vision API.

    Returns:
      None: Just prints the response in tree format.
    """
    pages = response["responses"][0]["fullTextAnnotation"]["pages"]
    for pg, page in enumerate(pages):
        tree_text_page = f'Page {pg}'
        print(tree_text_page)

        blocks = page["blocks"]
        for b, block in enumerate(blocks):
            tree_text_block = f'Block {b}'
            tree_symbol_block = "└──" if b == len(blocks)-1 else "├──"
            print(f'{tree_symbol_block} {tree_text_block}'.rjust(6+len(tree_text_block)))

            paragraphs = block["paragraphs"]
            for p, paragraph in enumerate(paragraphs):
                tree_text_para = f'Paragraph {p}'
                tree_symbol_block = "|" if b != len(blocks)-1 else ""
                tree_symbol_para = "└──" if p == len(paragraphs)-1 else "├──"
                print(tree_symbol_block.rjust(3)+f'{tree_symbol_para} {tree_text_para}'.rjust(9+len(tree_text_para)))

                words = paragraph["words"]
                for w, word in enumerate(words):
                    text = "".join(map(lambda x: x["text"], word["symbols"]))
                    tree_text_word = f'Word {w}: {text}'
                    tree_symbol_word = "└──" if w == len(words)-1 else "├──"
                    tree_symbol_block = "|" if b != len(blocks)-1 else ""
                    tree_symbol_para = "|" if p != len(paragraphs)-1 else ""
                    print(tree_symbol_block.rjust(3)+tree_symbol_para.rjust(6)+f'{tree_symbol_word} {tree_text_word}'.rjust(9+len(tree_text_word)))


def get_vision_api_response(img_pth, API_URL, save_to=None):
    """This function reads the image from the given path and calls the Vision
    API to get the recognized text as JSON response and converts into Python Dict
    before returning it to the user.

    Args:
      img_pth (str): The path of the image to be sent to Vision API.
      API_URL (str): The URL of the Vision API.
      save_to (str, optional): The path where the response is to be saved as JSON.

    Returns:
      Dict: Python Dict having the response returned by Vision API.
    """
    # encode
    img_bgr = cv2.imread(img_pth)
    is_success, buffer = cv2.imencode(".jpg", img_bgr)
    io_buf = io.BytesIO(buffer)

    # # decode
    # decode_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), -1)

    # converting image to base64
    encoded_string = base64.b64encode(io_buf.read())
    encoded_string = str(encoded_string)[2:-1]

    # sending base64 string with request to Vision API
    payload = {
        "requests": [
            {
                "image": {"content": f"{encoded_string}"},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["en-t-i0-handwrit"]}
            }
        ]
    }

    # sending post request and getting response object
    r = requests.post(url=API_URL, data=str(payload))

    data = json.loads(r.text)

    if save_to: json.dump(data, open(save_to, "w"), indent=4)

    return data



class Dir:
    """This class contains the methods that return the list of files and folders name in a
    given directory path.

    Attribures:
      dir_pth (str): The path of the directory.
    """
    def __init__(self, dir_pth):
        self.dir_pth = dir_pth

    def all(self):
        """This function returns the list of all the files and folders name in a directory.

        Returns:
          list: A list that contains the name of all files and folders.
        """
        return os.listdir(self.dir_pth)

    def files(self, sort_by=None):
        """This function returns the list of files name only in a directory.

        Args:
          sort_by (str, optional): The string to apply sorting based on it.
            For example, if the argument passed is "num", it sorts the file names based
            on the first group of continuous number characters present in file names.
            If no argument is provided then it doesn't sort the file names.

        Returns:
          list: A list that contains the name of files only.
        """
        x = [*filter(lambda l: os.path.isfile(f"{self.dir_pth}/{l}"), os.listdir(self.dir_pth))]
        if sort_by == "num":
            x.sort(key=lambda l: int(re.findall('\d+', l)[0]))
        return x

    def dirs(self):
        """This function returns the list of folders only in a directory.
        
        Returns:
          list: A list that contains the name of folders only.
        """
        return [*filter(lambda l: os.path.isdir(f"{self.dir_pth}/{l}"), os.listdir(self.dir_pth))]



def rgb(img):
    """This function takes grayscale or color image(BGR) and converts it into
    RGB colorspace.

    Args:
      img (numpy.ndarray): An image in form of numpy array.

    Returns:
      numpy.ndarray: An image converted into RGB in form of numpy array.
    """
    if img.ndim == 2:
        # IT'S A GRAYSCALE IMAGE
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # IT'S A COLOR IMAGE
        return img[..., ::-1]



def bgr(img):
    """This function takes grayscale or color image(RGB) and converts it into
    BGR colorspace.

    Args:
      img (numpy.ndarray): An image in form of numpy array.

    Returns:
      numpy.ndarray: An image converted into BGR in form of numpy array.
    """
    if img.ndim == 2:
        # IT'S A GRAYSCALE IMAGE
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        # IT'S A COLOR IMAGE
        return img[..., ::-1]