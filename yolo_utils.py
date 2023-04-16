def get_bbox(prediciton):
    """
    Returns the dimensions and position of a bounding box predicted by YOLOv8
    """
    try:
        x, y, w, h = prediciton[0].boxes.xywh[0].squeeze().tolist()
        return x, y, w, h
    except:
        return (0, 0, 0, 0)


def get_circle(bbox):
    """
    Get the centroid and the radius of bbox given
    its upper left corner position, its width and its height
    """
    x0, y0, w, h = bbox

    centr_x = int(x0 + w / 2)
    centr_y = int(y0 + h / 2)
    radius = int((w + h) / 4)

    return centr_x, centr_y, radius
