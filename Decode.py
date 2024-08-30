import math
import cv2
import numpy as np
import Misc
import time

def vertex_sort(vertices):  # order vertex in a clockwise manner
    if vertices.ndim == 3:
        vertices = vertices.reshape((vertices.shape[0], vertices.shape[2]))

    centroid = np.mean(vertices, axis=0)

    def angle_from_center(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

    angles = np.array([angle_from_center(point) for point in vertices])
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]

    # sorted_vertices = sorted(vertices, key=angle_from_center)
    return sorted_vertices


def vertex_clean_e4v(vertices):  # eliminate the vertex that far from other vertices
    sorted_vertices = vertex_sort(vertices)

    def angle(a, b, c):
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot_product = ba[0] * bc[0] + ba[1] * bc[1]

        mba = np.sqrt(pow(ba[0], 2) + pow(ba[1], 2))
        mbc = np.sqrt(pow(bc[0], 2) + pow(bc[1], 2))
        cos = dot_product / (mba * mbc)

        return math.acos(cos) * 180 / 3.1415926

    for i in range(0, 4):
        pt = i
        lpt = Misc.ring_buffer(i - 1, 4)
        npt = Misc.ring_buffer(i + 1, 4)
        if angle(sorted_vertices[lpt], sorted_vertices[pt], sorted_vertices[npt]) < 80:
            min_index = i
            temp = [sorted_vertices[Misc.ring_buffer(min_index + 1, 4)], sorted_vertices[Misc.ring_buffer(min_index - 1, 4)]]
            center = np.mean(temp, axis=0)
            vector = center - sorted_vertices[Misc.ring_buffer(min_index - 2, 4)]
            new_vertex = center + vector
            for vertex in vertices:  # Calculate new vertex
                if vertex[0][0] == sorted_vertices[i][0] and vertex[0][1] == sorted_vertices[i][1]:
                    vertex[0][0] = new_vertex[0]
                    vertex[0][1] = new_vertex[1]
            return vertices
    return vertices


def vertex_clean_l4v(vertices):
    def cal_distance(point1, point2):
        x2 = pow(point1[0] - point2[0], 2)
        y2 = pow(point1[1] - point2[1], 2)
        return np.sqrt(x2 + y2)

    def cal_newVertex(v1, v2, v3):
        temp = [v1, v3]
        center = np.mean(temp, axis=0)
        vector = center - v2
        new_vertex = center + vector
        return new_vertex

    distances = []

    # Calculate the average distance between vertices for every vertex
    for i in range(0, len(vertices)):
        distance = 0
        for vertex in vertices:
            d = cal_distance(vertices[i][0], vertex[0])
            distance += d
        distances.append([distance / len(vertices), i])

    # Only keep the last three vertice have the most average distances for vertices > 4
    distances.sort()

    delete_list = []
    while len(distances) > 4:
        delete_list.append(distances[0][1])
        distances.pop(0)
    delete_list.sort()

    # Find out vertices on the diagonal
    d1 = cal_distance(vertices[distances[1][1]][0], vertices[distances[2][1]][0])
    d2 = cal_distance(vertices[distances[1][1]][0], vertices[distances[3][1]][0])
    d3 = cal_distance(vertices[distances[3][1]][0], vertices[distances[2][1]][0])

    if d1 > d2:
        if d1 > d3:  # d1 max
            o = [1, 3, 2]
        else:  # d3 max
            o = [2, 1, 3]
    else:
        if d2 > d3:  # d2 max
            o = [1, 2, 3]
        else:  # d3 max
            o = [2, 1, 3]

    # Calculate the new vertex
    new_vertex = cal_newVertex(vertices[distances[o[0]][1]][0], vertices[distances[o[1]][1]][0],
                               vertices[distances[o[2]][1]][0])
    vertices[distances[0][1]][0][0] = new_vertex[0]
    vertices[distances[0][1]][0][1] = new_vertex[1]
    vertices = np.delete(vertices, delete_list, 0)
    return vertices


def sum_neighborhood(img, points, range, vertex_center):
    rows, cols = img.shape
    # Calculate the range of rows and columns to sum
    sums = []
    coordinates = []
    for (x, y) in points:
        vector = [x - vertex_center[0], y - vertex_center[1]]
        nx = int(vertex_center[0] + 0.8 * vector[0])
        ny = int(vertex_center[1] + 0.8 * vector[1])
        row_min = max(ny - range, 0)
        row_max = min(ny + range + 1, rows)
        col_min = max(nx - range, 0)
        col_max = min(nx + range + 1, cols)

        # Extract the neighborhood
        neighborhood = img[row_min:row_max, col_min:col_max]
        sums.append(np.sum(abs(neighborhood - 128)))
        coordinates.append([nx, ny])
    min_index = np.argmin(sums)
    coordinates.pop(min_index)
    return coordinates


def find_center(contours):
    centers = []

    # Remove the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)  # Index of the largest contour

    index = 0
    # Calculate contour centroid
    for contour in contours:
        # Calculate moments for each contour
        if index != max_index:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                # Calculate centroid coordinates
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append([cx, cy])
        index += 1
    return centers


def find_indicator(vertices, centers, input, vertex_center):
    indicator_coordinates = []
    for (x, y) in centers:
        res = cv2.pointPolygonTest(vertices, (x, y), False)
        if res >= 0:
            indicator_coordinates.append((x, y))

    if len(indicator_coordinates) != 3:
        indicator_coordinates = sum_neighborhood(input, vertices, 5, vertex_center)

    def cal_distance(point1, point2):
        x2 = pow(point1[0] - point2[0], 2)
        y2 = pow(point1[1] - point2[1], 2)
        return np.sqrt(x2 + y2)

    distances = []
    for i in range(0, 3):
        i1 = Misc.ring_buffer(i - 1, 3)
        i2 = Misc.ring_buffer(i + 1, 3)
        d1 = cal_distance(indicator_coordinates[i1], indicator_coordinates[i])
        d2 = cal_distance(indicator_coordinates[i2], indicator_coordinates[i])
        distances.append(d1 + d2)
    index = np.argmin(distances)
    middle_vertex = indicator_coordinates[index]

    ds = []
    for i in range(0, 4):
        ds.append(cal_distance(vertices[i], middle_vertex))
    index = np.argmin(ds)
    # box = np.array(vertices[index], vertices[Misc.ring_buffer(index + 1, 4)],
    # vertices[Misc.ring_buffer(index + 2, 4)],
    # vertices[Misc.ring_buffer(index + 3, 4)])
    box = np.array([[vertices[index][0], vertices[index][1]],
                    [vertices[Misc.ring_buffer(index + 1, 4)][0], vertices[Misc.ring_buffer(index + 1, 4)][1]],
                    [vertices[Misc.ring_buffer(index + 2, 4)][0], vertices[Misc.ring_buffer(index + 2, 4)][1]],
                    [vertices[Misc.ring_buffer(index + 3, 4)][0], vertices[Misc.ring_buffer(index + 3, 4)][1]]])

    return indicator_coordinates, box


def outline(camera_output, model_output, scale_factor, draw_vertices, draw_indicator, blur, save):

    if camera_output is not None:
        threshold_value = 80
        if blur:
            threshold_value = 160
            model_output = cv2.GaussianBlur(model_output, (3, 3), 0)
        # Denoise detected area
        _, cleaned_model_output = cv2.threshold(model_output, threshold_value, 255, cv2.THRESH_BINARY)

        # Find the contour of the detected area
        contour, _ = cv2.findContours(cleaned_model_output, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        centers = find_center(contour)
        marker_outline = max(contour, key=cv2.contourArea)

        approx = cv2.approxPolyDP(marker_outline, 15, True)

        # Find similar polygons
        if len(approx) >= 4:
            if len(approx) > 4:
                approx = vertex_clean_l4v(approx)
            else:
                approx = vertex_clean_e4v(approx)
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            box = vertex_sort(box)
            center = np.mean(box, axis=0)

            aa = cv2.cvtColor(camera_output, cv2.COLOR_GRAY2RGB)
            if draw_vertices:
                for (x, y) in box:
                    cv2.circle(aa, (x, y), 2, (0, 255, 0), 2)

            # Detect the direction indicator positions
            indicators, box = find_indicator(box, centers, camera_output, center)

            if draw_indicator:
                for (x, y) in indicators:
                    cv2.circle(aa, (x, y), 5, (0, 0, 255), 1)

            # Extend output range to get full info of the marker
            expanded_box = []
            for point in box:
                vector = point - center
                expanded_point = center + vector * scale_factor
                expanded_box.append(expanded_point)
            expanded_box = np.int0(expanded_box)

            polygon_contour = expanded_box.reshape((-1, 1, 2))  # Only used for using event stream files

            # marker = warp(camera_output, expanded_box, scale_factor, smooth)

            return expanded_box, True, polygon_contour
        else:
            return 0, False, 0


def warp(camera_output, vertex, scale_factor, smooth):
    # Get the marker from the camera output according to prediction result of NN
    mask = np.zeros((260, 346, 1), dtype=np.uint8)
    cv2.fillPoly(mask, [vertex], (255, 255, 255))
    result = cv2.bitwise_and(mask, camera_output, mask)

    # Homography warping according to the detected direction indicators
    width = int(scale_factor * 100)
    warped_img_corner = np.array([[0, 0], [width, 0], [width, width], [0, width]], dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(vertex, warped_img_corner)
    marker = cv2.warpPerspective(result, homography_matrix, (width, width))

    if smooth:
        marker = Misc.image_smooth(marker)

    return marker


def divide(marker):

    if marker is not None:
        _, white_mask = cv2.threshold(marker, 148, 255, cv2.THRESH_BINARY)
        _, black_mask = cv2.threshold(marker, 108, 255, cv2.THRESH_BINARY_INV)

        dark_gray_image = np.full_like(marker, 255)

        on = cv2.bitwise_and(marker, dark_gray_image, mask=white_mask)
        off = cv2.bitwise_or(marker, dark_gray_image, mask=black_mask)

        return on, off


def extract(index, input, optimized_vertex, basic_vertex, basic_contour, prd_res, scale_factor, indicator):
    # Extract content in the detected area
    mask = np.zeros((260, 346, 1), dtype=np.uint8)
    cv2.fillPoly(mask, [optimized_vertex], (255, 255, 255))
    result = cv2.bitwise_and(mask, input, mask)

    # Warp according to the homography
    width = int(scale_factor * 100)
    warped_img_corner = np.array([[0, 0], [width, 0], [width, width], [0, width]], dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(optimized_vertex, warped_img_corner)
    corrected_image = cv2.warpPerspective(result, homography_matrix, (width, width))
    corrected_image = Misc.image_smooth(corrected_image)
    # cv2.imwrite("./frames/detected_no_scale/" + str(index) + ".png", corrected_image)
    cv2.imwrite("./frames/Test3/res/" + str(index) + ".png", corrected_image)

    Misc.divide(corrected_image, index)

    prd_res = cv2.cvtColor(prd_res, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(prd_res, basic_contour, -1, (0, 255, 0), 1)
    cv2.drawContours(prd_res, basic_vertex, -1, (0, 255, 0), 10)
    # cv2.polylines(prd_res, [basic_vertex], isClosed=True, color=(255, 0, 0), thickness=1)
    cv2.polylines(prd_res, [optimized_vertex], isClosed=True, color=(0, 0, 255), thickness=1)
    # cv2.imwrite("./frames/prediction/Contours3/" + str(index) + ".png", prd_res)

    # cv2.imwrite("./frames/indicator/" + str(index) + ".png", indicator)


def refine_l1(events, is_on, path, save_l1, save_l2, save_decode):

    if events is not None:
        if save_l1:
            aa = cv2.cvtColor(events, cv2.COLOR_GRAY2RGB)

        sum_rows = np.sum(events, axis=1)
        sum_columns = np.sum(events, axis=0)

        white_threshold = 45
        mid_threshold = 35
        black_threshold = 20

        hmove = False
        vmove = False

        # Horizontal line for last segmentation
        done = False
        lhcheck = 0
        for i in range(0, len(sum_rows), 1):
            if sum_rows[i] >= 255 * mid_threshold:
                if i >= 66:
                    if save_l1:
                        cv2.line(aa, (0, 69), (69, 69), (0, 0, 255), 1)
                    vmove = True
                    lhcheck = 69
                    done = True
                    break
                else:
                    for j in range(1, 4):
                        if sum_rows[i + j] < 255 * black_threshold:
                            if sum_rows[i] >= 255 * white_threshold:
                                if save_l1:
                                    cv2.line(aa, (0, i + j - 1), (69, i + j - 1), (0, 0, 255), 1)
                                vmove = True
                                done = True
                                lhcheck = i + j - 1
                                break
                            else:
                                lhcheck = i + j - 1
                                if i < 5:
                                    if save_l1:
                                        cv2.line(aa, (0, i + j - 1), (69, i + j - 1), (0, 0, 255), 1)
                                    done = True
                                    break
                    if done:
                        break
        if not done and lhcheck != 0:
            if save_l1:
                cv2.line(aa, (0, lhcheck), (69, lhcheck), (0, 0, 255), 1)
            vmove = True

        # Vertical line for last segmentation
        done = False
        lvcheck = 0
        for i in range(0, len(sum_columns), 1):
            if sum_columns[i] >= 255 * mid_threshold:
                if i >= 66:
                    if save_l1:
                        cv2.line(aa, (69, 0), (69, 69), (0, 0, 255), 1)
                    hmove = True
                    done = True
                    lvcheck = 69
                    break
                else:
                    for j in range(1, 4):
                        if sum_columns[i + j] < 255 * black_threshold:
                            if sum_columns[i] >= 255 * white_threshold:
                                if save_l1:
                                    cv2.line(aa, (i + j - 1, 0), (i + j - 1, 69), (0, 0, 255), 1)
                                hmove = True
                                done = True
                                lvcheck = i + j - 1
                                break
                            else:
                                lvcheck = i + j - 1
                                if i < 5:
                                    if save_l1:
                                        cv2.line(aa, (i + j - 1, 0), (i + j - 1, 69), (0, 0, 255), 1)
                                    done = True
                                    break
                    if done:
                        break
        if not done and lvcheck != 0:
            if save_l1:
                cv2.line(aa, (lvcheck, 0), (lvcheck, 69), (0, 0, 255), 1)
            hmove = True

        # Horizontal line for first segmentation
        done = False
        fhcheck = 0
        for i in range(0, len(sum_rows), 1):
            if sum_rows[i] <= 255 * mid_threshold:
                if i >= 66:
                    if i < lhcheck:
                        if save_l1:
                            cv2.line(aa, (0, 66), (69, 66), (255, 0, 255), 1)
                        fhcheck = 66
                        done = True
                        break
                else:
                    for j in range(1, 4):
                        if sum_rows[i + j] > 255 * white_threshold:
                            if sum_rows[i] <= 255 * black_threshold:
                                if save_l1:
                                    cv2.line(aa, (0, i + j - 1), (69, i + j - 1), (255, 0, 255), 1)
                                done = True
                                fhcheck = i + j - 1
                                break
                            else:
                                fhcheck = i + j - 1
                    if done:
                        break
        if not done and fhcheck != 0 and fhcheck < lhcheck:
            if save_l1:
                cv2.line(aa, (0, fhcheck), (69, fhcheck), (255, 0, 255), 1)
        if fhcheck == 0 and lhcheck != 0:
            if save_l1:
                cv2.line(aa, (0, 0), (69, 0), (255, 0, 255), 1)

        # Vertical line for first segmentation
        done = False
        fvcheck = 0
        for i in range(0, len(sum_columns), 1):
            if sum_columns[i] <= 255 * mid_threshold:
                if i >= 65:
                    if i < lvcheck:
                        if save_l1:
                            cv2.line(aa, (65, 0), (65, 69), (255, 0, 255), 1)
                        fvcheck = 65
                        done = True
                        break
                else:
                    for j in range(1, 5):
                        if sum_columns[i + j] > 255 * white_threshold:
                            if sum_columns[i] <= 255 * black_threshold:
                                p = i + j - 2
                                if p < 0:
                                    p = 0
                                if save_l1:
                                    cv2.line(aa, (p, 0), (p, 69), (255, 0, 255), 1)
                                done = True
                                fvcheck = p
                                break
                            else:
                                fvcheck = i + j - 2
                                if fvcheck < 0:
                                    fvcheck = 0
                    if done:
                        break
        if fvcheck != 0:
            if not done and fvcheck < lvcheck:
                if save_l1:
                    cv2.line(aa, (fvcheck, 0), (fvcheck, 69), (255, 0, 255), 1)
            elif done and fvcheck > lvcheck:
                if save_l1:
                    cv2.line(aa, (0, 0), (0, 69), (255, 0, 255), 1)
                fvcheck = 0
        elif fvcheck == 0 and lvcheck != 0:
            if save_l1:
                cv2.line(aa, (0, 0), (0, 69), (255, 0, 255), 1)

        if abs(lvcheck - fvcheck) > 25:
            lvcheck = 0
            fvcheck = 0
        if abs(lhcheck - fhcheck) > 25:
            lhcheck = 0
            fhcheck = 0

        if save_l1:
            if is_on:
                cv2.imwrite(path[0], aa)
            else:
                cv2.imwrite(path[1], aa)
        return refine_l2(lhcheck, lvcheck, fhcheck, fvcheck, hmove, vmove, events, save_l2, is_on, path, save_decode)


def refine_l2(lhcheck, lvcheck, fhcheck, fvcheck, hmove, vmove, image, save_l2, is_on, path, save_decode):
    if hmove and vmove:
        if lvcheck < 35:
            if lhcheck < 35:
                crop_image = image[lhcheck:69, lvcheck:69]
            else:
                crop_image = image[0:fhcheck, lvcheck:69]
        else:
            if lhcheck < 35:
                crop_image = image[lhcheck:69, 0:fvcheck]
            else:
                crop_image = image[0:fhcheck, 0:fvcheck]
    elif hmove and not vmove:
        if lvcheck < 35:
            crop_image = image[lhcheck:69, lvcheck:69]
        else:
            crop_image = image[lhcheck:69, 0:fvcheck]
    elif not hmove and vmove:
        if lhcheck < 35:
            crop_image = image[lhcheck:69, lvcheck:69]
        else:
            crop_image = image[0:fhcheck, lvcheck:69]
    else:
        crop_image = image[lhcheck:69, lvcheck:69]

    weights = np.ones((crop_image.shape[0], crop_image.shape[1]))
    weights[:2, :] = 0  # Set the first two rows to 0
    weights[-2:, :] = 0  # Set the last two rows to 0
    weights[:, :2] = 0  # Set the first two columns to 0
    weights[:, -2:] = 0  # Set the last two columns to 0

    weighted_image = crop_image * weights

    sum_rows = np.sum(weighted_image, axis=1)
    sum_columns = np.sum(weighted_image, axis=0)
    # show_pic(weighted_image)
    density_rows = np.sum(sum_rows / weights.shape[1])
    density_columns = np.sum(sum_columns / weights.shape[0])

    if density_columns < 1200 or density_rows < 1200:
        check_threshold = 1
    elif density_columns < 1800 or density_rows < 1800:
        check_threshold = 2
    else:
        check_threshold = 4

    if density_columns < 1200 or density_rows < 1200:
        check_iteration = 3
    else:
        check_iteration = 5
    if save_l2:
        aa = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2RGB)

    y1, y2, x1, x2 = 0, 0, 0, 0

    for i in range(0, len(sum_columns), 1):
        if sum_columns[i] > 0:
            count = 0
            for j in range(1, 1 + check_iteration):
                if sum_columns[i + j] > 255 * check_threshold:
                    count += 1
            if count == check_iteration:
                x1 = i + 1
                if save_l2:
                    cv2.line(aa, (x1, 0), (x1, 69), (0, 0, 255), 1)

                break

    for i in range(len(sum_columns) - 1, -1, -1):
        if sum_columns[i] > 0:
            count = 0
            for j in range(1, 1 + check_iteration):
                if sum_columns[i - j] >= 255 * check_threshold:
                    count += 1
            if count == check_iteration:
                x2 = i + 1
                if save_l2:
                    cv2.line(aa, (x2, 0), (x2, 69), (0, 0, 255), 1)

                break

    for i in range(0, len(sum_rows), 1):
        if sum_rows[i] > 0:
            count = 0
            for j in range(1, 1 + check_iteration):
                if sum_rows[i + j] >= 255 * check_threshold:
                    count += 1
            if count == check_iteration:
                if save_l2:
                    cv2.line(aa, (0, i), (69, i), (0, 0, 255), 1)
                y1 = i
                break

    for i in range(len(sum_rows) - 1, -1, -1):
        if sum_rows[i] > 0:
            count = 0
            for j in range(1, 1 + check_iteration):
                if sum_rows[i - j] >= 255 * check_threshold:
                    count += 1
            if count == check_iteration:
                if save_l2:
                    cv2.line(aa, (0, i), (69, i), (0, 0, 255), 1)
                y2 = i
                break

    crop_image = crop_image[y1:y2, x1:x2]
    new_sum_row = np.sum(crop_image, axis=1)
    new_sum_column = np.sum(crop_image, axis=0)
    x1, x2, y1, y2 = 0, crop_image.shape[1], 0, crop_image.shape[0]

    for i in range(crop_image.shape[0]):
        if new_sum_row[i] == 0:
            y1 += 1
        else:
            break

    for i in range(crop_image.shape[0] - 1, -1, -1):
        if new_sum_row[i] == 0:
            y2 -= 1
        else:
            break

    for i in range(crop_image.shape[1]):
        if new_sum_column[i] == 0:
            x1 += 1
        else:
            break

    for i in range(crop_image.shape[1] - 1, -1, -1):
        if new_sum_column[i] == 0:
            x2 -= 1
        else:
            break

    if x1 != 0 or x2 != crop_image.shape[1] or y1 != 0 or y2 != crop_image.shape[0]:
        crop_image = crop_image[y1:y2, x1:x2]

    if save_l2:
        if is_on:
            cv2.imwrite(path[2], aa)
        else:
            cv2.imwrite(path[3], aa)

    return decode(x1, x2, y1, y2, crop_image, save_decode, is_on, path)


def decode(x1, x2, y1, y2, image, save_decode, is_on, path):
    result_table = []
    wpp = []
    scale_factor = 4
    image = cv2.resize(image, (image.shape[1] * scale_factor, image.shape[0] * scale_factor))

    tw = x2 - x1
    th = y2 - y1
    w, h = int(tw / 4 * scale_factor), int(th / 4 * scale_factor)

    wp = 0
    count = 0

    for i in range(0, 4):
        for j in range(0, 4):
            xx, yy = j * w, i * h
            roi = image[yy: yy + h, xx: xx + w]
            _, binary_mask = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(binary_mask)
            p = round(white_pixels / (w * h), 4)

            Misc.twoD_array(wpp, i, j, p)
            wp += white_pixels / (w * h)
            count += 1

    wp /= count

    for i in range(0, 4):
        for j in range(0, 4):

            if wpp[i][j] >= wp:
                result = 1
            else:
                result = 0

            Misc.twoD_array(result_table, i, j, result)

    if save_decode:
        aa = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for i in range(0, 4):
            for j in range(0, 4):
                xx, yy = j * w, i * h

                position = (xx + 10, yy + 20)  # The position where you want to start the text
                font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
                font_scale = 0.4  # Font scale (size of the fo  nt)
                color_decode = (255, 0, 255)  # Color of the text in BGR (blue, green, red)
                color_weight = (255, 255, 0)  # Color of the text in BGR (blue, green, red)
                color_threshold = (0, 255, 255)  # Color of the text in BGR (blue, green, red)
                thickness = 1  # Thickness of the font

                row_line_start = (xx, yy)
                row_line_end = (xx + w, yy)

                column_line_start = (xx, yy)
                column_line_end = (xx, yy + h)
                cv2.line(aa, row_line_start, row_line_end, (0, 255, 0), 1)
                cv2.line(aa, column_line_start, column_line_end, (0, 0, 255), 1)
                cv2.putText(aa, str(int(result_table[i][j])), position, font, font_scale, color_decode, thickness, cv2.LINE_AA)
                cv2.putText(aa, str(wpp[i][j]), (xx, yy + 30), font, font_scale, color_weight, thickness, cv2.LINE_AA)
                cv2.putText(aa, str(round(wp, 2)), (3 * w, 3 * h + 40), font, font_scale, color_threshold, thickness, cv2.LINE_AA)
        if is_on:
            pat = path[4]
            cv2.imwrite(pat, aa)
        else:
            cv2.imwrite(path[5], aa)
    return result_table


def blur_clear(on, off):
    on = cv2.GaussianBlur(on, (3, 3), 0)
    _, on = cv2.threshold(on, 200, 255, cv2.THRESH_BINARY)

    off = cv2.GaussianBlur(off, (3, 3), 0)
    _, off = cv2.threshold(off, 200, 255, cv2.THRESH_BINARY)

    return on, off

def decode_message(camera_output, model_output, index):

    if camera_output is not None:
        path = []
        scale_factor = 0.7
        model_output = cv2.resize(model_output, [346, 260])
        vertices, valid, _ = outline(camera_output, model_output, scale_factor, False, False, False, False)

        if valid:
            marker = warp(camera_output, vertices, scale_factor, False)
            on_events, off_events = divide(marker)
            on_events, off_events = blur_clear(on_events, off_events)

            on_result = refine_l1(on_events, True, path, False, False, False)
            off_result = refine_l1(off_events, False, path, False, False, False)

            decode_result = ""
            for i in range(0, 4):
                for j in range(0, 4):
                    decode_result += str(on_result[i][j] | off_result[i][j])

            print(decode_result)

def decode_message_test(index):
    camera_output = cv2.imread("./frames/RndMsg/camera_outputs/" + str(index).zfill(5) + ".png",
                               cv2.IMREAD_GRAYSCALE)  # path = "./frames/" + str(index).zfill(10) + ".png"
    if camera_output is not None:
        model_output = cv2.resize(cv2.imread("./frames/RndMsg/model_outputs/" + str(index) + ".png", cv2.IMREAD_GRAYSCALE), [346, 260])  # path = "./frames/prediction/Res/" + str(index) + ".png"
        on_path_l1 = "./frames/RndMsg/res/on/contour/" + str(index) + ".png"
        off_path_l1 = "./frames/RndMsg/res/off/contour/" + str(index) + ".png"
        on_path_l2 = "./frames/RndMsg/res/on/refined/" + str(index) + ".png"
        off_path_l2 = "./frames/RndMsg/res/off/refined/" + str(index) + ".png"
        on_path_decode = "./frames/RndMsg/res/on/decode/" + str(index) + ".png"
        off_path_decode = "./frames/RndMsg/res/off/decode/" + str(index) + ".png"
        # off_event = cv2.imread("./frames/Test2/res/off/off_blur_clean/" + str(index) + ".png", cv2.IMREAD_GRAYSCALE)
        # on_event = cv2.imread("./frames/Test2/res/on/on_blur_clean/" + str(index) + ".png", cv2.IMREAD_GRAYSCALE)

        path = [on_path_l1, off_path_l1, on_path_l2, off_path_l2, on_path_decode, off_path_decode]
        scale_factor = 0.7

        vertices, valid, _ = outline(camera_output, model_output, scale_factor, False, False, False, False)

        if valid:
            marker = warp(camera_output, vertices, scale_factor, False)
            on_events, off_events = divide(marker)
            on_events, off_events = blur_clear(on_events, off_events)

            on_result = refine_l1(on_events, True, path, True, True, True)
            off_result = refine_l1(off_events, False, path, True, True, True)

            decode_result = ""
            for i in range(0, 4):
                for j in range(0, 4):
                    decode_result += str(on_result[i][j] | off_result[i][j])

            print(decode_result)


def main():
    for i in range(0, 597):
        print(i)
        # outline(i, False, True)
        decode_message_test(i)



if __name__ == '__main__':
    main()
