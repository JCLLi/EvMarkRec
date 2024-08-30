import os

import dv_processing as dv
import cv2
import numpy as np
import Decode
import Misc

# Used to get visible marker moving trend within a event frame
def time_order():
    aedat_file_path = "../DVS/EVStest-2024_06_07_15_15_01.aedat4"

    reader = dv.io.MonoCameraRecording(aedat_file_path)

    print(f"Checking available streams in [{aedat_file_path}] for camera name [{reader.getCameraName()}]:")

    if reader.isEventStreamAvailable():
        print("Event stream yes")
    else:
        print("Event stream no")

    if reader.isFrameStreamAvailable():
        # Check the resolution of frame stream
        resolution = reader.getFrameResolution()

        # Print that the stream is available and its resolution
        print(f"  * Frame stream with resolution [{resolution[0]}x{resolution[1]}]")
    else:
        print("No frame stream")

    if reader.isRunning():
        frame_old = reader.getNextFrame()
        index = 0
        while reader.isRunning() and index < 260:
            img = np.zeros((260, 346, 3), dtype=np.uint8)
            old_events = np.zeros((260, 346, 3), dtype=np.uint8)
            new_events = np.zeros((260, 346, 3), dtype=np.uint8)
            frame_new = reader.getNextFrame()
            events = reader.getEventsTimeRange(frame_old.timestamp, frame_new.timestamp)
            coordinates = events.coordinates()
            timestamps = events.timestamps()
            go, vertex = Decode.outline(index, False, False, False, False, False, False) # there is a problem here: input types needed to be changed accordingly

            if go:
                fc = []
                fs = []
                index2 = 0
                for (x, y) in coordinates:
                    res = cv2.pointPolygonTest(vertex, (x, y), False)
                    if res >= 0:
                        fc.append([x, y])
                        fs.append(timestamps[index2])
                    index2 += 1
                a = 0
                coordinates_map = {}
                min_time = min(fs)
                index2 = 0
                for (x, y) in fc:
                    if (x, y) in coordinates_map:
                        coordinates_map[(x, y)][0] += 1
                        coordinates_map[(x, y)][1] += fs[index2] - min_time
                    else:
                        coordinates_map[(x, y)] = [1]
                        coordinates_map[(x, y)].append(fs[index2] - min_time)
                    index2 += 1

                for key in coordinates_map:
                    coordinates_map[key][1] = coordinates_map[key][1] / coordinates_map[key][0]

                sorted_items = sorted(coordinates_map.items(), key=lambda item: item[1][1])
                coordinates_map = tuple(sorted_items)

                centroid_old = [0, 0]
                centroid_new = [0, 0]
                number_new = 0
                number_old = 0
                for key, value in coordinates_map:
                    if a / len(coordinates_map) >= 0.4:
                        centroid_new[0] += key[0]
                        centroid_new[1] += key[1]
                        number_new += 1
                        img[key[1], key[0]] = [0, 255, 0]
                        new_events[key[1], key[0]] = [0, 255, 0]
                    else:
                        centroid_old[0] += key[0]
                        centroid_old[1] += key[1]
                        number_old += 1
                        img[key[1], key[0]] = [0, 0, 255]
                        old_events[key[1], key[0]] = [0, 0, 255]
                    a += 1

                cox = int(centroid_old[0] / number_old)
                coy = int(centroid_old[1] / number_old)
                cnx = int(centroid_new[0] / number_new)
                cny = int(centroid_new[1] / number_new)

                cv2.circle(img, (cox, coy), 2, [255, 0, 255], 3)
                cv2.circle(img, (cnx, cny), 2, [0, 255, 255], 3)

                cv2.circle(old_events, (cox, coy), 2, [255, 0, 255], 3)
                cv2.circle(new_events, (cnx, cny), 2, [0, 255, 255], 3)

            cv2.imwrite(f"./frames/compare/{index:010d}.png", img)
            cv2.imwrite(f"./frames/cn/{index}.png", new_events)
            cv2.imwrite(f"./frames/co/{index}.png", old_events)
            frame_old = frame_new
            index += 1


def get_frame():
    aedat_file_path_event = "../DVS/EVStest-2024_08_19_11_58_16.aedat4"

    reader = dv.io.MonoCameraRecording(aedat_file_path_event)

    print(f"Checking available streams in [{aedat_file_path_event}] for camera name [{reader.getCameraName()}]:")

    if reader.isEventStreamAvailable():
        print("yes")
    else:
        print("no")

    accumulator = dv.Accumulator(reader.getEventResolution())
    accumulator.setMinPotential(0.0)
    accumulator.setMaxPotential(1.0)
    accumulator.setNeutralPotential(0.5)
    accumulator.setEventContribution(0.22806699573993683)
    accumulator.setDecayFunction(dv.Accumulator.Decay.STEP)
    accumulator.setDecayParam(1e+10)
    accumulator.setIgnorePolarity(False)
    accumulator.setSynchronousDecay(False)


    if reader.isFrameStreamAvailable():
        # Check the resolution of frame stream
        resolution = reader.getFrameResolution()
        print(f"  * Frame stream with resolution [{resolution[0]}x{resolution[1]}]")
    else:
        print("No event stream")
    if reader.isRunning():
        frame_old = reader.getNextFrame()
        index = 0
        while reader.isRunning():
            frame_new = reader.getNextFrame()
            events = reader.getEventsTimeRange(frame_old.timestamp, frame_new.timestamp)

            accumulator.accept(events)
            event_frame = accumulator.generateFrame()
            print(f"Getting events between {frame_old.timestamp} to {frame_new.timestamp}")
            cv2.imwrite("./Dataset/v4/Marker/" + str(index) + ".png", event_frame.image)
            cv2.imwrite("./Dataset/v4/Label/" + str(index) + ".png", frame_new.image)

            frame_old = frame_new
            index += 1
