# from pathlib import Path
#
# dir = Path("C:/Users/admin/Desktop/git/PPE-Detection/services/model/dataset/train/labels")
# paths = list(dir.glob(pattern="*.txt"))
#
# bugsy = []
# normal = []
# min_value = 1e9
# max_value = -1e9
# max_file = ""
# for file in paths:
#     with open(file, "r") as f:
#         for line in f:
#             length = len(line.split())
#             if length == 5 and file not in normal:
#                 normal.append(file)
#             elif length > 5:
#                 min_value = min(min_value, length)
#                 max_value = max(max_value, length)
#                 if max_value == length: max_file = file
#                 if file not in bugsy: bugsy.append(file)
#
# print("Normal Total:", len(normal))
# print("Bugsy Total:", len(bugsy))
# print("Bugsy:", bugsy)
# print("Min line:", min_value)
# print("Max line:", max_value)
# print("Max line file:", max_file)
#
# rect1 = [1, 2, 3, 4]
# rect2 = [2, 3, 6, 8]
#
# width1 = rect1[2] - rect1[0]
# width2 = rect2[2] - rect2[0]
#
# height1 = rect1[3] - rect1[1]
# height2 = rect2[3] - rect2[1]
#
# area_rect1 = width1 * height1
# area_rect2 = width2 * height2
#
# top_left_x = max(rect1[0], rect2[0])
# top_left_y = max(rect1[1], rect2[1])
# bottom_right_x = min(rect1[2], rect2[2])
# bottom_right_y = min(rect1[3], rect2[3])
#
# width_is = bottom_right_x - top_left_x
# height_is = bottom_right_y - top_left_y
#
# intersect_area = max(0, width_is) * max(0, height_is)
# union_area = area_rect1 + area_rect2 - intersect_area
#
# print(intersect_area / union_area)
