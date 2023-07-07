/*
 * types.hpp
 *
 *  Created on: 07 Jun 2023
 *      Author: jurie
 */

#ifndef SRC_NSRC_EVENTS_HPP_
#define SRC_NSRC_EVENTS_HPP_

#define WINDOW_EVENT_CAPTURE 0
#define WINDOW_EVENT_DRAW 1
#define WINDOW_EVENT_SAVE_IMAGE 2
#define WINDOW_EVENT_EXIT 27
#define WINDOW_EVENT_CLEAR_TAB 3
#define WINDOW_EVENT_CHANGE_MODE (0x00000000 | (1>>8))
#define WINDOW_EVENT_FULL_KEY 5


#define WINDOW_NAME_KEYING "Keying Window"
#define WINDOW_NAME_SETTINGS "Settings"
#define WINDOW_NAME_MAIN "Main Window"
#define WINDOW_NAME_MASK "Mask Preview"
#define WINDOW_NAME_OUTPUT "Output Window"

#define OBJECT_PREPROCESSOR "preprocessor"
#define OBJECT_SNAPSHOT "snapshot"
#define OBJECT_LOOKUP "lookup"
#define OBJECT_CHROMMA_MASK "chromma mask"
#define OBJECT_YOLO_MASK "yolo mask"
#define OBJECT_KEYER "keyer"
#define OBJECT_INPUT "input"

#define WINDOW_TRACKBAR_BLENDING "Blending"
#define WINDOW_TRACKBAR_DELAY "Delay"
#define WINDOW_TRACKBAR_ERODE "Erode"
#define WINDOW_TRACKBAR_DILATE "Dilate"
#define WINDOW_TRACKBAR_OUTER_DIAM "Outer Diam"
#define WINDOW_TRACKBAR_UV_DIAM "UV Diam"
#define WINDOW_TRACKBAR_LUM_DEPTH "Lum Depth"
#define WINDOW_TRACKBAR_UV "UV"
#define WINDOW_TRACKBAR_LUM "LUM"
#define WINDOW_TRACKBAR_KEYTOP "Key Top"
#define WINDOW_TRACKBAR_KEYBOTTOM "Key Bottom"
#define WINDOW_TRACKBAR_BRIGHTNESS "Brightness"
#define WINDOW_TRACKBAR_SAT "saturation"


#define WINDOW_MODE_KEYER 0
#define WINDOW_MODE_CLEAR 1

#define VK_LCONTROL 0
#define VK_F1 190
#define VK_F2 191
#define VK_F3 192
#define VK_F4 193
#define VK_F5 194
#define VK_F10 199
#endif /* SRC_NSRC_EVENTS_HPP_ */
