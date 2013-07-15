import threading
import sys
import select
import thread
import time

from dc1394 cimport *

import codebench.events as events

import numpy as np
cimport numpy as np
from cpython cimport Py_INCREF, Py_DECREF

cdef dict color_coding = {
                           DC1394_COLOR_CODING_MONO8      : "MONO8",
                           DC1394_COLOR_CODING_YUV411     : "YUV411",
                           DC1394_COLOR_CODING_YUV422     : "YUV422",
                           DC1394_COLOR_CODING_YUV444     : "YUV444",
                           DC1394_COLOR_CODING_RGB8       : "RGB8",
                           DC1394_COLOR_CODING_MONO16     : "MONO16",
                           DC1394_COLOR_CODING_RGB16      : "RGB16",
                           DC1394_COLOR_CODING_MONO16S    : "MONO16S",
                           DC1394_COLOR_CODING_RGB16S     : "RGB16S",
                           DC1394_COLOR_CODING_RAW8       : "RAW8",
                           DC1394_COLOR_CODING_RAW16      : "RAW16" }


VIDEO_MODE_1600x1200_RGB8 = DC1394_VIDEO_MODE_1600x1200_RGB8
VIDEO_MODE_1280x960_RGB8 = DC1394_VIDEO_MODE_1280x960_RGB8
VIDEO_MODE_1024x768_RGB8 = DC1394_VIDEO_MODE_1024x768_RGB8
VIDEO_MODE_800x600_RGB8 = DC1394_VIDEO_MODE_800x600_RGB8
VIDEO_MODE_640x480_RGB8 = DC1394_VIDEO_MODE_640x480_RGB8
VIDEO_MODE_1600x1200_YUV422 = DC1394_VIDEO_MODE_1600x1200_YUV422
VIDEO_MODE_1280x960_YUV422 = DC1394_VIDEO_MODE_1280x960_YUV422
VIDEO_MODE_1024x768_YUV422 = DC1394_VIDEO_MODE_1024x768_YUV422
VIDEO_MODE_800x600_YUV422 = DC1394_VIDEO_MODE_800x600_YUV422
VIDEO_MODE_640x480_YUV422 = DC1394_VIDEO_MODE_640x480_YUV422
VIDEO_MODE_640x480_YUV411 = DC1394_VIDEO_MODE_640x480_YUV411
VIDEO_MODE_1280x960_MONO8 = DC1394_VIDEO_MODE_1280x960_MONO8
VIDEO_MODE_1024x768_MONO8 = DC1394_VIDEO_MODE_1024x768_MONO8
VIDEO_MODE_800x600_MONO8 = DC1394_VIDEO_MODE_800x600_MONO8
VIDEO_MODE_640x480_MONO8 = DC1394_VIDEO_MODE_640x480_MONO8
VIDEO_MODE_1600x1200_MONO16 = DC1394_VIDEO_MODE_1600x1200_MONO16
VIDEO_MODE_1280x960_MONO16 = DC1394_VIDEO_MODE_1280x960_MONO16
VIDEO_MODE_1600x1200_MONO8 = DC1394_VIDEO_MODE_1600x1200_MONO8
VIDEO_MODE_1024x768_MONO16 = DC1394_VIDEO_MODE_1024x768_MONO16
VIDEO_MODE_800x600_MONO16 = DC1394_VIDEO_MODE_800x600_MONO16
VIDEO_MODE_640x480_MONO16 = DC1394_VIDEO_MODE_640x480_MONO16


FRAMERATE_1_875 = DC1394_FRAMERATE_1_875
FRAMERATE_3_75 = DC1394_FRAMERATE_3_75
FRAMERATE_7_5 = DC1394_FRAMERATE_7_5
FRAMERATE_15 = DC1394_FRAMERATE_15
FRAMERATE_30 = DC1394_FRAMERATE_30
FRAMERATE_60 = DC1394_FRAMERATE_60
FRAMERATE_120 = DC1394_FRAMERATE_120
FRAMERATE_240 = DC1394_FRAMERATE_240


ISO_SPEED_100 = DC1394_ISO_SPEED_100
ISO_SPEED_200 = DC1394_ISO_SPEED_200
ISO_SPEED_400 = DC1394_ISO_SPEED_400
ISO_SPEED_800 = DC1394_ISO_SPEED_800
ISO_SPEED_1600 = DC1394_ISO_SPEED_1600
ISO_SPEED_3200 = DC1394_ISO_SPEED_3200

OPERATION_MODE_LEGACY = DC1394_OPERATION_MODE_LEGACY
OPERATION_MODE_1394B = DC1394_OPERATION_MODE_1394B

cdef list DC1394ISOSpeedTable = [
                        DC1394_ISO_SPEED_100,
                        DC1394_ISO_SPEED_200,
                        DC1394_ISO_SPEED_400,
                        DC1394_ISO_SPEED_800,
                        DC1394_ISO_SPEED_1600,
                        DC1394_ISO_SPEED_3200 ]


cdef dict DC1394NumpyColorCoding = {
                        DC1394_COLOR_CODING_MONO8       : np.uint8(),
                        DC1394_COLOR_CODING_YUV411      : np.uint8(),
                        DC1394_COLOR_CODING_YUV422      : np.uint8(),
                        DC1394_COLOR_CODING_YUV444      : np.uint8(),
                        DC1394_COLOR_CODING_RGB8        : np.dtype([("R", np.uint8), ("G", np.uint8), ("B", np.uint8)]),
                        DC1394_COLOR_CODING_MONO16      : np.uint16(),
                        DC1394_COLOR_CODING_RGB16       : np.dtype([("R", np.uint16), ("G", np.uint16), ("B", np.uint16)]),
                        DC1394_COLOR_CODING_MONO16S     : np.int16(),
                        DC1394_COLOR_CODING_RGB16S      : np.int16(),
                        DC1394_COLOR_CODING_RAW8        : np.uint8(),
                        DC1394_COLOR_CODING_RAW16       : np.uint16(),
}

cdef dict DC1394NumpyColorCoding2 = {
            DC1394_COLOR_CODING_MONO8       : ("u8", 1),
            DC1394_COLOR_CODING_YUV411      : ("u8", 1),
            DC1394_COLOR_CODING_YUV422      : ("u8", 3),
            DC1394_COLOR_CODING_YUV444      : ("u8", 1),
            DC1394_COLOR_CODING_RGB8        : ("u8", 3),
            DC1394_COLOR_CODING_MONO16      : ("u16", 1),
            DC1394_COLOR_CODING_RGB16       : ("u16", 3),
            DC1394_COLOR_CODING_MONO16S     : ("i16", 1),
            DC1394_COLOR_CODING_RGB16S      : ("i16", 3),
            DC1394_COLOR_CODING_RAW8        : ("u8", 1),
            DC1394_COLOR_CODING_RAW16       : ("u16", 1),
}



class DC1394Error(Exception): pass


cdef inline bint DC1394SafeCall(dc1394error_t error, bint raise_event=True) except -1:
    cdef const_char_ptr errstr
    cdef bint return_value = True
    if DC1394_SUCCESS != error:
        errstr = dc1394_error_get_string(error)
        if raise_event:
            raise DC1394Error("%s - no %d" % (errstr, error))
        return_value = False
    return return_value


cdef inline dict __dc1394_array_interface__(dc1394video_frame_t * frame):
    cdef str dtype
    cdef uint8_t nbytes
    (dtype, nbytes) = DC1394NumpyColorCoding2[frame.color_coding]
    cdef str endianess = ">%s" % dtype
    if (frame.little_endian == DC1394_TRUE):
        endianess = "%s%s" % ("<", dtype)


    return dict(data= < np.intp_t > frame.image,
                shape=(frame.size[1], frame.size[0], nbytes),
                strides=(frame.stride, nbytes, 1),
                version=3,
                typestr=endianess)

cdef class FrameObject(object):
    def __init__(self, interface):
        self.__array_interface__ = interface

cdef class DC1394Context(object):
    """
    This object represent a DC1394 context which is needed for further calling
    of dc1394 function
    """
    cdef dc1394_t * dc1394

    def __cinit__(self):
        self.dc1394 = dc1394_new()

    def __dealloc__(self):
        dc1394_free(self.dc1394)

    def __get_number_of_devices__(self):
        cdef dc1394camera_list_t * camera_list
        DC1394SafeCall(dc1394_camera_enumerate(self.dc1394, & camera_list))

        return camera_list.num

    numberOfDevices = property(__get_number_of_devices__)

    cpdef list enumerateCameras(self):
        cdef dc1394camera_list_t * camera_lst
        cdef list return_value

        DC1394SafeCall(dc1394_camera_enumerate(self.dc1394, & camera_lst))
        return_value = [camera_lst.ids[i] for i in xrange(camera_lst.num)]
        dc1394_camera_free_list(camera_lst)
        return return_value

    cpdef createCamera(self, int cid= -1, int64_t guid= -1):
        camdesc = None
        enumCam = self.enumerateCameras()
        if not enumCam:
            return None
        if cid >= len(enumCam):
            return None
        if guid > 0:
            for cam in enumCam:
                if cam['guid'] == guid:
                    camdesc = cam
                    break
            if not camdesc:
                return None
        if not camdesc:
            camdesc = enumCam[cid]
        cam = None
        try:
            cam = DC1394Camera(self, camdesc['guid'], unit=camdesc['unit'])
        except Exception as e:
            print("Error: %s" % e)
            print("The list of camera %s" % self.enumerateCameras())
        return cam

class DC1394CameraServer(object):
    _instance = None

    def __new__(cls):
        # Singleton
        if not cls._instance:
            # private variable
            cls.dct_camera = {}
            cls.select_list = {}
            cls.thread_select = None
            cls.in_execution = False

            # instance class
            cls._instance = super(DC1394CameraServer, cls).__new__(cls)
        return cls._instance

    def add_camera(self, camera):
        if camera in self.dct_camera:
            return False
        fileno = camera.fileno
        self.dct_camera[camera] = fileno
        self.select_list[fileno] = camera
        if not self.in_execution:
            self.in_execution = True
            thread.start_new_thread(self._execution, ())
        return True

    def remove_camera(self, camera):
        if camera not in self.dct_camera:
            return False
        fileno = self.dct_camera[camera]
        del self.dct_camera[camera]
        del self.select_list[fileno]
        if not self.select_list:
            self.in_execution = False

        return True

    def _execution(self):
        begin_time = time.time()
        # TODO use semaphore to stop or start the execution
        while self.in_execution:
            # timeout of 1 second. Not normal to don't receive 1 frame after 1 second.
            rlist, wlist, xlist = select.select(self.select_list, [], [], 1)
            for fileno in rlist:
                if fileno in self.select_list:
                    camera = self.select_list[fileno]
                    camera.capture_image()
                else:
                    print("error from camera server video1394 select on file id %s" % fileno)
            """
            actual_time = time.time()
            if not rlist:
                print("list of fileno is empty")
            if actual_time - begin_time > 1.5:
                begin_time = actual_time
                print("Timeout select camera.")
            for key, value in self.select_list.items():
                actual_fileno = value.fileno
                if key != actual_fileno:
                    print("The fileno of camera %s has change. %s to %s." % (value, key, actual_fileno))
            """

    def close(self):
        self.in_execution = False

cdef class DC1394Camera(object):
    cdef dc1394camera_t * cam
    cdef DC1394Context ctx
    cdef dict available_modes
    cdef bint stop_event
    cdef bint running
    cdef bint force_rgb8
    cdef object __grab_event__
    cdef object __init_event__
    cdef object __stop_event__
    cdef object capture_loop
    cdef object cam_server
    cdef dict available_features
    cdef dict unavailable_features
    cdef dict available_features_string
    cdef uint8_t pixel_convert[480000 * 3]
    cdef np.ndarray arr

    def __dealloc__(self):
        if self.transmission == DC1394_ON:
            self.transmission = DC1394_OFF
        if self.cam:
            dc1394_camera_free(self.cam)

    def __cinit__(self, DC1394Context ctx, uint64_t guid, int unit= -1):
        self.ctx = ctx
        self.cam_server = DC1394CameraServer()
        if unit != -1:
            self.cam = dc1394_camera_new_unit(ctx.dc1394, guid, unit)
        else:
            self.cam = dc1394_camera_new(ctx.dc1394, guid);
        if not self.cam:
            raise DC1394Error("No camera detected, guid %d." % guid)

        self.populate_capabilities()

        self.force_rgb8 = False
        self.running = False
        self.stop_event = False
        self.__grab_event__ = events.Event()
        self.__init_event__ = events.Event()
        self.__stop_event__ = events.Event()

    def initialize(self, reset_bus=True, mode=None, framerate=None, iso_speed=None,
              operation_mode=None):
        # initialize the camera
        self.transmission = DC1394_OFF
        if reset_bus:
            self.resetBus()
        if operation_mode is not None:
            self.operationMode = operation_mode
        else:
            try:
                self.operationMode = DC1394_OPERATION_MODE_1394B
                self.isoSpeed = DC1394_ISO_SPEED_800
            except:
                self.operationMode = DC1394_OPERATION_MODE_LEGACY
                self.isoSpeed = DC1394_ISO_SPEED_400
        if iso_speed is not None:
            self.isoSpeed = iso_speed
        if framerate is not None:
            self.framerate = framerate
        if mode is not None:
            self.mode = mode

    def start(self, force_rgb8=False):
        cdef dc1394video_frame_t * frame

        if (self.running):
            raise RuntimeError("Camera Already Running")
        self.force_rgb8 = force_rgb8
        self.stop_event = False

        self.__init_event__()
        self.transmission = DC1394_OFF
        # Comments from cc1394Setup : Capture - We currently allocate the channel and not
        #               the iso bandwidth. Program crashes may leave a channel occupied.
        num_buffer = 10
        DC1394SafeCall(dc1394_capture_setup(self.cam, num_buffer, DC1394_CAPTURE_FLAGS_CHANNEL_ALLOC))
        self.transmission = DC1394_ON

        dc1394_capture_dequeue(self.cam, DC1394_CAPTURE_POLICY_WAIT, & frame)
        if self.force_rgb8:
            dtype = DC1394NumpyColorCoding[DC1394_COLOR_CODING_RGB8]
        else:
            dtype = DC1394NumpyColorCoding[frame.color_coding]
        self.arr = np.ndarray(shape=(frame.size[1], frame.size[0], dtype.itemsize) , dtype=np.uint8)
        self.arr.dtype = dtype
        Py_INCREF(dtype)

        self.cam_server.add_camera(self)
        self.running = True

    def capture_image(self):
        cdef dc1394video_frame_t * frame
        if not DC1394SafeCall(dc1394_capture_dequeue(self.cam, DC1394_CAPTURE_POLICY_POLL, & frame), raise_event=False):
            return

        self.format_image(frame)

        self.__grab_event__(self.arr, frame.timestamp)

        dc1394_capture_enqueue(self.cam, frame)

    cdef void format_image(self, dc1394video_frame_t * frame):
        # cdef uint8_t jo = frame.size[0] * frame.size[1] * dtype.itemsize
        # TODO do a malloc and be dynamic
        # 600 * 800 * 3
        if self.force_rgb8 and frame.color_coding == DC1394_COLOR_CODING_YUV422:
            dc1394_convert_to_RGB8(frame.image, self.pixel_convert, frame.size[1], frame.size[0], DC1394_BYTE_ORDER_UYVY, DC1394_COLOR_CODING_YUV422, 8);
            self.arr.data = < char *> self.pixel_convert
            return

        self.arr.data = < char *> frame.image

    def stop(self):
        try:
            DC1394SafeCall(dc1394_capture_stop(self.cam))
            self.transmission = DC1394_OFF
        except:
            # ignore error, if camera crash, just stop the event
            pass
        self.cam_server.remove_camera(self)
        self.running = False
        self.__stop_event__()

    cdef void populate_capabilities(self):
        cdef dc1394video_modes_t modes
        cdef dc1394framerates_t framerates
        cdef float framerate

        self.available_features = {}
        self.available_features_string = {}
        self.unavailable_features = {}
        self.available_modes = {}

        cdef dc1394featureset_t featureset
        DC1394SafeCall(dc1394_feature_get_all(self.cam, & featureset))

        cdef dc1394feature_info_t featureinfo
        cdef const_char_ptr feature_name
        for i in range(DC1394_FEATURE_NUM):
            featureinfo = featureset.feature[i]
            if (featureinfo.available == DC1394_TRUE):
                self.available_features[featureinfo.id] = featureinfo
                feature_name = dc1394_feature_get_string(featureinfo.id)
                self.available_features_string[feature_name] = featureinfo
            else:
                self.unavailable_features[featureinfo.id] = featureinfo

        DC1394SafeCall(dc1394_video_get_supported_modes(self.cam, & modes))
        for m in [modes.modes[i] for i in xrange(modes.num)]:
            try:
                DC1394SafeCall(dc1394_video_get_supported_framerates (self.cam, m, & framerates))
            except:
                break
            fmlist = []
            for j in range(framerates.num):
                fmlist.append(framerates.framerates[j])
            self.available_modes[m] = fmlist

    def get_dict_available_features(self):
        return self.available_features_string

    def get_property(self, name):
        cdef uint32_t ret_value
        feature = self.available_features_string.get(name, None)
        if not feature:
            raise DC1394Error("[%s] not available" % name)
        id = feature["id"]
        dc1394_feature_get_value(self.cam, id, & ret_value)
        return ret_value

    def get_property_is_auto(self, name):
        cdef dc1394feature_mode_t value
        feature = self.available_features_string.get(name, None)
        if not feature:
            raise DC1394Error("[%s] not available" % name)
        id = feature["id"]
        dc1394_feature_get_mode(self.cam, id, & value)
        return value == DC1394_FEATURE_MODE_AUTO

    def set_property(self, name, value):
        feature = self.available_features_string.get(name, None)
        if not feature:
            raise DC1394Error("[%s] not available" % name)
        id = feature["id"]

        # set section
        DC1394SafeCall(dc1394_feature_set_mode(self.cam, id, DC1394_FEATURE_MODE_MANUAL))

        if id == DC1394_FEATURE_WHITE_BALANCE:
            raise DC1394Error("[%s] cannot change value of white balance. Use set_whitebalance" % name)

        if value < feature['min'] or value > feature['max']:
            raise DC1394Error("[%s] value out of range" % name)

        DC1394SafeCall(dc1394_feature_set_value(self.cam, id, value))

    def set_property_auto(self, name, value):
        feature = self.available_features_string.get(name, None)
        if not feature:
            raise DC1394Error("[%s] not available" % name)
        id = feature["id"]

        if value:
            DC1394SafeCall(dc1394_feature_set_mode(self.cam, id, DC1394_FEATURE_MODE_AUTO))
        else:
            DC1394SafeCall(dc1394_feature_set_mode(self.cam, id, DC1394_FEATURE_MODE_MANUAL))

    def set_whitebalance(self, RV_value=None, BU_value=None):
        cdef uint32_t actual_rv_value
        cdef uint32_t actual_bu_value
        name = "White Balance"
        feature = self.available_features_string.get(name, None)
        if not feature:
            raise DC1394Error("[%s] not available" % name)
        id = feature["id"]
        if id != DC1394_FEATURE_WHITE_BALANCE:
            raise DC1394Error("[%s] wrong feature, use set_property" % name)

        DC1394SafeCall(dc1394_feature_set_mode(self.cam, id, DC1394_FEATURE_MODE_MANUAL))
        dc1394_feature_whitebalance_get_value(self.cam, & actual_bu_value, & actual_rv_value)

        if RV_value is not None:
            if RV_value < feature['min'] or RV_value > feature['max']:
                raise DC1394Error("[%s] RV_value out of range" % name)
        else:
            RV_value = actual_rv_value

        if BU_value is not None:
            if BU_value < feature['min'] or BU_value > feature['max']:
                raise DC1394Error("[%s] BU_value out of range" % name)
        else:
            BU_value = actual_bu_value

        DC1394SafeCall(dc1394_feature_whitebalance_set_value(self.cam, BU_value, RV_value))


    property fileno:
        def __get__(self):
            return dc1394_capture_get_fileno(self.cam)

    property available_features:
        def __get__(self):
            return self.available_features

    property available_features_string:
        def __get__(self):
            return self.available_features_string

    # -------------------------------------------------------------------------
    def __repr__(self):
        return '<DC1394Camera vendor="%s" model="%s"/>' % (self.cam.vendor, self.cam.model)

    # -------------------------------------------------------------------------
    property grabEvent:
        def __get__(self):
            return self.__grab_event__

    property initEvent:
        def __get__(self):
            return self.__init_event__

    property stopEvent:
        def __get__(self):
            return self.__stop_event__
    # -------------------------------------------------------------------------
    property bandwitdh:
        def __get__(self):
            cdef uint32_t bandwidth
            DC1394SafeCall(dc1394_video_get_bandwidth_usage(self.cam, & bandwidth))
            return bandwidth

    # -------------------------------------------------------------------------
    property multishot:
        def __get__(self):
            cdef dc1394bool_t pwr
            cdef uint32_t frames
            DC1394SafeCall(dc1394_video_get_multi_shot(self.cam, & pwr, & frames))
            return frames if (pwr == DC1394_TRUE) else 0

        def __set__(self, uint32_t frames):
            cdef dc1394bool_t pwr = DC1394_TRUE if (frames > 0) else DC1394_FALSE
            DC1394SafeCall(dc1394_video_set_multi_shot(self.cam, frames, pwr))

    # -------------------------------------------------------------------------
    property brightness:
        def __get__(self):
            if DC1394_FEATURE_BRIGHTNESS not in self.available_features:
                raise DC1394Error("[brightness] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_BRIGHTNESS, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_BRIGHTNESS not in self.available_features:
                raise DC1394Error("[brightness] not available")

            feature = self.available_features[DC1394_FEATURE_BRIGHTNESS]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[brightness] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[brightness] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_BRIGHTNESS, value))

    # -------------------------------------------------------------------------
    property exposure:
        def __get__(self):
            if DC1394_FEATURE_EXPOSURE not in self.available_features:
                raise DC1394Error("[exposure] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_EXPOSURE, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_EXPOSURE not in self.available_features:
                raise DC1394Error("[exposure not available")

            feature = self.available_features[DC1394_FEATURE_EXPOSURE]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[exposure] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[exposure] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_EXPOSURE, value))


    # -------------------------------------------------------------------------
    property sharpness:
        def __get__(self):
            if DC1394_FEATURE_SHARPNESS not in self.available_features:
                raise DC1394Error("[sharpness] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_SHARPNESS, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_SHARPNESS not in self.available_features:
                raise DC1394Error("[sharpness not available")

            feature = self.available_features[DC1394_FEATURE_SHARPNESS]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[sharpness] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[sharpness] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_SHARPNESS, value))


    # -------------------------------------------------------------------------
    property whiteBalance:
        def __get__(self):
            if DC1394_FEATURE_WHITE_BALANCE not in self.available_features:
                raise DC1394Error("[whiteBalance] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_WHITE_BALANCE, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_WHITE_BALANCE not in self.available_features:
                raise DC1394Error("[whiteBalance not available")

            feature = self.available_features[DC1394_FEATURE_WHITE_BALANCE]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[whiteBalance] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[whiteBalance] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_WHITE_BALANCE, value))

    # -------------------------------------------------------------------------
    property hue:
        def __get__(self):
            if DC1394_FEATURE_HUE not in self.available_features:
                raise DC1394Error("[hue] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_HUE, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_HUE not in self.available_features:
                raise DC1394Error("[hue not available")

            feature = self.available_features[DC1394_FEATURE_HUE]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[hue] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[hue] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_HUE, value))


    # -------------------------------------------------------------------------
    property saturation:
        def __get__(self):
            if DC1394_FEATURE_SATURATION not in self.available_features:
                raise DC1394Error("[saturation] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_SATURATION, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_SATURATION not in self.available_features:
                raise DC1394Error("[saturation not available")

            feature = self.available_features[DC1394_FEATURE_SATURATION]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[saturation] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[saturation] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_SATURATION, value))


    # -------------------------------------------------------------------------
    property gamma:
        def __get__(self):
            if DC1394_FEATURE_GAMMA not in self.available_features:
                raise DC1394Error("[gamma] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_GAMMA, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_GAMMA not in self.available_features:
                raise DC1394Error("[gamma not available")

            feature = self.available_features[DC1394_FEATURE_GAMMA]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[gamma] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[gamma] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_GAMMA, value))


    # -------------------------------------------------------------------------
    property shutter:
        def __get__(self):
            if DC1394_FEATURE_SHUTTER not in self.available_features:
                raise DC1394Error("[shutter] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_SHUTTER, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_SHUTTER not in self.available_features:
                raise DC1394Error("[shutter not available")

            feature = self.available_features[DC1394_FEATURE_SHUTTER]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[shutter] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[shutter] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_SHUTTER, value))


    # -------------------------------------------------------------------------
    property gain:
        def __get__(self):
            if DC1394_FEATURE_GAIN not in self.available_features:
                raise DC1394Error("[gain] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_GAIN, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_GAIN not in self.available_features:
                raise DC1394Error("[gain not available")

            feature = self.available_features[DC1394_FEATURE_GAIN]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[gain] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[gain] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_GAIN, value))


    # -------------------------------------------------------------------------
    property iris:
        def __get__(self):
            if DC1394_FEATURE_IRIS not in self.available_features:
                raise DC1394Error("[iris] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_IRIS, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_IRIS not in self.available_features:
                raise DC1394Error("[iris not available")

            feature = self.available_features[DC1394_FEATURE_IRIS]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[iris] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[iris] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_IRIS, value))


    # -------------------------------------------------------------------------
    property focus:
        def __get__(self):
            if DC1394_FEATURE_FOCUS not in self.available_features:
                raise DC1394Error("[focus] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_FOCUS, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_FOCUS not in self.available_features:
                raise DC1394Error("[focus not available")

            feature = self.available_features[DC1394_FEATURE_FOCUS]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[focus] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[focus] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_FOCUS, value))


    # -------------------------------------------------------------------------
    property temperature:
        def __get__(self):
            if DC1394_FEATURE_TEMPERATURE not in self.available_features:
                raise DC1394Error("[temperature] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_TEMPERATURE, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_TEMPERATURE not in self.available_features:
                raise DC1394Error("[temperature not available")

            feature = self.available_features[DC1394_FEATURE_TEMPERATURE]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[temperature] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[temperature] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_TEMPERATURE, value))


    # -------------------------------------------------------------------------
    property trigger:
        def __get__(self):
            if DC1394_FEATURE_TRIGGER not in self.available_features:
                raise DC1394Error("[trigger] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_TRIGGER, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_TRIGGER not in self.available_features:
                raise DC1394Error("[trigger not available")

            feature = self.available_features[DC1394_FEATURE_TRIGGER]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[trigger] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[trigger] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_TRIGGER, value))


    # -------------------------------------------------------------------------
    property triggerDelay:
        def __get__(self):
            if DC1394_FEATURE_TRIGGER_DELAY not in self.available_features:
                raise DC1394Error("[triggerDelay] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_TRIGGER_DELAY, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_TRIGGER_DELAY not in self.available_features:
                raise DC1394Error("[triggerDelay not available")

            feature = self.available_features[DC1394_FEATURE_TRIGGER_DELAY]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[triggerDelay] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[triggerDelay] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_TRIGGER_DELAY, value))


    # -------------------------------------------------------------------------
    property whiteShading:
        def __get__(self):
            if DC1394_FEATURE_WHITE_SHADING not in self.available_features:
                raise DC1394Error("[whiteShading] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_WHITE_SHADING, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_WHITE_SHADING not in self.available_features:
                raise DC1394Error("[whiteShading not available")

            feature = self.available_features[DC1394_FEATURE_WHITE_SHADING]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[whiteShading] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[whiteShading] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_WHITE_SHADING, value))


    # -------------------------------------------------------------------------
    property frameRate:
        def __get__(self):
            if DC1394_FEATURE_FRAME_RATE not in self.available_features:
                raise DC1394Error("[frameRate] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_FRAME_RATE, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_FRAME_RATE not in self.available_features:
                raise DC1394Error("[frameRate not available")

            feature = self.available_features[DC1394_FEATURE_FRAME_RATE]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[frameRate] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[frameRate] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_FRAME_RATE, value))


    # -------------------------------------------------------------------------
    property zoom:
        def __get__(self):
            if DC1394_FEATURE_ZOOM not in self.available_features:
                raise DC1394Error("[zoom] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_ZOOM, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_ZOOM not in self.available_features:
                raise DC1394Error("[zoom not available")

            feature = self.available_features[DC1394_FEATURE_ZOOM]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[zoom] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[zoom] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_ZOOM, value))


    # -------------------------------------------------------------------------
    property pan:
        def __get__(self):
            if DC1394_FEATURE_PAN not in self.available_features:
                raise DC1394Error("[pan] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_PAN, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_PAN not in self.available_features:
                raise DC1394Error("[pan not available")

            feature = self.available_features[DC1394_FEATURE_PAN]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[pan] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[pan] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_PAN, value))


    # -------------------------------------------------------------------------
    property tilt:
        def __get__(self):
            if DC1394_FEATURE_TILT not in self.available_features:
                raise DC1394Error("[tilt] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_TILT, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_TILT not in self.available_features:
                raise DC1394Error("[tilt not available")

            feature = self.available_features[DC1394_FEATURE_TILT]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[tilt] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[tilt] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_TILT, value))


    # -------------------------------------------------------------------------
    property opticalFilter:
        def __get__(self):
            if DC1394_FEATURE_OPTICAL_FILTER not in self.available_features:
                raise DC1394Error("[opticalFilter] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_OPTICAL_FILTER, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_OPTICAL_FILTER not in self.available_features:
                raise DC1394Error("[opticalFilter not available")

            feature = self.available_features[DC1394_FEATURE_OPTICAL_FILTER]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[opticalFilter] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[opticalFilter] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_OPTICAL_FILTER, value))


    # -------------------------------------------------------------------------
    property captureSize:
        def __get__(self):
            if DC1394_FEATURE_CAPTURE_SIZE not in self.available_features:
                raise DC1394Error("[captureSize] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_CAPTURE_SIZE, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_CAPTURE_SIZE not in self.available_features:
                raise DC1394Error("[captureSize not available")

            feature = self.available_features[DC1394_FEATURE_CAPTURE_SIZE]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[captureSize] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[captureSize] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_CAPTURE_SIZE, value))


    # -------------------------------------------------------------------------
    property captureQuality:
        def __get__(self):
            if DC1394_FEATURE_CAPTURE_QUALITY not in self.available_features:
                raise DC1394Error("[captureQuality] not available")

            cdef uint32_t value
            dc1394_feature_get_value(self.cam, DC1394_FEATURE_CAPTURE_QUALITY, & value)
            return value

        def __set__(self, uint32_t value):
            if DC1394_FEATURE_CAPTURE_QUALITY not in self.available_features:
                raise DC1394Error("[captureQuality not available")

            feature = self.available_features[DC1394_FEATURE_CAPTURE_QUALITY]
            if feature['current_mode'] == DC1394_FEATURE_MODE_AUTO:
                raise DC1394Error("[captureQuality] Currently In Auto Mode")

            if value < feature['min'] or value > feature['max']:
                raise DC1394Error("[captureQuality] value out of range")

            DC1394SafeCall(dc1394_feature_set_value(self.cam, DC1394_FEATURE_CAPTURE_QUALITY, value))




    # -------------------------------------------------------------------------
    property oneshot:
        def __get__(self):
            cdef dc1394bool_t pwr
            DC1394SafeCall(dc1394_video_get_one_shot(self.cam, & pwr))
            return (pwr == DC1394_TRUE)

        def __set__(self, bint flag):
            cdef dc1394bool_t pwr = DC1394_TRUE if flag else DC1394_TRUE
            DC1394SafeCall(dc1394_video_set_one_shot(self.cam, pwr))

    # -------------------------------------------------------------------------
    property availableFramerates:
        def __get__(self):
            cdef float framerate
            cdef dict framerates = {}
            for f in self.available_modes[self.mode]:
                dc1394_framerate_as_float(f, & framerate)
                framerates[f] = framerate
            return framerates

    # -------------------------------------------------------------------------
    property availableModes:
        def __get__(self):
            return self.available_modes

    # -------------------------------------------------------------------------
    property mode:
        def __get__(self):
            cdef dc1394video_mode_t mode
            DC1394SafeCall(dc1394_video_get_mode(self.cam, & mode))
            return mode

        def __set__(self, dc1394video_mode_t mode):
            DC1394SafeCall(dc1394_video_set_mode(self.cam, mode))

    # -------------------------------------------------------------------------
    property framerate:
        def __get__(self):
            cdef dc1394framerate_t framerate
            DC1394SafeCall(dc1394_video_get_framerate(self.cam, & framerate))
            return framerate

        def __set__(self, dc1394framerate_t framerate):
            DC1394SafeCall(dc1394_video_set_framerate(self.cam, framerate))

    # -------------------------------------------------------------------------
    property isoSpeed:
        def __get__(self):
            cdef dc1394speed_t speed
            DC1394SafeCall(dc1394_video_get_iso_speed(self.cam, & speed))
            return speed

        def __set__(self, dc1394speed_t speed):
            DC1394SafeCall(dc1394_video_set_iso_speed(self.cam, speed))

    # -------------------------------------------------------------------------
    property operationMode:
        def __get__(self):
            cdef dc1394operation_mode_t mode
            DC1394SafeCall(dc1394_video_get_operation_mode(self.cam, & mode))
            return mode

        def __set__(self, dc1394operation_mode_t mode):
            DC1394SafeCall(dc1394_video_set_operation_mode(self.cam, mode))

    # -------------------------------------------------------------------------
    property transmission:
        def __set__(self, dc1394switch_t trans):
            DC1394SafeCall(dc1394_video_set_transmission(self.cam, trans))

        def __get__(self):
            cdef dc1394switch_t trans
            DC1394SafeCall(dc1394_video_get_transmission(self.cam, & trans))
            return trans

    # -------------------------------------------------------------------------
    property vendor:
        def __get__(self):
            return self.cam.vendor

    # -------------------------------------------------------------------------
    property model:
        def __get__(self):
            return self.cam.model

    # -------------------------------------------------------------------------
    property vendorID:
        def __get__(self):
            return self.cam.vendor_id

    # -------------------------------------------------------------------------
    property modelID:
        def __get__(self):
            return self.cam.model_id

    # -------------------------------------------------------------------------
    property SWVersion:
        def __get__(self):
            return (self.cam.unit_sw_version, self.cam.unit_sub_sw_version)

    # -------------------------------------------------------------------------
    property cycleTimer:
        def __get__(self):
            cdef uint32_t node, generation
            DC1394SafeCall(dc1394_camera_get_node(self.cam, & node, & generation))
            return (node, generation)

    # -------------------------------------------------------------------------
    property node:
        def __get__(self):
            cdef uint32_t cycle_timer
            cdef uint64_t local_time
            DC1394SafeCall(dc1394_read_cycle_timer(self.cam, & cycle_timer, & local_time))
            return (cycle_timer, local_time)

    # -------------------------------------------------------------------------
    property broadcast:
        def __set__(self, bint broadcast):
            cdef dc1394bool_t flag = DC1394_TRUE if broadcast else DC1394_FALSE
            DC1394SafeCall(dc1394_camera_set_broadcast(self.cam, flag))

        def __get__(self):
            cdef dc1394bool_t flag
            DC1394SafeCall(dc1394_camera_get_broadcast(self.cam, & flag))
            return (flag == DC1394_TRUE)

    # -------------------------------------------------------------------------
    property power:
        def __set__(self, bint power):
            cdef dc1394switch_t pwr = DC1394_ON if power else DC1394_OFF
            DC1394SafeCall(dc1394_camera_set_power(self.cam, pwr))

    # -------------------------------------------------------------------------
    property softwareTrigger:
        def __get__(self):
            cdef dc1394switch_t pwr
            DC1394SafeCall(dc1394_software_trigger_get_power(self.cam, & pwr))
            return (pwr == DC1394_ON)

        def __set__(self, bint flag):
            cdef dc1394switch_t pwr = DC1394_ON if flag else DC1394_OFF
            DC1394SafeCall(dc1394_software_trigger_set_power(self.cam, pwr))

    # -------------------------------------------------------------------------
    def print_info(self):
        DC1394SafeCall(dc1394_camera_print_info(self.cam, stderr))
        cdef uint32_t width, height
        cdef dc1394color_coding_t coding
        cdef float framerate
        cdef list props = []
        if self.cam.bmode_capable :
            props.append("bmode")
        if self.cam.one_shot_capable :
            props.append("one_shot")
        if self.cam.multi_shot_capable:
            props.append("multi_shot")
        if self.cam.can_switch_on_off:
            props.append("switch_on_off")
        if self.cam.has_vmode_error_status:
            props.append("vmode_error_status")
        if self.cam.has_feature_error_status:
            props.append("feature_error_status")


        print ("Software Version \t\t  :\tv%d.%d" % self.SWVersion)
        print ("Capabilities \t\t\t  :\t%s" % ", ".join(props))

        print "------ Camera supported modes ------"

        for j, m in enumerate(self.available_modes):
            DC1394SafeCall(dc1394_get_image_size_from_video_mode (self.cam, m, & width, & height))
            DC1394SafeCall(dc1394_get_color_coding_from_video_mode (self.cam, m, & coding))
            framerates = []
            for f in self.available_modes[m]:
                DC1394SafeCall(dc1394_framerate_as_float(f, & framerate))
                framerates.append(framerate)
            print ("%d \t\t\t\t  :\t%dx%d %6s @ %s" % (m, width, height, color_coding[coding], str(framerates)))

        print "------ Camera current mode ------"


        DC1394SafeCall(dc1394_get_image_size_from_video_mode (self.cam, self.mode, & width, & height))
        DC1394SafeCall(dc1394_get_color_coding_from_video_mode (self.cam, self.mode, & coding))
        DC1394SafeCall(dc1394_framerate_as_float(self.framerate, & framerate))
        print "%d \t\t\t\t  :\t%dx%d %s @ %d" % (self.mode, width, height, color_coding[coding], framerate)

        cdef dc1394featureset_t featureset
        DC1394SafeCall(dc1394_feature_get_all(self.cam, & featureset))
        DC1394SafeCall(dc1394_feature_print_all(& featureset, stderr))



    def resetBus(self):
        dc1394_reset_bus(self.cam);

    def resetToFactoryDefault(self):
        dc1394_camera_reset(self.cam)

#
# vim: filetype=pyrex
#
#
#

