import csv
import datetime
import json
import os
import traceback
from collections import defaultdict

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QFileDialog,
)
from ultralytics.utils.plotting import Annotator, colors

import REID.config.model_cfgs as cfgs
from REID.reid_outer_api import ReidPipeline
from GUI.libs import qt_sql
from GUI.libs.draw_box_api import draw_chinese_box


class ProcessThread(QObject):
    debug_msg = Signal(str)
    show_match_status = Signal(str)
    show_match_id = Signal(str)
    show_match_dist = Signal(str)
    progress_bar = Signal(int)
    show_img = Signal(np.ndarray)
    show_target_img = Signal(np.ndarray)
    table_info_list = Signal(list)

    def __init__(self, base_feat_lists, base_idx_lists, dims=1024, target_class="person", device_info="gpu"):
        QObject.__init__(self)
        self.reid_pipeline = ReidPipeline(base_feat_lists, base_idx_lists, dims=dims, target_class=target_class,
                                          device_info=device_info)
        self.proc_source_url = ''
        self.proc_source_type = None
        self.skip_frames = 3
        self.stop_dtc = False
        self.continue_dtc = True
        self.match_thresh = 0.20
        self.is_track = False
        self.is_show_no_match_item = False
        self.had_track_id_dict = dict()

    def reload_faiss(self, dims=cfgs.DIMS):
        base_feat_lists, base_idx_lists = qt_sql.load_sql_feat_info(cfgs.DB_PATH, cfgs.DB_NAME)
        self.reid_pipeline.reload_search_engine(base_feat_lists, base_idx_lists, dims)

    def proc_start_run_dir_type(self):
        # 初始化目录处理索引和文件列表
        proc_dir_index = 0
        start_img_list = os.listdir(self.proc_source_url)
        start_img_path_list = []
        # 过滤有效媒体文件（支持图片和视频格式）
        for e_img in start_img_list:
            if e_img.lower().endswith(".jpg") or e_img.lower().endswith(".png") or e_img.lower().endswith(".jpeg") or \
                    e_img.lower().endswith(".mp4") or e_img.lower().endswith(".mkv") or e_img.lower().endswith(
                ".avi") or e_img.lower().endswith(".flv"):
                start_img_path_list.append(os.path.join(self.proc_source_url, e_img))
        start_img_path_list = sorted(start_img_path_list)
        all_count = len(start_img_path_list)
        is_first_frame = True
        while True:
            if self.stop_dtc:
                break
            if self.continue_dtc:
                if len(start_img_path_list) < proc_dir_index + 1:
                    break
                # 图片文件处理分支
                target_file_path = start_img_path_list[proc_dir_index]
                if target_file_path.endswith(".jpg") or target_file_path.endswith(".png") or target_file_path.endswith(
                        ".jpeg"):
                    proc_img = cv2.imread(target_file_path)
                    proc_dir_index += 1
                    frame_count = 1
                    _inter_type = "image"
                # 视频文件处理分支
                elif target_file_path.endswith(".mp4") or target_file_path.endswith(
                        ".mkv") or target_file_path.endswith(".avi") or target_file_path.endswith(".flv"):
                    _inter_type = "video"
                    if is_first_frame:
                        self.reid_pipeline.reset_track()
                        track_history = defaultdict(list)
                        start_img = cv2.VideoCapture(target_file_path)
                        _fps = start_img.get(cv2.CAP_PROP_FPS)
                        flag_read, proc_img = start_img.read()
                        is_first_frame = False
                        self.had_track_id_dict = dict()
                        frame_count = 0
                    else:
                        flag_read, proc_img = start_img.read()
                        if not flag_read:
                            is_first_frame = True
                            proc_dir_index += 1
                            frame_count = 0
                            continue
                        else:
                            frame_count += 1
                else:
                    proc_dir_index += 1
                    continue
                _image = proc_img.copy()
                # 图片处理分支
                if _inter_type == "image":
                    # 执行目标检测
                    boxes, track_ids, labels = self.reid_pipeline.detect(proc_img,
                                                                         class_idx_list=self.reid_pipeline._target_class_idx_list,
                                                                         format=_inter_type)
                    # 绘制检测框
                    self.draw_box(_image, boxes, labels)
                # 视频处理分支
                else:
                    # 跳帧处理
                    if frame_count % self.skip_frames != 0:
                        continue
                    # 带跟踪的检测
                    boxes, track_ids, labels = self.reid_pipeline.detect(proc_img,
                                                                         class_idx_list=self.reid_pipeline._target_class_idx_list,
                                                                         format=_inter_type, is_track=self.is_track)
                    filter_bbox_list = []
                    filter_trackid_list = []
                    had_search_trackid_list = []
                    if self.is_track and track_ids is not None:
                        self.draw_track(_image, boxes, track_ids, labels, track_history)
                        # 跟踪ID管理
                        for bbox, track_id in zip(boxes, track_ids):
                            if track_id not in self.had_track_id_dict:
                                self.had_track_id_dict[track_id] = [bbox, None]
                                filter_bbox_list.append(bbox)
                                filter_trackid_list.append(track_id)
                            else:
                                if self.had_track_id_dict[track_id][1] is not None:
                                    had_search_label = self.had_track_id_dict[track_id][1]
                                    had_search_trackid_list.append([bbox, had_search_label])
                        # 绘制历史匹配结果
                        _image = self.draw_match(_image, [row[0] for row in had_search_trackid_list],
                                                 [row[1] for row in had_search_trackid_list])
                        # 过滤后的检测框
                        boxes = filter_bbox_list
                    else:
                        self.draw_box(_image, boxes, labels)
                # 特征搜索与匹配
                search_labels_list, search_dist_list, target_box_list, before_sort_list = self.reid_pipeline.search(
                    proc_img, boxes, self.match_thresh)
                if _inter_type == 'video':
                    happend_time = round((frame_count + 1) / _fps, 3)
                    # 更新跟踪ID的匹配结果
                    if self.is_track and track_ids is not None:
                        for _idx, _e_sort_box in enumerate(boxes):
                            if before_sort_list[_idx] != "unknown":
                                # 更新跟踪ID字典：格式 {track_id: [bbox, match_result]}
                                self.had_track_id_dict[filter_trackid_list[_idx]][1] = before_sort_list[_idx]
                        search_labels_list.extend([row[1] for row in had_search_trackid_list])
                        target_box_list.extend([row[0] for row in had_search_trackid_list])
                        search_dist_list.extend(["None"] * len(had_search_trackid_list))
                else:
                    happend_time = 0.0
                # 结果展示分支
                if len(target_box_list) > 0:
                    _image = self.draw_match(_image, target_box_list, search_labels_list)
                    for _idx, _e_dist in enumerate(search_dist_list):
                        rows = [target_file_path, self.reid_pipeline._target_class, frame_count, happend_time,
                                search_dist_list[_idx][0], "{}:{}".format("命中", search_labels_list[_idx]),
                                "[{}]".format(','.join(map(str, map(int, target_box_list[_idx][:4]))))]
                        self.table_info_list.emit(rows)
                        bb = target_box_list[_idx]
                        crop_img = proc_img[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2]), :]
                        self.show_target_img.emit(crop_img)
                        self.show_match_dist.emit(str(search_dist_list[_idx][0]))
                        self.show_match_id.emit(search_labels_list[_idx])
                        self.show_match_status.emit("命中")
                else:
                    if self.is_show_no_match_item:
                        rows = [target_file_path, self.reid_pipeline._target_class, frame_count, happend_time, "None",
                                "未命中", "None"]
                        self.table_info_list.emit(rows)
                    crop_img = np.full((128, 128, 3), 255, dtype=np.uint8)
                    self.show_target_img.emit(crop_img)
                    self.show_match_dist.emit("None")
                    self.show_match_id.emit("None")
                    self.show_match_status.emit("未命中")
                self.show_img.emit(_image)
                process_value = int((proc_dir_index) / all_count * 1000)
                self.progress_bar.emit(process_value)
                if process_value == 1000:
                    break

    def draw_track(self, frame, boxes, track_ids, clss, track_history):
        for box, track_id, cls in zip(boxes, track_ids, clss):
            annotator = Annotator(frame, example=str(cfgs.YOLO_LABELS))
            annotator.box_label(box, "track_id:{}".format(track_id), color=colors(track_id, True))
            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center   
            track = track_history[track_id]  # Tracking Lines plot
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=colors(track_id, True), thickness=4)

    def draw_box(self, _image, boxes, labels):
        for _idx, (box, cls) in enumerate(zip(boxes, labels)):
            annotator = Annotator(_image, example=str(cfgs.YOLO_LABELS))
            annotator.box_label(box, self.reid_pipeline._target_class, color=colors(int(cls), True))

    def draw_match(self, _image, boxes, match_ids):
        for _idx, (box, cls) in enumerate(zip(boxes, match_ids)):
            _image = draw_chinese_box(_image, font="./models/SimHei.ttf", box=box, label=cls, color=(0, 255, 0))
        return _image

    def proc_start_run_media_type(self):
        _fps = 0
        all_count = 0
        if self.proc_source_type == 'image':
            start_img = cv2.imread(self.proc_source_url)
            all_count = 1
        elif self.proc_source_type == 'video':
            start_img = cv2.VideoCapture(self.proc_source_url)
            _fps = start_img.get(cv2.CAP_PROP_FPS)
            all_count = start_img.get(cv2.CAP_PROP_FRAME_COUNT)
            track_history = defaultdict(list)
        frame_count = 1
        self.had_track_id_dict = dict()
        while True:
            if self.stop_dtc:
                break
            _inter_type = "image"
            if self.continue_dtc:
                if self.proc_source_type == 'video':
                    _inter_type = "video"
                    flag_read, proc_img = start_img.read()
                    frame_count += 1
                    if not flag_read:
                        self.progress_bar.emit(1000)
                        break
                else:
                    proc_img = start_img
                _image = proc_img.copy()
                if _inter_type == "image":
                    boxes, track_ids, labels = self.reid_pipeline.detect(proc_img,
                                                                         class_idx_list=self.reid_pipeline._target_class_idx_list,
                                                                         format=_inter_type)
                    self.draw_box(_image, boxes, labels)
                else:
                    # sample frame
                    if frame_count % self.skip_frames != 0:
                        continue
                    boxes, track_ids, labels = self.reid_pipeline.detect(proc_img,
                                                                         class_idx_list=self.reid_pipeline._target_class_idx_list,
                                                                         format=_inter_type, is_track=self.is_track)
                    filter_bbox_list = []
                    filter_trackid_list = []
                    had_search_trackid_list = []
                    if self.is_track and track_ids is not None:
                        self.draw_track(_image, boxes, track_ids, labels, track_history)
                        for bbox, track_id in zip(boxes, track_ids):
                            if track_id not in self.had_track_id_dict:
                                self.had_track_id_dict[track_id] = [bbox, None]
                                filter_bbox_list.append(bbox)
                                filter_trackid_list.append(track_id)
                            else:
                                if self.had_track_id_dict[track_id][1] is not None:
                                    had_search_label = self.had_track_id_dict[track_id][1]
                                    had_search_trackid_list.append([bbox, had_search_label])
                        _image = self.draw_match(_image, [row[0] for row in had_search_trackid_list],
                                                 [row[1] for row in had_search_trackid_list])
                        boxes = filter_bbox_list
                    else:
                        self.draw_box(_image, boxes, labels)
                search_labels_list, search_dist_list, target_box_list, before_sort_list = self.reid_pipeline.search(
                    proc_img, boxes, self.match_thresh)
                if _inter_type == 'video':
                    happend_time = round((frame_count + 1) / _fps, 3)
                    if self.is_track and track_ids is not None:
                        for _idx, _e_sort_box in enumerate(boxes):
                            if before_sort_list[_idx] != "unknown":
                                self.had_track_id_dict[filter_trackid_list[_idx]][1] = before_sort_list[_idx]
                        ## had_search_trackid
                        search_labels_list.extend([row[1] for row in had_search_trackid_list])
                        target_box_list.extend([row[0] for row in had_search_trackid_list])
                        search_dist_list.extend(["None"] * len(had_search_trackid_list))
                else:
                    happend_time = 0.0
                if len(target_box_list) > 0:
                    _image = self.draw_match(_image, target_box_list, search_labels_list)
                    for _idx, _e_dist in enumerate(search_dist_list):
                        rows = [self.proc_source_url, self.reid_pipeline._target_class, frame_count, happend_time,
                                search_dist_list[_idx][0], "{}:{}".format("命中", search_labels_list[_idx]),
                                "[{}]".format(','.join(map(str, map(int, target_box_list[_idx][:4]))))]
                        self.table_info_list.emit(rows)
                        # 目标区域裁剪与显示
                        bb = target_box_list[_idx]
                        crop_img = proc_img[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2]), :]
                        # 更新右侧信息面板
                        self.show_target_img.emit(crop_img)
                        self.show_match_dist.emit(str(search_dist_list[_idx][0]))
                        self.show_match_id.emit(search_labels_list[_idx])
                        self.show_match_status.emit("命中")
                # 无匹配结果处理
                else:
                    if self.is_show_no_match_item:
                        rows = [self.proc_source_url, self.reid_pipeline._target_class, frame_count, happend_time,
                                "None", "未命中", "None"]
                        self.table_info_list.emit(rows)
                    crop_img = np.full((128, 128, 3), 255, dtype=np.uint8)
                    self.show_target_img.emit(crop_img)
                    self.show_match_dist.emit("None")
                    self.show_match_id.emit("None")
                    self.show_match_status.emit("未命中")

                self.show_img.emit(_image)
                process_value = int(frame_count / all_count * 1000)
                self.progress_bar.emit(process_value)
                if process_value == 1000:
                    break

    def proc_start_run_camera_type(self):
        start_img = cv2.VideoCapture(self.proc_source_url)
        _fps = start_img.get(cv2.CAP_PROP_FPS)
        all_count = start_img.get(cv2.CAP_PROP_FRAME_COUNT)

    def proc_start_run_func(self):
        try:
            if self.proc_source_type == 'image' or self.proc_source_type == 'video':
                self.proc_start_run_media_type()
            elif self.proc_source_type == 'dir':
                self.proc_start_run_dir_type()
            elif self.proc_source_type == 'camera':
                self.proc_start_run_camera_type()
        except Exception as e:
            print(traceback.print_exc())
            self.debug_msg.emit("%s" % e)
        finally:
            self.progress_bar.emit(1000)


class PageProcess:
    main_process_thread = Signal()
    is_save_video = False
    is_save_csv = False
    video_writer = None

    def set_proc_page(self):
        # 初始化摄像头列表和特征数据库
        self.available_cameras = self.find_available_cameras()
        base_feat_lists, base_idx_lists = qt_sql.load_sql_feat_info(cfgs.DB_PATH, cfgs.DB_NAME)
        # 创建处理线程实例
        self.proc_class = ProcessThread(base_feat_lists, base_idx_lists, dims=1280, target_class="person",
                                        device_info="gpu")
        self.reid_pipeline = self.proc_class.reid_pipeline
        self.process_file_button.clicked.connect(self.proc_open_file_func)
        self.init_process_camera()
        self.process_camera_edit.activated.connect(self.proc_open_camera_func)
        self._process_info_model = QStandardItemModel(self)
        self._process_info_model.setHorizontalHeaderLabels(
            ['来源', '类型', "帧数", "时间", '距离', '是否命中', "位置[x1,y1,x2,y2]"])
        self.process_table_show.setModel(self._process_info_model)
        # self.process_table_show.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 信号-槽连接：处理线程 → 界面组件
        self.proc_class.debug_msg.connect(lambda x: self.show_status(x))
        self.proc_class.show_img.connect(lambda x: self.show_image(x, self.det_pano_img))
        self.proc_class.show_target_img.connect(lambda x: self.show_image(x, self.match_show_img))
        self.proc_class.show_match_id.connect(lambda x: self.match_show_id.setText(x))
        self.proc_class.show_match_status.connect(lambda x: self.match_show_status.setText(x))
        self.proc_class.show_match_dist.connect(lambda x: self.match_show_dist.setText(x))
        self.proc_class.progress_bar.connect(lambda x: self.show_progress_bar(x))
        self.proc_class.table_info_list.connect(lambda x: self.show_table_proc_stage(x))

        # 创建并配置工作线程
        self.process_thread = QThread()
        self.main_process_thread.connect(self.proc_class.proc_start_run_func)
        self.proc_class.moveToThread(self.process_thread)

        # 按钮事件绑定
        self.proc_run_button.clicked.connect(self.proc_run_or_continue)
        self.proc_stop_button.clicked.connect(self.proc_stop)

        self.istrack_checkbox.toggled.connect(self.istrack_checkbox_setting)
        self.save_media_checkbox.toggled.connect(self.save_media_checkbox_setting)
        self.save_csv_checkbox.toggled.connect(self.save_csv_checkbox_setting)

    def init_process_camera(self):
        for camera_index in self.available_cameras:
            self.process_camera_edit.addItem(f"   摄像头{camera_index}")

    def find_available_cameras(self):
        available_cameras = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                available_cameras.append(index)
            cap.release()
            index += 1
        return available_cameras

    def register_video_writer(self):
        date_now = datetime.datetime.now()
        year = date_now.year
        month = date_now.month
        day = date_now.day
        hour = date_now.hour
        second = date_now.second
        result_dir = f"{year}_{month}_{hour}_{day}_{second}.mp4"
        file_path = os.path.join("./outputs/video", result_dir)
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        video_writer = cv2.VideoWriter(file_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       max(int(25 // self.proc_class.skip_frames), 25),
                                       (1280, 720))
        return video_writer

    def istrack_checkbox_setting(self):
        self.proc_class.reid_pipeline.reset_track()
        if self.istrack_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: {} state had been changed to False.'.format('istrack_checkbox'))
            self.proc_class.is_track = False
        elif self.istrack_checkbox.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: {} state had been changed to True.'.format('istrack_checkbox'))
            self.proc_class.is_track = True
            self.proc_class.had_track_id_dict = dict()

    def save_media_checkbox_setting(self):
        if self.save_media_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: {} state had been changed to False.'.format('save_media_checkbox'))
            self.is_save_video = False
        elif self.save_media_checkbox.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: {} state had been changed to True.'.format('save_media_checkbox'))
            self.is_save_video = True

    def save_csv_checkbox_setting(self):
        if self.save_csv_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: {} state had been changed to False.'.format('save_csv_checkbox'))
            self.is_save_csv = False
        elif self.save_csv_checkbox.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: {} state had been changed to True.'.format('save_csv_checkbox'))
            self.is_save_csv = True

    def proc_run_or_continue(self):
        if self.proc_class.proc_source_url == '':
            self.show_status('Please select the media video/image or dir source before starting detection...')
            self.proc_run_button.setChecked(False)
        else:
            self.proc_class.stop_dtc = False
            if self.proc_run_button.isChecked():
                self.proc_run_button.setChecked(True)
                self.show_status("Reid process......")
                self.proc_class.continue_dtc = True
                self.process_thread.start()
                self.main_process_thread.emit()
            else:
                self.proc_class.continue_dtc = False
                self.show_status("Pause...")
                self.proc_run_button.setChecked(False)

    def proc_stop(self):
        if self.process_thread.isRunning():
            self.process_thread.quit()
        self.proc_class.stop_dtc = True
        self.proc_class.reid_pipeline.reset_track()
        self.proc_run_button.setChecked(False)
        self.progress_bar.setValue(0)
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def show_status(self, msg):
        print(msg)

    def show_progress_bar(self, x):
        if x == 1000:
            self.proc_stop()
            if self.is_save_csv:
                self.save_as_csv()
        else:
            self.progress_bar.setValue(x)

    def set_sample_frame(self, num):
        self.proc_class.skip_frames = num

    def show_table_proc_stage(self, rows):
        items = [QStandardItem(str(item)) for item in rows]
        self._process_info_model.appendRow(items)
        self.process_table_show.scrollToBottom()

    def set_dist_thresh(self, dist):
        self.dist_rank_thresh = dist

    def proc_open_camera_func(self):
        print("proc_open_camera_func")
        name = self.process_camera_edit.currentIndex()
        self.proc_class.proc_source_url = name
        self.proc_class.proc_source_type = 'video'
        self.process_file_edit.setText("请选择media文件")
        self.proc_stop()

    def proc_open_file_func(self):
        config_file = './config/proc_fold.json'
        if os.path.exists(config_file):
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            open_fold = config['open_fold']
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()
        else:
            config = dict()
            open_fold = config['open_fold'] = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold,
                                              "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png *.jpeg)")
        if name:
            self.proc_class.proc_source_url = name
            suffix_name = os.path.basename(name)
            config['open_fold'] = os.path.dirname(name)
            if suffix_name.endswith(".jpg") or suffix_name.endswith(".png") or suffix_name.endswith(".jpeg"):
                self.proc_class.proc_source_type = 'image'
            if suffix_name.endswith(".mp4") or suffix_name.endswith(".mkv") or suffix_name.endswith(
                    ".avi") or suffix_name.endswith(".flv"):
                self.proc_class.proc_source_type = 'video'
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.process_file_edit.setText(name)
            self.process_camera_edit.clear()
            self.init_process_camera()
            self.proc_stop()

    def show_image(self, img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))
            print(traceback.print_exc())
        finally:
            if label.objectName() == "det_pano_img" and self.is_save_video:
                if self.video_writer is None:
                    self.video_writer = self.register_video_writer()
                img_src_ = cv2.resize(img_src, (1280, 720))
                self.video_writer.write(img_src_)

    def save_as_csv(self):
        date_now = datetime.datetime.now()
        year = date_now.year
        month = date_now.month
        day = date_now.day
        hour = date_now.hour
        second = date_now.second
        result_dir = f"{year}_{month}_{hour}_{day}_{second}.csv"
        file_path = os.path.join("./outputs/csv/", result_dir)
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [self._process_info_model.horizontalHeaderItem(i).text()
                      for i in range(self._process_info_model.columnCount())]
            writer.writerow(header)  # 写入表头
            for row in range(self._process_info_model.rowCount()):
                row_data = []
                for column in range(self._process_info_model.columnCount()):
                    item = self._process_info_model.item(row, column)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append("")  # 如果项目为空，则写入空字符串
                writer.writerow(row_data)
