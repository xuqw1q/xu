import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from ffpyplayer.player import MediaPlayer
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import queue
import cv2
import os
from collections import deque


class FFPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Educational Video Player (Slide Detection) - Audio-Video Sync Optimized")
        self.root.geometry("1200x800")

        self.player = None
        self.video_path = None
        self.playing = False
        self.paused = False
        self.duration = 0.0
        self.current_pos = 0.0
        self.slider_updating = False
        self.img_on_canvas = None
        self.seeking = False
        self.play_thread = None
        self.gui_thread = None
        self.should_stop = False

        # ===== 优化的音视频同步参数 =====
        self.video_fps = 25.0
        self.last_frame_time = 0.0
        self.frame_start_time = 0.0
        self.playback_start_pos = 0.0

        # 同步纠正参数
        self.sync_threshold = 0.100  # 100ms阈值开始纠正
        self.max_sync_offset = 0.200  # 200ms内尝试软纠正
        self.hard_sync_threshold = 0.500  # 500ms以上强制重新同步

        # 同步统计
        self.sync_corrections = 0
        self.sync_stats = deque(maxlen=50)
        self.last_sync_time = 0.0

        # 帧时间控制
        self.target_frame_interval = 1.0 / 30.0  # 目标30fps显示
        self.last_display_time = 0.0
        self.frame_skip_count = 0

        # 播放器稳定性参数
        self.player_restart_threshold = 2.0  # 2秒以上偏移重启播放器
        self.consecutive_sync_failures = 0
        self.max_sync_failures = 5

        # New sync debugging parameters
        self.audio_time = 0.0
        self.video_time = 0.0
        self.system_time_offset = 0.0
        self.last_sync_check_time = 0.0
        self.sync_history = deque(maxlen=15)

        # Slide detection related
        self.slides_detected = []
        self.slide_buttons = []
        self.detection_in_progress = False
        self.detection_thread = None

        # Current focused slide information
        self.current_slide_index = -1
        self.slide_start_time = 0.0
        self.slide_end_time = 0.0
        self.is_slide_focused = False

        # 优化的帧队列
        self.frame_queue = queue.Queue(maxsize=2)  # 减小队列大小提高响应性
        self.canvas_width = 800
        self.canvas_height = 450

        # Enhanced thread locks
        self.player_lock = threading.RLock()
        self.seek_lock = threading.Lock()
        self.seek_in_progress = False
        self.sync_lock = threading.Lock()  # 新增同步锁

        # Create interface
        self.setup_ui()
        self.update_progress()
        self.start_gui_update_thread()
        self.start_sync_monitor_thread()  # 新增同步监控线程

    def start_sync_monitor_thread(self):
        """启动同步监控线程 - 主动纠正同步偏差"""

        def sync_monitor_loop():
            while not self.should_stop:
                if self.playing and not self.seeking and not self.seek_in_progress:
                    self.check_and_correct_sync()
                time.sleep(0.1)  # 100ms检查一次

        sync_thread = threading.Thread(target=sync_monitor_loop, daemon=True)
        sync_thread.start()

    def check_and_correct_sync(self):
        """检查并主动纠正同步偏差"""
        try:
            with self.player_lock:
                if not self.player:
                    return

                # 获取当前播放位置
                pts = self.player.get_pts()
                if pts is None:
                    return

                current_time = time.time()
                actual_video_time = float(pts)

                # 计算期望的视频时间
                elapsed_time = current_time - self.frame_start_time
                expected_video_time = self.playback_start_pos + elapsed_time

                # 计算同步偏差
                sync_offset = actual_video_time - expected_video_time

                with self.sync_lock:
                    self.sync_stats.append({
                        'time': current_time,
                        'offset': sync_offset,
                        'actual': actual_video_time,
                        'expected': expected_video_time
                    })

                # 更新显示
                self.update_sync_display(sync_offset)

                # 同步纠正逻辑
                abs_offset = abs(sync_offset)

                if abs_offset > self.hard_sync_threshold:
                    # 严重偏差：强制重新同步
                    print(f"Hard sync correction: {sync_offset * 1000:.0f}ms")
                    self.hard_sync_correction(actual_video_time)
                    self.consecutive_sync_failures = 0

                elif abs_offset > self.max_sync_offset:
                    # 中等偏差：软纠正
                    self.consecutive_sync_failures += 1
                    if self.consecutive_sync_failures > self.max_sync_failures:
                        print(f"Multiple sync failures, hard reset: {sync_offset * 1000:.0f}ms")
                        self.hard_sync_correction(actual_video_time)
                        self.consecutive_sync_failures = 0
                    else:
                        self.soft_sync_correction(sync_offset)

                elif abs_offset > self.sync_threshold:
                    # 轻微偏差：微调
                    self.micro_sync_correction(sync_offset)
                    self.consecutive_sync_failures = max(0, self.consecutive_sync_failures - 1)
                else:
                    # 同步正常
                    self.consecutive_sync_failures = max(0, self.consecutive_sync_failures - 1)

        except Exception as e:
            pass

    def hard_sync_correction(self, actual_time):
        """强制同步纠正：重置时间基线"""
        try:
            with self.sync_lock:
                self.frame_start_time = time.time()
                self.playback_start_pos = actual_time
                self.sync_corrections += 1
                print(f"Hard sync reset to {actual_time:.2f}s")
        except Exception as e:
            pass

    def soft_sync_correction(self, offset):
        """软同步纠正：调整播放速度或跳帧"""
        try:
            with self.player_lock:
                if not self.player:
                    return

                # 如果视频落后太多，跳过一些帧来追赶
                if offset < -0.3:  # 视频落后300ms以上
                    self.skip_frames_to_catch_up(abs(offset))

                # 调整时间基线（温和调整）
                adjustment = offset * 0.1  # 只调整偏差的10%，避免震荡
                with self.sync_lock:
                    self.playback_start_pos += adjustment

        except Exception as e:
            pass

    def micro_sync_correction(self, offset):
        """微同步纠正：细微调整"""
        try:
            # 对于小偏差，通过调整帧显示时间来纠正
            with self.sync_lock:
                # 微调时间基线
                adjustment = offset * 0.05  # 只调整5%
                self.playback_start_pos += adjustment
        except Exception as e:
            pass

    def skip_frames_to_catch_up(self, delay):
        """通过跳帧来追赶同步"""
        try:
            frames_to_skip = min(5, int(delay * self.video_fps))  # 最多跳5帧

            with self.player_lock:
                if not self.player:
                    return

                for _ in range(frames_to_skip):
                    frame, val = self.player.get_frame()
                    if val == 'eof' or frame is None:
                        break

                self.frame_skip_count += frames_to_skip
                print(f"Skipped {frames_to_skip} frames to catch up")

        except Exception as e:
            pass

    def update_sync_display(self, offset):
        """更新同步状态显示"""
        try:
            abs_offset = abs(offset)
            offset_ms = offset * 1000

            if abs_offset < self.sync_threshold:
                color = "green"
                status = "Sync: Normal"
            elif abs_offset < self.max_sync_offset:
                color = "orange"
                status = f"Sync: Minor Offset"
            elif abs_offset < self.hard_sync_threshold:
                color = "red"
                status = f"Sync: Major Offset"
            else:
                color = "darkred"
                status = f"Sync: Critical Offset"

            self.root.after(0, lambda: self.sync_label.config(text=status, fg=color))
            self.root.after(0, lambda: self.sync_detail_label.config(
                text=f"A/V: {offset_ms:.0f}ms | Corrections: {self.sync_corrections}"))

        except Exception as e:
            pass

    def create_optimized_player(self):
        """创建优化的播放器实例"""
        try:
            # 尝试使用优化的音视频同步参数
            ff_opts = {
                'sync': 'audio',  # 以音频为准同步
                'framedrop': '1',  # 允许丢帧
                'autoexit': '0',  # 不自动退出
                'avioflags': 'direct',  # 直接访问
            }

            player = MediaPlayer(self.video_path, ff_opts=ff_opts)
            return player

        except Exception as e:
            try:
                # 备用方案：基础播放器
                player = MediaPlayer(self.video_path)
                return player
            except Exception as e2:
                raise e2

    def setup_ui(self):
        # Create main layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left main frame (video and controls)
        left_main_frame = tk.Frame(main_frame)
        left_main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Video area
        video_frame = tk.Frame(left_main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True)

        # Video canvas
        self.canvas = tk.Canvas(video_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack()

        # Control buttons area
        self.control_frame = tk.Frame(video_frame)
        self.control_frame.pack(fill=tk.X, pady=5)

        self.btn_open = tk.Button(self.control_frame, text="Import Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_play = tk.Button(self.control_frame, text="Play", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_detect = tk.Button(self.control_frame, text="Detect Slides", command=self.detect_slides,
                                    state=tk.DISABLED)
        self.btn_detect.pack(side=tk.LEFT, padx=5)

        # Exit slide focus mode button
        self.btn_exit_focus = tk.Button(self.control_frame, text="Show Full Progress",
                                        command=self.exit_slide_focus, state=tk.DISABLED)
        self.btn_exit_focus.pack(side=tk.LEFT, padx=5)

        # 新增：手动同步纠正按钮
        self.btn_sync_reset = tk.Button(self.control_frame, text="Reset Sync",
                                        command=self.manual_sync_reset, bg="yellow")
        self.btn_sync_reset.pack(side=tk.LEFT, padx=5)

        # Sync status display area
        sync_frame = tk.Frame(self.control_frame)
        sync_frame.pack(side=tk.LEFT, padx=10)

        self.sync_label = tk.Label(sync_frame, text="Sync: Normal", fg="green")
        self.sync_label.pack()

        self.sync_detail_label = tk.Label(sync_frame, text="A/V: 0ms | Corrections: 0",
                                          font=("Arial", 8), fg="blue")
        self.sync_detail_label.pack()

        # Progress bar and time display
        self.progress_frame = tk.Frame(video_frame)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.time_label = tk.Label(self.progress_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT, padx=5)

        # Slide range label
        self.slide_range_label = tk.Label(self.progress_frame, text="", fg="blue", font=("Arial", 9))
        self.slide_range_label.pack(side=tk.LEFT, padx=5)

        self.scale = ttk.Scale(self.progress_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=600)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Bind mouse events
        self.scale.bind("<Button-1>", self.on_slider_click)
        self.scale.bind("<B1-Motion>", self.on_slider_drag)
        self.scale.bind("<ButtonRelease-1>", self.on_slider_release)

        # Slide detection status
        self.detection_status_frame = tk.Frame(left_main_frame)
        self.detection_status_frame.pack(fill=tk.X, pady=(10, 0))

        self.detection_status_label = tk.Label(self.detection_status_frame, text="Slide Detection Status: Not Started",
                                               fg="blue")
        self.detection_status_label.pack()

        self.detection_progress = ttk.Progressbar(self.detection_status_frame, mode='determinate')
        self.detection_progress.pack(fill=tk.X, pady=5)

        # Right side: Slide buttons area - 改进版本
        right_frame = tk.Frame(main_frame, width=320)  # 增加一些宽度
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # Slide button area title with slide count
        self.slides_title_frame = tk.Frame(right_frame)
        self.slides_title_frame.pack(fill=tk.X, pady=(0, 10))

        self.slides_label = tk.Label(self.slides_title_frame, text="Quick Jump to Slides:",
                                     font=("Arial", 12, "bold"))
        self.slides_label.pack(anchor=tk.W)

        self.slides_count_label = tk.Label(self.slides_title_frame, text="(No slides detected)",
                                           font=("Arial", 9), fg="gray")
        self.slides_count_label.pack(anchor=tk.W)

        # 滚动条控制提示
        scroll_hint_label = tk.Label(right_frame, text="↕ Use mouse wheel or scrollbar to navigate",
                                     font=("Arial", 8), fg="blue")
        scroll_hint_label.pack(anchor=tk.W, pady=(0, 5))

        # Create enhanced scrollable slide button container
        slides_border_frame = tk.Frame(right_frame, relief=tk.SUNKEN, borderwidth=2, bg="gray90")
        slides_border_frame.pack(fill=tk.BOTH, expand=True)

        # 创建带更明显滚动条的容器
        self.slides_canvas = tk.Canvas(slides_border_frame, bg="white", highlightthickness=0)

        # 设置更宽更明显的滚动条
        self.slides_scrollbar = ttk.Scrollbar(slides_border_frame, orient="vertical",
                                              command=self.slides_canvas.yview)

        self.slides_frame = tk.Frame(self.slides_canvas, bg="white")

        # 配置滚动功能
        self.slides_frame.bind(
            "<Configure>",
            lambda e: self.slides_canvas.configure(scrollregion=self.slides_canvas.bbox("all"))
        )

        self.slides_canvas.create_window((0, 0), window=self.slides_frame, anchor="nw")
        self.slides_canvas.configure(yscrollcommand=self.slides_scrollbar.set)

        # 布局，确保滚动条明显可见
        self.slides_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.slides_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定更好的鼠标滚轮事件
        self.bind_scrolling_events()

        # 添加键盘导航支持
        self.setup_keyboard_navigation()

        # 在slides_frame中添加一个默认提示
        self.no_slides_label = tk.Label(self.slides_frame,
                                        text="No slides detected yet.\nClick 'Detect Slides' to analyze video.",
                                        font=("Arial", 10), fg="gray", bg="white",
                                        justify=tk.CENTER, wraplength=280)
        self.no_slides_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=50)

    def bind_scrolling_events(self):
        """绑定更好的滚动事件"""

        # 为不同平台绑定鼠标滚轮事件
        def on_mousewheel(event):
            # 检查鼠标是否在slides区域内
            if self.is_mouse_in_slides_area(event):
                # Windows和Linux
                if event.delta:
                    self.slides_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                # MacOS
                elif event.num == 4:
                    self.slides_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.slides_canvas.yview_scroll(1, "units")

        # 绑定多种滚轮事件
        self.root.bind_all("<MouseWheel>", on_mousewheel)  # Windows
        self.root.bind_all("<Button-4>", on_mousewheel)  # Linux/MacOS
        self.root.bind_all("<Button-5>", on_mousewheel)  # Linux/MacOS

        # 为slides_canvas特别绑定
        self.slides_canvas.bind("<MouseWheel>", on_mousewheel)
        self.slides_canvas.bind("<Button-4>", on_mousewheel)
        self.slides_canvas.bind("<Button-5>", on_mousewheel)

    def is_mouse_in_slides_area(self, event):
        """检查鼠标是否在slides区域内"""
        try:
            # 获取slides_canvas的屏幕坐标
            canvas_x = self.slides_canvas.winfo_rootx()
            canvas_y = self.slides_canvas.winfo_rooty()
            canvas_width = self.slides_canvas.winfo_width()
            canvas_height = self.slides_canvas.winfo_height()

            mouse_x = event.x_root
            mouse_y = event.y_root

            return (canvas_x <= mouse_x <= canvas_x + canvas_width and
                    canvas_y <= mouse_y <= canvas_y + canvas_height)
        except:
            return True  # 如果检查失败，允许滚动

    def setup_keyboard_navigation(self):
        """设置键盘导航"""

        def on_key_press(event):
            if not self.slides_detected:
                return

            # 上下箭头键导航slides
            if event.keysym == "Up":
                self.slides_canvas.yview_scroll(-3, "units")
            elif event.keysym == "Down":
                self.slides_canvas.yview_scroll(3, "units")
            elif event.keysym == "Page_Up":
                self.slides_canvas.yview_scroll(-10, "units")
            elif event.keysym == "Page_Down":
                self.slides_canvas.yview_scroll(10, "units")
            elif event.keysym == "Home":
                self.slides_canvas.yview_moveto(0)
            elif event.keysym == "End":
                self.slides_canvas.yview_moveto(1)

        # 绑定键盘事件
        self.root.bind("<Up>", on_key_press)
        self.root.bind("<Down>", on_key_press)
        self.root.bind("<Page_Up>", on_key_press)
        self.root.bind("<Page_Down>", on_key_press)
        self.root.bind("<Home>", on_key_press)
        self.root.bind("<End>", on_key_press)

    def manual_sync_reset(self):
        """手动重置同步"""
        if self.playing and self.player:
            try:
                with self.player_lock:
                    pts = self.player.get_pts()
                    if pts is not None:
                        self.hard_sync_correction(float(pts))
                        messagebox.showinfo("Sync Reset", "Audio-video sync has been reset")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset sync: {str(e)}")

    def play_loop_improved(self):
        """优化的播放循环 - 重点解决音视频同步"""
        sync_check_interval = 0.1  # 100ms检查一次同步
        last_sync_check = 0
        frame_count = 0

        while not self.should_stop:
            if not self.playing or self.seeking or self.seek_in_progress:
                time.sleep(0.02)
                continue

            loop_start_time = time.time()

            try:
                with self.player_lock:
                    if not self.player:
                        break

                    frame, val = self.player.get_frame()

                if val == 'eof':
                    self.root.after(0, self.on_playback_finished)
                    break

                if frame is not None:
                    frame_count += 1

                    # 更新当前位置
                    with self.player_lock:
                        if self.player and not self.seek_in_progress:
                            try:
                                pts = self.player.get_pts()
                                if pts is not None:
                                    self.current_pos = float(pts)
                            except:
                                pass

                    # 控制帧显示频率
                    current_time = time.time()
                    if current_time - self.last_display_time >= self.target_frame_interval:
                        try:
                            # 清理队列避免积压
                            while not self.frame_queue.empty():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    break

                            self.frame_queue.put_nowait(frame)
                            self.last_display_time = current_time

                        except queue.Full:
                            # 队列满时跳过当前帧
                            pass

                    # 优化的帧时间控制
                    img, t = frame
                    if t > 0:
                        # 使用播放器提供的时间，但限制范围避免卡顿
                        sleep_time = min(max(t, 0.001), 0.1)  # 1ms到100ms之间
                        time.sleep(sleep_time)
                    else:
                        # 备用时间控制
                        time.sleep(0.033)  # 约30fps
                else:
                    time.sleep(0.01)

            except Exception as e:
                if not self.should_stop:
                    time.sleep(0.01)

    def start_playback(self):
        """优化的播放启动"""
        try:
            with self.player_lock:
                if self.player is None:
                    self.player = self.create_optimized_player()

                    # 获取视频信息
                    if self.duration <= 0:
                        metadata = self.player.get_metadata()
                        if metadata and 'duration' in metadata and metadata['duration'] is not None:
                            self.duration = float(metadata['duration'])
                            self.scale.configure(to=self.duration)

                    # 如果需要从特定位置开始播放
                    if self.current_pos > 0:
                        try:
                            self.player.seek(self.current_pos, relative=False)
                            time.sleep(0.1)
                        except:
                            pass

                # 设置播放状态
                if hasattr(self.player, 'set_pause'):
                    self.player.set_pause(False)

            self.playing = True
            self.paused = False
            self.btn_play.config(text="Pause")

            # 重置同步基线
            self.reset_sync_timing(self.current_pos)

            # 启动播放线程
            if self.play_thread is None or not self.play_thread.is_alive():
                self.play_thread = threading.Thread(target=self.play_loop_improved, daemon=True)
                self.play_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Playback failed: {str(e)}")

    def reset_sync_timing(self, new_position=None):
        """重置同步时间基线"""
        with self.sync_lock:
            current_time = time.time()
            if new_position is not None:
                self.playback_start_pos = float(new_position)
            else:
                self.playback_start_pos = self.current_pos

            self.frame_start_time = current_time
            self.last_sync_check_time = 0.0
            self.sync_history.clear()
            self.consecutive_sync_failures = 0
            print(f"Sync timing reset to position {self.playback_start_pos:.2f}s")

    def exit_slide_focus(self):
        """Exit slide focus mode"""
        self.is_slide_focused = False
        self.current_slide_index = -1
        self.scale.configure(from_=0, to=self.duration)
        self.slide_range_label.config(text="")
        self.btn_exit_focus.config(state=tk.DISABLED)

        for btn in self.slide_buttons:
            btn.config(bg="lightgray", relief=tk.RAISED)

    def jump_to_slide(self, target_time, slide_index):
        """跳转到幻灯片"""
        if not self.video_path:
            return

        self.current_slide_index = slide_index - 1
        self.is_slide_focused = True

        self.slide_start_time = target_time
        if self.current_slide_index < len(self.slides_detected) - 1:
            self.slide_end_time = self.slides_detected[self.current_slide_index + 1]
        else:
            self.slide_end_time = self.duration

        self.scale.configure(from_=self.slide_start_time, to=self.slide_end_time)
        self.scale.set(target_time)

        start_str = self.format_time(self.slide_start_time)
        end_str = self.format_time(self.slide_end_time)
        self.slide_range_label.config(text=f"[Slide {slide_index}: {start_str}-{end_str}]")
        self.btn_exit_focus.config(state=tk.NORMAL)

        for i, btn in enumerate(self.slide_buttons):
            if i == slide_index - 1:
                btn.config(bg="lightblue", relief=tk.SUNKEN)
            else:
                btn.config(bg="lightgray", relief=tk.RAISED)

        was_playing = self.playing
        seek_thread = threading.Thread(
            target=self.perform_optimized_seek,
            args=(target_time, was_playing),
            daemon=True
        )
        seek_thread.start()

    def perform_optimized_seek(self, target_pos, was_playing):
        """优化的seek操作"""
        with self.seek_lock:
            if self.seek_in_progress:
                return
            self.seek_in_progress = True

        try:
            self.current_pos = target_pos
            self.root.after(0, lambda: self.scale.set(target_pos))

            # 暂停播放
            self.playing = False

            # 清理帧队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            with self.player_lock:
                # 重新创建播放器以确保seek准确性
                if self.player:
                    try:
                        self.player.close_player()
                    except:
                        pass
                    self.player = None
                    time.sleep(0.05)

                # 创建新播放器
                self.player = self.create_optimized_player()
                time.sleep(0.05)

                # 执行seek
                try:
                    self.player.seek(target_pos, relative=False)
                    self.wait_for_seek_verification(target_pos)
                except Exception as e:
                    print(f"Seek error: {e}")

            # 更新时间显示
            self.root.after(0, lambda: self.update_time_display(target_pos, self.duration))

            # 恢复播放状态
            if was_playing:
                with self.player_lock:
                    if self.player:
                        try:
                            actual_pts = self.player.get_pts()
                            if actual_pts is not None:
                                self.reset_sync_timing(float(actual_pts))
                            else:
                                self.reset_sync_timing(target_pos)
                        except:
                            self.reset_sync_timing(target_pos)

                self.playing = True
                self.paused = False
                self.root.after(0, lambda: self.btn_play.config(text="Pause"))

                with self.player_lock:
                    if self.player and hasattr(self.player, 'set_pause'):
                        try:
                            self.player.set_pause(False)
                        except:
                            pass

                if self.play_thread is None or not self.play_thread.is_alive():
                    self.play_thread = threading.Thread(target=self.play_loop_improved, daemon=True)
                    self.play_thread.start()
            else:
                self.reset_sync_timing(target_pos)
                self.paused = True
                self.root.after(0, lambda: self.btn_play.config(text="Play"))

        except Exception as e:
            print(f"Optimized seek error: {e}")
        finally:
            def release_lock():
                self.seek_in_progress = False
                self.seeking = False
                self.slider_updating = False

            self.root.after(100, release_lock)

    def wait_for_seek_verification(self, target_pos, max_attempts=10):
        """等待seek完成并验证"""
        for attempt in range(max_attempts):
            try:
                with self.player_lock:
                    if not self.player:
                        break

                    frame, val = self.player.get_frame()
                    if val == 'eof' or frame is None:
                        continue

                    try:
                        current_pts = self.player.get_pts()
                        if current_pts is not None:
                            pos_diff = abs(float(current_pts) - target_pos)
                            if pos_diff <= 1.0:  # 1秒误差容忍
                                try:
                                    self.frame_queue.put_nowait(frame)
                                except queue.Full:
                                    pass
                                return
                    except:
                        pass

                    if attempt == 0:
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass

                time.sleep(0.02)
            except:
                time.sleep(0.02)

    def open_video(self):
        """打开视频文件"""
        file_path = filedialog.askopenfilename(
            title="Select Educational Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("MP4 Files", "*.mp4"),
                ("AVI Files", "*.avi"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.video_path = file_path
        self.stop_playback()
        self.reset_player()
        self.clear_slide_buttons()
        self.exit_slide_focus()

        try:
            temp_player = MediaPlayer(self.video_path)
            self.duration = 0.0

            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if self.video_fps <= 0 or self.video_fps > 120:
                    self.video_fps = 25.0
                cap.release()
            else:
                self.video_fps = 25.0

            metadata = temp_player.get_metadata()
            if metadata and 'duration' in metadata and metadata['duration'] is not None:
                self.duration = float(metadata['duration'])

            temp_player.close_player()

            if self.duration > 0:
                self.scale.configure(to=self.duration)
                self.btn_detect.config(state=tk.NORMAL)
            else:
                self.duration = 600.0
                self.scale.configure(to=self.duration)

            self.scale.set(0)
            self.update_time_display(0.0, self.duration)

        except Exception as e:
            self.duration = 600.0
            self.video_fps = 25.0
            self.scale.configure(to=self.duration)
            messagebox.showerror("Error", f"Cannot get video information: {str(e)}")

    def detect_slides(self):
        """检测幻灯片切换点"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first")
            return

        if self.detection_in_progress:
            messagebox.showinfo("Info", "Slide detection is in progress...")
            return

        self.detection_in_progress = True
        self.btn_detect.config(state=tk.DISABLED, text="Detecting...")
        self.detection_status_label.config(text="Slide Detection Status: Analyzing...", fg="orange")

        self.detection_thread = threading.Thread(target=self.perform_slide_detection, daemon=True)
        self.detection_thread.start()

    def perform_slide_detection(self):
        """智能幻灯片检测：区分幻灯片切换与教师手写注释"""

        try:
            self.slides_detected.clear()
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file for analysis")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else self.duration
            skip_frames = max(1, int(fps * 1))
            min_slide_duration = 2  # 幻灯片间最小间隔

            # 保存历史帧数据
            prev_frame_gray = None
            prev_hist = None
            frame_history = []  # 保存最近几帧的变化特征

            # 手写内容检测相关
            writing_detection_active = False
            writing_start_time = 0
            consecutive_writing_frames = 0

            slide_times = [0.0]  # 第一帧作为第一张幻灯片
            frame_count = 0
            processed_frames = 0

            # 区域重要性配置
            region_importance = {
                'global': 1.0,  # 全局对比
                'edges': 1.3,  # 边缘检测
                'histdiff': 1.2,  # 直方图差异
                'movement': 0.6  # 移动检测（低权重减少手写误判）
            }

            self.root.after(0, lambda: self.detection_progress.config(maximum=total_frames // skip_frames, value=0))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue

                processed_frames += 1
                current_time = frame_count / fps

                # 更新UI进度
                if processed_frames % 10 == 0:
                    progress = processed_frames
                    self.root.after(0, lambda p=progress: self.detection_progress.config(value=p))
                    self.root.after(0, lambda t=current_time: self.detection_status_label.config(
                        text=f"Detection Progress: {current_time:.1f}s / {video_duration:.1f}s"))

                # 基础图像处理
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 240))

                # 边缘检测（对形状变化敏感，对文字添加不太敏感）
                edges = cv2.Canny(gray, 50, 150)
                edge_pixels = np.sum(edges > 0)

                # 计算全局特征
                hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                if prev_frame_gray is not None and prev_hist is not None:
                    # 计算指标
                    frame_diff = cv2.absdiff(prev_frame_gray, gray)
                    mean_diff = np.mean(frame_diff)

                    # 计算变化区域的占比
                    change_mask = frame_diff > 30  # 阈值决定什么算"变化"
                    changed_area_ratio = np.sum(change_mask) / (gray.shape[0] * gray.shape[1])

                    # 计算变化区域的集中度
                    if np.sum(change_mask) > 0:
                        # 获取变化区域的标记
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                            change_mask.astype(np.uint8))

                        # 计算最大连通区域占比
                        if num_labels > 1:  # 0是背景
                            largest_label = 1
                            largest_area = stats[1, cv2.CC_STAT_AREA]

                            for i in range(2, num_labels):
                                if stats[i, cv2.CC_STAT_AREA] > largest_area:
                                    largest_area = stats[i, cv2.CC_STAT_AREA]
                                    largest_label = i

                            largest_change_ratio = largest_area / np.sum(change_mask)
                        else:
                            largest_change_ratio = 1.0
                    else:
                        largest_change_ratio = 0.0

                    # 直方图相关性（对整体内容变化敏感）
                    hist_corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    hist_diff_score = (1.0 - hist_corr) * 100

                    # 边缘差异（对形状/结构变化敏感）
                    if hasattr(self, '_prev_edges'):
                        edge_diff_ratio = abs(edge_pixels - self._prev_edges) / (self._prev_edges + 1)
                    else:
                        edge_diff_ratio = 0.0
                    self._prev_edges = edge_pixels

                    # 汇总当前帧变化特征
                    frame_features = {
                        'time': current_time,
                        'mean_diff': mean_diff,
                        'hist_diff': hist_diff_score,
                        'changed_area': changed_area_ratio,
                        'change_concentration': largest_change_ratio,
                        'edge_diff': edge_diff_ratio
                    }

                    # 保存到历史中
                    frame_history.append(frame_features)
                    if len(frame_history) > 15:  # 保留15帧历史
                        frame_history.pop(0)

                    # ===== 幻灯片变化VS手写内容的智能识别 =====

                    # 手写内容特征:
                    # 1. 小区域变化 (changed_area_ratio小)
                    # 2. 变化区域集中 (largest_change_ratio大)
                    # 3. 连续多帧小变化
                    # 4. 直方图差异小

                    is_writing_like = (
                            changed_area_ratio < 0.15 and  # 变化区域小于15%
                            largest_change_ratio > 0.6 and  # 变化集中
                            hist_diff_score < 30 and  # 颜色分布变化小
                            mean_diff < 20  # 整体差异小
                    )

                    # 幻灯片切换特征:
                    # 1. 大区域变化
                    # 2. 直方图差异大
                    # 3. 边缘结构变化大

                    # 综合变化分数 (slide change score)
                    slide_score = (
                            mean_diff * region_importance['global'] +
                            hist_diff_score * region_importance['histdiff'] +
                            edge_diff_ratio * 100 * region_importance['edges']
                    )

                    # 减去手写特征的影响
                    if is_writing_like:
                        consecutive_writing_frames += 1
                        slide_score *= max(0.3, 1.0 - (consecutive_writing_frames * 0.1))  # 连续手写帧越多，分数越低
                    else:
                        consecutive_writing_frames = max(0, consecutive_writing_frames - 1)

                    # 分析最近几帧的变化模式
                    recent_changes = []
                    if len(frame_history) >= 3:
                        for f in frame_history[-3:]:
                            recent_changes.append(f['mean_diff'])

                    # 手写特征: 渐进式变化
                    gradual_change = False
                    if len(recent_changes) >= 3:
                        if all(5 < x < 25 for x in recent_changes):  # 连续小变化
                            gradual_change = True

                    # 进一步抑制手写误判
                    if gradual_change:
                        slide_score *= 0.5

                    # 最终判断
                    is_slide_change = False

                    # 明确是幻灯片切换的情况
                    if (
                            (slide_score > 45 and changed_area_ratio > 0.25) or  # 大变化+大面积
                            (hist_diff_score > 60) or  # 颜色分布大变化
                            (edge_diff_ratio > 0.4) or  # 边缘结构大变化
                            (slide_score > 80)  # 极高分数
                    ):
                        is_slide_change = True

                    # 如果确定是幻灯片变化，且满足最小时间间隔要求
                    if is_slide_change and (current_time - slide_times[-1]) >= min_slide_duration:
                        slide_times.append(current_time)
                        consecutive_writing_frames = 0  # 重置手写计数

                # 更新上一帧数据
                prev_frame_gray = gray.copy()
                prev_hist = hist.copy()

            cap.release()

            def update_slides_data():
                self.slides_detected = slide_times.copy()
                self.create_slide_buttons()
                self.update_slides_count_display()
                self.detection_status_label.config(
                    text=f"Detection complete: Found {len(self.slides_detected)} slides", fg="green")

            self.root.after(0, update_slides_data)

        except Exception as e:
            error_msg = f"Slide detection failed: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Detection Error", error_msg))
            self.root.after(0, lambda: self.detection_status_label.config(
                text="Detection failed", fg="red"))
        finally:
            self.detection_in_progress = False
            self.root.after(0, lambda: self.btn_detect.config(state=tk.NORMAL, text="Re-detect"))
            self.root.after(0, lambda: self.detection_progress.config(value=0))

    def update_slides_count_display(self):
        """更新幻灯片数量显示"""
        count = len(self.slides_detected)
        if count == 0:
            text = "(No slides detected)"
            color = "gray"
        else:
            text = f"({count} slides detected)"
            color = "darkgreen"

        self.slides_count_label.config(text=text, fg=color)

    def create_slide_buttons(self):
        """创建幻灯片按钮"""
        self.clear_slide_buttons()

        # 隐藏默认提示标签
        if hasattr(self, 'no_slides_label'):
            self.no_slides_label.pack_forget()

        if not self.slides_detected:
            # 如果没有slides，显示提示
            if hasattr(self, 'no_slides_label'):
                self.no_slides_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=50)
            return

        for i, slide_time in enumerate(self.slides_detected):
            start_time_str = self.format_time(slide_time)

            if i < len(self.slides_detected) - 1:
                end_time = self.slides_detected[i + 1]
            else:
                end_time = self.duration

            end_time_str = self.format_time(end_time)
            duration_str = self.format_time(end_time - slide_time)

            # 更紧凑的按钮文本
            button_text = f" Slide {i + 1}\n Time Range: {start_time_str} - {end_time_str}\n Duration: {duration_str}"

            btn = tk.Button(
                self.slides_frame,
                text=button_text,
                command=lambda t=slide_time, idx=i + 1: self.jump_to_slide(t, idx),
                width=22, height=3, font=("Arial", 9, "normal"),  # 缩小尺寸和字体
                relief=tk.RAISED, bd=1, bg="lightgray",  # 减小边框
                cursor="hand2",  # 改变鼠标样式
                wraplength=200,  # 减小文本换行宽度
                justify=tk.LEFT,
                padx=2, pady=1  # 减小内部边距
            )

            btn.pack(fill=tk.X, pady=1, padx=5)  # 减小按钮间距
            self.slide_buttons.append(btn)

            # 添加鼠标悬停效果
            def on_enter(event, button=btn):
                button.config(bg="lightblue")

            def on_leave(event, button=btn):
                if self.current_slide_index != i:
                    button.config(bg="lightgray")

            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)

        # 更新滚动区域
        self.slides_canvas.update_idletasks()
        self.slides_canvas.configure(scrollregion=self.slides_canvas.bbox("all"))

    def clear_slide_buttons(self):
        """清除幻灯片按钮"""
        for btn in self.slide_buttons:
            btn.destroy()
        self.slide_buttons.clear()

        # 重新显示默认提示
        if hasattr(self, 'no_slides_label'):
            self.no_slides_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=50)

    def toggle_play(self):
        """播放/暂停切换"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first")
            return

        if not self.playing:
            self.start_playback()
        else:
            self.pause_playback()

    def pause_playback(self):
        """暂停播放"""
        self.playing = False
        self.paused = True
        self.btn_play.config(text="Play")

        with self.player_lock:
            if self.player:
                try:
                    if hasattr(self.player, 'set_pause'):
                        self.player.set_pause(True)
                    pts = self.player.get_pts()
                    if pts is not None:
                        self.current_pos = float(pts)
                except:
                    pass

    def stop_playback(self):
        """停止播放"""
        self.should_stop = True
        self.playing = False
        self.paused = False

        with self.player_lock:
            if self.player:
                try:
                    self.player.close_player()
                except:
                    pass
                self.player = None

    def reset_player(self):
        """重置播放器"""
        self.should_stop = True
        self.playing = False
        self.paused = False
        self.current_pos = 0.0
        self.seek_in_progress = False

        with self.player_lock:
            if self.player:
                try:
                    self.player.close_player()
                except:
                    pass
                self.player = None

        self.canvas.delete("all")
        self.should_stop = False

    def start_gui_update_thread(self):
        """GUI更新线程"""

        def gui_update_loop():
            while not self.should_stop:
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    if frame is not None:
                        self.display_frame_safe(frame)
                except queue.Empty:
                    continue
                except:
                    time.sleep(0.01)

        if self.gui_thread is None or not self.gui_thread.is_alive():
            self.gui_thread = threading.Thread(target=gui_update_loop, daemon=True)
            self.gui_thread.start()

    def display_frame_safe(self, frame):
        """安全显示帧"""
        try:
            img, t = frame
            w, h = img.get_size()
            arr = np.frombuffer(img.to_bytearray()[0], dtype=np.uint8)
            frame_array = arr.reshape(h, w, 3)

            pil_image = Image.fromarray(frame_array)
            if w != self.canvas_width or h != self.canvas_height:
                pil_image = pil_image.resize((self.canvas_width, self.canvas_height))

            photo = ImageTk.PhotoImage(pil_image)
            self.root.after_idle(self.update_canvas_safe, photo)
        except:
            pass

    def update_canvas_safe(self, photo):
        """安全更新画布"""
        try:
            if self.img_on_canvas is None:
                self.img_on_canvas = self.canvas.create_image(
                    self.canvas_width // 2, self.canvas_height // 2, image=photo)
            else:
                self.canvas.itemconfig(self.img_on_canvas, image=photo)
            self.canvas.photo_ref = photo
        except:
            pass

    def update_progress(self):
        """更新进度条"""
        try:
            if self.playing and not self.slider_updating and not self.seeking:
                if self.is_slide_focused:
                    if self.current_pos >= self.slide_end_time:
                        next_slide_index = self.current_slide_index + 2
                        if next_slide_index <= len(self.slides_detected):
                            next_slide_time = self.slides_detected[self.current_slide_index + 1]
                            self.jump_to_slide(next_slide_time, next_slide_index)
                        else:
                            self.exit_slide_focus()
                    else:
                        self.scale.set(self.current_pos)
                else:
                    if self.duration > 0:
                        self.scale.set(self.current_pos)

                self.update_time_display(self.current_pos, self.duration)
        except:
            pass

        if not self.should_stop:
            self.root.after(200, self.update_progress)

    def update_time_display(self, current, total):
        """更新时间显示"""
        current_str = self.format_time(current or 0.0)
        total_str = self.format_time(total or 0.0)
        try:
            self.time_label.config(text=f"{current_str} / {total_str}")
        except:
            pass

    def format_time(self, seconds):
        """格式化时间"""
        try:
            if seconds is None or seconds <= 0:
                return "00:00"
            seconds = float(seconds)
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"
        except:
            return "00:00"

    def on_slider_click(self, event):
        """滑块点击"""
        self.slider_updating = True

    def on_slider_drag(self, event):
        """滑块拖动"""
        self.slider_updating = True

    def on_slider_release(self, event):
        """滑块释放"""
        if not self.video_path:
            self.slider_updating = False
            return

        target_pos = float(self.scale.get())

        if self.is_slide_focused:
            target_pos = max(self.slide_start_time, min(target_pos, self.slide_end_time))
        else:
            target_pos = max(0, min(target_pos, self.duration))

        was_playing = self.playing
        seek_thread = threading.Thread(target=self.perform_optimized_seek,
                                       args=(target_pos, was_playing), daemon=True)
        seek_thread.start()

    def on_playback_finished(self):
        """播放完成"""
        self.playing = False
        self.paused = False
        self.btn_play.config(text="Play")
        self.current_pos = 0.0
        self.scale.set(0)

        for btn in self.slide_buttons:
            btn.config(bg="lightgray", relief=tk.RAISED)

        if self.is_slide_focused:
            self.exit_slide_focus()

        with self.player_lock:
            if self.player:
                try:
                    self.player.close_player()
                except:
                    pass
                self.player = None


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(True, True)
    app = FFPlayer(root)


    def on_closing():
        try:
            app.should_stop = True
            app.stop_playback()
            time.sleep(0.2)
        except:
            pass
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()