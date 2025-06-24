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

        # Audio-video sync related parameters
        self.video_fps = 25.0
        self.last_frame_time = 0.0
        self.frame_start_time = 0.0  # System time when playback started
        self.playback_start_pos = 0.0  # Video position when playback started
        self.sync_threshold = 0.040

        # New sync debugging parameters
        self.audio_time = 0.0
        self.video_time = 0.0
        self.system_time_offset = 0.0
        self.last_sync_check_time = 0.0
        self.sync_history = deque(maxlen=30)  # Store last 30 sync data points

        # Slide detection related
        self.slides_detected = []
        self.slide_buttons = []
        self.detection_in_progress = False
        self.detection_thread = None

        # New: Current focused slide information
        self.current_slide_index = -1
        self.slide_start_time = 0.0
        self.slide_end_time = 0.0
        self.is_slide_focused = False

        # Optimized frame processing
        self.frame_queue = queue.Queue(maxsize=3)
        self.canvas_width = 800
        self.canvas_height = 450

        # Enhanced thread locks
        self.player_lock = threading.RLock()
        self.seek_lock = threading.Lock()
        self.seek_in_progress = False

        # Create interface
        self.setup_ui()
        self.update_progress()
        self.start_gui_update_thread()

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

        # New: Exit slide focus mode button
        self.btn_exit_focus = tk.Button(self.control_frame, text="Show Full Progress",
                                        command=self.exit_slide_focus, state=tk.DISABLED)
        self.btn_exit_focus.pack(side=tk.LEFT, padx=5)

        # Sync status display area
        sync_frame = tk.Frame(self.control_frame)
        sync_frame.pack(side=tk.LEFT, padx=10)

        self.sync_label = tk.Label(sync_frame, text="Sync: Normal", fg="green")
        self.sync_label.pack()

        self.sync_detail_label = tk.Label(sync_frame, text="A/V: 0ms", font=("Arial", 8), fg="blue")
        self.sync_detail_label.pack()

        # Progress bar and time display
        self.progress_frame = tk.Frame(video_frame)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.time_label = tk.Label(self.progress_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT, padx=5)

        # New: Slide range label
        self.slide_range_label = tk.Label(self.progress_frame, text="", fg="blue", font=("Arial", 9))
        self.slide_range_label.pack(side=tk.LEFT, padx=5)

        self.scale = ttk.Scale(self.progress_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=600)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Bind mouse events
        self.scale.bind("<Button-1>", self.on_slider_click)
        self.scale.bind("<B1-Motion>", self.on_slider_drag)
        self.scale.bind("<ButtonRelease-1>", self.on_slider_release)

        # Slide detection status (below video controls)
        self.detection_status_frame = tk.Frame(left_main_frame)
        self.detection_status_frame.pack(fill=tk.X, pady=(10, 0))

        self.detection_status_label = tk.Label(self.detection_status_frame, text="Slide Detection Status: Not Started",
                                               fg="blue")
        self.detection_status_label.pack()

        self.detection_progress = ttk.Progressbar(self.detection_status_frame, mode='determinate')
        self.detection_progress.pack(fill=tk.X, pady=5)

        # Right side: Slide buttons area
        right_frame = tk.Frame(main_frame, width=300)  # 增加宽度以容纳时间区间
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # Slide button area title
        slides_label = tk.Label(right_frame, text="Quick Jump to Slides:", font=("Arial", 12, "bold"))
        slides_label.pack(anchor=tk.W, pady=(0, 10))

        # Create bordered slide button container with scrollbar
        slides_border_frame = tk.Frame(right_frame, relief=tk.SUNKEN, borderwidth=2)
        slides_border_frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollable button area - vertical scroll only
        self.slides_canvas = tk.Canvas(slides_border_frame, bg="white")
        self.slides_scrollbar = ttk.Scrollbar(slides_border_frame, orient="vertical",
                                              command=self.slides_canvas.yview)
        self.slides_frame = tk.Frame(self.slides_canvas)

        self.slides_frame.bind(
            "<Configure>",
            lambda e: self.slides_canvas.configure(scrollregion=self.slides_canvas.bbox("all"))
        )

        self.slides_canvas.create_window((0, 0), window=self.slides_frame, anchor="nw")
        self.slides_canvas.configure(yscrollcommand=self.slides_scrollbar.set)

        # Layout scrollbar and canvas
        self.slides_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.slides_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind mouse wheel events
        self.slides_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.slides_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.slides_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Handle mouse wheel events"""
        # Check if mouse is over slides canvas
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget == self.slides_canvas or widget in self.slides_canvas.winfo_children():
            # Windows
            if event.delta:
                self.slides_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            # Linux
            else:
                if event.num == 4:
                    self.slides_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.slides_canvas.yview_scroll(1, "units")

    def exit_slide_focus(self):
        """Exit slide focus mode and restore full progress bar"""
        self.is_slide_focused = False
        self.current_slide_index = -1

        # Restore progress bar range to full video
        self.scale.configure(from_=0, to=self.duration)

        # Update display
        self.slide_range_label.config(text="")
        self.btn_exit_focus.config(state=tk.DISABLED)

        # Reset all button colors
        for btn in self.slide_buttons:
            btn.config(bg="lightgray", relief=tk.RAISED)

    def jump_to_slide(self, target_time, slide_index):
        """Jump to specified slide and update progress bar range"""
        if not self.video_path:
            return

        # Update current slide index
        self.current_slide_index = slide_index - 1  # Convert to 0-based index
        self.is_slide_focused = True

        # Calculate slide start and end time
        self.slide_start_time = target_time

        # If not the last slide, end time is the next slide's start time
        if self.current_slide_index < len(self.slides_detected) - 1:
            self.slide_end_time = self.slides_detected[self.current_slide_index + 1]
        else:
            # Last slide, end time is video end
            self.slide_end_time = self.duration

        # Update progress bar range
        self.scale.configure(from_=self.slide_start_time, to=self.slide_end_time)
        self.scale.set(target_time)

        # Update slide range display
        start_str = self.format_time(self.slide_start_time)
        end_str = self.format_time(self.slide_end_time)
        self.slide_range_label.config(text=f"[Slide {slide_index}: {start_str}-{end_str}]")

        # Enable exit focus button
        self.btn_exit_focus.config(state=tk.NORMAL)

        # Update button colors
        for i, btn in enumerate(self.slide_buttons):
            if i == slide_index - 1:
                btn.config(bg="lightblue", relief=tk.SUNKEN)
            else:
                btn.config(bg="lightgray", relief=tk.RAISED)

        # Perform jump
        was_playing = self.playing
        seek_thread = threading.Thread(
            target=self.perform_seek_improved,
            args=(target_time, was_playing),
            daemon=True
        )
        seek_thread.start()

    def update_progress(self):
        """Update progress bar - modified to support slide focus mode"""
        try:
            if self.playing and not self.slider_updating and not self.seeking and not self.seek_in_progress:
                # Check if beyond current slide range
                if self.is_slide_focused:
                    if self.current_pos >= self.slide_end_time:
                        # Automatically jump to next slide
                        next_slide_index = self.current_slide_index + 2  # +2 because display index starts from 1
                        if next_slide_index <= len(self.slides_detected):
                            next_slide_time = self.slides_detected[self.current_slide_index + 1]
                            self.jump_to_slide(next_slide_time, next_slide_index)
                        else:
                            # Already the last slide, exit focus mode
                            self.exit_slide_focus()
                    else:
                        # Normal progress bar update
                        self.scale.set(self.current_pos)
                else:
                    # Non-focus mode, normal update
                    if self.duration and self.duration > 0:
                        self.scale.set(self.current_pos)

                self.update_time_display(self.current_pos, self.duration or 0.0)
        except Exception as e:
            if not self.should_stop:
                pass

        if not self.should_stop:
            self.root.after(200, self.update_progress)

    def on_slider_release(self, event):
        """Optimized slider jump functionality - considers slide focus mode"""
        if not self.video_path:
            self.slider_updating = False
            return

        if self.seek_in_progress:
            self.slider_updating = False
            return

        target_pos = float(self.scale.get())

        # In slide focus mode, limit jump range
        if self.is_slide_focused:
            if target_pos < self.slide_start_time:
                target_pos = self.slide_start_time
            elif target_pos > self.slide_end_time:
                target_pos = self.slide_end_time
        else:
            # Normal mode range limits
            if target_pos < 0:
                target_pos = 0
            if target_pos > self.duration:
                target_pos = self.duration

        was_playing = self.playing
        seek_thread = threading.Thread(target=self.perform_seek_improved, args=(target_pos, was_playing), daemon=True)
        seek_thread.start()

    def create_slide_buttons(self):
        """Create slide jump buttons with time interval display"""
        self.clear_slide_buttons()

        if not self.slides_detected:
            return

        # Use vertical layout - one button per row
        for i, slide_time in enumerate(self.slides_detected):
            start_time_str = self.format_time(slide_time)

            # Calculate end time for this slide
            if i < len(self.slides_detected) - 1:
                # Not the last slide, end time is next slide's start time
                end_time = self.slides_detected[i + 1]
            else:
                # Last slide, end time is video duration
                end_time = self.duration

            end_time_str = self.format_time(end_time)

            # Create button text with time interval
            button_text = f"Slide {i + 1}\n[{start_time_str} - {end_time_str}]"

            btn = tk.Button(
                self.slides_frame,
                text=button_text,
                command=lambda t=slide_time, idx=i + 1: self.jump_to_slide(t, idx),
                width=25,  # 增加宽度以容纳时间区间
                height=3,  # 增加高度以容纳两行文本
                font=("Arial", 9),
                relief=tk.RAISED,
                bd=2,
                bg="lightgray",
                anchor="w",  # Left align text
                justify=tk.LEFT  # Left justify multi-line text
            )

            # Add double-click event handler
            btn.bind("<Double-Button-1>", lambda e, idx=i + 1: self.on_slide_double_click(idx))

            btn.pack(fill=tk.X, pady=2, padx=5)
            self.slide_buttons.append(btn)

        self.slides_frame.update_idletasks()
        self.slides_canvas.update_idletasks()
        self.slides_canvas.configure(scrollregion=self.slides_canvas.bbox("all"))

        # Ensure canvas starts at top
        self.slides_canvas.yview_moveto(0)

    def on_slide_double_click(self, slide_index):
        """Handle double-click on slide button"""
        if self.is_slide_focused and self.current_slide_index == slide_index - 1:
            # If double-clicking the currently focused slide, exit focus mode
            self.exit_slide_focus()

    def print_sync_debug_info(self, pts, system_time):
        """Print detailed sync debug information"""
        if pts is None:
            return

        # Calculate various times
        current_system_time = system_time
        playback_elapsed = current_system_time - self.frame_start_time  # System time elapsed since playback started
        expected_video_time = self.playback_start_pos + playback_elapsed  # Expected video time
        actual_video_time = float(pts)  # Actual video time

        # Audio-video offset
        av_offset = actual_video_time - expected_video_time

        # Update time records
        self.audio_time = expected_video_time
        self.video_time = actual_video_time

        # Save sync history
        sync_data = {
            'timestamp': current_system_time,
            'av_offset': av_offset,
            'expected': expected_video_time,
            'actual': actual_video_time,
            'system_elapsed': playback_elapsed
        }
        self.sync_history.append(sync_data)

        # Update interface display every 500ms
        if current_system_time - self.last_sync_check_time > 0.5:
            self.last_sync_check_time = current_system_time

            # Update interface display
            sync_color = "green"
            sync_status = "Sync: Normal"

            if abs(av_offset) > 0.1:  # 100ms
                sync_color = "orange"
                sync_status = f"Sync: Offset {av_offset * 1000:.0f}ms"

            if abs(av_offset) > 0.2:  # 200ms
                sync_color = "red"
                sync_status = f"Sync: Severe Offset {av_offset * 1000:.0f}ms"

            # Update sync status display
            try:
                self.sync_label.config(text=sync_status, fg=sync_color)
                self.sync_detail_label.config(text=f"A/V: {av_offset * 1000:.0f}ms")
            except:
                pass

    def reset_sync_timing(self, new_position=None):
        """Reset sync time baseline - this is a key function"""
        current_time = time.time()
        if new_position is not None:
            self.playback_start_pos = float(new_position)
        else:
            self.playback_start_pos = self.current_pos

        self.frame_start_time = current_time
        self.last_sync_check_time = 0.0  # Force next check
        self.sync_history.clear()

    def open_video(self):
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

        # Reset slide focus state
        self.exit_slide_focus()

        # Get video information
        try:
            temp_player = MediaPlayer(self.video_path)
            self.duration = 0.0

            # Get video frame rate
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

            if self.duration <= 0:
                frame_count = 0
                while frame_count < 100:
                    frame, val = temp_player.get_frame()
                    if val == 'eof':
                        break
                    if frame is not None:
                        frame_count += 1
                        try:
                            new_metadata = temp_player.get_metadata()
                            if new_metadata and 'duration' in new_metadata and new_metadata['duration'] is not None:
                                self.duration = float(new_metadata['duration'])
                                break
                        except:
                            pass

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
        """Detect slide change points"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first")
            return

        if self.detection_in_progress:
            messagebox.showinfo("Info", "Slide detection is in progress...")
            return

        self.detection_in_progress = True
        self.btn_detect.config(state=tk.DISABLED, text="Detecting...")
        self.detection_status_label.config(text="Slide Detection Status: Analyzing...", fg="orange")

        # Execute detection in new thread
        self.detection_thread = threading.Thread(target=self.perform_slide_detection, daemon=True)
        self.detection_thread.start()

    def perform_slide_detection(self):
        """Execute main slide detection logic"""
        try:
            self.slides_detected.clear()

            cap = cv2.VideoCapture(self.video_path)

            if not cap.isOpened():
                raise Exception("Cannot open video file for analysis")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else self.duration

            skip_frames = max(1, int(fps * 0.5))
            threshold = 0.1
            min_slide_duration = 1.0

            prev_hist = None
            prev_edges = None
            slide_times = [0.0]  # Default first slide at beginning
            frame_count = 0
            processed_frames = 0

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

                if processed_frames % 10 == 0:
                    progress = processed_frames
                    self.root.after(0, lambda p=progress: self.detection_progress.config(value=p))
                    self.root.after(0, lambda t=current_time: self.detection_status_label.config(
                        text=f"Detection Progress: {current_time:.1f}s / {video_duration:.1f}s"))

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 240))

                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                edges = cv2.Canny(gray, 50, 150)
                edge_count = np.sum(edges > 0)

                if prev_hist is not None and prev_edges is not None:
                    correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    edge_change = abs(edge_count - prev_edges) / max(prev_edges, 1)

                    scene_change = False

                    if correlation < threshold:
                        scene_change = True

                    if edge_change > 0.3:
                        scene_change = True

                    chi_square = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                    if chi_square > 30000:
                        scene_change = True

                    if scene_change:
                        if len(slide_times) == 0 or (current_time - slide_times[-1]) >= min_slide_duration:
                            slide_times.append(current_time)

                prev_hist = hist.copy()
                prev_edges = edge_count

            cap.release()

            def update_slides_data():
                self.slides_detected = slide_times.copy()
                self.create_slide_buttons()
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

    def clear_slide_buttons(self):
        """Clear all slide buttons"""
        for btn in self.slide_buttons:
            btn.destroy()
        self.slide_buttons.clear()

    def reset_player(self):
        self.should_stop = True
        self.playing = False
        self.paused = False
        self.current_pos = 0.0
        self.seek_in_progress = False
        self.last_frame_time = 0.0
        self.frame_start_time = 0.0
        self.playback_start_pos = 0.0
        self.sync_history.clear()
        self.btn_play.config(text="Play")
        self.btn_detect.config(state=tk.DISABLED)

        self.clear_frame_queue()

        with self.player_lock:
            if self.player:
                try:
                    self.player.close_player()
                except:
                    pass
            self.player = None

        self.canvas.delete("all")
        self.img_on_canvas = None
        time.sleep(0.1)
        self.should_stop = False

    def clear_frame_queue(self):
        """Safely clear frame queue"""
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def toggle_play(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first")
            return

        if not self.playing:
            self.start_playback()
        else:
            self.pause_playback()

    def start_playback(self):
        try:
            with self.player_lock:
                if self.player is None:
                    # Try to create player with optimized parameters
                    try:
                        self.player = MediaPlayer(self.video_path, ff_opts={'sync': 'audio'})
                    except Exception as e:
                        try:
                            self.player = MediaPlayer(self.video_path)
                        except Exception as e2:
                            raise e2

                    if self.duration <= 0 or self.duration == 600.0:
                        metadata = self.player.get_metadata()
                        if metadata and 'duration' in metadata and metadata['duration'] is not None:
                            new_duration = float(metadata['duration'])
                            if new_duration > 0:
                                self.duration = new_duration
                                self.scale.configure(to=self.duration)

                    if self.current_pos > 0:
                        try:
                            self.player.seek(self.current_pos, relative=False)
                            time.sleep(0.1)
                        except Exception as e:
                            pass

                if hasattr(self.player, 'set_pause'):
                    self.player.set_pause(False)

            self.playing = True
            self.paused = False
            self.btn_play.config(text="Pause")

            # Reset sync time baseline
            self.reset_sync_timing(self.current_pos)

            if self.play_thread is None or not self.play_thread.is_alive():
                self.play_thread = threading.Thread(target=self.play_loop_improved, daemon=True)
                self.play_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Playback failed: {str(e)}")

    def pause_playback(self):
        """Correct pause functionality"""
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
                except Exception as e:
                    pass

    def stop_playback(self):
        self.should_stop = True
        self.playing = False
        self.paused = False

        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)

        with self.player_lock:
            if self.player:
                try:
                    self.player.close_player()
                except:
                    pass
                self.player = None

    def play_loop_improved(self):
        """Improved playback loop - fixes audio-video sync issues"""
        frame_time_target = 1.0 / max(self.video_fps, 1.0)
        sync_samples = deque(maxlen=10)
        dropped_frames = 0
        total_frames = 0

        while not self.should_stop:
            if not self.playing or self.seeking or self.seek_in_progress:
                time.sleep(0.02)
                continue

            try:
                with self.player_lock:
                    if not self.player:
                        break

                    frame, val = self.player.get_frame()

                if val == 'eof':
                    self.root.after(0, self.on_playback_finished)
                    break

                if frame is not None:
                    total_frames += 1
                    current_system_time = time.time()

                    # Get player timestamp and perform sync debugging
                    with self.player_lock:
                        if self.player and not self.seek_in_progress:
                            try:
                                pts = self.player.get_pts()
                                if pts is not None:
                                    self.current_pos = float(pts)
                                    self.print_sync_debug_info(pts, current_system_time)
                            except Exception as e:
                                pass

                    # Simplified frame queue management
                    img, t = frame

                    try:
                        # If queue is full, remove old frames
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                                dropped_frames += 1
                            except queue.Empty:
                                pass

                        self.frame_queue.put_nowait(frame)

                    except queue.Full:
                        dropped_frames += 1
                        continue

                    # Simplified timing control
                    if t > 0:
                        # For low frame rate videos, use player-provided time interval
                        sleep_time = min(t, 0.2)  # Max sleep 200ms
                        if sleep_time > 0.005:  # Min sleep 5ms
                            time.sleep(sleep_time)
                    else:
                        # When no time info, use calculated frame interval
                        time.sleep(min(frame_time_target, 0.1))
                else:
                    # No frame, wait
                    time.sleep(0.01)

            except Exception as e:
                if not self.should_stop:
                    pass
                time.sleep(0.02)

    def start_gui_update_thread(self):
        """GUI update thread - adapted for low frame rate videos"""

        def gui_update_loop():
            last_update_time = 0
            # Adjust GUI update frequency based on video frame rate, but don't exceed 30fps
            target_gui_fps = min(30, max(10, self.video_fps * 2))
            gui_frame_interval = 1.0 / target_gui_fps

            while True:
                if self.should_stop:
                    time.sleep(0.1)
                    continue

                current_time = time.time()

                # Control GUI update frequency
                if current_time - last_update_time < gui_frame_interval:
                    time.sleep(0.005)
                    continue

                try:
                    frame = self.frame_queue.get(timeout=0.2)  # Increase timeout for low frame rate
                    if frame is not None:
                        self.display_frame_safe(frame)
                        last_update_time = current_time
                except queue.Empty:
                    continue
                except Exception as e:
                    if not self.should_stop:
                        pass
                    time.sleep(0.01)

        if self.gui_thread is None or not self.gui_thread.is_alive():
            self.gui_thread = threading.Thread(target=gui_update_loop, daemon=True)
            self.gui_thread.start()

    def display_frame_safe(self, frame):
        """Safe frame display"""
        try:
            img, t = frame
            w, h = img.get_size()

            arr = np.frombuffer(img.to_bytearray()[0], dtype=np.uint8)
            frame_array = arr.reshape(h, w, 3)

            pil_image = Image.fromarray(frame_array)
            if w != self.canvas_width or h != self.canvas_height:
                pil_image = pil_image.resize(
                    (self.canvas_width, self.canvas_height),
                    Image.Resampling.LANCZOS
                )

            photo = ImageTk.PhotoImage(pil_image)
            self.root.after_idle(self.update_canvas_safe, photo)

        except Exception as e:
            if not self.should_stop:
                pass

    def update_canvas_safe(self, photo):
        """Safe canvas update"""
        try:
            if self.should_stop:
                return

            if self.img_on_canvas is None:
                self.img_on_canvas = self.canvas.create_image(
                    self.canvas_width // 2, self.canvas_height // 2, image=photo
                )
            else:
                self.canvas.itemconfig(self.img_on_canvas, image=photo)

            self.canvas.photo_ref = photo

        except Exception as e:
            if not self.should_stop:
                pass

    def update_time_display(self, current, total):
        current = current or 0.0
        total = total or 0.0
        current_str = self.format_time(current)
        total_str = self.format_time(total)
        try:
            self.time_label.config(text=f"{current_str} / {total_str}")
        except:
            pass

    def format_time(self, seconds):
        try:
            if seconds is None or seconds <= 0:
                return "00:00"
            seconds = float(seconds)
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"
        except (TypeError, ValueError):
            return "00:00"

    def on_slider_click(self, event):
        """Slider click event"""
        self.slider_updating = True
        current_value = self.scale.get()

    def on_slider_drag(self, event):
        """Slider drag event"""
        self.slider_updating = True
        current_value = self.scale.get()
        if self.duration > 0:
            time_str = self.format_time(current_value)
            if hasattr(self, '_last_drag_value') and abs(current_value - self._last_drag_value) > 0.5:
                self._last_drag_value = current_value

    def perform_seek_improved(self, target_pos, was_playing):
        """Improved seek implementation - key function to fix sync issues"""
        with self.seek_lock:
            if self.seek_in_progress:
                self.slider_updating = False
                return

            self.seek_in_progress = True

        try:
            self.current_pos = target_pos
            self.root.after(0, lambda: self.scale.set(target_pos))

            self.playing = False
            self.clear_frame_queue()

            with self.player_lock:
                # Force recreation of player to ensure seek accuracy
                if self.player:
                    try:
                        self.player.close_player()
                    except:
                        pass
                    self.player = None
                    time.sleep(0.1)  # Give player time to fully close

                try:
                    self.player = MediaPlayer(self.video_path, ff_opts={'sync': 'audio'})
                except Exception as e:
                    self.player = MediaPlayer(self.video_path)

                time.sleep(0.1)

                try:
                    self.player.seek(target_pos, relative=False)
                    self.wait_for_seek_completion_with_verification(target_pos)
                    self.root.after(0, lambda: self.update_time_display(target_pos, self.duration))
                except Exception as e:
                    raise

            time.sleep(0.1)

            if was_playing:
                # Reset sync time baseline after seek, using verified actual position
                with self.player_lock:
                    if self.player:
                        try:
                            actual_pts = self.player.get_pts()
                            if actual_pts is not None:
                                verified_pos = float(actual_pts)
                                self.reset_sync_timing(verified_pos)
                                self.current_pos = verified_pos
                            else:
                                self.reset_sync_timing(target_pos)
                        except:
                            self.reset_sync_timing(target_pos)
                    else:
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
                # Reset sync baseline even in paused state, preparing for next playback
                with self.player_lock:
                    if self.player:
                        try:
                            actual_pts = self.player.get_pts()
                            if actual_pts is not None:
                                verified_pos = float(actual_pts)
                                self.reset_sync_timing(verified_pos)
                                self.current_pos = verified_pos
                            else:
                                self.reset_sync_timing(target_pos)
                        except:
                            self.reset_sync_timing(target_pos)
                    else:
                        self.reset_sync_timing(target_pos)

                self.paused = True
                self.root.after(0, lambda: self.btn_play.config(text="Play"))
                with self.player_lock:
                    if self.player and hasattr(self.player, 'set_pause'):
                        try:
                            self.player.set_pause(True)
                        except:
                            pass

        except Exception as e:
            try:
                with self.player_lock:
                    if not self.player:
                        try:
                            self.player = MediaPlayer(self.video_path, ff_opts={'sync': 'audio'})
                        except Exception as e:
                            self.player = MediaPlayer(self.video_path)
            except Exception as recovery_error:
                pass

        finally:
            def release_slider_lock():
                self.seek_in_progress = False
                self.seeking = False
                self.slider_updating = False

            self.root.after(300, release_slider_lock)

    def wait_for_seek_completion_with_verification(self, target_pos, max_attempts=30):
        """Wait for seek completion and verify position accuracy"""
        successful_frame_count = 0
        required_success = 3  # Need 3 consecutive frames at correct position to consider seek successful

        for attempt in range(max_attempts):
            try:
                with self.player_lock:
                    if not self.player:
                        break

                    frame, val = self.player.get_frame()

                    if val == 'eof':
                        break

                    if frame is not None:
                        # Verify if current position is close to target position
                        try:
                            current_pts = self.player.get_pts()
                            if current_pts is not None:
                                pos_diff = abs(float(current_pts) - target_pos)

                                if pos_diff <= 2.0:  # Allow 2 second error
                                    successful_frame_count += 1

                                    if successful_frame_count >= required_success:
                                        try:
                                            self.frame_queue.put_nowait(frame)
                                        except queue.Full:
                                            self.clear_frame_queue()
                                            self.frame_queue.put_nowait(frame)
                                        return
                                else:
                                    successful_frame_count = 0
                            else:
                                pass
                        except Exception as e:
                            pass

                        # Save first valid frame in case verification fails
                        if attempt == 0:
                            try:
                                self.frame_queue.put_nowait(frame)
                            except queue.Full:
                                self.clear_frame_queue()
                                self.frame_queue.put_nowait(frame)

                time.sleep(0.05)

            except Exception as e:
                if attempt > 15:
                    break
                time.sleep(0.05)

    def on_playback_finished(self):
        """Playback finished"""
        self.playing = False
        self.paused = False
        self.btn_play.config(text="Play")
        self.current_pos = 0.0
        self.scale.set(0)
        self.sync_label.config(text="Sync: Normal", fg="green")
        self.sync_detail_label.config(text="A/V: 0ms")

        # Reset slide button highlighting
        for btn in self.slide_buttons:
            btn.config(bg="lightgray", relief=tk.RAISED)

        # Exit slide focus mode
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