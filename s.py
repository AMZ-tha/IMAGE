#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai Coin Scanner — Desktop Edition
Logic: cv2.HoughCircles + HSV Color Analysis (จากโค้ดต้นแบบ)
UI:    PyQt5 Cyberpunk

ติดตั้ง:
    pip install PyQt5 opencv-python numpy

รัน:
    python thai_coin_scanner.py
"""

import sys, os, csv, time, datetime
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QMessageBox, QCheckBox
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal
from PyQt5.QtGui   import QImage, QPixmap, QPalette, QColor

# ─────────────────────────────────────────────
#  STYLESHEET  (Cyberpunk Dark)
# ─────────────────────────────────────────────
STYLE = """
QMainWindow,QWidget{background:#04060D;color:#C8E0F8;font-family:'Segoe UI',sans-serif;font-size:13px;}
QLabel{color:#C8E0F8;}
QPushButton{background:transparent;border:1px solid rgba(0,220,255,0.35);border-radius:5px;
            color:#90C8E8;padding:7px 14px;letter-spacing:1px;}
QPushButton:hover{background:rgba(0,220,255,0.08);border-color:#00DCFF;color:#00DCFF;}
QPushButton#btn_cam{border-color:rgba(0,255,136,0.4);color:#60D890;}
QPushButton#btn_cam:hover{background:rgba(0,255,136,0.08);border-color:#00FF88;color:#00FF88;}
QPushButton#btn_stop{border-color:rgba(255,56,88,0.4);color:#E06080;}
QPushButton#btn_stop:hover{background:rgba(255,56,88,0.08);border-color:#FF3858;color:#FF3858;}
QPushButton#btn_save{border-color:rgba(255,208,0,0.4);color:#D4A800;}
QPushButton#btn_save:hover{background:rgba(255,208,0,0.08);border-color:#FFD000;color:#FFD000;}
QGroupBox{border:1px solid rgba(0,220,255,0.15);border-radius:6px;margin-top:10px;padding-top:6px;
          color:#00DCFF;font-size:10px;letter-spacing:2px;}
QGroupBox::title{subcontrol-origin:margin;left:10px;}
QTableWidget{background:#060A15;border:1px solid rgba(0,220,255,0.12);border-radius:4px;
             gridline-color:rgba(0,220,255,0.06);color:#C8E0F8;
             selection-background-color:rgba(0,220,255,0.12);}
QTableWidget::item{padding:4px 8px;border-bottom:1px solid rgba(0,220,255,0.05);}
QHeaderView::section{background:#0A1020;color:#00DCFF;border:none;
                     border-bottom:1px solid rgba(0,220,255,0.2);
                     padding:5px 8px;font-size:10px;letter-spacing:1px;}
QScrollBar:vertical{background:transparent;width:5px;}
QScrollBar::handle:vertical{background:rgba(0,220,255,0.2);border-radius:2px;min-height:20px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
QTextEdit{background:#020408;border:1px solid rgba(0,220,255,0.1);border-radius:4px;
          color:#3A8A5A;font-family:'Consolas',monospace;font-size:11px;}
QSpinBox,QDoubleSpinBox{background:transparent;border:1px solid rgba(0,220,255,0.2);
                        border-radius:4px;padding:3px 6px;color:#90C8E8;}
QCheckBox{color:#5070A0;spacing:6px;}
QCheckBox::indicator{width:14px;height:14px;border:1px solid rgba(0,220,255,0.3);border-radius:3px;}
QCheckBox::indicator:checked{background:rgba(0,220,255,0.3);border-color:#00DCFF;}
QStatusBar{background:#020408;color:#3A5A7A;border-top:1px solid rgba(0,220,255,0.1);font-size:11px;}
QStatusBar::item{border:none;}
"""


# ─────────────────────────────────────────────
#  CORE DETECTION  (Logic จากโค้ดต้นแบบ)
# ─────────────────────────────────────────────
def process_frame(frame, dp=1.2, minDist=50, param1=100, param2=35,
                  minRadius=20, maxRadius=100, show_debug=False,
                  brightness=0, contrast=1.0):
    """
    รับ BGR frame → คืน (output_frame, total_value, coin_list)
    ใช้ 2-Pass:
      Pass 1: เก็บข้อมูลทุกวงกลม + วิเคราะห์สี
      Pass 2: จำแนกเหรียญเงิน (5/1 บาท) โดยเทียบขนาดกันเอง
    """
    # ── Brightness / Contrast ─────────────────────────
    #   result = clip(frame * contrast + brightness, 0, 255)
    if brightness != 0 or contrast != 1.0:
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    output = frame.copy()

    # ── helpers สี ──────────────────────────────────
    def sample_zone(cx, cy, r_in, r_out):
        mask_out = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask_in  = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask_out, (cx, cy), max(1, int(r_out)), 255, -1)
        cv2.circle(mask_in,  (cx, cy), max(1, int(r_in)),  255, -1)
        mask = cv2.subtract(mask_out, mask_in)
        bgr  = cv2.mean(frame, mask=mask)[:3]
        px   = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
        hsv  = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
        return int(hsv[0]), int(hsv[1]), int(hsv[2])

    def is_golden(h, s):
        return s > 35 and (8 <= h <= 35)

    def is_silver(s):
        return s < 55

    # ── pre-processing ──────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    if show_debug:
        edges = cv2.Canny(blur, 50, 150)
        ec = np.zeros_like(frame); ec[:,:,1] = edges
        output = cv2.addWeighted(output, 0.65, ec, 0.7, 0)

    detected = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=minDist,
        param1=param1, param2=param2,
        minRadius=minRadius, maxRadius=maxRadius,
    )

    if detected is None:
        # HUD ว่าง
        cv2.rectangle(output,(0,0),(frame.shape[1],40),(2,4,10),-1)
        cv2.line(output,(0,40),(frame.shape[1],40),(0,220,255),1)
        cv2.putText(output,"TOTAL : 0.00 Baht    Coins: 0",(16,28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,220,255),2,cv2.LINE_AA)
        s_=22; H_,W_=frame.shape[:2]
        for cx_,cy_,dx,dy in [(0,0,1,1),(W_,0,-1,1),(0,H_,1,-1),(W_,H_,-1,-1)]:
            cv2.line(output,(cx_,cy_),(cx_+dx*s_,cy_),(0,220,255),2)
            cv2.line(output,(cx_,cy_),(cx_,cy_+dy*s_),(0,220,255),2)
        return output, 0.0, []

    detected = np.uint16(np.around(detected))
    max_r = max(c[2] for c in detected[0])

    # ══════════════════════════════════════════════════
    #  PASS 1 — เก็บข้อมูลทุกวงกลม
    # ══════════════════════════════════════════════════
    raw = []   # list of dict per circle
    for (x, y, r) in detected[0]:
        h_in,  s_in,  _ = sample_zone(x, y, 0,       r*0.45)
        h_full,s_full,v = sample_zone(x, y, 0,       r*0.75)
        h_out, s_out,  _ = sample_zone(x, y, r*0.60, r*0.92)

        inner_gold  = is_golden(h_in,  s_in)
        inner_silv  = is_silver(s_in)
        outer_gold  = is_golden(h_out, s_out)
        outer_silv  = is_silver(s_out)
        full_gold   = is_golden(h_full, s_full) and s_full > 30

        # จำแนก "กลุ่มสี" ก่อน — ยังไม่ตัดสิน 5/1
        if inner_gold and outer_silv and (s_in - s_out) > 20:
            group = "bimetal"       # 10 บาท
        elif full_gold:
            group = "gold"          # 2 บาท / สตางค์
        elif inner_silv and outer_silv:
            group = "silver"        # 5 บาท หรือ 1 บาท ← ยังไม่รู้
        else:
            group = "unknown"

        raw.append({
            "x":int(x),"y":int(y),"r":int(r),
            "r_ratio": r/max_r,
            "group": group,
            "s_in": s_in, "s_out": s_out,
            "h_in": h_in, "h_out": h_out,
            "h_full": h_full, "s_full": s_full, "v": v,
        })

    # ══════════════════════════════════════════════════
    #  PASS 2 — จำแนกขั้นสุดท้าย
    #
    #  เหรียญเงิน (5 บาท vs 1 บาท) — ขนาดจริง:
    #    5 บาท = 24 mm   1 บาท = 20 mm   อัตราส่วน = 1.20
    #
    #  วิธีเทียบ (เรียงความแม่นยำ):
    #  A) มีเหรียญเงิน ≥ 2 อัน → เทียบกันเอง: ใหญ่กว่า median = 5, เล็กกว่า = 1
    #  B) มีเหรียญเงิน 1 อัน + มีเหรียญอ้างอิง (2b/10b/50s) → เทียบอัตราส่วน
    #  C) มีเหรียญเงินอัน เดียว ไม่มีอ้างอิง → ใช้ V (ความสว่าง) ช่วย:
    #     5 บาท มีผิวหยาบ/เทาเข้มกว่า (V ต่ำกว่า), 1 บาท ขาวกว่า (V สูงกว่า)
    # ══════════════════════════════════════════════════

    # แยก silver group
    silver_items = [d for d in raw if d["group"] == "silver"]
    silver_rs    = sorted([d["r"] for d in silver_items])

    # หา median radius ของ silver
    if len(silver_rs) >= 2:
        silver_median = silver_rs[len(silver_rs)//2]
    else:
        silver_median = None

    # หา reference radius จากเหรียญอ้างอิงที่รู้ขนาดจริง (mm)
    # 10b=26, 2b=22, 50s=18  → ใช้ค่าเฉลี่ยถ่วงน้ำหนัก
    REF_MM = {"bimetal": 26.0, "gold": 22.0}   # ใช้ 2b เป็น proxy ของ gold
    ref_px_per_mm = None
    ref_items = [d for d in raw if d["group"] in REF_MM]
    if ref_items:
        # คำนวณ px/mm จากเหรียญอ้างอิง
        ppms = [d["r"] / (REF_MM[d["group"]] / 2) for d in ref_items]
        ref_px_per_mm = sum(ppms) / len(ppms)

    coin_list   = []
    total_value = 0.0

    for d in raw:
        r         = d["r"]
        r_ratio   = d["r_ratio"]
        group     = d["group"]
        coin_value = 0
        coin_type  = "unknown"

        if group == "bimetal":
            # ── 10 บาท ──
            coin_value = 10
            coin_type  = "10b"

        elif group == "gold":
            # ── 2 บาท / 50 สต / 25 สต ──
            # ขนาดจริง: 2b=22mm, 50s=18mm, 25s=16mm
            if ref_px_per_mm:
                mm = r / ref_px_per_mm * 2   # diameter
                if mm >= 20:
                    coin_value = 2;    coin_type = "2b"
                elif mm >= 16:
                    coin_value = 0.50; coin_type = "50s"
                else:
                    coin_value = 0.25; coin_type = "25s"
            else:
                # fallback ใช้ r_ratio
                if r_ratio > 0.75:
                    coin_value = 2;    coin_type = "2b"
                elif r_ratio > 0.58:
                    coin_value = 0.50; coin_type = "50s"
                else:
                    coin_value = 0.25; coin_type = "25s"

        elif group == "silver":
            # ── 5 บาท หรือ 1 บาท ──
            # ขนาดจริง: 5b=24mm, 1b=20mm → อัตราส่วน 1.20

            if silver_median is not None:
                # วิธี A: เทียบ median ของ silver เอง
                # ถ้าใหญ่กว่าหรือเท่า median → 5 บาท, เล็กกว่า → 1 บาท
                coin_value = 5 if r >= silver_median else 1
                coin_type  = "5b" if r >= silver_median else "1b"

            elif ref_px_per_mm:
                # วิธี B: เทียบกับเหรียญอ้างอิงรู้ขนาดจริง
                # 5b=24mm → r_px = ref_px_per_mm * 12
                # 1b=20mm → r_px = ref_px_per_mm * 10
                # threshold = midpoint = ref_px_per_mm * 11
                threshold = ref_px_per_mm * 11.0
                coin_value = 5 if r >= threshold else 1
                coin_type  = "5b" if r >= threshold else "1b"

            else:
                # วิธี C: อัน เดียว ไม่มีอ้างอิง
                # ใช้ค่า V (brightness): 1 บาท ขาวกว่า (V สูง), 5 บาท เทากว่า (V ต่ำ)
                # threshold ประมาณ V=170 (ปรับได้ตามแสง)
                v_val = d["v"]
                coin_value = 1 if v_val > 170 else 5
                coin_type  = "1b(V)" if v_val > 170 else "5b(V)"

        else:
            # unknown fallback
            if r_ratio > 0.88:
                coin_value = 5;  coin_type = "5b?"
            elif r_ratio > 0.72:
                coin_value = 2;  coin_type = "2b?"
            else:
                coin_value = 1;  coin_type = "1b?"

        total_value += coin_value

        # ════════════════════════════════════════════════
        #  วาด Overlay
        # ════════════════════════════════════════════════
        COLOR_MAP = {
            "10b":   (0,  210, 255),   # cyan-gold
            "2b":    (0,  180, 255),   # amber
            "50s":   (20, 150, 255),   # orange
            "25s":   (30, 120, 210),   # orange dim
            "5b":    (210,240, 255),   # silver bright
            "1b":    (150,190, 230),   # silver dim
            "5b(V)": (200,230, 240),
            "1b(V)": (140,180, 220),
            "5b?":   (180,200, 200),
            "2b?":   (0,  160, 220),
            "1b?":   (120,160, 200),
            "unknown":(80, 80, 180),
        }
        x_, y_ = d["x"], d["y"]
        ring_color = COLOR_MAP.get(coin_type, (120,120,200))

        glow = output.copy()
        cv2.circle(glow, (x_,y_), r+7, ring_color, -1)
        output = cv2.addWeighted(output, 1.0, glow, 0.07, 0)

        cv2.circle(output, (x_,y_), r, ring_color, 2)
        cv2.circle(output, (x_,y_), max(4,int(r*0.45)), (255,255,255), 1)
        cv2.circle(output, (x_,y_), max(4,int(r*0.90)), (160,160,160), 1)
        cv2.circle(output, (x_,y_), 3, ring_color, -1)

        for deg in (0,90,180,270):
            a = np.radians(deg)
            cv2.line(output,
                     (int(x_+r*np.cos(a)),   int(y_+r*np.sin(a))),
                     (int(x_+(r+9)*np.cos(a)),int(y_+(r+9)*np.sin(a))),
                     ring_color, 1)

        label = f"{coin_value:.2f} THB"
        (tw,th_),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.46,1)
        lx = x_-tw//2; ly = y_-r-9
        cv2.rectangle(output,(lx-5,ly-th_-4),(lx+tw+5,ly+4),(3,5,14),-1)
        cv2.rectangle(output,(lx-5,ly-th_-4),(lx+tw+5,ly+4),ring_color,1)
        cv2.putText(output,label,(lx,ly),
                    cv2.FONT_HERSHEY_SIMPLEX,0.46,ring_color,1,cv2.LINE_AA)

        # debug: type + r(px) + V
        dbg = f"{coin_type} r={r}px V={d['v']}"
        cv2.putText(output,dbg,(x_-36,y_+r+14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.28,(80,120,160),1)

        coin_list.append({
            "x":x_,"y":y_,"r":r,
            "value": coin_value,
            "type":  coin_type,
            "h": d["h_full"], "s": d["s_full"], "v": d["v"],
            "s_in": d["s_in"], "s_out": d["s_out"],
            "r_ratio": round(float(r_ratio),3),
        })

    # HUD bar
    bar = f"TOTAL : {total_value:.2f} Baht    Coins: {len(coin_list)}"
    cv2.rectangle(output,(0,0),(frame.shape[1],40),(2,4,10),-1)
    cv2.line(output,(0,40),(frame.shape[1],40),(0,220,255),1)
    cv2.putText(output,bar,(16,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,220,255),2,cv2.LINE_AA)

    s_=22; H_,W_=frame.shape[:2]
    for cx_,cy_,dx,dy in [(0,0,1,1),(W_,0,-1,1),(0,H_,1,-1),(W_,H_,-1,-1)]:
        cv2.line(output,(cx_,cy_),(cx_+dx*s_,cy_),(0,220,255),2)
        cv2.line(output,(cx_,cy_),(cx_,cy_+dy*s_),(0,220,255),2)

    return output, total_value, coin_list


# ─────────────────────────────────────────────
#  CAMERA THREAD
# ─────────────────────────────────────────────
class CameraThread(QThread):
    frame_signal = pyqtSignal(object, float, list)
    error_signal = pyqtSignal(str)

    def __init__(self, cam_index=0, params=None):
        super().__init__()
        self.cam_index = cam_index
        self.params    = params or {}   # main thread update ได้ตลอด
        self.running   = False

    def update_params(self, new_params):
        """เรียกจาก main thread ได้เลย — มีผลทันทีใน frame ถัดไป"""
        self.params = new_params

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.error_signal.emit(f"ไม่สามารถเปิดกล้อง index {self.cam_index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            # อ่าน params ล่าสุดทุก frame → slider ขยับมีผลทันที
            out, total, coins = process_frame(frame, **self.params)
            self.frame_signal.emit(out, total, coins)
            time.sleep(0.03)
        cap.release()

    def stop(self):
        self.running = False
        self.wait(2000)


# ─────────────────────────────────────────────
#  VIDEO LABEL
# ─────────────────────────────────────────────
class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 340)
        self.setStyleSheet("background:#000;border:1px solid rgba(0,220,255,0.2);border-radius:6px;")
        self._fps = 0.0; self._fc = 0; self._lt = time.time()

    def show_frame(self, bgr):
        self._fc += 1
        now = time.time()
        if now - self._lt >= 1.0:
            self._fps = self._fc / (now - self._lt)
            self._fc = 0; self._lt = now
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi   = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qi)
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thai Coin Scanner  ·  HoughCircles + HSV")
        self.resize(1120, 720)
        self.setMinimumSize(860, 560)
        self.cam_thread    = None
        self.current_coins = []
        self.current_total = 0.0
        self.scan_history  = []
        self._last_bgr     = None
        self._build_ui()
        self.setStyleSheet(STYLE)
        self.statusBar().showMessage("🔬  Ready  |  cv2.HoughCircles + HSV Color Analysis  |  No AI")

    def _lbl(self, t):
        l = QLabel(t); l.setStyleSheet("font-size:11px;color:#5070A0;"); return l

    def _big(self, t):
        l = QLabel(t)
        l.setStyleSheet("font-size:18px;font-weight:700;color:#00DCFF;")
        l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return l

    def _make_slider(self, mn, mx, val):
        from PyQt5.QtWidgets import QSlider
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(mn); sl.setMaximum(mx); sl.setValue(val)
        sl.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:rgba(0,220,255,0.15);border-radius:2px;}"
            "QSlider::handle:horizontal{width:14px;height:14px;margin:-5px 0;"
            "background:#00DCFF;border-radius:7px;}"
            "QSlider::sub-page:horizontal{background:rgba(0,220,255,0.45);border-radius:2px;}"
        )
        return sl

    def _val_lbl(self, txt):
        l = QLabel(txt)
        l.setStyleSheet("font-size:12px;font-weight:700;color:#00DCFF;"
                        "min-width:38px;font-family:Consolas,monospace;")
        l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return l

    def _get_params(self):
        return dict(
            dp         = self.sl_dp.value()   / 10.0,
            minDist    = self.sl_dist.value(),
            param1     = self.sl_p1.value(),
            param2     = self.sl_p2.value(),
            minRadius  = self.sl_minr.value(),
            maxRadius  = self.sl_maxr.value(),
            show_debug = self.chk_dbg.isChecked(),
            brightness = self.sl_bright.value(),          # -100 .. +100
            contrast   = self.sl_contrast.value() / 10.0, # 0.1 .. 3.0 (×0.1)
        )

    def _on_slider(self):
        """เรียกทุกครั้งที่ slider/checkbox เปลี่ยน — update label + push params real-time"""
        self.vl_dp.setText(     f"{self.sl_dp.value()/10:.1f}")
        self.vl_dist.setText(   str(self.sl_dist.value()))
        self.vl_p1.setText(     str(self.sl_p1.value()))
        self.vl_p2.setText(     str(self.sl_p2.value()))
        self.vl_minr.setText(   str(self.sl_minr.value()))
        self.vl_maxr.setText(   str(self.sl_maxr.value()))
        bv = self.sl_bright.value()
        self.vl_bright.setText( ("+"+str(bv) if bv>0 else str(bv)))
        self.vl_contrast.setText(f"{self.sl_contrast.value()/10:.1f}x")
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.update_params(self._get_params())

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        lay  = QHBoxLayout(root); lay.setContentsMargins(8,8,8,8); lay.setSpacing(8)

        # ── LEFT ──────────────────────────────── #
        left = QWidget(); ll = QVBoxLayout(left)
        ll.setContentsMargins(0,0,0,0); ll.setSpacing(6)
        lay.addWidget(left, stretch=3)

        t1 = QLabel("⬡  THAI COIN SCANNER")
        t1.setStyleSheet("font-size:17px;font-weight:700;color:#FFD000;letter-spacing:3px;")
        t2 = QLabel("cv2.HoughCircles  ·  HSV Color Analysis  ·  Real-time Sliders")
        t2.setStyleSheet("font-size:9px;color:#3A6A8A;letter-spacing:2px;")
        ll.addWidget(t1); ll.addWidget(t2)

        self.vid = VideoLabel()
        ll.addWidget(self.vid, stretch=5)

        # ── SLIDERS ───────────────────────────── #
        pg  = QGroupBox("🎛  HOUGHCIRCLES PARAMS  —  Real-time")
        pgL = QGridLayout(pg); pgL.setSpacing(5)
        pgL.setColumnMinimumWidth(0, 110)
        pgL.setColumnStretch(1, 1)
        pgL.setColumnMinimumWidth(2, 42)

        def add_row(row, name, slider, val_lbl):
            pgL.addWidget(self._lbl(name), row, 0)
            pgL.addWidget(slider,          row, 1)
            pgL.addWidget(val_lbl,         row, 2)

        self.sl_dp   = self._make_slider(5,  30,  12);  self.vl_dp   = self._val_lbl("1.2")
        self.sl_dist = self._make_slider(10, 200, 50);  self.vl_dist = self._val_lbl("50")
        self.sl_p1   = self._make_slider(20, 300, 100); self.vl_p1   = self._val_lbl("100")
        self.sl_p2   = self._make_slider(5,  120, 35);  self.vl_p2   = self._val_lbl("35")
        self.sl_minr = self._make_slider(5,  150, 20);  self.vl_minr = self._val_lbl("20")
        self.sl_maxr = self._make_slider(30, 400, 120); self.vl_maxr = self._val_lbl("120")

        add_row(0, "dp (×0.1)",            self.sl_dp,   self.vl_dp)
        add_row(1, "minDist",              self.sl_dist, self.vl_dist)
        add_row(2, "param1 (Canny hi)",    self.sl_p1,   self.vl_p1)
        add_row(3, "param2 (↓=จับมากขึ้น)", self.sl_p2,  self.vl_p2)
        add_row(4, "minRadius",            self.sl_minr, self.vl_minr)
        add_row(5, "maxRadius",            self.sl_maxr, self.vl_maxr)

        for sl in (self.sl_dp, self.sl_dist, self.sl_p1,
                   self.sl_p2, self.sl_minr, self.sl_maxr):
            sl.valueChanged.connect(self._on_slider)

        self.chk_dbg = QCheckBox("Show Canny Edge Debug Overlay")
        self.chk_dbg.stateChanged.connect(self._on_slider)
        pgL.addWidget(self.chk_dbg, 6, 0, 1, 3)
        ll.addWidget(pg)

        # ── IMAGE ADJUST ──────────────────────────── #
        ag  = QGroupBox("🌟  IMAGE ADJUST  —  Real-time")
        agL = QGridLayout(ag); agL.setSpacing(5)
        agL.setColumnMinimumWidth(0, 110)
        agL.setColumnStretch(1, 1)
        agL.setColumnMinimumWidth(2, 48)

        # Brightness  -100 .. +100  default 0
        self.sl_bright   = self._make_slider(-100, 100, 0)
        self.vl_bright   = self._val_lbl("0")
        self.sl_bright.valueChanged.connect(self._on_slider)
        agL.addWidget(self._lbl("☀ Brightness"), 0, 0)
        agL.addWidget(self.sl_bright,             0, 1)
        agL.addWidget(self.vl_bright,             0, 2)

        # Contrast  1..30 (÷10 = 0.1x..3.0x)  default 10 = 1.0x
        self.sl_contrast = self._make_slider(1, 30, 10)
        self.vl_contrast = self._val_lbl("1.0x")
        self.sl_contrast.valueChanged.connect(self._on_slider)
        agL.addWidget(self._lbl("◑ Contrast"),   1, 0)
        agL.addWidget(self.sl_contrast,           1, 1)
        agL.addWidget(self.vl_contrast,           1, 2)

        # Reset button
        from PyQt5.QtWidgets import QPushButton as _PB
        btn_reset = _PB("↺  Reset Image")
        btn_reset.setStyleSheet("padding:4px 10px;font-size:11px;")
        btn_reset.clicked.connect(self._reset_image_adj)
        agL.addWidget(btn_reset, 2, 0, 1, 3)
        ll.addWidget(ag)

        # Log
        lg  = QGroupBox("EVENT LOG"); lgL = QVBoxLayout(lg)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(85)
        lgL.addWidget(self.log); ll.addWidget(lg)

        # ── RIGHT ─────────────────────────────── #
        right = QWidget(); right.setFixedWidth(268)
        rl    = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
        lay.addWidget(right)

        # Controls
        cg  = QGroupBox("CONTROLS"); cgL = QVBoxLayout(cg); cgL.setSpacing(5)
        r1  = QHBoxLayout()
        self.btn_cam  = QPushButton("📷  เปิดกล้อง"); self.btn_cam.setObjectName("btn_cam")
        self.btn_cam.clicked.connect(self.start_cam)
        self.btn_stop = QPushButton("⬛  หยุด");      self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self.stop_cam); self.btn_stop.setEnabled(False)
        r1.addWidget(self.btn_cam); r1.addWidget(self.btn_stop); cgL.addLayout(r1)

        r2 = QHBoxLayout()
        b_img  = QPushButton("📁  อัพโหลดรูป"); b_img.clicked.connect(self.load_image)
        b_snap = QPushButton("📸  Snapshot");   b_snap.clicked.connect(self.snapshot)
        r2.addWidget(b_img); r2.addWidget(b_snap); cgL.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(self._lbl("Cam index:"))
        self.sp_cam = QSpinBox(); self.sp_cam.setRange(0,10); self.sp_cam.setValue(0)
        r3.addWidget(self.sp_cam); cgL.addLayout(r3)
        rl.addWidget(cg)

        # Stats
        sg  = QGroupBox("LIVE STATS"); sgL = QGridLayout(sg); sgL.setSpacing(4)
        self.lbl_n     = self._big("0")
        self.lbl_fps   = self._big("--")
        self.lbl_total = self._big("฿0.00")
        self.lbl_total.setStyleSheet("font-size:22px;font-weight:700;color:#FFD000;")
        sgL.addWidget(self._lbl("เหรียญ"), 0, 0); sgL.addWidget(self.lbl_n,     0, 1)
        sgL.addWidget(self._lbl("FPS"),    0, 2); sgL.addWidget(self.lbl_fps,   0, 3)
        sgL.addWidget(self._lbl("มูลค่า"), 1, 0, 1, 2)
        sgL.addWidget(self.lbl_total,              1, 2, 1, 2)
        rl.addWidget(sg)

        # Coins table
        ctg  = QGroupBox("DETECTED COINS"); ctgL = QVBoxLayout(ctg)
        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["มูลค่า", "type", "S_in", "S_out", "r_ratio"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setShowGrid(False); self.tbl.setMaximumHeight(200)
        ctgL.addWidget(self.tbl); rl.addWidget(ctg)

        # History
        hg  = QGroupBox("SCAN HISTORY"); hgL = QVBoxLayout(hg)
        self.htbl = QTableWidget(0, 3)
        self.htbl.setHorizontalHeaderLabels(["เวลา", "รวม", "จำนวน"])
        self.htbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.htbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.htbl.verticalHeader().setVisible(False); self.htbl.setMaximumHeight(110)
        hgL.addWidget(self.htbl); rl.addWidget(hg)

        self.btn_save = QPushButton("💾  บันทึก CSV"); self.btn_save.setObjectName("btn_save")
        self.btn_save.clicked.connect(self.save_csv)
        rl.addWidget(self.btn_save); rl.addStretch()

    # ── Camera ── #
    def start_cam(self):
        if self.cam_thread and self.cam_thread.isRunning(): return
        self.cam_thread = CameraThread(self.sp_cam.value(), self._get_params())
        self.cam_thread.frame_signal.connect(self._on_frame)
        self.cam_thread.error_signal.connect(lambda m: QMessageBox.warning(self,"Error",m))
        self.cam_thread.start()
        self.btn_cam.setEnabled(False); self.btn_stop.setEnabled(True)
        self._log("🟢  กล้องเริ่มทำงาน")
        self.statusBar().showMessage("📷  Camera active  |  Real-time scanning...")

    def stop_cam(self):
        if self.cam_thread: self.cam_thread.stop(); self.cam_thread = None
        self.btn_cam.setEnabled(True); self.btn_stop.setEnabled(False)
        self._log("🔴  หยุดกล้อง")
        self.statusBar().showMessage("⬛  Camera stopped")

    def _on_frame(self, bgr, total, coins):
        self._last_bgr = bgr; self.current_coins = coins; self.current_total = total
        self.vid.show_frame(bgr)
        self._update_ui(total, coins)

    # ── Image ── #
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "เลือกรูปภาพ", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path: return
        self.stop_cam()
        bgr = cv2.imread(path)
        if bgr is None: QMessageBox.warning(self,"Error","อ่านไฟล์ไม่ได้"); return
        out, total, coins = process_frame(bgr, **self._get_params())
        self._last_bgr = out; self.vid.show_frame(out)
        self._update_ui(total, coins); self._add_history(total, coins)
        self._log(f"📁  {os.path.basename(path)}  →  {len(coins)} เหรียญ  ฿{total:.2f}")
        self.statusBar().showMessage(f"📁  {os.path.basename(path)}  |  {len(coins)} coins  |  ฿{total:.2f}")

    # ── Snapshot ── #
    def snapshot(self):
        if self._last_bgr is None: QMessageBox.information(self,"Info","ยังไม่มีภาพ"); return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path,_ = QFileDialog.getSaveFileName(self,"บันทึก Snapshot",f"snap_{ts}.jpg","JPEG (*.jpg)")
        if path:
            cv2.imwrite(path, self._last_bgr)
            self._add_history(self.current_total, self.current_coins)
            self._log(f"📸  Snapshot → {os.path.basename(path)}")

    # ── CSV ── #
    def save_csv(self):
        if not self.scan_history: QMessageBox.information(self,"Info","ยังไม่มีประวัติ"); return
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path,_ = QFileDialog.getSaveFileName(self,"บันทึก CSV",f"coins_{ts}.csv","CSV (*.csv)")
        if not path: return
        try:
            with open(path,"w",newline="",encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["Timestamp","Total(THB)","CoinCount","10b","5b","2b","1b","0.5b","0.25b"])
                for rec in self.scan_history:
                    cnt = {"10":0,"5":0,"2":0,"1":0,"0.5":0,"0.25":0}
                    for c in rec["coins"]:
                        k = str(c["value"]); cnt[k] = cnt.get(k,0)+1
                    w.writerow([rec["ts"],f"{rec['total']:.2f}",len(rec["coins"]),
                                 cnt["10"],cnt["5"],cnt["2"],cnt["1"],cnt["0.5"],cnt["0.25"]])
            self._log(f"💾  CSV → {os.path.basename(path)}")
            QMessageBox.information(self,"สำเร็จ",f"บันทึกแล้ว:\n{path}")
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

    # ── helpers ── #
    def _update_ui(self, total, coins):
        self.lbl_n.setText(str(len(coins)))
        self.lbl_fps.setText(f"{self.vid._fps:.1f}")
        self.lbl_total.setText(f"฿{total:.2f}")
        self.tbl.setRowCount(len(coins))
        for i, c in enumerate(coins):
            gold = c["value"] in (10, 0.50, 0.25)
            col  = QColor("#FFD000") if gold else QColor("#00DCFF")
            vals = [
                f"฿{c['value']:.2f}",
                c.get("type", "?"),
                str(c.get("s_in",  "-")),
                str(c.get("s_out", "-")),
                str(c.get("r_ratio", "-")),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v); item.setTextAlignment(Qt.AlignCenter)
                if j == 0: item.setForeground(col)
                self.tbl.setItem(i, j, item)

    def _add_history(self, total, coins):
        if not coins: return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.scan_history.append({"ts":ts,"total":total,"coins":coins})
        row = self.htbl.rowCount(); self.htbl.insertRow(row)
        self.htbl.setItem(row,0,QTableWidgetItem(ts))
        i1 = QTableWidgetItem(f"฿{total:.2f}"); i1.setForeground(QColor("#FFD000"))
        self.htbl.setItem(row,1,i1)
        self.htbl.setItem(row,2,QTableWidgetItem(str(len(coins))))
        self.htbl.scrollToBottom()

    def _reset_image_adj(self):
        """Reset brightness/contrast กลับค่าเริ่มต้น"""
        self.sl_bright.setValue(0)
        self.sl_contrast.setValue(10)
        self._on_slider()
        self._log("↺  Reset Brightness=0  Contrast=1.0x")

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f'<span style="color:#2A6A4A">[{ts}]</span>  {msg}')

    def closeEvent(self, e):
        self.stop_cam(); e.accept()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,     QColor("#04060D"))
    pal.setColor(QPalette.WindowText, QColor("#C8E0F8"))
    pal.setColor(QPalette.Base,       QColor("#060A15"))
    pal.setColor(QPalette.Text,       QColor("#C8E0F8"))
    pal.setColor(QPalette.Button,     QColor("#080C18"))
    pal.setColor(QPalette.ButtonText, QColor("#90C8E8"))
    pal.setColor(QPalette.Highlight,  QColor(0, 220, 255, 80))
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())