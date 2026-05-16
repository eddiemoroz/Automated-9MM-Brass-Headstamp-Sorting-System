"""
Ammunition Headstamp Sorter — Raspberry Pi 5
480x320 landscape display
Merges raspberry_pi_software.py inference pipeline with a clean UI.
"""

import os
import json
import queue
import threading
import time
import subprocess
import numpy as np
import cv2
import serial

import pygame
import pygame.freetype

# ── Config / paths ─────────────────────────────────────────────────────────────
MODEL_PATH = "/home/morozed/Desktop/headstampSorting/model_float32.tflite"
CLASS_PATH = "/home/morozed/Desktop/headstampSorting/headstamp_efficientnet_with_unknown_classes.json"
INPUT_DIR  = "/home/morozed/Desktop/headstampSorting/capturedImages"
CAPTURE_PATH      = os.path.join(INPUT_DIR, "captured.jpg")
UART_PORT         = "/dev/ttyAMA0"
BAUD_RATE         = 115200

DISPLAY_W         = 480
DISPLAY_H         = 320

# Accuracy slider: user-facing 70-90 maps to internal 0.90-0.95
DISPLAY_MIN, DISPLAY_MAX, DISPLAY_STEP = 70, 90, 5
INTERNAL_MIN, INTERNAL_MAX             = 0.90, 0.95

# Image processing constants (from raspberry_pi_software.py)
CROP_W, CROP_H    = 750, 750
CENTER_X          = 1770
CENTER_Y          = 1080
PRIMER_RADIUS     = 150
OUTER_MASK_RADIUS = 310
IMG_SIZE          = (320, 320)
CONTRAST          = 1.0

def display_to_internal(d: int) -> float:
    t = (d - DISPLAY_MIN) / (DISPLAY_MAX - DISPLAY_MIN)
    return INTERNAL_MIN + t * (INTERNAL_MAX - INTERNAL_MIN)

# ── Palette — gunmetal gray theme ──────────────────────────────────────────────
BG            = ( 69,  77,  84)   # gunmetal gray
PANEL_DARK    = ( 50,  56,  62)   # header/footer panels
BIN_NORMAL    = ( 88,  98, 107)   # bin boxes — contrasts with BG
BIN_UNKNOWN   = (155,  45,  45)   # red for unknown bin 6
BIN_ACTIVE    = ( 55, 118, 168)   # blue highlight for active bin
BIN_EDGE      = ( 38,  43,  49)
TEXT_WHITE    = (245, 245, 245)
TEXT_DIM      = (175, 182, 190)
ACCENT_LABEL  = (195, 208, 220)

BTN_START     = ( 72, 160,  90)   # soft green
BTN_START_HI  = ( 90, 185, 108)
BTN_STOP      = (188,  60,  60)   # pleasant red
BTN_STOP_HI   = (215,  82,  82)
BTN_DISABLED  = ( 75,  83,  90)
BTN_TEXT      = (245, 245, 245)

ESTOP_BG      = (175,  28,  28)
ESTOP_TEXT    = (255, 240, 240)
ESTOP_SUB     = (255, 195, 195)

# ── Shared state ───────────────────────────────────────────────────────────────
class SorterState:
    def __init__(self):
        self.lock          = threading.Lock()
        self.bins          = {i: {"name": None, "count": 0} for i in range(1, 7)}
        self.total         = 0
        self.last_label    = "—"
        self.last_conf     = 0.0
        self.last_bin      = 0
        self.avg_ms        = 0.0
        self.display_acc   = 80       # user-facing %
        self.system_state  = "IDLE"   # IDLE | RUNNING | STOPPED | ESTOP
        self.estop_active  = False
        self.paused_for_start = False # True after STOPESTOP — wait for user

STATE = SorterState()

_ser        = None
_ser_lock   = threading.Lock()

# ── Single queue that uart_listener puts ALL inbound messages onto ─────────────
# inference_worker reads from this queue instead of calling _ser.readline() itself.
_uart_rx_queue = queue.Queue()

def uart_send(msg: str):
    with _ser_lock:
        if _ser:
            try:
                _ser.write((msg + "\n").encode())
            except Exception:
                pass

# ── Image processing pipeline (mirrors raspberry_pi_software.py) ───────────────
def capture_image(save_path: str):
    subprocess.run(["rpicam-still", "-t 100", "-n", "-o", save_path], check=False)

def crop_center(img):
    x1 = int(CENTER_X - CROP_W / 2)
    y1 = int(CENTER_Y - CROP_H / 2)
    return img[max(0, y1):y1 + CROP_H, max(0, x1):x1 + CROP_W]

def mask_regions(img):
    h, w = img.shape
    cx, cy = w // 2, h // 2
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), OUTER_MASK_RADIUS, 255, -1)
    cv2.circle(mask, (cx, cy), PRIMER_RADIUS,      0,   -1)
    return cv2.bitwise_and(img, mask)

def apply_clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)

def morphological_gradient(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def process_image(path: str):
    """
    Full preprocessing pipeline that saves intermediate steps
    for debugging to the 'capturedImages' folder.
    """
    image = cv2.imread(path)
    if image is None:
        return None

    # Step 1: Crop
    cropped = crop_center(image)
    cv2.imwrite(os.path.join(INPUT_DIR, "01_cropped.jpg"), cropped)

    # Step 2: Grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(INPUT_DIR, "02_gray.jpg"), gray)

    # Step 3: Masking (Removing background and primer hole)
    masked = mask_regions(gray)
    cv2.imwrite(os.path.join(INPUT_DIR, "03_masked.jpg"), masked)

    # Step 4: CLAHE (Contrast Enhancement)
    enhanced = apply_clahe(masked)
    cv2.imwrite(os.path.join(INPUT_DIR, "04_enhanced.jpg"), enhanced)

    # Step 5: Edge Detection (Morphological Gradient)
    edges = morphological_gradient(enhanced)
    edges = cv2.convertScaleAbs(edges, alpha=CONTRAST, beta=0)
    cv2.imwrite(os.path.join(INPUT_DIR, "05_edges.jpg"), edges)

    # Step 6: Blending (Overlaying edges back onto the image)
    blended = cv2.addWeighted(enhanced, 0.6, edges, 0.8, 0)
    cv2.imwrite(os.path.join(INPUT_DIR, "06_blended.jpg"), blended)

    # Step 7: Final Resizing (What actually goes into the AI)
    final = cv2.resize(blended, IMG_SIZE)
    final_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join(INPUT_DIR, "07_final_input.jpg"), final_rgb)

    return final_rgb

# ── UART listener — owns ALL serial reads; routes messages to the right place ──
#
#   ESTOP/STOPESTOP  → update STATE directly (time-critical, must never be delayed)
#   everything else  → push onto _uart_rx_queue for inference_worker to consume
#
def uart_listener():
    while True:
        time.sleep(0.01)
        if _ser is None:
            continue
        try:
            if _ser.in_waiting > 0:
                raw = _ser.readline()
                msg = raw.decode("utf-8", errors="replace").strip()
                msg_up = msg.upper()

                if msg_up == "STARTESTOP":
                    with STATE.lock:
                        STATE.estop_active     = True
                        STATE.system_state     = "ESTOP"
                        STATE.paused_for_start = False
                elif msg_up == "STOPESTOP":
                    with STATE.lock:
                        STATE.estop_active     = False
                        STATE.paused_for_start = True
                        STATE.system_state     = "STOPPED"
                else:
                    # All other messages (e.g. "ready", "image", "setup complete")
                    # are forwarded to the inference worker via the queue.
                    _uart_rx_queue.put(msg_up)
        except Exception:
            pass

# ── Inference worker ───────────────────────────────────────────────────────────
def inference_worker():
    global _ser

    try:
        _ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
    except Exception:
        _ser = None

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)

    class_names = []
    if os.path.exists(CLASS_PATH):
        with open(CLASS_PATH, "r") as f:
            cn = json.load(f)
        class_names = [cn[str(i)] for i in range(len(cn))] if isinstance(cn, dict) else cn

    interpreter = inp_details = out_details = None
    try:
        from ai_edge_litert.interpreter import Interpreter
        if os.path.exists(MODEL_PATH):
            interpreter = Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            inp_details = interpreter.get_input_details()
            out_details = interpreter.get_output_details()
    except Exception:
        interpreter = None

    def run_inference(img_array):
        if interpreter is None:
            time.sleep(0.1)
            lbl = class_names[np.random.randint(0, len(class_names))] if class_names else "MOCK"
            return lbl, float(np.random.uniform(0.70, 0.99))
        arr = np.expand_dims(np.array(img_array, dtype=np.float32), axis=0)
        interpreter.set_tensor(inp_details[0]["index"], arr)
        interpreter.invoke()
        preds = interpreter.get_tensor(out_details[0]["index"])[0]
        idx   = int(np.argmax(preds))
        label = class_names[idx] if idx < len(class_names) else "UNKNOWN"
        return label, float(preds[idx])

    def assign_bin(label: str, conf: float, threshold: float) -> int:
        with STATE.lock:
            bins_snap = {k: v["name"] for k, v in STATE.bins.items()}
        if label.lower() == "unknown" or conf < threshold:
            return 6
        for b, name in bins_snap.items():
            if name == label:
                return b
        for b in range(1, 6):
            if bins_snap[b] is None:
                return b
        return 6

    def wait_for_queue_msg(keyword: str, timeout: float = 10.0) -> bool:
        """
        Block until a message containing `keyword` arrives on _uart_rx_queue,
        or until the system is no longer RUNNING, or until timeout.
        Returns True if the keyword was found, False otherwise.
        Discards any messages that don't match (they are not re-queued).
        """
        deadline = time.monotonic() + timeout
        while True:
            with STATE.lock:
                if STATE.system_state != "RUNNING":
                    return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                msg = _uart_rx_queue.get(timeout=min(remaining, 0.1))
                if keyword in msg:
                    return True
                # Non-matching message — discard and keep waiting
            except queue.Empty:
                pass

    timing_total  = 0.0
    timing_count  = 0
    first_cycle   = True   # True until the setup handshake completes

    while True:  # ── outer loop: one round per brass ────────────────────────

        # 1. Wait until the system is in RUNNING state
        while True:
            with STATE.lock:
                go = STATE.system_state == "RUNNING"
            if go:
                break
            time.sleep(0.05)

        # 2. Wait for "READY" from the ESP32 (routed here via the queue).
        #    On the very first cycle the ESP32 runs two move45deg() calls plus
        #    delays during setup, so give it a generous timeout.  Subsequent
        #    cycles only need to wait for a single move.
        ready_timeout = 60.0 if first_cycle else 30.0
        ready = wait_for_queue_msg("READY", timeout=ready_timeout)
        if not ready:
            # System was stopped/ESTOPed before we got READY — restart outer loop
            continue

        first_cycle = False   # setup handshake done; normal timeouts from here

        # 3. Read threshold from shared state
        with STATE.lock:
            current_threshold = display_to_internal(STATE.display_acc)

        # 4. Capture and pre-process image
        capture_image(CAPTURE_PATH)
        t0        = time.perf_counter()
        processed = process_image(CAPTURE_PATH)
        if processed is None:
            uart_send("6")
            continue

        # 5. Run inference
        label, conf = run_inference(processed)
        bin_num     = assign_bin(label, conf, current_threshold)
        ms          = (time.perf_counter() - t0) * 1000

        timing_total += ms
        timing_count += 1

        display_label = label if bin_num < 6 else "Unknown"
        with STATE.lock:
            info = STATE.bins[bin_num]
            if bin_num < 6 and info["name"] is None:
                info["name"] = label
            info["count"]    += 1
            STATE.total      += 1
            STATE.last_label  = display_label
            STATE.last_conf   = conf
            STATE.last_bin    = bin_num
            STATE.avg_ms      = timing_total / timing_count

        # 6. Tell ESP32 which bin to use.
        # No need to wait for "image" here — the ESP32 sends "ready" only after
        # it has fully finished moving (disc + SG90), so the wait_for_queue_msg
        # ("READY") at the top of the loop is the natural pacing gate.
        uart_send(str(bin_num))

# ── Drawing helpers ────────────────────────────────────────────────────────────
def rounded_rect(surf, color, rect, r=6, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=r)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, width=border, border_radius=r)

def draw_text(surf, font, text, color, x, y, anchor="topleft"):
    s, _ = font.render(text, color)
    r = s.get_rect(**{anchor: (x, y)})
    surf.blit(s, r)
    return r

# ── Main UI ────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    pygame.mouse.set_visible(False)

    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.FULLSCREEN)
    pygame.display.set_caption("Automated 9mm Brass Headstamp Sorter")
    clock = pygame.time.Clock()

    ft = pygame.freetype
    FONT_B  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_R  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        f_title = ft.Font(FONT_B, 13)
        f_label = ft.Font(FONT_R,  9)
        f_count = ft.Font(FONT_B, 18)
        f_bin   = ft.Font(FONT_R,  8)
        f_btn   = ft.Font(FONT_B, 12)
        f_stat  = ft.Font(FONT_R,  8)
        f_estop = ft.Font(FONT_B, 30)
        f_esub  = ft.Font(FONT_B, 13)
    except Exception:
        fb = ft.SysFont("monospace", 11)
        f_title = f_label = f_count = f_bin = f_btn = f_stat = f_estop = f_esub = fb

    # ── Layout (480×320) ──────────────────────────────────────────────────────
    HEADER_H  = 28
    STATS_Y   = HEADER_H + 2
    STATS_H   = 20
    BIN_TOP   = STATS_Y + STATS_H + 4   # ~54
    FOOTER_Y  = 270
    FOOTER_H  = DISPLAY_H - FOOTER_Y    # 50px footer

    # Bin grid: 3 cols × 2 rows
    COLS  = 3
    ROWS  = 2
    GAP   = 4
    BIN_W = (DISPLAY_W - GAP * (COLS + 1)) // COLS     # ~154
    BIN_H = (FOOTER_Y - BIN_TOP - GAP * (ROWS + 1)) // ROWS  # ~100

    def bin_rect(i):
        col = (i - 1) % COLS
        row = (i - 1) // COLS
        x   = GAP + col * (BIN_W + GAP)
        y   = BIN_TOP + GAP + row * (BIN_H + GAP)
        return pygame.Rect(x, y, BIN_W, BIN_H)

    # Footer buttons — centered
    BTN_W, BTN_H = 105, 36
    btn_start = pygame.Rect(DISPLAY_W // 2 - BTN_W - 6, FOOTER_Y + (FOOTER_H - BTN_H) // 2, BTN_W, BTN_H)
    btn_stop  = pygame.Rect(DISPLAY_W // 2 + 6,          FOOTER_Y + (FOOTER_H - BTN_H) // 2, BTN_W, BTN_H)

    # Accuracy bar — left of footer
    ACC_X       = 6
    ACC_Y_TOP   = FOOTER_Y + 6
    ACC_BAR_W   = 78
    ACC_BAR_H   = 9
    ACC_BAR_Y   = ACC_Y_TOP + 12

    start_ever_pressed = False
    btn_hover = {"start": False, "stop": False}

    # Start threads
    threading.Thread(target=inference_worker, daemon=True).start()
    threading.Thread(target=uart_listener,    daemon=True).start()

    running = True
    while running:
        clock.tick(30)

        # ── Events ────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

            elif ev.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                if ev.type == pygame.FINGERDOWN:
                    pos = (int(ev.x * DISPLAY_W), int(ev.y * DISPLAY_H))
                else:
                    pos = ev.pos

                with STATE.lock:
                    cur_state = STATE.system_state
                    estop     = STATE.estop_active

                if estop: continue

                if btn_start.collidepoint(pos):
                    can_start = (not start_ever_pressed) or (cur_state == "STOPPED")
                    if can_start:
                        with STATE.lock:
                            STATE.system_state = "RUNNING"
                            STATE.paused_for_start = False
                        time.sleep(0.1)
                        if not start_ever_pressed:
                            uart_send("SETUP")
                            start_ever_pressed = True
                        else:
                            uart_send("6")

                elif btn_stop.collidepoint(pos):
                    if cur_state == "RUNNING":
                        uart_send("STOP")
                        with STATE.lock:
                            STATE.system_state = "STOPPED"

                # Accuracy bar tap — remap click position to a step value
                else:
                    bar_hit = pygame.Rect(ACC_X, ACC_BAR_Y - 6, ACC_BAR_W + 4, ACC_BAR_H + 12)
                    if bar_hit.collidepoint(pos):
                        rel     = max(0, min(pos[0] - ACC_X, ACC_BAR_W))
                        steps   = (DISPLAY_MAX - DISPLAY_MIN) // DISPLAY_STEP
                        step_i  = round(rel / ACC_BAR_W * steps)
                        new_val = DISPLAY_MIN + step_i * DISPLAY_STEP
                        with STATE.lock:
                            STATE.display_acc = max(DISPLAY_MIN, min(DISPLAY_MAX, new_val))

            elif ev.type == pygame.MOUSEMOTION:
                btn_hover["start"] = btn_start.collidepoint(ev.pos)
                btn_hover["stop"]  = btn_stop.collidepoint(ev.pos)

        # ── Snapshot ──────────────────────────────────────────────────────────
        with STATE.lock:
            bins         = {k: dict(v) for k, v in STATE.bins.items()}
            total        = STATE.total
            last_label   = STATE.last_label
            last_conf    = STATE.last_conf
            last_bin     = STATE.last_bin
            avg_ms       = STATE.avg_ms
            display_acc  = STATE.display_acc
            sys_state    = STATE.system_state
            estop_active = STATE.estop_active
            paused       = STATE.paused_for_start

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        # ════════════════════ EMERGENCY STOP OVERLAY ═════════════════════════
        if estop_active:
            screen.fill(ESTOP_BG)
            draw_text(screen, f_estop, "EMERGENCY STOP",
                      ESTOP_TEXT, DISPLAY_W // 2, 100, "center")
            draw_text(screen, f_esub, "ENGAGED",
                      ESTOP_TEXT, DISPLAY_W // 2, 150, "center")
            draw_text(screen, f_stat, "Awaiting STOPESTOP signal...",
                      ESTOP_SUB,  DISPLAY_W // 2, 200, "center")
            pygame.display.flip()
            continue

        # ════════════════════════ HEADER ════════════════════════════════════
        pygame.draw.rect(screen, PANEL_DARK, (0, 0, DISPLAY_W, HEADER_H))
        pygame.draw.line(screen, BIN_EDGE, (0, HEADER_H - 1), (DISPLAY_W, HEADER_H - 1), 1)
        draw_text(screen, f_title, "AUTOMATED 9MM BRASS HEADSTAMP SORTER", TEXT_WHITE,
                  DISPLAY_W // 2, HEADER_H // 2, "center")

        # Status badge top-right
        STATE_COLORS = {
            "IDLE":    ( 90, 100, 110),
            "RUNNING": ( 55, 165,  80),
            "STOPPED": (185,  70,  55),
            "ESTOP":   (215,  28,  28),
        }
        sc = STATE_COLORS.get(sys_state, (90, 100, 110))
        badge_surf, _ = f_stat.render(sys_state, TEXT_WHITE)
        bw = badge_surf.get_width() + 10
        pygame.draw.rect(screen, sc, (DISPLAY_W - bw - 4, 5, bw, 16), border_radius=4)
        screen.blit(badge_surf, (DISPLAY_W - bw - 4 + 5, 7))

        # ════════════════════════ STATS ROW ══════════════════════════════════
        pygame.draw.rect(screen, PANEL_DARK, (0, STATS_Y, DISPLAY_W, STATS_H))
        pygame.draw.line(screen, BIN_EDGE, (0, STATS_Y + STATS_H - 1),
                         (DISPLAY_W, STATS_Y + STATS_H - 1), 1)

        draw_text(screen, f_stat,  "TOTAL SORTED",          ACCENT_LABEL, 8,   STATS_Y + 2)
        draw_text(screen, f_label, str(total),               TEXT_WHITE,   8,   STATS_Y + 11)
        draw_text(screen, f_stat,  f"AVG {avg_ms:.0f}ms" if avg_ms > 0 else "",
                  TEXT_DIM, 75, STATS_Y + 11)

        if last_bin:
            draw_text(screen, f_stat,  "LAST", ACCENT_LABEL, 160, STATS_Y + 2)
            result_str = f"{last_label}  {last_conf*100:.0f}%  →  BIN {last_bin}"
            draw_text(screen, f_stat, result_str, TEXT_WHITE, 160, STATS_Y + 11)

        # Paused notice
        if paused and sys_state == "STOPPED":
            notice_surf, _ = f_stat.render("PAUSED — Press START", (255, 220, 90))
            screen.blit(notice_surf, (DISPLAY_W - notice_surf.get_width() - 8, STATS_Y + 6))

        # ════════════════════════ BIN GRID ═══════════════════════════════════
        for i in range(1, 7):
            br    = bin_rect(i)
            info  = bins[i]
            is_uk = (i == 6)
            is_ac = (i == last_bin)

            # Background color
            if is_uk:
                bg_c  = BIN_UNKNOWN
                ed_c  = (200, 55, 55)
            elif is_ac:
                bg_c  = BIN_ACTIVE
                ed_c  = ( 90, 160, 210)
            else:
                bg_c  = BIN_NORMAL
                ed_c  = BIN_EDGE

            rounded_rect(screen, bg_c, br, r=7, border=2, border_color=ed_c)

            # Bin label (top-left)
            bin_lbl = f"BIN {i}" if not is_uk else "UNKNOWN"
            draw_text(screen, f_bin, bin_lbl, TEXT_WHITE, br.x + 6, br.y + 5)

            # Class name (center-top area)
            name = info["name"] if info["name"] else "—"
            if len(name) > 11:
                name = name[:11]
            draw_text(screen, f_label, name, TEXT_WHITE, br.centerx, br.y + 20, "center")

            # Count — large, vertically centered lower half
            draw_text(screen, f_count, str(info["count"]),
                      TEXT_WHITE, br.centerx, br.centery + 14, "center")

        # ════════════════════════ FOOTER ════════════════════════════════════
        pygame.draw.rect(screen, PANEL_DARK, (0, FOOTER_Y, DISPLAY_W, FOOTER_H))
        pygame.draw.line(screen, BIN_EDGE, (0, FOOTER_Y), (DISPLAY_W, FOOTER_Y), 1)

        # ── Accuracy bar ──────────────────────────────────────────────────────
        draw_text(screen, f_stat, f"CONFIDENCE  {display_acc}%",
                  ACCENT_LABEL, ACC_X, ACC_Y_TOP)

        # Track
        pygame.draw.rect(screen, BIN_EDGE,
                         (ACC_X, ACC_BAR_Y, ACC_BAR_W, ACC_BAR_H), border_radius=4)
        # Fill
        fill_frac = (display_acc - DISPLAY_MIN) / (DISPLAY_MAX - DISPLAY_MIN)
        fill_w    = max(4, int(ACC_BAR_W * fill_frac))
        pygame.draw.rect(screen, BTN_START,
                         (ACC_X, ACC_BAR_Y, fill_w, ACC_BAR_H), border_radius=4)
        # Step notches
        n_steps = (DISPLAY_MAX - DISPLAY_MIN) // DISPLAY_STEP
        for si in range(n_steps + 1):
            nx = ACC_X + int(si / n_steps * ACC_BAR_W)
            pygame.draw.line(screen, TEXT_DIM,
                             (nx, ACC_BAR_Y - 2), (nx, ACC_BAR_Y + ACC_BAR_H + 2), 1)

        # Internal threshold label
        int_t = display_to_internal(display_acc)
        draw_text(screen, f_stat, f"(AI {int_t*100:.0f}%)", TEXT_DIM,
                  ACC_X, ACC_BAR_Y + ACC_BAR_H + 3)

        # ── START button ──────────────────────────────────────────────────────
        can_start = (not start_ever_pressed) or (sys_state == "STOPPED")
        if can_start:
            sc_col  = BTN_START_HI if btn_hover["start"] else BTN_START
            ed_col  = ( 35, 100, 55)
        else:
            sc_col  = BTN_DISABLED
            ed_col  = BIN_EDGE

        rounded_rect(screen, sc_col, btn_start, r=8, border=2, border_color=ed_col)
        draw_text(screen, f_btn, "START", BTN_TEXT, btn_start.centerx, btn_start.centery, "center")

        # ── STOP button ───────────────────────────────────────────────────────
        can_stop = (sys_state == "RUNNING")
        if can_stop:
            sp_col  = BTN_STOP_HI if btn_hover["stop"] else BTN_STOP
            sep_col = (120, 38, 38)
        else:
            sp_col  = BTN_DISABLED
            sep_col = BIN_EDGE

        rounded_rect(screen, sp_col, btn_stop, r=8, border=2, border_color=sep_col)
        draw_text(screen, f_btn, "STOP", BTN_TEXT, btn_stop.centerx, btn_stop.centery, "center")

        pygame.display.flip()

    # Cleanup
    with STATE.lock:
        STATE.system_state = "IDLE"
    if _ser:
        _ser.close()
    pygame.quit()

if __name__ == "__main__":
    main()
