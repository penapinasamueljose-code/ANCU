import sys
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QMessageBox, QFrame, QHBoxLayout
)
from PyQt5.QtCore import QTimer, Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QFont, QPen

# Ocultar consola en Windows si se ejecuta sin entorno
if sys.platform == "win32":
    import ctypes
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd != 0:
        ctypes.windll.user32.ShowWindow(hwnd, 0)

STYLE_SHEET = """
QMainWindow { background-color: #0F172A; }
QFrame#card { background-color: #1E293B; border-radius: 16px; padding: 15px; }
QLabel { color: #F8FAFC; font-family: 'Segoe UI', sans-serif; font-size: 13px; font-weight: 500; }
QLabel#title { font-size: 18px; font-weight: bold; color: #38BDF8; }
QComboBox { background-color: #334155; color: #F8FAFC; border: 1px solid #475569; border-radius: 8px; padding: 6px 12px; font-size: 13px; }
QComboBox::drop-down { border: none; }
QSlider::groove:horizontal { height: 8px; background: #334155; border-radius: 4px; }
QSlider::sub-page:horizontal { background: #38BDF8; border-radius: 4px; }
QSlider::handle:horizontal { background: #F8FAFC; width: 20px; margin-top: -6px; margin-bottom: -6px; border-radius: 10px; }
QPushButton#ancBtn { background-color: #0284C7; color: white; font-weight: bold; font-size: 14px; border-radius: 10px; padding: 12px; border: none; }
QPushButton#ancBtn:hover { background-color: #0369A1; }
QPushButton#ancBtn:checked { background-color: #EF4444; }
QPushButton#ancBtn:checked:hover { background-color: #DC2626; }
"""

class GynzyGaugeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_value = 0.0
        self.target_value = 0.0
        self.threshold_value = 60.0
        self.setMinimumSize(320, 200)

    def set_values(self, current, threshold):
        self.target_value = current
        self.threshold_value = threshold
        self.current_value += (self.target_value - self.current_value) * 0.25
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        size = min(width, height * 1.5) - 40

        rect = QRectF((width - size) / 2, height - size / 1.5 - 20, size, size)

        pen_width = 22
        pen = QPen()
        pen.setWidth(pen_width)
        pen.setCapStyle(Qt.RoundCap)

        angle_green = int((self.threshold_value / 100.0) * 180 * 16)
        pen.setColor(QColor("#10B981"))
        painter.setPen(pen)
        painter.drawArc(rect, (180 * 16) - angle_green, angle_green)

        angle_red = int(((100 - self.threshold_value) / 100.0) * 180 * 16)
        pen.setColor(QColor("#EF4444"))
        painter.setPen(pen)
        painter.drawArc(rect, 0, angle_red)

        center_x = rect.center().x()
        center_y = rect.center().y()
        radius = size / 2

        current_angle_rad = np.radians(180 - (min(self.current_value, 100) / 100.0 * 180))
        needle_length = radius - 18
        needle_x = center_x + needle_length * np.cos(current_angle_rad)
        needle_y = center_y - needle_length * np.sin(current_angle_rad)

        needle_pen = QPen(QColor("#F8FAFC"), 3)
        painter.setPen(needle_pen)
        painter.drawLine(int(center_x), int(center_y), int(needle_x), int(needle_y))

        painter.setBrush(QColor("#38BDF8"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(center_x - 8), int(center_y - 8), 16, 16)

        painter.setFont(QFont("Segoe UI", 13, QFont.Bold))
        if self.current_value > self.threshold_value:
            painter.setPen(QColor("#EF4444"))
            text = f"¡RUIDO ELEVADO! {int(self.current_value)}%"
        else:
            painter.setPen(QColor("#10B981"))
            text = f"Ruido Estable: {int(self.current_value)}%"

        painter.drawText(int(width / 2 - 80), int(height - 10), text)


class CombinedGynzyANCApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Retorno Limpio & Control Gynzy")
        self.setGeometry(100, 100, 520, 720)
        self.setStyleSheet(STYLE_SHEET)

        self.is_anc_active = False
        self.stream_in = None
        self.stream_out = None
        self.audio_buffer = np.zeros((2048, 1), dtype=np.float32)
        self.noise_level = 0.0
        self.threshold = 60
        self.volume_gain = 1.0
        self.noise_profile = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout()

        title = QLabel("Procesamiento de Audio en Tiempo Real")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        card_layout.addWidget(QLabel("Micrófono de Captura:"))
        self.combo_input = QComboBox()
        card_layout.addWidget(self.combo_input)

        card_layout.addWidget(QLabel("Salida de Auriculares:"))
        self.combo_output = QComboBox()
        card_layout.addWidget(self.combo_output)

        card_layout.addWidget(QLabel("Modo de Procesamiento:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("🔊 Modo Transparencia Limpia (FFT + Noise Gate)", "clean")
        self.combo_mode.addItem("🔄 Inversión de Fase (180° Inverted ANC)", "invert")
        card_layout.addWidget(self.combo_mode)

        self.populate_audio_devices()

        self.gauge = GynzyGaugeWidget()
        card_layout.addWidget(self.gauge)

        card_layout.addWidget(QLabel("Sensibilidad del Detector / Umbral Puerta de Ruido:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(20, 95)
        self.slider.setValue(self.threshold)
        self.slider.valueChanged.connect(self.update_threshold)
        card_layout.addWidget(self.slider)

        self.lbl_threshold = QLabel(f"Umbral de Alerta: {self.threshold}%")
        self.lbl_threshold.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.lbl_threshold)

        # Control de Volumen de Salida
        card_layout.addWidget(QLabel("Volumen de Retorno:"))
        self.slider_vol = QSlider(Qt.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(80)
        self.slider_vol.valueChanged.connect(self.update_volume)
        card_layout.addWidget(self.slider_vol)

        self.btn_toggle_anc = QPushButton("Activar Proceso")
        self.btn_toggle_anc.setObjectName("ancBtn")
        self.btn_toggle_anc.setCheckable(True)
        self.btn_toggle_anc.clicked.connect(self.toggle_anc)
        card_layout.addWidget(self.btn_toggle_anc)

        card.setLayout(card_layout)
        main_layout.addWidget(card)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.monitor_audio)
        self.timer.start(50)

    def populate_audio_devices(self):
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.combo_input.addItem(f"🎤 {dev['name']}", i)
            if dev['max_output_channels'] > 0:
                self.combo_output.addItem(f"🎧 {dev['name']}", i)

    def update_threshold(self, val):
        self.threshold = val
        self.lbl_threshold.setText(f"Umbral de Alerta: {self.threshold}%")

    def update_volume(self, val):
        self.volume_gain = val / 100.0

    def monitor_audio(self):
        input_device_idx = self.combo_input.currentData()
        if input_device_idx is None:
            return

        try:
            device_info = sd.query_devices(input_device_idx, 'input')
            samplerate = int(device_info['default_samplerate'])

            rec = sd.rec(
                int(0.04 * samplerate),
                samplerate=samplerate,
                channels=1,
                device=input_device_idx,
                blocking=False
            )
            sd.wait()

            rms = np.sqrt(np.mean(rec**2)) * 320
            self.noise_level = (self.noise_level * 0.5) + (min(rms, 100) * 0.5)
            self.gauge.set_values(self.noise_level, self.threshold)

        except Exception:
            pass

    def audio_callback(self, indata, frames, time, status):
        """ Procesa el audio aplicando filtrado espectral o inversión de fase (180°) """
        if self.is_anc_active:
            try:
                mode = self.combo_mode.currentData()
                signal = indata[:, 0].astype(np.float32)

                if mode == "invert":
                    # Inversión de fase matemática pura (Onda invertida 180°)
                    processed_signal = -1.0 * signal
                else:
                    # Sustracción espectral + Puerta de ruido
                    fft_signal = np.fft.rfft(signal)
                    magnitude = np.abs(fft_signal)
                    phase = np.angle(fft_signal)

                    if self.noise_profile is None:
                        self.noise_profile = magnitude
                    else:
                        self.noise_profile = 0.98 * self.noise_profile + 0.02 * magnitude

                    cleaned_magnitude = np.maximum(magnitude - (self.noise_profile * 2.0), 0)
                    cleaned_fft = cleaned_magnitude * np.exp(1j * phase)
                    processed_signal = np.fft.irfft(cleaned_fft)

                    # Puerta de ruido
                    rms_val = np.sqrt(np.mean(processed_signal**2))
                    gate_threshold = (self.threshold / 100.0) * 0.05
                    if rms_val < gate_threshold:
                        processed_signal *= 0.05

                # Aplicar control de volumen de salida
                processed_signal *= self.volume_gain
                self.audio_buffer = processed_signal.reshape(-1, 1).astype(np.float32)

            except Exception:
                self.audio_buffer = indata
        else:
            self.audio_buffer = np.zeros((frames, 1), dtype=np.float32)

    def out_callback(self, outdata, frames, time, status):
        if self.is_anc_active:
            try:
                outdata[:] = self.audio_buffer
            except Exception:
                outdata.fill(0)
        else:
            outdata.fill(0)

    def toggle_anc(self, checked):
        self.is_anc_active = checked
        if self.is_anc_active:
            self.noise_profile = None
            self.btn_toggle_anc.setText("Desactivar Proceso")
            self.start_processing()
        else:
            self.btn_toggle_anc.setText("Activar Proceso")
            self.stop_processing()

    def start_processing(self):
        input_idx = self.combo_input.currentData()
        output_idx = self.combo_output.currentData()

        try:
            self.stream_in = sd.InputStream(
                device=input_idx,
                channels=1,
                samplerate=44100,
                blocksize=2048,
                callback=self.audio_callback
            )
            self.stream_out = sd.OutputStream(
                device=output_idx,
                channels=1,
                samplerate=44100,
                blocksize=2048,
                callback=self.out_callback
            )

            self.stream_in.start()
            self.stream_out.start()

        except Exception as e:
            QMessageBox.critical(self, "Error de Audio", f"No se pudo iniciar el proceso: {e}")
            self.btn_toggle_anc.setChecked(False)
            self.is_anc_active = False

    def stop_processing(self):
        if hasattr(self, 'stream_in') and self.stream_in and self.stream_in.active:
            self.stream_in.stop()
            self.stream_in.close()
        if hasattr(self, 'stream_out') and self.stream_out and self.stream_out.active:
            self.stream_out.stop()
            self.stream_out.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CombinedGynzyANCApp()
    window.show()
    sys.exit(app.exec_())
