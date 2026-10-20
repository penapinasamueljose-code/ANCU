"""ANCU Desktop 0.2.0-preview — reducción digital de ruido, no ANC acústico.
Requiere Python 3.11+, numpy >=2, sounddevice y PyQt5. Audio local, sin red.
Basado en el archivo ANCU_code.txt proporcionado por el propietario del proyecto.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

VERSION = "0.2.0-preview"
RATE = 48_000
HOP = 256
SIZE = HOP * 2


@dataclass(frozen=True)
class Parameters:
    strength: float = 0.6
    volume: float = 0.2
    bypass: bool = False


class SpectralDenoiser:
    """STFT con ventana sqrt-Hann, 50% de solapamiento y perfil de ruido.

    El estado pertenece exclusivamente al callback. No se consultan widgets ni
    se hace I/O aquí. NumPy/Python NO ofrecen garantías de tiempo real duro.
    El retardo algorítmico es HOP/RATE (5.33 ms); NO es la latencia total.
    """
    def __init__(self) -> None:
        import numpy as np
        self.np = np
        self.window = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(SIZE) / SIZE))
        self.history = np.zeros(SIZE, dtype=np.float64)
        self.analysis = np.zeros(SIZE, dtype=np.float64)
        self.tail = np.zeros(HOP, dtype=np.float64)
        self.spectrum = np.zeros(SIZE // 2 + 1, dtype=np.complex128)
        self.inverse = np.zeros(SIZE, dtype=np.float64)
        self.noise = np.full(SIZE // 2 + 1, 1e-8)
        self.gain = np.ones(SIZE // 2 + 1)
        self.calibration_left = 94
        self.calibration_count = 0
        self.previous_volume = 0.0
        self.wet = 1.0
        self.ramp = np.linspace(0, 1, HOP)

    def calibrate(self) -> None:
        self.calibration_left = 94
        self.calibration_count = 0

    def process(self, signal: Any, output: Any, parameters: Parameters) -> None:
        np = self.np
        self.history[:HOP] = self.history[HOP:]
        self.history[HOP:] = signal
        np.nan_to_num(self.history, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.multiply(self.history, self.window, out=self.analysis)
        np.fft.rfft(self.analysis, out=self.spectrum)
        power = self.spectrum.real ** 2 + self.spectrum.imag ** 2
        if self.calibration_left:
            self.calibration_count += 1
            self.noise += (power - self.noise) / self.calibration_count
            self.calibration_left -= 1
        else:
            # No aprender automáticamente los picos de voz como ruido.
            stationary = power < self.noise * 2.5 + 1e-9
            self.noise[stationary] = 0.98 * self.noise[stationary] + 0.02 * power[stationary]
        strength = max(0.0, min(1.0, parameters.strength))
        floor = 1.0 - 0.94 * strength
        target = np.sqrt(np.maximum(1 - (1 + strength) * self.noise / (power + 1e-12), floor * floor))
        if strength == 0:
            target.fill(1)
        target[1:-1] = (target[:-2] + 2 * target[1:-1] + target[2:]) * 0.25
        self.gain *= 0.65
        self.gain += target * 0.35
        self.spectrum *= self.gain
        np.fft.irfft(self.spectrum, n=SIZE, out=self.inverse)
        self.inverse *= self.window
        wet_signal = self.inverse[:HOP] + self.tail
        self.tail[:] = self.inverse[HOP:]
        target_wet = 0.0 if parameters.bypass else 1.0
        wet_ramp = self.wet + (target_wet - self.wet) * self.ramp
        result = wet_signal * wet_ramp + self.history[:HOP] * (1 - wet_ramp)
        volume = max(0.0, min(1.0, parameters.volume))
        result *= self.previous_volume + (volume - self.previous_volume) * self.ramp
        self.previous_volume, self.wet = volume, target_wet
        np.clip(result, -0.98, 0.98, out=result)
        output[:] = result


class AudioEngine:
    """Un único stream dúplex: sin grabaciones paralelas ni búfer compartido.

    La GUI cambia una referencia a Parameters (inmutable). El callback toma
    una instantánea por bloque. Las métricas son orientativas, no telemetría cloud.
    """
    def __init__(self) -> None:
        self.parameters = Parameters()
        self.stream: Any = None
        self.dsp: SpectralDenoiser | None = None
        self.input_db = -90.0
        self.processing_ms = 0.0
        self.xruns = 0
        self.error = ""
        self.calibration_requested = False
        self.reported_latency_ms: float | None = None
        self.channels = 2

    @staticmethod
    def devices() -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [(i, str(d["name"])) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        outputs = [(i, str(d["name"])) for i, d in enumerate(devices) if d["max_output_channels"] > 0]
        return inputs, outputs

    def start(self, input_device: int, output_device: int) -> None:
        import sounddevice as sd
        self.stop()
        self.dsp = SpectralDenoiser()
        self.channels = min(2, int(sd.query_devices(output_device)["max_output_channels"]))
        sd.check_input_settings(device=input_device, channels=1, dtype="float32", samplerate=RATE)
        sd.check_output_settings(device=output_device, channels=self.channels, dtype="float32", samplerate=RATE)
        self.error, self.xruns = "", 0
        stream = sd.Stream(device=(input_device, output_device), samplerate=RATE,
                           blocksize=HOP, channels=(1, self.channels), dtype="float32",
                           latency="low", callback=self._callback)
        try:
            stream.start()
            self.stream = stream
            latency = stream.latency
            self.reported_latency_ms = (sum(latency) + HOP / RATE) * 1000
        except Exception:
            stream.close()
            self.dsp = None
            raise

    def _callback(self, indata: Any, outdata: Any, frames: int, timing: Any, status: Any) -> None:
        if status:
            self.xruns += 1
        if frames != HOP or self.dsp is None:
            outdata.fill(0)
            self.error = "El controlador no entregó bloques de 256 muestras. Cambia de dispositivo."
            return
        try:
            started = time.perf_counter()
            if self.calibration_requested:
                self.dsp.calibrate()
                self.calibration_requested = False
            params = self.parameters
            self.dsp.process(indata[:, 0], outdata[:, 0], params)
            for channel in range(1, self.channels):
                outdata[:, channel] = outdata[:, 0]
            rms = float(self.dsp.np.sqrt(self.dsp.np.mean(indata[:, 0].astype(float) ** 2)))
            self.input_db = max(-90.0, min(0.0, 20 * math.log10(max(rms, 1e-9))))
            self.processing_ms = 0.9 * self.processing_ms + 0.1 * (time.perf_counter() - started) * 1000
        except Exception as exc:
            outdata.fill(0)
            self.error = f"El audio se detuvo por un error de procesamiento: {type(exc).__name__}."

    def stop(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        self.dsp = None
        self.input_db = -90.0
        self.reported_latency_ms = None


def launch_ui() -> int:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
    from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
        QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
        QPushButton, QSlider, QVBoxLayout, QWidget)

    app = QApplication(sys.argv)
    app.setApplicationName("ANCU")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet("""
    QWidget { background:#09090b; color:#ededf1; font-family:'Segoe UI'; }
    QLabel { background:transparent; }
    QFrame#card { background:#111115; border:1px solid #25252c; border-radius:18px; }
    QLabel#muted { color:#96939f; font-size:12px; }
    QLabel#eyebrow { color:#baa1f7; font-size:11px; }
    QLabel#title { font-size:30px; font-weight:600; }
    QComboBox { background:#19191f; border:1px solid #32313a; border-radius:9px; padding:11px; min-height:17px; }
    QComboBox QAbstractItemView { background:#19191f; color:#ededf1; selection-background-color:#473566; }
    QComboBox::drop-down { border:none; width:25px; }
    QPushButton { background:#1a191f; border:1px solid #33313c; border-radius:9px; padding:11px 16px; }
    QPushButton:hover { background:#282331; border-color:#6d578f; }
    QPushButton:disabled { color:#595560; border-color:#26242c; }
    QPushButton#primary { background:#b89af7; color:#1b112b; border:1px solid #cbb2ff; font-weight:600; }
    QPushButton#primary:hover { background:#cbb2ff; }
    QPushButton:checked { border-color:#a78bfa; color:#c7adff; background:#241c35; }
    QSlider::groove:horizontal { background:#2b2734; height:5px; border-radius:2px; }
    QSlider::sub-page:horizontal { background:#a78bfa; border-radius:2px; }
    QSlider::handle:horizontal { background:#dacaff; border:3px solid #a78bfa; width:12px; margin:-6px 0; border-radius:8px; }
    QCheckBox { color:#a9a3b2; spacing:9px; background:transparent; }
    QCheckBox::indicator { width:15px; height:15px; border:1px solid #5c526d; border-radius:4px; }
    QCheckBox::indicator:checked { background:#a78bfa; }
    """)

    class Waveform(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.level = 0.0
            self.phase = 0.0
            self.active = False
            self.setMinimumHeight(105)
            self.setStyleSheet("background:transparent")

        def paintEvent(self, event: Any) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            width, height = self.width(), self.height()
            painter.setPen(QPen(QColor("#24202e"), 1))
            painter.drawLine(0, height // 2, width, height // 2)
            amplitude = (7 + self.level * 29) if self.active else 2
            for layer in range(3):
                path = QPainterPath()
                for x in range(0, width + 1, 3):
                    envelope = math.sin(math.pi * x / max(1, width)) ** 1.5
                    y = height / 2 + math.sin(x * 0.032 + self.phase + layer * 0.5) * amplitude * envelope
                    if x == 0:
                        path.moveTo(x, y)
                    else:
                        path.lineTo(x, y)
                color = QColor("#b89af7" if layer < 2 else "#6bd5c4")
                color.setAlpha(190 if layer == 0 else 65)
                painter.setPen(QPen(color, 1.6))
                painter.drawPath(path)

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.engine = AudioEngine()
            self.running = False
            self.setWindowTitle(f"ANCU · Tu espacio, a tu ritmo · {VERSION}")
            self.resize(900, 790)
            self.setMinimumSize(720, 700)
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(35, 25, 35, 24)
            layout.setSpacing(16)
            brand = QHBoxLayout()
            logo = QLabel("≋  ancu.")
            logo.setStyleSheet("font-size:27px;font-weight:600;color:#c3a8ff")
            brand.addWidget(logo)
            brand.addStretch()
            version = QLabel(f"DESKTOP PREVIEW  /  {VERSION}")
            version.setObjectName("muted")
            brand.addWidget(version)
            layout.addLayout(brand)
            title = QLabel("Tu espacio, a tu ritmo.")
            title.setObjectName("title")
            layout.addWidget(title)
            subtitle = QLabel("Elige tu micrófono, ponte los audífonos y deja que tu voz tenga más espacio.")
            subtitle.setObjectName("muted")
            subtitle.setWordWrap(True)
            layout.addWidget(subtitle)
            card = QFrame()
            card.setObjectName("card")
            inside = QVBoxLayout(card)
            inside.setContentsMargins(24, 20, 24, 18)
            status_row = QHBoxLayout()
            self.status = QLabel("●  Todo listo para empezar")
            self.status.setObjectName("eyebrow")
            status_row.addWidget(self.status)
            status_row.addStretch()
            local = QLabel("AUDIO LOCAL · SIN NUBE")
            local.setObjectName("muted")
            status_row.addWidget(local)
            inside.addLayout(status_row)
            self.wave = Waveform()
            inside.addWidget(self.wave)
            self.meter = QLabel("El micrófono está apagado. Tú decides cuándo escuchar.")
            self.meter.setObjectName("muted")
            inside.addWidget(self.meter)
            layout.addWidget(card)
            settings = QFrame()
            settings.setObjectName("card")
            grid = QGridLayout(settings)
            grid.setContentsMargins(24, 20, 24, 22)
            grid.setHorizontalSpacing(24)
            grid.setVerticalSpacing(12)
            grid.addWidget(QLabel("¿Desde dónde te escuchamos?"), 0, 0)
            grid.addWidget(QLabel("¿Dónde quieres escuchar?"), 0, 1)
            self.input = QComboBox()
            self.output = QComboBox()
            self.input.setAccessibleName("Micrófono de entrada")
            self.output.setAccessibleName("Audífonos de salida")
            grid.addWidget(self.input, 1, 0)
            grid.addWidget(self.output, 1, 1)
            self.strength_label = QLabel("Limpieza de ruido · 60%")
            grid.addWidget(self.strength_label, 2, 0)
            self.volume_label = QLabel("Volumen de retorno · 20%")
            grid.addWidget(self.volume_label, 2, 1)
            self.strength = QSlider(Qt.Horizontal)
            self.volume = QSlider(Qt.Horizontal)
            for slider, value in ((self.strength, 60), (self.volume, 20)):
                slider.setRange(0, 100)
                slider.setValue(value)
                slider.valueChanged.connect(self.update_parameters)
            self.strength.setAccessibleName("Intensidad de reducción de ruido")
            self.volume.setAccessibleName("Volumen de retorno")
            grid.addWidget(self.strength, 3, 0)
            grid.addWidget(self.volume, 3, 1)
            tip = QLabel("Más natural ← → Más limpieza")
            tip.setObjectName("muted")
            grid.addWidget(tip, 4, 0)
            safety = QLabel("Empieza suave. Tu oído va primero.")
            safety.setObjectName("muted")
            grid.addWidget(safety, 4, 1)
            layout.addWidget(settings)
            row = QHBoxLayout()
            self.compare = QPushButton("Comparar sin filtro")
            self.compare.setCheckable(True)
            self.compare.setEnabled(False)
            self.compare.toggled.connect(self.update_parameters)
            self.calibrate = QPushButton("Aprender el ambiente")
            self.calibrate.setEnabled(False)
            self.calibrate.clicked.connect(self.learn_noise)
            self.refresh = QPushButton("Actualizar dispositivos")
            self.refresh.clicked.connect(self.populate_devices)
            for button in (self.compare, self.calibrate, self.refresh):
                row.addWidget(button)
            layout.addLayout(row)
            self.headphones = QCheckBox("Estoy usando audífonos y he bajado el volumen antes de empezar.")
            layout.addWidget(self.headphones)
            self.start_button = QPushButton("Empezar a escuchar")
            self.start_button.setObjectName("primary")
            self.start_button.setMinimumHeight(50)
            self.start_button.clicked.connect(self.toggle)
            layout.addWidget(self.start_button)
            self.hint = QLabel("Al empezar, deja medio segundo de silencio para aprender el ruido del ambiente.")
            self.hint.setObjectName("muted")
            self.hint.setWordWrap(True)
            layout.addWidget(self.hint)
            note = QLabel("Reducción digital del micrófono; no cancela físicamente el ruido que llega al oído.\n48 kHz · mono · Python / NumPy / PortAudio. Sin transmisión móvil ni anuncios en esta versión.")
            note.setObjectName("muted")
            note.setWordWrap(True)
            layout.addWidget(note)
            self.populate_devices()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(50)  # Solo dibuja/lee métricas. Nunca abre una captura.

        def populate_devices(self) -> None:
            if self.running:
                return
            self.input.clear()
            self.output.clear()
            try:
                import sounddevice as sd
                inputs, outputs = self.engine.devices()
                defaults = sd.default.device
                for combo, items, default in ((self.input, inputs, defaults[0]), (self.output, outputs, defaults[1])):
                    for index, name in items:
                        combo.addItem(name, index)
                    found = combo.findData(default)
                    if found >= 0:
                        combo.setCurrentIndex(found)
                if not inputs or not outputs:
                    self.hint.setText("No encontramos una entrada y una salida. Conecta tus dispositivos y pulsa Actualizar.")
            except Exception as exc:
                self.hint.setText(f"No pudimos enumerar tus dispositivos: {exc}")

        def update_parameters(self, *_: Any) -> None:
            self.engine.parameters = Parameters(self.strength.value() / 100, self.volume.value() / 100, self.compare.isChecked())
            self.strength_label.setText(f"Limpieza de ruido · {self.strength.value()}%")
            self.volume_label.setText(f"Volumen de retorno · {self.volume.value()}%")
            self.compare.setText("Volver al filtro" if self.compare.isChecked() else "Comparar sin filtro")

        def learn_noise(self) -> None:
            self.engine.calibration_requested = True
            self.hint.setText("Danos medio segundo de silencio. Estamos aprendiendo el sonido del ambiente.")

        def toggle(self) -> None:
            if self.running:
                self.stop()
                return
            if not self.headphones.isChecked():
                QMessageBox.information(self, "Tu oído va primero", "Usa audífonos para evitar acoples. Baja el volumen y confirma la casilla antes de empezar.")
                return
            if self.input.currentData() is None or self.output.currentData() is None:
                QMessageBox.information(self, "Nos falta un dispositivo", "Conecta un micrófono y una salida de audio; después actualiza la lista.")
                return
            try:
                self.update_parameters()
                self.engine.start(self.input.currentData(), self.output.currentData())
                self.running = True
                self.start_button.setText("Dejar de escuchar")
                for button in (self.compare, self.calibrate):
                    button.setEnabled(True)
                for widget in (self.input, self.output, self.refresh):
                    widget.setEnabled(False)
                self.hint.setText("Estamos escuchando el ambiente. Deja medio segundo de silencio; después habla con naturalidad.")
            except Exception as exc:
                self.stop()
                QMessageBox.warning(self, "No pudimos abrir el audio", f"{exc}\n\nPrueba otro dispositivo o configura su formato a 48 kHz en Windows. No se ha iniciado una grabación.")

        def stop(self) -> None:
            try:
                self.engine.stop()
            except Exception:
                logging.exception("Error cerrando el dispositivo")
            self.running = False
            self.start_button.setText("Empezar a escuchar")
            for button in (self.compare, self.calibrate):
                button.setEnabled(False)
            for widget in (self.input, self.output, self.refresh):
                widget.setEnabled(True)
            self.status.setText("●  Tu micrófono está en pausa")
            self.wave.active = False
            self.meter.setText("El micrófono está apagado. Tú decides cuándo escuchar.")
            self.hint.setText("Cuando quieras, volvemos a escuchar. No guardamos ni enviamos tu audio.")

        def tick(self) -> None:
            if self.running and self.engine.error:
                error, self.engine.error = self.engine.error, ""
                self.stop()
                QMessageBox.warning(self, "Hemos pausado el audio", error)
            if self.running:
                learning = bool(self.engine.dsp and self.engine.dsp.calibration_left)
                self.status.setText("●  Aprendiendo el ambiente…" if learning else "●  Tu voz tiene más espacio" if not self.compare.isChecked() else "●  Escuchando la señal sin filtro")
                self.wave.active = True
                self.wave.phase += 0.18
                self.wave.level += (max(0, (self.engine.input_db + 65) / 65) - self.wave.level) * 0.2
                latency = self.engine.reported_latency_ms
                delay_text = f"{latency:.1f} ms estimados por el controlador" if latency is not None else "retardo no disponible"
                self.meter.setText(f"Entrada {self.engine.input_db:.0f} dBFS  ·  DSP {self.engine.processing_ms:.2f} ms  ·  {delay_text}  ·  incidencias {self.engine.xruns}")
            self.wave.update()

        def closeEvent(self, event: Any) -> None:
            self.timer.stop()
            self.stop()
            event.accept()

    window = Window()
    window.show()
    if "--screenshot" in sys.argv:
        destination = sys.argv[sys.argv.index("--screenshot") + 1]
        def capture() -> None:
            window.grab().save(destination)
            window.close()
            app.quit()
        QTimer.singleShot(400, capture)
    return app.exec_()


if __name__ == "__main__":
    try:
        raise SystemExit(launch_ui())
    except ModuleNotFoundError as exc:
        message = (f"Falta una dependencia: {exc.name}.\n\nAbre una terminal en esta carpeta y ejecuta:\n"
                   "py -m pip install -r requirements.txt\n\nDespués ejecuta: py ancu_core.pyw")
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, message, "ANCU · Instalación", 0x30)
        else:
            print(message, file=sys.stderr)
        raise SystemExit(1)
