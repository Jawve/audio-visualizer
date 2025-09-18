# pages/visualizer_page.py
from __future__ import annotations
import os
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QComboBox, QSlider, QListWidget, QListWidgetItem, QSplitter, QGroupBox,
    QFormLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QSizePolicy, QMessageBox
)

from backgrounds import AnimatedBackground
from widgets.bars_widget import BarsWidget

# ---------------- Layer item ----------------
class LayerItem(QGraphicsPixmapItem):
    def __init__(self, pix: QPixmap, name: str):
        super().__init__(pix)
        self.setFlags(self.ItemIsMovable | self.ItemIsSelectable | self.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.name = name
        self._handle = 12
        self._resizing = False

    def hoverMoveEvent(self, ev):
        if not self.isSelected():
            self.setCursor(Qt.OpenHandCursor); return
        br = self.boundingRect().bottomRight()
        grip = QRectF(br.x()-self._handle, br.y()-self._handle, self._handle, self._handle)
        self.setCursor(Qt.SizeFDiagCursor if grip.contains(ev.pos()) else Qt.OpenHandCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.isSelected():
            br = self.boundingRect().bottomRight()
            grip = QRectF(br.x()-self._handle, br.y()-self._handle, self._handle, self._handle)
            if grip.contains(ev.pos()):
                self._resizing = True; ev.accept(); return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._resizing:
            br = self.boundingRect().bottomRight()
            local = ev.pos()
            sx = max(0.2, local.x()/max(1e-3, br.x()))
            sy = max(0.2, local.y()/max(1e-3, br.y()))
            self.setScale((sx+sy)/2.0); ev.accept(); return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._resizing = False
        super().mouseReleaseEvent(ev)

# ---------------- Page ----------------
class VisualizerPage(QWidget):
    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self.hub = hub

        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        self.bg = AnimatedBackground(mode="animated", parent=self)
        root.addWidget(self.bg)

        overlay = QWidget(self.bg); overlay.setAttribute(Qt.WA_TranslucentBackground, True)
        overlay_layout = QVBoxLayout(overlay); overlay_layout.setContentsMargins(8,8,8,8); overlay_layout.setSpacing(6)
        bg_layout = QVBoxLayout(self.bg); bg_layout.setContentsMargins(0,0,0,0); bg_layout.addWidget(overlay)

        # --- top bar: visualizer + pop button
        top_row = QHBoxLayout()
        self._top_row = top_row

        self.bars = BarsWidget(bar_count=48, smoothing=0.65)
        self.bars.set_mirrored(True)
        self.bars.setMinimumHeight(100)
        self.bars.setMaximumHeight(140)
        self.bars.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_pop = QPushButton("Pop out visualizer")
        self.btn_pop.setFixedHeight(34)
        self.btn_pop.clicked.connect(self._pop_out_visualizer)

        top_row.addWidget(self.bars, 1)
        top_row.addWidget(self.btn_pop, 0)
        overlay_layout.addLayout(top_row)

        # --- middle split: left controls + center sandbox
        split = QSplitter(); split.setOrientation(Qt.Horizontal); split.setChildrenCollapsible(False)
        overlay_layout.addWidget(split, 1)

        # left controls
        left = QWidget(); left.setAttribute(Qt.WA_TranslucentBackground, True)
        left_lay = QVBoxLayout(left); left_lay.setContentsMargins(6,6,6,6); left_lay.setSpacing(8)

        dev_box = QGroupBox("Playback Device"); dev_form = QFormLayout(dev_box)
        self.cmb_device = QComboBox()
        dev_form.addRow("Device:", self.cmb_device)

        ctrl_box = QGroupBox("Global Controls"); ctrl_form = QFormLayout(ctrl_box)
        self.sld_sens = QSlider(Qt.Horizontal); self.sld_sens.setRange(5, 400); self.sld_sens.setValue(100)
        self.sld_low  = QSlider(Qt.Horizontal); self.sld_low.setRange(0, 100); self.sld_low.setValue(0)
        self.sld_high = QSlider(Qt.Horizontal); self.sld_high.setRange(0, 100); self.sld_high.setValue(100)
        ctrl_form.addRow("Sensitivity", self.sld_sens)
        ctrl_form.addRow("EQ Low %", self.sld_low)
        ctrl_form.addRow("EQ High %", self.sld_high)

        lay_box = QGroupBox("Layers"); lay_lay = QVBoxLayout(lay_box)
        self.list_layers = QListWidget()
        row = QHBoxLayout()
        self.btn_add = QPushButton("Add Image"); self.btn_up = QPushButton("Up"); self.btn_down = QPushButton("Down"); self.btn_del = QPushButton("Delete")
        row.addWidget(self.btn_add); row.addWidget(self.btn_up); row.addWidget(self.btn_down); row.addWidget(self.btn_del)
        lay_lay.addWidget(self.list_layers); lay_lay.addLayout(row)

        sel_box = QGroupBox("Selected Layer Controls"); sel_form = QFormLayout(sel_box)
        self.sld_layer_sens = QSlider(Qt.Horizontal); self.sld_layer_sens.setRange(5, 400); self.sld_layer_sens.setValue(100)
        self.sld_layer_low  = QSlider(Qt.Horizontal); self.sld_layer_low.setRange(0, 100); self.sld_layer_low.setValue(0)
        self.sld_layer_high = QSlider(Qt.Horizontal); self.sld_layer_high.setRange(0, 100); self.sld_layer_high.setValue(100)
        sel_form.addRow("Layer Sensitivity", self.sld_layer_sens)
        sel_form.addRow("Layer EQ Low %", self.sld_layer_low)
        sel_form.addRow("Layer EQ High %", self.sld_layer_high)

        left_lay.addWidget(dev_box)
        left_lay.addWidget(ctrl_box)
        left_lay.addWidget(lay_box, 1)
        left_lay.addWidget(sel_box)
        split.addWidget(left)

        # center sandbox
        center = QWidget(); center.setAttribute(Qt.WA_TranslucentBackground, True)
        center_lay = QVBoxLayout(center); center_lay.setContentsMargins(0,0,0,0)
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        center_lay.addWidget(self.view)
        split.addWidget(center)
        split.setSizes([340, 860])

        # signals
        self.hub.devicesChanged.connect(self._fill_devices)
        self.hub.levelsUpdated.connect(self._feed_levels)
        self.hub.currentDeviceChanged.connect(self._on_device_changed)

        self.cmb_device.currentTextChanged.connect(self._device_selected)
        self.sld_sens.valueChanged.connect(lambda v: self.hub.set_sensitivity(v/100.0))
        self.sld_low.valueChanged.connect(self._apply_global_eq)
        self.sld_high.valueChanged.connect(self._apply_global_eq)

        self.btn_add.clicked.connect(self._add_image_layer)
        self.btn_del.clicked.connect(self._delete_selected_layer)
        self.btn_up.clicked.connect(lambda: self._move_selected(-1))
        self.btn_down.clicked.connect(lambda: self._move_selected(+1))
        self.list_layers.currentRowChanged.connect(self._on_layer_selected)
        self.sld_layer_sens.valueChanged.connect(self._apply_layer_controls)
        self.sld_layer_low.valueChanged.connect(self._apply_layer_controls)
        self.sld_layer_high.valueChanged.connect(self._apply_layer_controls)

        self.layer_params = {}

    # ------- adapters -------
    def _feed_levels(self, vals: list[float]):
        # Forward to BarsWidget; also keep it “live”
        self.bars.update_levels(vals)
        self.bars.note_idle_tick()

    def _fill_devices(self, names: list[str]):
        self.cmb_device.blockSignals(True)
        self.cmb_device.clear()
        self.cmb_device.addItems(names)
        # keep the current selection if possible
        self.cmb_device.blockSignals(False)

    def _device_selected(self, name: str):
        # when a real device is picked, use 'plasma'; when default/none, use 'viridis'
        self.hub.set_device(name)
        if name.strip().lower() == "default output":
            self.bars.set_colormap("viridis")
        else:
            self.bars.set_colormap("plasma")

    def _on_device_changed(self, name: str):
        # tooltip for quick confirmation
        self.bars.setToolTip(f"Device: {name}")

    def _apply_global_eq(self):
        lo = min(self.sld_low.value(), self.sld_high.value())/100.0
        hi = max(self.sld_low.value(), self.sld_high.value())/100.0
        self.hub.set_eq_range(lo, hi)

    # ------- layers -------
    def _add_image_layer(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not fn: return
        pix = QPixmap(fn)
        if pix.isNull():
            QMessageBox.warning(self, "Error", "Failed to load image."); return
        item = LayerItem(pix, os.path.basename(fn))
        item.setPos(50, 50)
        self.scene.addItem(item)

        self.layer_params[item] = {"sens": 1.0, "low": 0.0, "high": 1.0}
        lw = QListWidgetItem(item.name)
        self.list_layers.addItem(lw)
        self.list_layers.setCurrentItem(lw)
        self.scene.clearSelection(); item.setSelected(True)

    def _cur_layer(self) -> LayerItem | None:
        r = self.list_layers.currentRow()
        if r < 0: return None
        name = self.list_layers.item(r).text()
        for it in sorted(self.scene.items(), key=lambda i: i.zValue(), reverse=True):
            if isinstance(it, LayerItem) and it.name == name:
                return it
        return None

    def _delete_selected_layer(self):
        it = self._cur_layer()
        if not it: return
        self.scene.removeItem(it)
        for i in range(self.list_layers.count()):
            if self.list_layers.item(i).text() == it.name:
                self.list_layers.takeItem(i); break
        self.layer_params.pop(it, None)

    def _move_selected(self, delta: int):
        it = self._cur_layer()
        if not it: return
        it.setZValue(it.zValue() + delta)
        items = [(jt.name, jt.zValue()) for jt in self.scene.items() if isinstance(jt, LayerItem)]
        items.sort(key=lambda t: t[1], reverse=True)
        self.list_layers.clear()
        for name, _ in items:
            self.list_layers.addItem(QListWidgetItem(name))
        for i in range(self.list_layers.count()):
            if self.list_layers.item(i).text() == it.name:
                self.list_layers.setCurrentRow(i); break

    def _on_layer_selected(self, row: int):
        it = self._cur_layer()
        if not it: return
        maxz = max([i.zValue() for i in self.scene.items() if isinstance(i, LayerItem)] + [0])
        it.setZValue(maxz + 1)
        p = self.layer_params.get(it, {"sens":1.0, "low":0.0, "high":1.0})
        self.sld_layer_sens.blockSignals(True)
        self.sld_layer_low.blockSignals(True)
        self.sld_layer_high.blockSignals(True)
        self.sld_layer_sens.setValue(int(p["sens"]*100))
        self.sld_layer_low.setValue(int(p["low"]*100))
        self.sld_layer_high.setValue(int(p["high"]*100))
        self.sld_layer_sens.blockSignals(False)
        self.sld_layer_low.blockSignals(False)
        self.sld_layer_high.blockSignals(False)

    def _apply_layer_controls(self):
        it = self._cur_layer()
        if not it: return
        p = self.layer_params.get(it)
        if not p: return
        p["sens"] = self.sld_layer_sens.value()/100.0
        lo = min(self.sld_layer_low.value(), self.sld_layer_high.value())/100.0
        hi = max(self.sld_layer_low.value(), self.sld_layer_high.value())/100.0
        p["low"], p["high"] = lo, hi

    # ------- pop-out -------
    def _pop_out_visualizer(self):
        if getattr(self, "_bars_popup", None) is not None:
            self._bars_popup.raise_(); self._bars_popup.activateWindow(); return
        self.bars.setParent(None)
        self._bars_popup = VisualizerWindow(self.hub, parent=None)
        self._bars_popup.setAttribute(Qt.WA_DeleteOnClose, True)
        self._bars_popup.set_external_bars(self.bars)

        def _restore():
            self.bars.setParent(None)
            self._top_row.insertWidget(0, self.bars, 1)
            self._bars_popup = None
        self._bars_popup.destroyed.connect(_restore)
        self._bars_popup.show()

class VisualizerWindow(QWidget):
    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visualizer")
        self.resize(700, 220)
        lay = QVBoxLayout(self)
        self.bars = BarsWidget(bar_count=48, smoothing=0.65)
        self.bars.set_mirrored(True)
        lay.addWidget(self.bars)
        hub.levelsUpdated.connect(self.bars.update_levels)

    def set_external_bars(self, bars_widget: BarsWidget):
        lay = self.layout()
        while lay.count():
            it = lay.takeAt(0); w = it.widget()
            if w: w.setParent(None)
        self.bars = bars_widget
        lay.addWidget(self.bars)
