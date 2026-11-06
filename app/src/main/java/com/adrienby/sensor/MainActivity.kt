package com.adrienby.sensor

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.StatFs
import android.os.SystemClock
import android.util.Range
import android.view.HapticFeedbackConstants
import android.view.View
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.GridLayout
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

class MainActivity : AppCompatActivity() {

    // ---- choices shown in the panel -------------------------------------------------------
    private val rooms = listOf(
        "Hallway", "Living room", "Kitchen", "Bedroom", "Bathroom", "Office", "Balcony", "Other"
    )
    private val lights = listOf(
        "Daylight", "Evening, lights on", "Night, lights on", "Night, lights off", "Mixed"
    )
    private val takeTypes = listOf(
        "Walkthrough", "Room sweep", "Transitions", "Interactions", "Free"
    )
    private val objects = listOf(
        "Door", "Light", "Fridge", "Tap", "Cupboard", "Drawer", "Window", "Curtain", "Other"
    )
    private val actions = listOf(
        "Open", "Close", "On", "Off", "Push", "Pull", "Pick up", "Put down", "Mark"
    )

    // ---- views ----------------------------------------------------------------------------
    private lateinit var previewView: PreviewView
    private lateinit var recText: TextView
    private lateinit var statsText: TextView
    private lateinit var motionText: TextView
    private lateinit var roomSpinner: Spinner
    private lateinit var lightSpinner: Spinner
    private lateinit var typeSpinner: Spinner
    private lateinit var lockFocus: CheckBox
    private lateinit var recordButton: Button
    private lateinit var savingText: TextView
    private lateinit var lastEventText: TextView
    private lateinit var undoButton: Button
    private lateinit var coverageText: TextView
    private lateinit var setupGroup: LinearLayout
    private val objectButtons = ArrayList<Button>()
    private val actionButtons = ArrayList<Button>()

    // ---- state ----------------------------------------------------------------------------
    private enum class State { IDLE, RECORDING, FINALIZING, SAVING }

    private var state = State.IDLE
    private var selectedObject = 0
    private var camera: Camera? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var cameraMeta = JSONObject()
    private var cameraSettings = JSONObject()
    private var recStartedAt = 0L

    @Volatile private var activeSession: CaptureSession? = null

    private lateinit var imu: ImuLogger
    private val io = Executors.newSingleThreadExecutor()
    private val ui = Handler(Looper.getMainLooper())

    private val frameCounter = AtomicLong(0)
    private var lastStatsAt = 0L
    private var lastFrameCount = 0L
    private var lastImuCounts = LongArray(4)
    private var statsLine = ""

    private val prefs by lazy { getSharedPreferences("coverage", Context.MODE_PRIVATE) }

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else recText.text = "Camera permission is needed"
        }

    /** Called by CameraX for every frame the camera produces. */
    private val frameCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            frameCounter.incrementAndGet()
            activeSession?.onFrame(result)
        }
    }

    // =======================================================================================
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        goImmersive()

        previewView = findViewById(R.id.preview)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        recText = findViewById(R.id.recText)
        statsText = findViewById(R.id.statsText)
        motionText = findViewById(R.id.motionText)
        roomSpinner = findViewById(R.id.roomSpinner)
        lightSpinner = findViewById(R.id.lightSpinner)
        typeSpinner = findViewById(R.id.typeSpinner)
        lockFocus = findViewById(R.id.lockFocus)
        recordButton = findViewById(R.id.recordButton)
        savingText = findViewById(R.id.savingText)
        lastEventText = findViewById(R.id.lastEventText)
        undoButton = findViewById(R.id.undoButton)
        coverageText = findViewById(R.id.coverageText)
        setupGroup = findViewById(R.id.setupGroup)

        // Keep the panels clear of the front-camera cutout.
        val root = findViewById<View>(R.id.root)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val i = insets.getInsets(WindowInsetsCompat.Type.displayCutout())
            v.setPadding(i.left, 0, i.right, 0)
            insets
        }

        // A stray back swipe must not end a take.
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (state == State.IDLE) {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                } else {
                    Toast.makeText(
                        this@MainActivity, "Stop the recording first", Toast.LENGTH_SHORT
                    ).show()
                }
            }
        })

        setupSpinner(roomSpinner, rooms, prefs.getInt("room", 0))
        setupSpinner(lightSpinner, lights, prefs.getInt("light", 0))
        setupSpinner(typeSpinner, takeTypes, prefs.getInt("type", 0))
        lockFocus.isChecked = prefs.getBoolean("lockFocus", false)

        buildGrid(findViewById(R.id.objectGrid), objects, objectButtons) { i, _ ->
            selectedObject = i
            refreshObjectButtons()
        }
        buildGrid(findViewById(R.id.actionGrid), actions, actionButtons) { i, v ->
            onTag(actions[i], v)
        }
        refreshObjectButtons()

        recordButton.setOnClickListener {
            when (state) {
                State.IDLE -> startRecording()
                State.RECORDING -> stopRecording()
                else -> {}
            }
        }
        undoButton.setOnClickListener { onUndo() }

        imu = ImuLogger(this)
        updateCoverage()
        applyState(State.IDLE)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    override fun onStart() {
        super.onStart()
        imu.start()
        ui.post(tick)
    }

    override fun onStop() {
        // Leaving the app ends the take cleanly instead of losing it.
        if (state == State.RECORDING) stopRecording()
        ui.removeCallbacks(tick)
        imu.stop()
        super.onStop()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) goImmersive()
    }

    // =======================================================================================
    // Camera

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({ bindCamera(future.get()) }, ContextCompat.getMainExecutor(this))
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun bindCamera(provider: ProcessCameraProvider) {
        val selector = CameraSelector.DEFAULT_BACK_CAMERA
        val info = selector.filter(provider.availableCameraInfos).firstOrNull()
        val c2 = info?.let { Camera2CameraInfo.from(it) }

        // Fixed 30 fps if the camera supports it: steady frame timing, capped exposure.
        val fpsRanges = c2?.getCameraCharacteristic(
            CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
        )
        val fixed30 = fpsRanges?.any { it.lower == 30 && it.upper == 30 } == true

        cameraSettings = JSONObject().apply {
            put("electronic_stabilization", "off (requested)")
            put("optical_stabilization", "device default")
            put("fps_range", if (fixed30) "30-30" else "device default")
            put("autofocus", "continuous video (unless locked at start)")
            put("exposure", "auto")
        }
        cameraMeta = if (c2 != null) describeCamera(c2) else JSONObject()

        // 16:9 preview so the framing on screen matches the 16:9 video exactly.
        val previewBuilder = Preview.Builder().setResolutionSelector(
            ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .build()
        )
        Camera2Interop.Extender(previewBuilder).apply {
            setCaptureRequestOption(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF
            )
            if (fixed30) setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(30, 30)
            )
        }
        val preview = previewBuilder.build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        val recorder = Recorder.Builder()
            .setQualitySelector(
                QualitySelector.from(
                    Quality.FHD, FallbackStrategy.lowerQualityOrHigherThan(Quality.HD)
                )
            )
            .setTargetVideoEncodingBitRate(20_000_000)
            .build()
        val videoBuilder = VideoCapture.Builder(recorder)
        Camera2Interop.Extender(videoBuilder).apply {
            setCaptureRequestOption(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF
            )
            if (fixed30) setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(30, 30)
            )
            setSessionCaptureCallback(frameCallback)
        }
        val vc = videoBuilder.build()

        try {
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, selector, preview, vc)
            videoCapture = vc
        } catch (e: Exception) {
            recText.text = "Camera error"
            Toast.makeText(this, "Camera failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun describeCamera(c2: Camera2CameraInfo): JSONObject {
        val o = JSONObject()
        o.put("camera_id", c2.cameraId)
        c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.let {
            o.put("focal_lengths_mm", JSONArray(it.map { f -> f.toDouble() }))
        }
        c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.let {
            o.put("sensor_size_mm", JSONArray(listOf(it.width.toDouble(), it.height.toDouble())))
        }
        c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)?.let {
            o.put("pixel_array", JSONArray(listOf(it.width, it.height)))
        }
        c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.let {
            o.put("active_array_ltrb", JSONArray(listOf(it.left, it.top, it.right, it.bottom)))
        }
        c2.getCameraCharacteristic(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.let {
            o.put("intrinsics_fx_fy_cx_cy_s", JSONArray(it.map { v -> v.toDouble() }))
        }
        c2.getCameraCharacteristic(CameraCharacteristics.LENS_DISTORTION)?.let {
            o.put("distortion_k1_k2_k3_p1_p2", JSONArray(it.map { v -> v.toDouble() }))
        }
        c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)?.let {
            o.put(
                "timestamp_source",
                if (it == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) "realtime"
                else "unknown"
            )
        }
        c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_ORIENTATION)?.let {
            o.put("sensor_orientation_deg", it)
        }
        c2.getCameraCharacteristic(CameraCharacteristics.LENS_POSE_ROTATION)?.let {
            o.put("lens_pose_rotation_xyzw", JSONArray(it.map { v -> v.toDouble() }))
        }
        c2.getCameraCharacteristic(CameraCharacteristics.LENS_POSE_TRANSLATION)?.let {
            o.put("lens_pose_translation_m", JSONArray(it.map { v -> v.toDouble() }))
        }
        return o
    }

    private fun lockFocusAtCenter() {
        val cam = camera ?: return
        val point = SurfaceOrientedMeteringPointFactory(1f, 1f).createPoint(0.5f, 0.5f)
        val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF)
            .disableAutoCancel()
            .build()
        cam.cameraControl.startFocusAndMetering(action)
    }

    // =======================================================================================
    // Recording

    private fun startRecording() {
        val vc = videoCapture ?: run {
            Toast.makeText(this, "Camera not ready yet", Toast.LENGTH_SHORT).show()
            return
        }
        val room = rooms[roomSpinner.selectedItemPosition]
        val light = lights[lightSpinner.selectedItemPosition]
        val take = takeTypes[typeSpinner.selectedItemPosition]
        prefs.edit()
            .putInt("room", roomSpinner.selectedItemPosition)
            .putInt("light", lightSpinner.selectedItemPosition)
            .putInt("type", typeSpinner.selectedItemPosition)
            .putBoolean("lockFocus", lockFocus.isChecked)
            .apply()

        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val id = "${stamp}_${slug(room)}_${slug(take)}"
        val dir = File(getExternalFilesDir(null), "sessions/$id")
        dir.mkdirs()

        val s = CaptureSession(dir, id)
        s.meta.apply {
            put("session", id)
            put("room", room)
            put("lighting", light)
            put("take_type", take)
            put("focus_locked", lockFocus.isChecked)
            put("app_version", appVersion())
            put("device", JSONObject().apply {
                put("manufacturer", Build.MANUFACTURER)
                put("model", Build.MODEL)
                put("android", Build.VERSION.RELEASE)
                put("sdk", Build.VERSION.SDK_INT)
            })
            put("camera", cameraMeta)
            put("camera_settings", cameraSettings)
            put("imu_sensors", imu.describe())
            put("clock", "elapsedRealtimeNanos")
            put("record_call_ns", SystemClock.elapsedRealtimeNanos())
            put("record_call_unix_ms", System.currentTimeMillis())
        }

        // Start logging before the video starts, so no frame or IMU sample is missed.
        activeSession = s
        imu.sink = s
        if (lockFocus.isChecked) lockFocusAtCenter()

        val options = FileOutputOptions.Builder(File(dir, "video.mp4")).build()
        recording = vc.output
            .prepareRecording(this, options)
            .start(ContextCompat.getMainExecutor(this)) { event -> onRecordEvent(event, s) }

        recStartedAt = SystemClock.elapsedRealtime()
        lastEventText.text = getString(R.string.no_tags)
        applyState(State.RECORDING)
    }

    private fun stopRecording() {
        recording?.stop()
        recording = null
        applyState(State.FINALIZING)
    }

    private fun onRecordEvent(event: VideoRecordEvent, s: CaptureSession) {
        when (event) {
            is VideoRecordEvent.Start -> {
                s.videoStartNs = SystemClock.elapsedRealtimeNanos()
                recStartedAt = SystemClock.elapsedRealtime()
            }
            is VideoRecordEvent.Finalize -> {
                val endNs = SystemClock.elapsedRealtimeNanos()
                // Stop feeding this take before closing its files.
                if (activeSession === s) activeSession = null
                if (imu.sink === s) imu.sink = null
                s.meta.put("video_finalize_ns", endNs)
                s.meta.put("video_duration_s",
                    event.recordingStats.recordedDurationNanos / 1e9)
                s.meta.put("video_bytes", event.recordingStats.numBytesRecorded)
                s.meta.put("video_error", if (event.hasError()) event.error else 0)
                event.cause?.let { s.meta.put("video_error_message", it.toString()) }
                camera?.cameraControl?.cancelFocusAndMetering()
                saveSession(s, event.recordingStats.recordedDurationNanos / 1e9)
            }
            else -> {}
        }
    }

    private fun saveSession(s: CaptureSession, durationS: Double) {
        applyState(State.SAVING)
        savingText.text = "Saving…"
        val room = s.meta.optString("room", "Other")
        io.execute {
            val result = try {
                s.close()
                val path = Exporter.export(this, s.dir, s.id) { pct ->
                    ui.post { savingText.text = "Saving… $pct%" }
                }
                s.dir.deleteRecursively()
                "Saved to $path"
            } catch (e: Exception) {
                "Save failed (${e.message}). Files kept in the app folder: ${s.dir.name}"
            }
            ui.post {
                if (result.startsWith("Saved")) addCoverage(room, durationS)
                savingText.text = result
                applyState(State.IDLE)
            }
        }
    }

    // =======================================================================================
    // Interaction tags

    private fun onTag(action: String, v: View) {
        val s = activeSession
        if (state != State.RECORDING || s == null) return
        v.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
        val obj = objects[selectedObject]
        val e = s.addEvent(slug(obj), slug(action), SystemClock.elapsedRealtimeNanos())
        val t = e.optDouble("t_video_s", Double.NaN)
        val time = if (t.isNaN()) "" else String.format(Locale.US, " @ %.1f s", t)
        lastEventText.text = "${s.eventCount()} tags · last: $obj → $action$time"
    }

    private fun onUndo() {
        val s = activeSession ?: return
        val removed = s.undoLastEvent(SystemClock.elapsedRealtimeNanos())
        lastEventText.text = if (removed == null) "Nothing to undo"
        else "${s.eventCount()} tags · removed ${removed.optString("object")} → ${removed.optString("action")}"
    }

    // =======================================================================================
    // UI helpers

    private val tick = object : Runnable {
        override fun run() {
            updateStatus()
            ui.postDelayed(this, 250)
        }
    }

    private fun updateStatus() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastStatsAt >= 1000) {
            val dt = (now - lastStatsAt).coerceAtLeast(1) / 1000.0
            val frames = frameCounter.get()
            val fps = (frames - lastFrameCount) / dt
            lastFrameCount = frames
            val hz = IntArray(4)
            for (i in 0 until 4) {
                val c = imu.counts.get(i)
                hz[i] = ((c - lastImuCounts[i]) / dt).toInt()
                lastImuCounts[i] = c
            }
            lastStatsAt = now
            val freeGb = try {
                StatFs(getExternalFilesDir(null)!!.path).availableBytes / 1e9
            } catch (e: Exception) { -1.0 }
            statsLine = String.format(
                Locale.US,
                "camera %.0f fps · gyro %d Hz · accel %d Hz · rot %d Hz\nfree %.1f GB",
                fps, hz[0], hz[1], hz[3], freeGb
            )
        }
        val s = activeSession
        statsText.text = if (s != null && state == State.RECORDING)
            "$statsLine · frames ${s.frameCount} · imu ${s.imuCount}" else statsLine

        if (state == State.RECORDING) {
            val sec = (now - recStartedAt) / 1000
            recText.text = String.format(Locale.US, "● REC %02d:%02d", sec / 60, sec % 60)
            recText.setTextColor(ContextCompat.getColor(this, R.color.rec))
        }

        val speed = imu.rotationSpeed
        when {
            speed > 1.4 -> {
                motionText.text = "Too fast, slow down"
                motionText.setTextColor(ContextCompat.getColor(this, R.color.rec))
            }
            speed > 0.8 -> {
                motionText.text = "A bit fast"
                motionText.setTextColor(ContextCompat.getColor(this, R.color.warn))
            }
            else -> {
                motionText.text = "Motion OK"
                motionText.setTextColor(ContextCompat.getColor(this, R.color.ok))
            }
        }
    }

    private fun applyState(newState: State) {
        state = newState
        val idle = newState == State.IDLE
        val rec = newState == State.RECORDING
        setEnabledDeep(setupGroup, idle)
        recordButton.isEnabled = idle || rec
        recordButton.text = getString(if (rec) R.string.stop else R.string.record)
        recordButton.setBackgroundResource(if (rec) R.drawable.rec_active else R.drawable.rec_idle)
        recordButton.alpha = if (idle || rec) 1f else 0.5f
        for (b in actionButtons) {
            b.isEnabled = rec
            b.alpha = if (rec) 1f else 0.4f
        }
        undoButton.isEnabled = rec
        undoButton.alpha = if (rec) 1f else 0.4f
        when (newState) {
            State.IDLE -> {
                recText.text = getString(R.string.ready)
                recText.setTextColor(ContextCompat.getColor(this, R.color.text))
            }
            State.FINALIZING -> recText.text = "Finishing video…"
            State.SAVING -> recText.text = "Saving…"
            State.RECORDING -> savingText.text = ""
        }
    }

    private fun setEnabledDeep(v: View, enabled: Boolean) {
        v.isEnabled = enabled
        v.alpha = if (enabled) 1f else 0.5f
        if (v is LinearLayout) for (i in 0 until v.childCount) v.getChildAt(i).isEnabled = enabled
    }

    private fun setupSpinner(sp: Spinner, items: List<String>, selected: Int) {
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, items)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        sp.adapter = adapter
        sp.setSelection(selected.coerceIn(0, items.size - 1))
    }

    private fun buildGrid(
        grid: GridLayout,
        labels: List<String>,
        out: MutableList<Button>,
        onClick: (Int, View) -> Unit
    ) {
        val m = dp(3)
        labels.forEachIndexed { i, label ->
            val b = Button(this)
            b.text = label
            b.isAllCaps = false
            b.textSize = 13f
            b.setTextColor(ContextCompat.getColor(this, R.color.text))
            b.setBackgroundResource(R.drawable.btn_normal)
            b.minHeight = 0
            b.minimumHeight = 0
            b.setPadding(dp(2), 0, dp(2), 0)
            val lp = GridLayout.LayoutParams(
                GridLayout.spec(GridLayout.UNDEFINED, 1f),
                GridLayout.spec(GridLayout.UNDEFINED, 1f)
            )
            lp.width = 0
            lp.height = dp(42)
            lp.setMargins(m, m, m, m)
            b.layoutParams = lp
            b.setOnClickListener { v -> onClick(i, v) }
            grid.addView(b)
            out.add(b)
        }
    }

    private fun refreshObjectButtons() {
        objectButtons.forEachIndexed { i, b ->
            b.setBackgroundResource(
                if (i == selectedObject) R.drawable.btn_selected else R.drawable.btn_normal
            )
        }
    }

    private fun addCoverage(room: String, seconds: Double) {
        val key = "sec_" + slug(room)
        val total = prefs.getFloat(key, 0f) + seconds.toFloat()
        prefs.edit().putFloat(key, total).apply()
        updateCoverage()
    }

    private fun updateCoverage() {
        val sb = StringBuilder()
        var all = 0f
        for (r in rooms) {
            val sec = prefs.getFloat("sec_" + slug(r), 0f)
            all += sec
            if (sec > 0f) sb.append(String.format(Locale.US, "%-12s %5.1f min\n", r, sec / 60f))
        }
        sb.append(String.format(Locale.US, "%-12s %5.1f min", "Total", all / 60f))
        coverageText.text = sb.toString()
    }

    private fun goImmersive() {
        WindowCompat.getInsetsController(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    private fun appVersion(): String = try {
        packageManager.getPackageInfo(packageName, 0).versionName ?: "?"
    } catch (e: Exception) { "?" }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private fun slug(s: String): String =
        s.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "-").trim('-')
}
