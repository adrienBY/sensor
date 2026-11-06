package com.adrienby.sensor

import android.app.Activity
import android.net.Uri
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.RecordingConfig
import com.google.ar.core.RecordingStatus
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import java.util.concurrent.atomic.AtomicLong
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * AR mode: ARCore owns the camera, tracks the phone's 6-DoF pose, and records the video with its
 * Recording API. Every new camera frame's pose goes to the active [CaptureSession] (poses.csv).
 *
 * The preview is drawn with a minimal OpenGL renderer: ARCore needs a GL texture to stream the
 * camera into before it will produce frames.
 */
class ArCapture(private val activity: Activity, private val glView: GLSurfaceView) :
    GLSurfaceView.Renderer {

    var session: Session? = null
        private set

    /** Session that receives poses; null when not recording. */
    @Volatile var sink: CaptureSession? = null

    @Volatile var tracking = false
        private set
    @Volatile var trackingText = "AR starting…"
        private set

    val frameCounter = AtomicLong(0)

    /** Camera configuration chosen for this session, for meta.json. */
    @Volatile var cameraInfo = JSONObject()
        private set

    private var installRequested = false
    private var running = false

    // GL state (GL thread only)
    private var textureId = -1
    private var program = 0
    private var posAttr = 0
    private var uvAttr = 0
    private var boundSession: Session? = null
    private var viewW = 0
    private var viewH = 0
    @Volatile private var geometryDirty = true
    private var lastTimestamp = 0L
    private var intrinsicsLogged = false

    private val quad: FloatBuffer = floatBuffer(
        floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
    )
    private val uv: FloatBuffer = floatBuffer(FloatArray(8))

    init {
        glView.preserveEGLContextOnPause = true
        glView.setEGLContextClientVersion(2)
        glView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        glView.setRenderer(this)
        glView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        glView.onPause()
    }

    /**
     * Creates (first time) and resumes the ARCore session.
     * Returns null on success, "INSTALLING" while Google Play Services for AR is being installed,
     * or a message explaining why AR is not available.
     */
    fun resume(): String? {
        if (running) return null
        if (session == null) {
            try {
                val status = ArCoreApk.getInstance().requestInstall(activity, !installRequested)
                if (status == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                    installRequested = true
                    return "INSTALLING"
                }
                val s = Session(activity)
                configure(s)
                session = s
            } catch (e: UnavailableDeviceNotCompatibleException) {
                return "This phone is not supported by ARCore"
            } catch (e: UnavailableUserDeclinedInstallationException) {
                return "Google Play Services for AR was not installed"
            } catch (e: UnavailableArcoreNotInstalledException) {
                return "Google Play Services for AR is not installed"
            } catch (e: UnavailableApkTooOldException) {
                return "Update Google Play Services for AR from the Play Store"
            } catch (e: UnavailableSdkTooOldException) {
                return "This app's ARCore version is too old"
            } catch (e: Exception) {
                return "ARCore failed to start: ${e.message}"
            }
        }
        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            return "Camera not available for AR"
        } catch (e: Exception) {
            return "ARCore failed to resume: ${e.message}"
        }
        running = true
        geometryDirty = true
        glView.onResume()
        return null
    }

    fun pause() {
        if (!running) return
        running = false
        glView.onPause()
        session?.pause()
        tracking = false
        trackingText = "AR paused"
    }

    fun close() {
        pause()
        session?.close()
        session = null
    }

    private fun configure(s: Session) {
        // Prefer the largest 30 fps back-camera stream: that is what gets recorded.
        val filter = CameraConfigFilter(s)
        filter.setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30))
        filter.setFacingDirection(CameraConfig.FacingDirection.BACK)
        val configs = s.getSupportedCameraConfigs(filter)
        val best = configs.maxByOrNull {
            it.textureSize.width.toLong() * it.textureSize.height.toLong()
        }
        if (best != null) s.setCameraConfig(best)

        val cfg = Config(s)
        cfg.setFocusMode(Config.FocusMode.AUTO)
        cfg.setUpdateMode(Config.UpdateMode.LATEST_CAMERA_IMAGE)
        cfg.setPlaneFindingMode(Config.PlaneFindingMode.DISABLED)
        cfg.setLightEstimationMode(Config.LightEstimationMode.DISABLED)
        cfg.setDepthMode(Config.DepthMode.DISABLED)
        s.configure(cfg)

        val info = JSONObject()
        val c = s.cameraConfig
        info.put("camera_id", c.cameraId)
        info.put("texture_size", JSONArray(listOf(c.textureSize.width, c.textureSize.height)))
        info.put("image_size", JSONArray(listOf(c.imageSize.width, c.imageSize.height)))
        info.put("fps_range", JSONArray(listOf(c.fpsRange.lower, c.fpsRange.upper)))
        info.put("configs_available", configs.size)
        info.put("focus", "auto")
        info.put("pose_convention",
            "Camera.getPose(): world-from-camera, OpenGL camera axes (+X right, +Y up, -Z forward), " +
                "world is gravity-aligned with +Y up, origin where the AR session started")
        cameraInfo = info
    }

    fun startRecording(target: CaptureSession, video: File): String? {
        val s = session ?: return "AR session is not running"
        try {
            s.startRecording(
                RecordingConfig(s)
                    .setMp4DatasetUri(Uri.fromFile(video))
                    .setAutoStopOnPause(true)
            )
        } catch (e: Exception) {
            return "Could not start AR recording: ${e.message}"
        }
        if (s.recordingStatus != RecordingStatus.OK) {
            return "AR recording status: ${s.recordingStatus}"
        }
        intrinsicsLogged = false
        sink = target
        return null
    }

    /** Stops the ARCore recording (the MP4 is finalized when this returns). */
    fun stopRecording(): String? {
        sink = null
        val s = session ?: return "AR session is not running"
        return try {
            s.stopRecording()
            null
        } catch (e: Exception) {
            "AR recording did not stop cleanly: ${e.message}"
        }
    }

    // ---- GLSurfaceView.Renderer ---------------------------------------------------------

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        textureId = tex[0]
        val target = GLES11Ext.GL_TEXTURE_EXTERNAL_OES
        GLES20.glBindTexture(target, textureId)
        GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        program = buildProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        posAttr = GLES20.glGetAttribLocation(program, "a_Position")
        uvAttr = GLES20.glGetAttribLocation(program, "a_TexCoord")
        boundSession = null
        geometryDirty = true
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewW = width
        viewH = height
        geometryDirty = true
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        if (!running) return
        if (boundSession !== s) {
            s.setCameraTextureName(textureId)
            boundSession = s
        }
        if (geometryDirty && viewW > 0 && viewH > 0) {
            s.setDisplayGeometry(displayRotation(), viewW, viewH)
            geometryDirty = false
        }

        val frame = try {
            s.update()
        } catch (e: Exception) {
            return
        }
        val ts = frame.timestamp
        if (ts == 0L) return

        if (frame.hasDisplayGeometryChanged()) {
            quad.position(0)
            uv.position(0)
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quad,
                Coordinates2d.TEXTURE_NORMALIZED, uv
            )
        }
        drawBackground()

        if (ts == lastTimestamp) return
        lastTimestamp = ts
        frameCounter.incrementAndGet()

        val cam = frame.camera
        val state = cam.trackingState
        tracking = state == TrackingState.TRACKING
        trackingText = if (tracking) "AR tracking OK"
        else "AR not tracking: ${cam.trackingFailureReason.name.lowercase().replace('_', ' ')}"

        val target = sink ?: return
        val p = cam.pose
        val k = cam.textureIntrinsics
        val f = k.focalLength
        val c = k.principalPoint
        target.writeArFrame(
            ts, state.name, cam.trackingFailureReason.name,
            floatArrayOf(p.tx(), p.ty(), p.tz(), p.qx(), p.qy(), p.qz(), p.qw()),
            floatArrayOf(f[0], f[1], c[0], c[1])
        )
        if (!intrinsicsLogged) {
            intrinsicsLogged = true
            val img = cam.imageIntrinsics
            val o = JSONObject()
            o.put("texture_fx_fy_cx_cy", JSONArray(listOf(f[0], f[1], c[0], c[1]).map { it.toDouble() }))
            o.put("texture_dims", JSONArray(k.imageDimensions.toList()))
            o.put("image_fx_fy_cx_cy", JSONArray(
                listOf(img.focalLength[0], img.focalLength[1],
                    img.principalPoint[0], img.principalPoint[1]).map { it.toDouble() }))
            o.put("image_dims", JSONArray(img.imageDimensions.toList()))
            target.putMeta("ar_intrinsics", o)
        }
    }

    private fun drawBackground() {
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        quad.position(0)
        uv.position(0)
        GLES20.glVertexAttribPointer(posAttr, 2, GLES20.GL_FLOAT, false, 0, quad)
        GLES20.glVertexAttribPointer(uvAttr, 2, GLES20.GL_FLOAT, false, 0, uv)
        GLES20.glEnableVertexAttribArray(posAttr)
        GLES20.glEnableVertexAttribArray(uvAttr)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(posAttr)
        GLES20.glDisableVertexAttribArray(uvAttr)
        GLES20.glDepthMask(true)
    }

    @Suppress("DEPRECATION")
    private fun displayRotation(): Int = activity.windowManager.defaultDisplay.rotation

    companion object {
        private const val VERTEX_SHADER = """
            attribute vec4 a_Position;
            attribute vec2 a_TexCoord;
            varying vec2 v_TexCoord;
            void main() {
                gl_Position = a_Position;
                v_TexCoord = a_TexCoord;
            }
        """

        private const val FRAGMENT_SHADER = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 v_TexCoord;
            uniform samplerExternalOES sTexture;
            void main() {
                gl_FragColor = texture2D(sTexture, v_TexCoord);
            }
        """

        private fun floatBuffer(values: FloatArray): FloatBuffer {
            val b = ByteBuffer.allocateDirect(values.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            b.put(values)
            b.position(0)
            return b
        }

        private fun compile(type: Int, src: String): Int {
            val shader = GLES20.glCreateShader(type)
            GLES20.glShaderSource(shader, src.trimIndent())
            GLES20.glCompileShader(shader)
            return shader
        }

        private fun buildProgram(vs: String, fs: String): Int {
            val p = GLES20.glCreateProgram()
            GLES20.glAttachShader(p, compile(GLES20.GL_VERTEX_SHADER, vs))
            GLES20.glAttachShader(p, compile(GLES20.GL_FRAGMENT_SHADER, fs))
            GLES20.glLinkProgram(p)
            return p
        }
    }
}
