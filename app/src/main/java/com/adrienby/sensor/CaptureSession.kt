package com.adrienby.sensor

import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

/**
 * One recording take. Everything is written into [dir] while recording:
 *   video.mp4        – written by CameraX
 *   imu.csv          – every IMU sample
 *   frames.csv       – one row per camera frame (sensor timestamp, exposure, focus, intrinsics)
 *   events.log.jsonl – raw interaction taps as they happen (append-only, survives crashes)
 * and on close:
 *   events.json      – final interaction tags (undo applied)
 *   meta.json        – settings, device, camera calibration, clock anchors, counts
 *
 * All timestamps are Android elapsedRealtimeNanos (same clock as SensorEvent.timestamp and,
 * when the camera's timestamp source is "realtime", the camera SENSOR_TIMESTAMP).
 */
class CaptureSession(val dir: File, val id: String) : ImuLogger.Sink {

    val meta = JSONObject()

    private val imu = BufferedWriter(FileWriter(File(dir, "imu.csv")), 1 shl 17)
    private val frames = BufferedWriter(FileWriter(File(dir, "frames.csv")), 1 shl 16)
    private val eventLog = BufferedWriter(FileWriter(File(dir, "events.log.jsonl")))

    private val events = ArrayList<JSONObject>()
    private val sb = StringBuilder(128)
    private var closed = false

    var imuCount = 0L
        private set
    var frameCount = 0L
        private set
    var firstFrameNs = -1L
        private set
    var lastFrameNs = -1L
        private set

    /** Set from the VideoRecordEvent.Start callback (elapsedRealtimeNanos). */
    @Volatile var videoStartNs = -1L

    init {
        imu.write("timestamp_ns,sensor,x,y,z,w\n")
        frames.write(
            "frame_number,timestamp_ns,exposure_ns,iso,focus_distance_diopters," +
                "focal_length_mm,fx,fy,cx,cy,skew\n"
        )
    }

    @Synchronized
    override fun writeImu(sensor: String, timestampNs: Long, v: FloatArray) {
        if (closed) return
        sb.setLength(0)
        sb.append(timestampNs).append(',').append(sensor)
        for (i in 0 until 4) {
            sb.append(',')
            if (i < v.size) sb.append(v[i])
        }
        sb.append('\n')
        imu.write(sb.toString())
        imuCount++
    }

    @Synchronized
    fun onFrame(result: TotalCaptureResult) {
        if (closed) return
        val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
        if (firstFrameNs < 0) firstFrameNs = ts
        lastFrameNs = ts
        sb.setLength(0)
        sb.append(result.frameNumber).append(',').append(ts).append(',')
        result.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let { sb.append(it) }
        sb.append(',')
        result.get(CaptureResult.SENSOR_SENSITIVITY)?.let { sb.append(it) }
        sb.append(',')
        result.get(CaptureResult.LENS_FOCUS_DISTANCE)?.let { sb.append(it) }
        sb.append(',')
        result.get(CaptureResult.LENS_FOCAL_LENGTH)?.let { sb.append(it) }
        val k = result.get(CaptureResult.LENS_INTRINSIC_CALIBRATION)
        for (i in 0 until 5) {
            sb.append(',')
            if (k != null && i < k.size) sb.append(k[i])
        }
        sb.append('\n')
        frames.write(sb.toString())
        frameCount++
    }

    /** Returns the stored tag so the UI can show it. */
    @Synchronized
    fun addEvent(obj: String, action: String, nowNs: Long): JSONObject {
        val e = JSONObject()
        e.put("t_ns", nowNs)
        if (videoStartNs > 0) e.put("t_video_s", (nowNs - videoStartNs) / 1e9)
        e.put("object", obj)
        e.put("action", action)
        if (!closed) {
            events.add(e)
            eventLog.write(e.toString())
            eventLog.write("\n")
            eventLog.flush()
        }
        return e
    }

    /** Removes the most recent tag. Returns the removed tag, or null if none. */
    @Synchronized
    fun undoLastEvent(nowNs: Long): JSONObject? {
        if (closed || events.isEmpty()) return null
        val removed = events.removeAt(events.size - 1)
        val u = JSONObject()
        u.put("t_ns", nowNs)
        u.put("undo_t_ns", removed.getLong("t_ns"))
        eventLog.write(u.toString())
        eventLog.write("\n")
        eventLog.flush()
        return removed
    }

    @Synchronized
    fun eventCount(): Int = events.size

    /** Flushes everything and writes events.json + meta.json. Safe to call once. */
    @Synchronized
    fun close() {
        if (closed) return
        closed = true
        imu.flush(); imu.close()
        frames.flush(); frames.close()
        eventLog.flush(); eventLog.close()

        val ev = JSONObject()
        ev.put("session", id)
        ev.put("clock", "elapsedRealtimeNanos")
        ev.put("video_start_ns", videoStartNs)
        ev.put("events", JSONArray(events))
        File(dir, "events.json").writeText(ev.toString(2))

        meta.put("imu_samples", imuCount)
        meta.put("frames_logged", frameCount)
        meta.put("first_frame_ns", firstFrameNs)
        meta.put("last_frame_ns", lastFrameNs)
        meta.put("video_start_ns", videoStartNs)
        meta.put("event_count", events.size)
        File(dir, "meta.json").writeText(meta.toString(2))
    }
}
