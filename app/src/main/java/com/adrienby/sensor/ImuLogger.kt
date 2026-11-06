package com.adrienby.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicLongArray
import kotlin.math.sqrt

/**
 * Reads gyroscope, accelerometer, gravity and game-rotation-vector at the fastest rate the
 * phone allows, on its own thread. Every sample goes to [sink] (the active session, if any).
 * It also keeps a smoothed rotation speed for the "slow down" warning and per-sensor counters
 * for the live Hz display.
 */
class ImuLogger(context: Context) : SensorEventListener {

    interface Sink {
        fun writeImu(sensor: String, timestampNs: Long, v: FloatArray)
    }

    @Volatile var sink: Sink? = null

    /** Smoothed angular speed in rad/s. */
    @Volatile var rotationSpeed = 0.0
        private set

    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private var thread: HandlerThread? = null

    private val wanted = listOf(
        Sensor.TYPE_GYROSCOPE to "gyro",
        Sensor.TYPE_ACCELEROMETER to "accel",
        Sensor.TYPE_GRAVITY to "grav",
        Sensor.TYPE_GAME_ROTATION_VECTOR to "rot",
    )

    /** Sample counters in the same order as [wanted]. */
    val counts = AtomicLongArray(wanted.size)
    val names: List<String> = wanted.map { it.second }

    fun start() {
        if (thread != null) return
        val t = HandlerThread("imu").also { it.start() }
        thread = t
        val handler = Handler(t.looper)
        for ((type, _) in wanted) {
            sm.getDefaultSensor(type)?.let {
                sm.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, handler)
            }
        }
    }

    fun stop() {
        sm.unregisterListener(this)
        thread?.quitSafely()
        thread = null
    }

    override fun onSensorChanged(e: SensorEvent) {
        val idx = when (e.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> 0
            Sensor.TYPE_ACCELEROMETER -> 1
            Sensor.TYPE_GRAVITY -> 2
            Sensor.TYPE_GAME_ROTATION_VECTOR -> 3
            else -> return
        }
        counts.incrementAndGet(idx)
        if (idx == 0) {
            val v = e.values
            val mag = sqrt((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).toDouble())
            rotationSpeed = rotationSpeed * 0.97 + mag * 0.03
        }
        sink?.writeImu(names[idx], e.timestamp, e.values)
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}

    /** Sensor names, vendors and rates for meta.json. */
    fun describe(): JSONArray {
        val arr = JSONArray()
        for ((type, name) in wanted) {
            val s = sm.getDefaultSensor(type) ?: continue
            val o = JSONObject()
            o.put("id", name)
            o.put("name", s.name)
            o.put("vendor", s.vendor)
            o.put("min_delay_us", s.minDelay)
            o.put("resolution", s.resolution.toDouble())
            o.put("max_range", s.maximumRange.toDouble())
            arr.put(o)
        }
        return arr
    }
}
