package com.adrienby.sensor

import android.content.ContentValues
import android.content.Context
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.IOException

/**
 * Copies a finished session from the app's private folder to the public
 * Download/SensorCapture/<take id>/ folder, where a PC sees it over USB.
 */
object Exporter {

    /** [subPath] is the take's folder name under Download/SensorCapture/. */
    fun export(context: Context, dir: File, subPath: String, progress: (Int) -> Unit): String {
        val files = dir.listFiles()?.filter { it.isFile }?.sortedBy { it.length() } ?: emptyList()
        val total = files.sumOf { it.length() }.coerceAtLeast(1L)
        val relPath = Environment.DIRECTORY_DOWNLOADS + "/SensorCapture/" + subPath
        val resolver = context.contentResolver
        var done = 0L
        var lastPct = -1
        val buf = ByteArray(1 shl 20)

        for (f in files) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, f.name)
                put(MediaStore.MediaColumns.MIME_TYPE, mimeFor(f.name))
                put(MediaStore.MediaColumns.RELATIVE_PATH, relPath)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: throw IOException("Could not create ${f.name} in Downloads")
            val out = resolver.openOutputStream(uri)
                ?: throw IOException("Could not open ${f.name} for writing")
            out.use { o ->
                f.inputStream().use { input ->
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        o.write(buf, 0, n)
                        done += n
                        val pct = ((done * 100) / total).toInt()
                        if (pct != lastPct) {
                            lastPct = pct
                            progress(pct)
                        }
                    }
                }
            }
            val finish = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
            resolver.update(uri, finish, null, null)
        }
        return relPath
    }

    private fun mimeFor(name: String): String = when {
        name.endsWith(".mp4") -> "video/mp4"
        name.endsWith(".csv") -> "text/csv"
        name.endsWith(".json") -> "application/json"
        // Generic type so Android does not append its own extension to .jsonl files.
        else -> "application/octet-stream"
    }
}
