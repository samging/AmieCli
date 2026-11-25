package org.example.project.components.terminal.window.controller.modelLoaders.aoti

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

/**
 * The models a bundle carries on the aoti runtime, by name: each one's compiled package and its labels, as the bundle's
 * `AMP/aoti.json` names them, relative to AMP. A bundle's runner installs them before its script runs, and `initMod` then
 * opens the package as it is, instead of finding it by hashing the ONNX file it was compiled from, which the bundle doesn't carry.
 */
object BundledAoti {
    const val FILE_NAME = "aoti.json"

    class Entry(val pt2: File, val labels: File?)

    @Serializable
    data class Manifest(val format: Int = FORMAT, val models: Map<String, Model>) {
        companion object {
            const val FORMAT = 1
        }
    }

    /** One model: its package and, when the bundle has one, its labels file (the config.json of the model), both under AMP */
    @Serializable
    data class Model(val pt2: String, val labels: String? = null)

    private val json = Json { prettyPrint = true; ignoreUnknownKeys = true; encodeDefaults = true }

    @Volatile
    private var entries: Map<String, Entry> = emptyMap()

    fun entry(name: String): Entry? = entries[name.trim()]

    /** Reads the `aoti.json` of the AMP folder [amp], if there is one; a folder without it installs nothing */
    fun install(amp: File) {
        val file = File(amp, FILE_NAME)
        entries = if (!file.isFile) emptyMap() else {
            json.decodeFromString(Manifest.serializer(), file.readText()).models.mapValues { (_, model) ->
                Entry(File(amp, model.pt2), model.labels?.let { File(amp, it) })
            }
        }
    }

    fun encode(manifest: Manifest): String = json.encodeToString(Manifest.serializer(), manifest)
}
