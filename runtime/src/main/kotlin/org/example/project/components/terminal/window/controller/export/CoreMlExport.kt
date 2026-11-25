package org.example.project.components.terminal.window.controller.export

import java.io.File

/**
 * The CoreML model of a `.pt2`, for a bundle exported for macOS. It runs the conversion pipeline, seqPipeline.sh in the
 * AmieConversion folder unless `AMIE_COREML_PIPELINE` names another, with the source `.pt2` and the destination
 * `.mlpackage` (inside the bundle) as its two arguments. The pipeline prints the destination as its last line.
 */
internal object CoreMlExport {
    fun pipeline(): File = System.getenv("AMIE_COREML_PIPELINE")?.takeIf { it.isNotBlank() }?.let(::File)
        ?: File(System.getProperty("user.home"), "Documents/AmieConversion/seqPipeline.sh")

    fun convert(pt2: File, into: File) {
        val script = pipeline()
        check(script.isFile) { "no CoreML pipeline at ${script.path}; set AMIE_COREML_PIPELINE to one" }
        into.parentFile?.mkdirs()
        val process = ProcessBuilder("bash", script.path, pt2.absolutePath, into.absolutePath)
            .redirectOutput(ProcessBuilder.Redirect.DISCARD)
            .redirectError(ProcessBuilder.Redirect.INHERIT)
            .start()
        val code = process.waitFor()
        check(code == 0 && into.isDirectory) { "the CoreML pipeline failed for ${pt2.name} (exit $code)" }
    }
}
