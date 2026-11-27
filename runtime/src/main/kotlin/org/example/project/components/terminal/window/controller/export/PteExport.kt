package org.example.project.components.terminal.window.controller.export

import java.io.File
import java.nio.file.Files

/**
 * The ExecuTorch `.pte` of an ExportedProgram, for a bundle exported with `--pte`. It runs pt2Topte's own `main.py`, as it is
 * in `~/GitHubRepository/pt2Topte` unless `AMIE_PTE_CONVERTER` names another, with that folder's venv. The converter reads
 * `model.pt2` from its working directory and writes `model.pte` there, so the program is copied to a scratch folder under that
 * name and the `.pte` is copied out of it.
 *
 * The converter then runs its own smoke test on the input shape of its helper's metadata, not on the model's, so it exits 1
 * for a model of another input size after the `.pte` is written. That exit is not a failed conversion: the `.pte` is kept
 * and the exit is returned for the caller to report. Only a conversion that wrote no `.pte` fails.
 */
internal object PteExport {
    class Result(val exit: Int, val log: String)

    fun converter(): File = System.getenv("AMIE_PTE_CONVERTER")?.takeIf { it.isNotBlank() }?.let(::File)
        ?: File(System.getProperty("user.home"), "GitHubRepository/pt2Topte")

    fun convert(program: File, into: File): Result {
        val dir = converter()
        val python = File(dir, "venv/bin/python")
        val main = File(dir, "main.py")
        check(python.isFile && main.isFile) { "no pt2Topte converter at ${dir.path} (its venv and main.py); set AMIE_PTE_CONVERTER to one" }
        val work = Files.createTempDirectory("pte").toFile()
        try {
            program.copyTo(File(work, "model.pt2"))
            val log = File(work, "converter.log")
            val process = ProcessBuilder(python.absolutePath, main.absolutePath)
                .directory(work)
                .redirectErrorStream(true)
                .redirectOutput(log)
                .start()
            val code = process.waitFor()
            val pte = File(work, "model.pte")
            val text = log.takeIf { it.isFile }?.readText().orEmpty()
            check(pte.isFile && pte.length() > 0) { "the pt2Topte conversion of ${program.name} wrote no .pte (exit $code):\n${text.lines().filter { it.isNotBlank() }.takeLast(8).joinToString("\n")}" }
            into.parentFile?.mkdirs()
            pte.copyTo(into, overwrite = true)
            return Result(code, text)
        } finally {
            work.deleteRecursively()
        }
    }
}
