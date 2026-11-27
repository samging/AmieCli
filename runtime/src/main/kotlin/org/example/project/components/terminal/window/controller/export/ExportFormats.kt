package org.example.project.components.terminal.window.controller.export

/**
 * The formats a bundle of aoti models can carry, and the warning each one gets. The pt2 package is always carried for a
 * model the script opens on aoti; mlpackage (CoreML) and pte (ExecuTorch) are made from the same program when asked for.
 */
object ExportFormats {
    const val PT2_WARNING = "warning: the pt2 package might not be stable: it is an AOTInductor build of the torch on this machine, and may not load on another torch version"
    const val MLPACKAGE_WARNING = "warning: the mlpackage might not be stable: coremltools converts the program, and its output or its support can change between versions"
    const val PTE_WARNING = "warning: the pte might not be stable: pt2Topte converts the program, and its own smoke test checks another input shape than the model's"

    /** The warnings of the formats a bundle carries, in the order the formats are listed */
    fun warnings(pt2: Boolean, mlpackage: Boolean, pte: Boolean): List<String> = buildList {
        if (pt2) add(PT2_WARNING)
        if (mlpackage) add(MLPACKAGE_WARNING)
        if (pte) add(PTE_WARNING)
    }
}
