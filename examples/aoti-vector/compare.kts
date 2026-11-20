// ORT against AOTI on one numeric-vector model, through the same initMod calls and the same predict and answer.
// Run: AOTI_FFI_LIB=<libaoti_ffi> AOTI_PYTHON=<onnx2pt>/env/bin/python AOTI_COMPILER_SCRIPT=<onnx2pt>/onnx_aoti/onnx_aoti.py
//      bin/amie examples/aoti-vector/compare.kts
// VEC_MODEL names the ONNX file (default: vecnet.onnx beside this script), VEC_INPUT the numbers of one text.

val model = System.getenv("VEC_MODEL") ?: "examples/aoti-vector/vecnet.onnx"
val text = System.getenv("VEC_INPUT") ?: "0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4"
val calls = (System.getenv("VEC_CALLS") ?: "2000").toInt()
val rounds = 5

fun perCallUs(block: () -> Unit): Double {
    repeat(300) { block() }
    val times = (1..rounds).map {
        val t0 = System.nanoTime()
        repeat(calls) { block() }
        (System.nanoTime() - t0) / 1000.0 / calls
    }.sorted()
    return times[rounds / 2]
}

quietly {
    initMod(model, "ort").use { ort ->
        initMod(model, "aoti").use { aoti ->
            val ortAnswer = ort.answer(text)
            val aotiAnswer = aoti.answer(text)
            log("ort  predict: ${ort.predict(text)}   answer: $ortAnswer")
            log("aoti predict: ${aoti.predict(text)}   answer: $aotiAnswer")
            log("same answer: ${ortAnswer == aotiAnswer}")

            val ortPredict = perCallUs { ort.predict(text) }
            val aotiPredict = perCallUs { aoti.predict(text) }
            val ortAnswerUs = perCallUs { ort.answer(text) }
            val aotiAnswerUs = perCallUs { aoti.answer(text) }

            log("")
            log("median per call over $rounds rounds of $calls calls")
            log("predict   ort %.1f us   aoti %.1f us   aoti/ort %.2f".format(ortPredict, aotiPredict, aotiPredict / ortPredict))
            log("answer    ort %.1f us   aoti %.1f us   aoti/ort %.2f".format(ortAnswerUs, aotiAnswerUs, aotiAnswerUs / ortAnswerUs))
        }
    }
}
