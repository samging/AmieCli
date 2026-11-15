An interpreter for Kotlin scripts that train, ask, group and serve neural network models on your own machine (headless). \
open models in ONNX, TorchScript and TensorFlow. \
You can import models them from Hugging Face. \
Add your own functionality (backend service) amie can use jars (they're defined with certain interface [jars](www.github.com/samging/jar-jar). 

```bash
>>> amie package --help # lists the public github jar-jars (runs on workflow)
```

#### Build 

Runs on JDK 21.
`install.sh` searches variables (`AMIE_JAVA_HOME`, `JAVA_HOME`, the
Mac's `java_home`, Homebrew, sdkman, `/usr/lib/jvm`).

```bash
./gradlew :amie:installDist      # the distribution under amie/build/install/amie
amie/install.sh                  # builds and installs it: ~/.local/lib/amie, ~/.local/bin/amie
bin/amie --version               # from the checkout: builds when the sources changed, then runs
```

`amie/install.sh --prefix /usr/local`, `--no-build` and `--uninstall` are the other forms.
```bash
amie # after successful installation should prompt something alike
```
<img width="1756" height="662" alt="gitImg" src="https://github.com/user-attachments/assets/eba3d2c0-9d15-42f0-9058-6af7adf9d311" />

#### If implementing pipelining: 
The intent models need the encoder models: `scripts/setup.sh` downloads them, and `AMIE_MODELS_DIR` points
at another copy.

#### Documentation

```bash
amie :docs # OR:
```
[amie/README.md](amie/README.md) guide: every command line, the prompt, the scripting functions, the
exit codes. `amie docs` opens the learn book, chapter by chapter from the first line to a model served.



#### Examples

Whole projects in [examples/](examples), each with a README and a `setup.sh` that adds it to the AMP folder.

The commands: `run`, `new`, `example`, `model`, `group`, `package`, `env`, `template`, `project`, `session`, `hf`
(Hugging Face) and `docs`, the learn book. Everything amie reads or writes is in the AMP folder: `--amp`, else
`AMIE_AMP_DIR`, else the user's data folder (`~/Library/Application Support/amie pilot/AMP` on a Mac).


Under [gnu-license](LICENSE).
