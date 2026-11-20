# amie runs on a JDK 25. Sourced by a sh script, this finds one and exports JAVA_HOME to it: AMIE_JAVA_HOME, JAVA_HOME
# when it already is one, the Mac's java_home, Homebrew's openjdk@25, the Temurin install.sh unpacks under ~/.local/lib
# or /usr/local/lib, sdkman's, /usr/lib/jvm, one Gradle downloaded; else the java on the PATH when that is a 25. Gradle splices it into the built bin/amie; ./amie at the repository root
# sources it. Finding none, it says so and exits with 2.
amie_is25() { [ -n "$1" ] && [ -x "$1/bin/java" ] && "$1/bin/java" -version 2>&1 | head -1 | grep -q '"25[.\"]'; }
amie_jdk=""
for amie_candidate in "$AMIE_JAVA_HOME" "$JAVA_HOME" "$(/usr/libexec/java_home -v 25 2>/dev/null)" \
        /opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home /usr/local/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home \
        /home/linuxbrew/.linuxbrew/opt/openjdk@25 "$HOME"/.local/lib/jdk-25/Contents/Home "$HOME"/.local/lib/jdk-25 \
        /usr/local/lib/jdk-25/Contents/Home /usr/local/lib/jdk-25 "$HOME"/.sdkman/candidates/java/25* /usr/lib/jvm/*25* \
        "$HOME"/.gradle/jdks/*/Contents/Home "$HOME"/.gradle/jdks/*; do
    if amie_is25 "$amie_candidate"; then amie_jdk="$amie_candidate"; break; fi
done
if [ -n "$amie_jdk" ]; then
    JAVA_HOME="$amie_jdk"; export JAVA_HOME
elif java -version 2>&1 | head -1 | grep -q '"25[.\"]'; then
    unset JAVA_HOME
else
    echo "amie needs a JDK 25 and found none: install one (Temurin, Homebrew's openjdk@25, sdkman's 25-tem, ...) or set AMIE_JAVA_HOME to its folder." >&2
    exit 2
fi
