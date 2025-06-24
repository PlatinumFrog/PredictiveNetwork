#version 460 core
layout (location = 0) in vec4 aPosTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPosTexCoord.xy, 0.0, 1.0);
    TexCoord = aPosTexCoord.zw;
}