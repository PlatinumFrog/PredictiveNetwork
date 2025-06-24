#version 460 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D texture1;
void main() {
    float f = texture(texture1, TexCoord).r;
    float f1 = abs(f);
    FragColor = (f > 0.0 ? vec4(0.0, f1, 0.0, 1.0) : vec4(f1, 0.0, f1, 1.0));
}