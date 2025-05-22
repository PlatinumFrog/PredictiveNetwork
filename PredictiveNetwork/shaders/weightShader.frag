#version 460

in vec2 send2;
out vec4 ocolor;

void main() {
    ocolor = vec4(((send2.y > 0.0) ? vec3(-send2.x, send2.x, -send2.x): vec3(send2.x, -send2.x, -send2.x)), 1.0);
}
