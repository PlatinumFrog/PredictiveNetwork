#version 460

layout(location = 0) in vec4 line;
layout(location = 1) in float weightVal;
layout(location = 2) in float weightErr;
layout(location = 3) uniform vec4 circle;

out vec2 send1;
out vec4 tang;

void main() {
    send1 = vec2(weightVal, weightErr);
    tang = line.yxwz * vec2(0.5625, -1.0).xyxy;
    gl_Position = circle.xyxy + line * vec2(0.5625, 1.0).xyxy;
}
