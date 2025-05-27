#version 460

layout(points) in;
layout(triangle_strip, max_vertices = 8) out;

uniform vec4 circle;

in vec2 send1[];
in vec4 tang[];

out vec2 send2;

void main() {

    float thickness = 0.5;

    float b = clamp(1.0 - 0.5 * abs(send1[0].x), 0.5, 1.0);
    send2 = vec2(send1[0].x, 1.0);

    float rad = 0.5 * (circle.w - circle.z);
    float midrad = (circle.z + rad);

    vec4 v1 = midrad * gl_in[0].gl_Position;
    vec4 v2 = rad * tang[0];

    vec2 start = v1.xy + v2.xy;
    vec2 end = v1.zw - v2.zw;
    vec2 dir = rad * normalize(start - end);
    start -= dir;
    end += dir;
    vec2 nor = thickness * vec2(dir.y, -dir.x);
    nor *= vec2(0.5625, 1.0);
    dir *= vec2(0.5625, 1.0);

    gl_Position = vec4(start + nor, b, 1.0);
    EmitVertex();
    gl_Position = vec4(end + nor, b, 1.0);
    EmitVertex();
    gl_Position = vec4(start, b, 1.0);
    EmitVertex();
    gl_Position = vec4(end, b, 1.0);
    EmitVertex();
    EndPrimitive();

    b = clamp(1.0 - 0.5 * abs(send1[0].y), 0.5, 1.0);
    send2 = vec2(send1[0].y, -1.0);

    start = v1.xy - v2.xy;
    end = v1.zw + v2.zw;
    dir = rad * normalize(end - start);
    start += dir;
    end -= dir;
    nor = thickness * vec2(dir.y, -dir.x);
    nor *= vec2(0.5625, 1.0);
    dir *= vec2(0.5625, 1.0);

    gl_Position = vec4(end - nor, b, 1.0);
    EmitVertex();
    gl_Position = vec4(start - nor, b, 1.0);
    EmitVertex();
    gl_Position = vec4(end, b, 1.0);
    EmitVertex();
    gl_Position = vec4(start, b, 1.0);
    EmitVertex();
    EndPrimitive();
    
}
