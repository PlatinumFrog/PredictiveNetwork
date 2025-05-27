#version 460

layout(points) in;
layout(triangle_strip, max_vertices = 8) out;

uniform vec4 circle;

in vec2 send1[];
in vec4 tang[];

out vec2 send2;

void main() {
    float b = clamp(1.0 - 0.5 * abs(send1[0].x), 0.5, 1.0);
    send2 = vec2(send1[0].x, 1.0);

    float thickness = 0.5;

    float rad = 0.5 * (circle.w - circle.z);
    float midrad = (circle.z + rad);
    vec2 start = midrad * gl_in[0].gl_Position.xy + rad * tang[0].xy;
    vec2 end = midrad * gl_in[0].gl_Position.zw - rad * tang[0].zw;
    vec2 dir = rad * normalize(start - end);
    vec2 nor = thickness * vec2(dir.y, -dir.x);
    nor *= vec2(0.5625, 1.0);
    dir *= vec2(0.5625, 1.0);


    gl_Position = vec4(start - dir + nor, b, 1.0);
    EmitVertex();
    gl_Position = vec4(end + dir + nor, b, 1.0);
    EmitVertex();
    gl_Position = vec4(start - dir, b, 1.0);
    EmitVertex();
    gl_Position = vec4(end + dir, b, 1.0);
    EmitVertex();
    EndPrimitive();

    b = clamp(1.0 - 0.5 * abs(send1[0].y), 0.5, 1.0);
    send2 = vec2(send1[0].y, -1.0);

    start = midrad * gl_in[0].gl_Position.xy - rad * tang[0].xy;
    end = midrad * gl_in[0].gl_Position.zw + rad * tang[0].zw;
    dir = rad * normalize(end - start);
    nor = thickness * vec2(dir.y, -dir.x);
    nor *= vec2(0.5625, 1.0);
    dir *= vec2(0.5625, 1.0);

    gl_Position = vec4(end - dir - nor, b, 1.0);
    EmitVertex();
    gl_Position = vec4(start + dir - nor, b, 1.0);
    EmitVertex();
    gl_Position = vec4(end - dir, b, 1.0);
    EmitVertex();
    gl_Position = vec4(start + dir, b, 1.0);
    EmitVertex();
    EndPrimitive();
    
}
