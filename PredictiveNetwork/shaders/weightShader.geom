#version 460

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec4 circle;

in vec2 send1[];
in vec4 tang[];

out vec2 send2;

void main() { 
    float b = clamp(1.0 - 0.5 * abs(send1[0].x), 0.5, 1.0);
    send2 = send1[0];

	vec4 near = circle.z * gl_in[0].gl_Position;
    vec4 mult = (circle.w - circle.z) * send1[0].y * tang[0];
	vec4 offset1 = vec4(0.45, 0.45, -0.45, -0.45) * mult + near;
    vec4 offset2 = vec4(0.55, 0.55, -0.55, -0.55) * mult + near;
    gl_Position = vec4(offset1.xy, b, 1.0);
    EmitVertex();
    gl_Position = vec4(offset1.zw, b, 1.0);
    EmitVertex();
    gl_Position = vec4(offset2.xy, b, 1.0);
    EmitVertex();
    gl_Position = vec4(offset2.zw, b, 1.0);
    EmitVertex();
    EndPrimitive();
    
}
