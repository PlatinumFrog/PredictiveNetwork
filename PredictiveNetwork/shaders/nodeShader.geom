#version 460
layout(points) in;
layout(triangle_strip, max_vertices = 8) out;

uniform vec4 circle;

in vec2 bright[];

out vec2 texCoord;
out vec2 brightness;

void main() {
	vec2 b = clamp(0.5 - 0.5 * abs(bright[0]), 0.0, 0.5);
	brightness = vec2(bright[0].x, 1.0);

	vec4 v = gl_in[0].gl_Position;
	vec2 far = circle.w * v.xy;
	vec2 near = circle.z * v.xy;
	vec2 offset = (circle.w - circle.z) * v.zw;

	texCoord = vec2(1.0, -1.0);
	gl_Position = vec4(near, b.x, 1.0);
	EmitVertex();
	texCoord = vec2(1.0, 1.0);
	gl_Position = vec4(far, b.x, 1.0);
	EmitVertex();
	texCoord = vec2(-1.0, -1.0);
	gl_Position = vec4(near + offset, b.x, 1.0);
	EmitVertex();
	texCoord = vec2(-1.0, 1.0);
	gl_Position = vec4(far + offset, b.x, 1.0);
	EmitVertex();
	EndPrimitive();
	brightness = vec2(bright[0].y, -1.0);
	texCoord = vec2(1.0, -1.0);
	gl_Position = vec4(near - offset, b.y, 1.0);
	EmitVertex();
	texCoord = vec2(1.0, 1.0);
	gl_Position = vec4(far - offset, b.y, 1.0);
	EmitVertex();
	texCoord = vec2(-1.0, -1.0);
	gl_Position = vec4(near, b.y, 1.0);
	EmitVertex();
	texCoord = vec2(-1.0, 1.0);
	gl_Position = vec4(far, b.y, 1.0);
	EmitVertex();
	EndPrimitive();
}
