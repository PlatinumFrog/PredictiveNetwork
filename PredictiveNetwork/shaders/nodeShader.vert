#version 460
layout(location = 0) in vec2 vertices;
layout(location = 1) in vec2 brightness;
layout(location = 2) uniform vec4 circle;

out vec2 bright;

void main() {
	bright = brightness;
	
	gl_Position = vec4(circle.xy + vertices.xy * vec2(0.5625, 1.0), vertices.yx * vec2(0.5625, -1.0));
}
