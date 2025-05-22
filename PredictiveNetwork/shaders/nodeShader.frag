#version 460

in vec2 texCoord;
in vec2 brightness;

out vec4 ocolor;

void main() {
	float d = dot(texCoord, texCoord);
	ocolor = vec4(((brightness.y > 0.0) ? vec3(-brightness.x, brightness.x, -brightness.x): vec3(brightness.x, -brightness.x, -brightness.x)), 1.0);
	//ocolor = vec4(1.0);
	if(d > 1.0) discard;
}
