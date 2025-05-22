#ifndef SHADER_COMPILERS
#define SHADER_COMPILERS

#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <glad.h>

class Shader {
	static inline GLuint internalCompileShader(const uint16_t type, const std::string &filename) {
#ifdef _DEBUG
		std::cout << ">> Compiling " << filename << "..." << std::endl;
#endif
		std::ifstream shaderFile("shaders/" + filename, std::ios::binary);
#if _DEBUG
		if (!shaderFile.is_open()) {
			std::cerr << "ERROR: Could not open file: " << filename << std::endl;
			abort();
		}
#endif
		std::string shaderString((std::istreambuf_iterator<char>(shaderFile)), (std::istreambuf_iterator<char>()));
		shaderFile.close();
		GLuint shaderShaderID = glCreateShader(type);
		const GLchar* sString = shaderString.c_str();
		glShaderSource(shaderShaderID, 1, &sString, NULL);
		glCompileShader(shaderShaderID);
#ifdef _DEBUG
		// DEBUG
		GLint success;
		GLchar log[512];
		glGetShaderiv(shaderShaderID, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(shaderShaderID, 512, NULL, log);
			std::string t;
			switch (type) {
			case GL_VERTEX_SHADER: t = "Vertex"; break;
			case GL_TESS_CONTROL_SHADER: t = "Tessellation Control"; break;
			case GL_TESS_EVALUATION_SHADER: t = "Tessellation Evaluation"; break;
			case GL_GEOMETRY_SHADER: t = "Geometry"; break;
			case GL_FRAGMENT_SHADER: t = "Fragment"; break;
			default: t = "Unknown";
			}
			std::cerr << "ERROR: " << t << " shader " + std::to_string(shaderShaderID) + " failed to compile!\n_________________________________________\n" + log + '\n';
		} else std::cout << "<< " << filename << " successfully compiled with ID: " << shaderShaderID << "\n" << std::endl;
#endif 
		return shaderShaderID;
	}

public:
	static GLuint create(
		const std::string &vertexFilename,
		const std::string &fragmentFilename
	) {
		const GLuint vertexShaderID = internalCompileShader(GL_VERTEX_SHADER, vertexFilename);
		const GLuint fragmentShaderID = internalCompileShader(GL_FRAGMENT_SHADER, fragmentFilename);
#if _DEBUG
		std::cout << ">> Linking shader program...\n";
#endif
		GLuint shaderID = glCreateProgram();
		glAttachShader(shaderID, vertexShaderID);
		glAttachShader(shaderID, fragmentShaderID);
		glLinkProgram(shaderID);
		glDeleteShader(vertexShaderID);
		glDeleteShader(fragmentShaderID);
#ifdef _DEBUG
		GLint success;
		GLchar log[512];
		glGetProgramiv(shaderID, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shaderID, 512, NULL, log);
			std::cerr << "ERROR: Shader " + std::to_string(shaderID) + " failed to link!\n_________________________________________\n" + log + '\n';
			abort();
		} else std::cout << "<< " << vertexFilename << " and " << fragmentFilename << " were successfully linked.\n\nShader Program Succesfully created with ID: " << shaderID << "\n\n";
#endif
		
		return shaderID;
	}
	static GLuint create(
		const std::string& vertexFilename,
		const std::string& geometryFilename,
		const std::string& fragmentFilename
	) {
		const GLuint vertexShaderID = internalCompileShader(GL_VERTEX_SHADER, vertexFilename);
		const GLuint geometryShaderID = internalCompileShader(GL_GEOMETRY_SHADER, geometryFilename);
		const GLuint fragmentShaderID = internalCompileShader(GL_FRAGMENT_SHADER, fragmentFilename);
#if _DEBUG
		std::cout << ">> Linking shader program...\n";
#endif
		GLuint shaderID = glCreateProgram();
		glAttachShader(shaderID, vertexShaderID);
		glAttachShader(shaderID, geometryShaderID);
		glAttachShader(shaderID, fragmentShaderID);
		glLinkProgram(shaderID);
		glDeleteShader(vertexShaderID);
		glDeleteShader(geometryShaderID);
		glDeleteShader(fragmentShaderID);
#ifdef _DEBUG
		GLint success;
		GLchar log[512];
		glGetProgramiv(shaderID, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shaderID, 512, NULL, log);
			std::cerr << "ERROR: Shader " + std::to_string(shaderID) + " failed to link!\n_________________________________________\n" + log + '\n';
			abort();
		} else std::cout << "<< " << vertexFilename << ", " << geometryFilename << ", and " << fragmentFilename << " were successfully linked.\n\nShader Program Succesfully created with ID: " << shaderID << "\n\n";
#endif
		return shaderID;
	}
	static GLuint create(
		const std::string& vertexFilename,
		const std::string& tessContFilename,
		const std::string& tessEvalFilename,
		const std::string& fragmentFilename
	) {
		const GLuint vertexShaderID = internalCompileShader(GL_VERTEX_SHADER, vertexFilename);
		const GLuint tessContShaderID = internalCompileShader(GL_TESS_CONTROL_SHADER, tessContFilename);
		const GLuint tessEvalShaderID = internalCompileShader(GL_TESS_EVALUATION_SHADER, tessEvalFilename);
		const GLuint fragmentShaderID = internalCompileShader(GL_FRAGMENT_SHADER, fragmentFilename);
#if _DEBUG
		std::cout << ">> Linking shader program...\n";
#endif
		GLuint shaderID = glCreateProgram();
		glAttachShader(shaderID, vertexShaderID);
		glAttachShader(shaderID, tessContShaderID);
		glAttachShader(shaderID, tessEvalShaderID);
		glAttachShader(shaderID, fragmentShaderID);
		glLinkProgram(shaderID);
		glDeleteShader(vertexShaderID);
		glDeleteShader(tessContShaderID);
		glDeleteShader(tessEvalShaderID);
		glDeleteShader(fragmentShaderID);
#ifdef _DEBUG
		GLint success;
		GLchar log[512];
		glGetProgramiv(shaderID, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shaderID, 512, NULL, log);
			std::cerr << "ERROR: Shader " + std::to_string(shaderID) + " failed to link!\n_________________________________________\n" + log + '\n';
			abort();
		} else std::cout << "<< " << vertexFilename << ", " << tessContFilename << ", " << tessEvalFilename << ", and " << fragmentFilename << " were successfully linked.\n\nShader Program Succesfully created with ID : " << shaderID << "\n\n";
#endif
		return shaderID;
	}
	static GLuint create(
		const std::string& vertexFilename,
		const std::string& tessContFilename,
		const std::string& tessEvalFilename,
		const std::string& geometryFilename,
		const std::string& fragmentFilename
	) {
		const GLuint vertexShaderID = internalCompileShader(GL_VERTEX_SHADER, vertexFilename);
		const GLuint tessContShaderID = internalCompileShader(GL_TESS_CONTROL_SHADER, tessContFilename);
		const GLuint tessEvalShaderID = internalCompileShader(GL_TESS_EVALUATION_SHADER, tessEvalFilename);
		const GLuint geometryShaderID = internalCompileShader(GL_GEOMETRY_SHADER, geometryFilename);
		const GLuint fragmentShaderID = internalCompileShader(GL_FRAGMENT_SHADER, fragmentFilename);
#if _DEBUG
		std::cout << ">> Linking shader program...\n";
#endif
		GLuint shaderID = glCreateProgram();
		glAttachShader(shaderID, vertexShaderID);
		glAttachShader(shaderID, tessContShaderID);
		glAttachShader(shaderID, tessEvalShaderID);
		glAttachShader(shaderID, geometryShaderID);
		glAttachShader(shaderID, fragmentShaderID);
		glLinkProgram(shaderID);
		glDeleteShader(vertexShaderID);
		glDeleteShader(tessContShaderID);
		glDeleteShader(tessEvalShaderID);
		glDeleteShader(geometryShaderID);
		glDeleteShader(fragmentShaderID);
#ifdef _DEBUG
		GLint success;
		GLchar log[512];
		glGetProgramiv(shaderID, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shaderID, 512, NULL, log);
			std::cerr << "ERROR: Shader " + std::to_string(shaderID) + " failed to link!\n_________________________________________\n" + log + '\n';
			abort();
		} else std::cout << "<< " << vertexFilename << ", " << tessContFilename << ", " << tessEvalFilename << ", " << geometryFilename << ", and " << fragmentFilename << " were successfully linked.\n\nShader Program Succesfully created with ID: " << shaderID << "\n\n";
#endif
		return shaderID;
	}
};

#endif