#pragma once
#include <iostream>
#include <glad.h>
#include "shader.h"
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
//constexpr uint32_t nodeSize = 64u;


constexpr uint32_t activeSize = 128u;
constexpr uint32_t weightMatrixSize = activeSize * activeSize;
constexpr uint32_t weightTotalSize = 2u * weightMatrixSize;

constexpr float nodePositionMult = 2.0f / ((float)activeSize - 1.0f);
constexpr float nodeIndexMult = 1.0f / (float)activeSize;
constexpr float PI  = 3.14159265359f;
constexpr float TAU = 6.28318530718f;
constexpr float PD2 = 1.57079632679f;
constexpr float nodeRadiusMult = PD2 / (float)activeSize;

float activation(const float x) {
	return std::atan(x);
}

float activationd(const float x) {
	return 1.0f / ((x * x) + 1.0f);
}

struct Node { float value, error; };

class Network {
	GLuint
		nodePosVBO, nodeColVBO, nodeVAO, nodeShader,
		weightPosVBO, weightCol1VBO, weightCol2VBO, weightVAO, weightShader;

	float energy;

	// 64 Nodes
	Node act[activeSize];
	float weights[weightTotalSize];

public:
	Network():
		nodePosVBO(0u),
		nodeColVBO(0u),
		nodeVAO(0u),
		nodeShader(0u),
		weightPosVBO(0u),
		weightCol1VBO(0u),
		weightCol2VBO(0u),
		weightVAO(0u),
		weightShader(0u),
		energy(0.0f)
	{

		for (Node* n = act; n < act + activeSize; n++) *n = {0.0f, 0.0f};
		for (float* f = weights; f < weights + weightTotalSize; f++) *f = 0.0f;

		glGenBuffers(1u, &nodePosVBO);
		glGenBuffers(1u, &nodeColVBO);
		glGenVertexArrays(1u, &nodeVAO);
		nodeShader = Shader::create("nodeShader.vert", "nodeShader.geom", "nodeShader.frag");
		glBindVertexArray(nodeVAO);

		glBindBuffer(GL_ARRAY_BUFFER, nodePosVBO);
		float* actPos = new float[2u * activeSize];
		for (uint32_t i = 0u; i < activeSize; i++) {
			float j = (float)i * (TAU * nodeIndexMult);
			float x = std::cos(j);
			float y = std::sin(j);
			uint32_t id = 2u * i;
			actPos[id] = x;
			actPos[id + 1u] = y;
		}
		glBufferData(GL_ARRAY_BUFFER, 2ull * activeSize * sizeof(float), actPos, GL_STATIC_DRAW);
		delete[] actPos;
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, nodeColVBO);
		glBufferData(GL_ARRAY_BUFFER, 2ull * activeSize * sizeof(float), act, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, 0u);
		glBindVertexArray(0u);

		glGenBuffers(1u, &weightPosVBO);
		glGenBuffers(1u, &weightCol1VBO);
		glGenBuffers(1u, &weightCol2VBO);
		glGenVertexArrays(1u, &weightVAO);
		weightShader = Shader::create("weightShader.vert", "weightShader.geom", "weightShader.frag");

		glBindVertexArray(weightVAO);
		glBindBuffer(GL_ARRAY_BUFFER, weightPosVBO);
		float* weightPos = new float[4u * weightTotalSize];

		for (uint32_t i = 0u; i < activeSize; i++) {

			float j1 = (float)i * (TAU * nodeIndexMult);
			float x1 = std::cos(j1);
			float y1 = std::sin(j1);

			for (uint32_t j = 0u; j < activeSize; j++) {

				float j2 = (float)j * (TAU * nodeIndexMult);
				float x2 = std::cos(j2);
				float y2 = std::sin(j2);

				uint32_t id = (4u * ((i * activeSize) + j));

				weightPos[id] = x1;
				weightPos[id + 1u] = y1;
				weightPos[id + 2u] = x2;
				weightPos[id + 3u] = y2;
				id += 4u * weightMatrixSize;
				weightPos[id] = x2;
				weightPos[id + 1u] = y2;
				weightPos[id + 2u] = x1;
				weightPos[id + 3u] = y1;
			}
		}
		glBufferData(GL_ARRAY_BUFFER, 4ull * weightTotalSize * sizeof(float), weightPos, GL_STATIC_DRAW);
		delete[] weightPos;
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, weightCol1VBO);
		glBufferData(GL_ARRAY_BUFFER, weightTotalSize * sizeof(float), weights, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, weightCol2VBO);
		float* weightColor = new float[weightTotalSize];
		for (uint32_t i = 0u; i < weightMatrixSize; i++) weightColor[i] = 1.0f, weightColor[i + weightMatrixSize] = -1.0f;
		glBufferData(GL_ARRAY_BUFFER, weightTotalSize * sizeof(float), weightColor, GL_STATIC_DRAW);
		delete[] weightColor;
		glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);

		glBindBuffer(GL_ARRAY_BUFFER, 0u);
		glBindVertexArray(0u);

	}

	~Network() {
		
	}

	void train(
		float* input, 
		float* output, 
		uint32_t* indexesI, 
		uint32_t* indexesO, 
		uint32_t sizeI, 
		uint32_t sizeO
	) {
		for (uint32_t i = 0u; i < sizeI; i++) act[indexesI[i]].value = input[i];
		for (uint32_t i = 0u; i < sizeO; i++) act[indexesO[i]].value = output[i];

		float weightRate = 0.0f;
		for (uint32_t i = 0u; i < activeSize; i++) { float v = activation(act[i].value); weightRate += v * v; }
		weightRate = 1.0f / weightRate;

		for (uint32_t i = 0u; i < activeSize; i++) {
			uint32_t weightIDForwMin = (activeSize * i);
			uint32_t weightIDForwMax = weightIDForwMin + activeSize;
			uint32_t weightIDBackMin = weightIDForwMin + weightMatrixSize;
			uint32_t weightIDBackMax = weightIDBackMin + activeSize;
			Node n = act[i];
			float inter = activation(n.value);
			float interd = activationd(n.value);

			n.error = inter;
			for (uint32_t j = 0u, k = weightIDBackMin; j < activeSize && k < weightIDBackMax; j++, k++) {
				n.error -= weights[k] * activation(act[j].value);
			}
			float weightweight = weightRate * n.error;
			for (uint32_t j = 0u, k = weightIDBackMin; j < activeSize && k < weightIDBackMax; j++, k++) {
				weights[k] += weightweight * activation(act[j].value);
			}
			
			n.value -= n.error;
			for (uint32_t j = 0u, k = weightIDForwMin; j < activeSize && k < weightIDForwMax; j++, k++) {
				n.value += weights[k] * act[j].error;
			}
			n.value *= interd;
			inter *= weightRate;
			for (uint32_t j = 0u, k = weightIDForwMin; j < activeSize && k < weightIDForwMax; j++, k++) {
				weights[k] += act[j].error * inter;
			}
			act[i] = n;
		}
	}

	void run(
		float* input,
		uint32_t* indexesI,
		uint32_t sizeI
	) {
		for (uint32_t i = 0u; i < sizeI; i++) act[indexesI[i]].value = input[i];

		/*float weightRate = 0.0f;
		for (uint32_t i = 0u; i < activeSize; i++) { float v = softsign(act[i].value); weightRate += v * v; }
		weightRate = 1.0f / weightRate;*/

		for (uint32_t i = 0u; i < activeSize; i++) {
			uint32_t weightIDForwMin = (activeSize * i);
			uint32_t weightIDForwMax = weightIDForwMin + activeSize;
			uint32_t weightIDBackMin = weightIDForwMin + weightMatrixSize;
			uint32_t weightIDBackMax = weightIDBackMin + activeSize;
			Node n = act[i];
			float inter = activation(n.value);
			float interd = activationd(n.value);

			n.error = inter;
			for (uint32_t j = 0u, k = weightIDBackMin; j < activeSize && k < weightIDBackMax; j++, k++) {
				n.error -= weights[k] * activation(act[j].value);
			}
			/*float weightweight = weightRate * n.error;
			for (uint32_t j = 0u, k = weightIDBackMin; j < activeSize && k < weightIDBackMax; j++, k++) {
				weights[k] += weightweight * softsign(act[j].value);
			}*/

			n.value -= n.error;
			for (uint32_t j = 0u, k = weightIDForwMin; j < activeSize && k < weightIDForwMax; j++, k++) {
				n.value += weights[k] * act[j].error;
			}
			n.value *= interd;
			/*inter *= weightRate;
			for (uint32_t j = 0u, k = weightIDForwMin; j < activeSize && k < weightIDForwMax; j++, k++) {
				weights[k] += act[j].error * inter;
			}*/
			act[i] = n;
		}
	}

	void sleep() {

		float weightRate = 0.0f;
		for (uint32_t i = 0u; i < activeSize; i++) { float v = activation(act[i].value); weightRate += v * v; }
		weightRate = 1.0f / weightRate;

		for (uint32_t i = 0u; i < activeSize; i++) {
			uint32_t weightIDForwMin = (activeSize * i);
			uint32_t weightIDForwMax = weightIDForwMin + activeSize;
			uint32_t weightIDBackMin = weightIDForwMin + weightMatrixSize;
			uint32_t weightIDBackMax = weightIDBackMin + activeSize;
			Node n = act[i];
			float inter = activation(n.value);
			float interd = activationd(n.value);

			n.error = inter;
			for (uint32_t j = 0u, k = weightIDBackMin; j < activeSize && k < weightIDBackMax; j++, k++) {
				n.error -= weights[k] * activation(act[j].value);
			}
			float weightweight = weightRate * n.error;
			for (uint32_t j = 0u, k = weightIDBackMin; j < activeSize && k < weightIDBackMax; j++, k++) {
				weights[k] += weightweight * activation(act[j].value);
			}

			n.value -= n.error;
			for (uint32_t j = 0u, k = weightIDForwMin; j < activeSize && k < weightIDForwMax; j++, k++) {
				n.value += weights[k] * act[j].error;
			}
			n.value *= interd;
			inter *= weightRate;
			for (uint32_t j = 0u, k = weightIDForwMin; j < activeSize && k < weightIDForwMax; j++, k++) {
				weights[k] += act[j].error * inter;
			}
			act[i] = n;
		}
	}

	void print() {
		//for (uint32_t i = 0u; i < nodeSize; i++) std::cout << act[i].value << ", " << act[i + nodeSize].value << "\n";
		float sum1 = 0.0f;
		for (uint32_t i = 0u; i < activeSize; i++) sum1 += act[i].error * act[i].error;
		energy = sum1;
		float sum2 = 0.0f;
		for (uint32_t i = 0u; i < weightMatrixSize; i++) {
			float s = weights[i] - weights[i + weightMatrixSize];
			sum2 += s * s;
		}
		std::cout << "Energy: " << sum1 << ", Matrix Diversion: " << sum2 << '\n';
	}

	void draw() {

		static const float s = 1.0f / std::cos(nodeRadiusMult);
		static const float t = std::tan(nodeRadiusMult);
		static const float r1 = (s - t) / (s + t);
		static const float r2 = 1.0f;

		glBindBuffer(GL_ARRAY_BUFFER, nodeColVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0u, 2u * activeSize * sizeof(float), act);
		glBindBuffer(GL_ARRAY_BUFFER, 0u);

		glBindBuffer(GL_ARRAY_BUFFER, weightCol1VBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0u, weightTotalSize * sizeof(float), weights);
		glBindBuffer(GL_ARRAY_BUFFER, 0u);



		glUseProgram(weightShader);
		glBindVertexArray(weightVAO);
		glUniform4f(3, 0.0f, 0.0f, r1, r2);
		glDrawArrays(GL_POINTS, 0, weightTotalSize);

		glUseProgram(nodeShader);
		glBindVertexArray(nodeVAO);
		glUniform4f(2, 0.0f, 0.0f, r1, r2);
		glDrawArrays(GL_POINTS, 0, activeSize);
		
		glUseProgram(0u);
		glBindVertexArray(0u);
	}
};

