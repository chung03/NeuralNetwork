package com.neural;

import com.neural.NeuralNetwork.WEIGHT_INIT_FUNC;

import Jama.Matrix;

public class InputLayer extends Layer {
	public InputLayer(int numInputs, int numOutputs, double _momentumFactor, double _learningRate, WEIGHT_INIT_FUNC _weightInitFunc) {
		nodesList = new InputNode[numOutputs];
		
		for(int i = 0; i < numOutputs; ++i){
			nodesList[i] = new InputNode();
		}
	}
	
	@Override
	public Matrix takeInput(Matrix input){
		return input.copy();
	}
	
	@Override
	public Matrix takeInputPrime(Matrix input){
		return input.copy();
	}
	
	public int GetLayerLength(){
		return nodesList.length;
	}
}
