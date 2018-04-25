package com.neural;

import com.neural.NeuralNetwork.WEIGHT_INIT_FUNC;

public class SigmoidLayer extends Layer {
	public SigmoidLayer(int numInputs, int numOutputs, double _momentumFactor, double _learningRate, WEIGHT_INIT_FUNC _weightInitFunc) {
		nodesList = new SigmoidNode[numOutputs];
		
		for(int i = 0; i < numOutputs; ++i){
			nodesList[i] = new SigmoidNode(numInputs, _momentumFactor, _learningRate, _weightInitFunc);
		}
	}
	
	public int GetLayerLength(){
		return nodesList.length;
	}
}
