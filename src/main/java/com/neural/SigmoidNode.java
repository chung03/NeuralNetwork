package com.neural;

import com.neural.NeuralNetwork.WEIGHT_INIT_FUNC;

import Jama.Matrix;

public class SigmoidNode extends Node {

	public SigmoidNode(int numWeights, double _momentumFactor, double _learningRate, WEIGHT_INIT_FUNC _weightInitFunc){
		super(numWeights, _momentumFactor, _learningRate, _weightInitFunc);
	}
	
	public double takeInput(double inputs[]){
		double ex = Math.exp(-(sumWeightsAndBias(inputs)));
		return 1 / (1 + ex);
	}
	
	public double takeInputPrime(double inputs[]){
		double ex = Math.exp(-(sumWeightsAndBias(inputs)));
		return ex / (1 + 2*ex + ex*ex);
	}
	
	public void randomizeWeightsXavier(int numWeights, Matrix weights, WEIGHT_INIT_FUNC _weightInitFunc){
		for(int i = 0; i < numWeights; ++i){
			double num = Math.random();
			
			// Get -2/d to 2/d
			num = (((num * 2) - 1)/Math.sqrt(numWeights)) * 2;
			
			weights.set(i, 0, num);
		}
	}
}
