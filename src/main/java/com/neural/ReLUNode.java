package com.neural;

import com.neural.NeuralNetwork.WEIGHT_INIT_FUNC;

import Jama.Matrix;

// LEaky version of ReLU
public class ReLUNode extends Node {
	
	public ReLUNode(int numWeights, double _momentumFactor, double _learningRate, WEIGHT_INIT_FUNC _weightInitFunc){
		super(numWeights, _momentumFactor, _learningRate, _weightInitFunc);
	}
	
	@Override
	public double takeInput(Matrix inputs){
		double x = sumWeightsAndBias(inputs);
		return (x <= 0) ? 0.01 * x : x;
	}
	
	@Override
	public double takeInputPrime(Matrix inputs){
		return (sumWeightsAndBias(inputs) <= 0) ? 0.01 : 1;
	}
	
	@Override
	public void randomizeWeightsXavier(int numWeights, Matrix weights, WEIGHT_INIT_FUNC _weightInitFunc){
		for(int i = 0; i < numWeights; ++i){
			double num = Math.random();
			
			// Get -2/d to 2/d
			num = num * 2/Math.sqrt(numWeights);
			
			weights.set(i, 0, num);
		}
	}
}
