package com.neural;

import com.neural.NeuralNetwork.WEIGHT_INIT_FUNC;

import Jama.Matrix;

public class TanHNode extends Node {
	
	public TanHNode(int numWeights, double _momentumFactor, double _learningRate, WEIGHT_INIT_FUNC _weightInitFunc){
		super(numWeights, _momentumFactor, _learningRate, _weightInitFunc);
	}
	
	@Override
	public double takeInput(Matrix inputs){
		return Math.tanh(sumWeightsAndBias(inputs));
	}
	
	@Override
	public double takeInputPrime(Matrix inputs){		
		double tanh = Math.tanh(sumWeightsAndBias(inputs));
		return 1 - tanh*tanh;
	}
	
	@Override
	public void randomizeWeightsXavier(int numWeights, Matrix weights, WEIGHT_INIT_FUNC _weightInitFunc){
		for(int i = 0; i < numWeights; ++i){
			double num = Math.random();
			
			// Get -2/d to 2/d
			num = (((num * 2) - 1)/Math.sqrt(numWeights)) * 2;
			
			weights.set(i, 0, num);
		}
	}
}
