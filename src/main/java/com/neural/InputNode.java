package com.neural;

import com.neural.NeuralNetwork.WEIGHT_INIT_FUNC;

import Jama.Matrix;

public class InputNode extends Node{
	public InputNode(){
		super(0, 0, 0, null);
	}
	
	@Override
	public double takeInput(Matrix inputs){
		return inputs.get(0, 0);
	}
	
	@Override
	public double takeInputPrime(Matrix inputs){
		return inputs.get(0, 0);
	}
	
	@Override
	public void randomizeWeightsXavier(int numWeights, Matrix weights, WEIGHT_INIT_FUNC _weightInitFunc){
		// Do nothing
	}
}
