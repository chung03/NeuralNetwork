package com.neural;

import com.neural.NeuralNetwork.WEIGHT_INIT_FUNC;

import Jama.Matrix;

public class InputNode extends Node{
	public InputNode(){
		super(0, 0, 0, null);
	}
	
	public double takeInput(double inputs[]){
		return inputs[0];
	}
	
	public double takeInputPrime(double inputs[]){
		return inputs[0];
	}
	
	// Do nothing
	public void randomizeWeightsXavier(int numWeights, Matrix weights, WEIGHT_INIT_FUNC _weightInitFunc){}
}
