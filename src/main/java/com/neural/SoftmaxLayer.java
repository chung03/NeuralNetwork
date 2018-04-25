package com.neural;

import com.neural.NeuralNetwork.WEIGHT_INIT_FUNC;

import Jama.Matrix;

public class SoftmaxLayer extends Layer {
	public SoftmaxLayer(int numInputs, int numOutputs, double _momentumFactor, double _learningRate, WEIGHT_INIT_FUNC _weightInitFunc) {
		nodesList = new SigmoidNode[numOutputs];
		
		for(int i = 0; i < numOutputs; ++i){
			nodesList[i] = new SigmoidNode(numInputs, _momentumFactor, _learningRate, _weightInitFunc);
		}
	}
	
	@Override
	public Matrix takeInput(Matrix input){
		double inputSum = 0;
		Matrix output = new Matrix(input.getRowDimension(), 0, 0);
		
		for(int i = 0; i < input.getRowDimension(); ++i){
			double expX = Math.exp(input.get(i, 0));
			output.set(i, 0, expX);
			inputSum += expX;
		}
		
		return output.times(1/inputSum);
	}
	
	@Override
	public Matrix takeInputPrime(Matrix input){
		double inputSum = 0;
		
		for(int i = 0; i < input.getRowDimension(); ++i){
			inputSum += input.get(i, 0);
		}
		
		return input.copy();
	}
	
	public int GetLayerLength(){
		return nodesList.length;
	}
}
