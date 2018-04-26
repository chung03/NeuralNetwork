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
	
	private double getMaxOfInput(Matrix input){
		//Get the max of the inputs for numerical stability
		double inputMax = input.get(0, 0);
		for(int i = 1; i < input.getColumnDimension(); ++i){
			if(inputMax < input.get(0, i)){
				inputMax = input.get(0, i);
			}
		}
		
		return inputMax;
	}
	
	@Override
	public Matrix takeInput(Matrix input){
		double inputSum = 0;
		Matrix output = new Matrix(1, input.getColumnDimension(), 0);
		
		double inputMax = getMaxOfInput(input);
		
		for(int i = 0; i < input.getColumnDimension(); ++i){
			// Subtract max of input to prevent NaNs
			double expX = Math.exp(input.get(0, i) - inputMax);
			output.set(0, i, expX);
			inputSum += expX;
		}
		
		return output.times(1/inputSum);
	}
	
	@Override
	public Matrix takeInputPrime(Matrix input){
		double inputSum = 0;
		Matrix output = new Matrix(1, input.getColumnDimension(), 0);
		
		double inputMax = getMaxOfInput(input);
		
		for(int i = 0; i < input.getColumnDimension(); ++i){
			// Subtract max of input to prevent NaNs
			double expX = Math.exp(input.get(0, i) - inputMax);
			output.set(0, i, expX);
			inputSum += expX;
		}
		
		return output.times(1/inputSum);
	}
	
	public int GetLayerLength(){
		return nodesList.length;
	}
}
