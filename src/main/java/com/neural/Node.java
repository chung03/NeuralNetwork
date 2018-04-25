package com.neural;

import com.neural.NeuralNetwork.WEIGHT_INIT_FUNC;

import Jama.Matrix;

public abstract class Node {
	// Weights is a column vector
	protected Matrix weights;
	protected Matrix currentWeightMomentum;
	
	protected double bias;
	protected double currentBiasMomentum;
	
	protected double momentumFactor;
	protected double learningRate;
	protected WEIGHT_INIT_FUNC weightInitFunc;
	
	public double[][] getWeights(){
		return weights.transpose().getArrayCopy();
	}
	
	public void adjustWeights(Matrix weightAdjustment){
		currentWeightMomentum = currentWeightMomentum.times(momentumFactor).plus(weightAdjustment);
		weights.plusEquals(currentWeightMomentum.times(learningRate));
	}
	
	public void adjustBias(double deltaBias){
		currentBiasMomentum = currentBiasMomentum * momentumFactor + deltaBias;
		bias += learningRate * currentBiasMomentum;
	}
	
	public Node(int numWeights, double _momentumFactor, double _learningRate, WEIGHT_INIT_FUNC _weightInitFunc){
		momentumFactor = _momentumFactor;
		learningRate = _learningRate;
		weightInitFunc = _weightInitFunc;
		
		// If input node, no need to have bias or non-one weights
		if(numWeights > 0){
			if(weightInitFunc == WEIGHT_INIT_FUNC.XAVIER_MODIFIED){
				weights = new Matrix(numWeights, 1);
				randomizeWeightsXavier(numWeights, weights, _weightInitFunc);
			} else {
				// Set weights between -0.5 and 0.5
				weights = Matrix.random(numWeights, 1);
				weights.minusEquals(new Matrix(numWeights, 1, 0.5));
			}
			
			bias = 0;
			currentWeightMomentum = new Matrix(numWeights, 1, 0);
		}
	}
	
	protected double sumWeightsAndBias(Matrix inputs){
		Matrix output = inputs.times(weights);
		return output.get(0, 0) + bias;
	}
	
	public abstract void randomizeWeightsXavier(int numWeights, Matrix weights, WEIGHT_INIT_FUNC _weightInitFunc);
	
	public abstract double takeInput(Matrix inputs);
	
	public abstract double takeInputPrime(Matrix inputs);
}
