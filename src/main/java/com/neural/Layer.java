package com.neural;

import Jama.Matrix;

public abstract class Layer {
	
	protected Node nodesList[];
	
	public Layer() {}
	
	public Matrix takeInput(Matrix input){
		Matrix ret = new Matrix(1, nodesList.length);
		
		for(int i = 0; i < nodesList.length; ++i){
			ret.set(0, i, nodesList[i].takeInput(input));
		}
		
		return ret;
	}
	
	
	public Matrix takeInputPrime(Matrix input){
		Matrix ret = new Matrix(1, nodesList.length);
		
		for(int i = 0; i < nodesList.length; ++i){
			ret.set(0, i, nodesList[i].takeInputPrime(input));
		}
		
		return ret;
	}
	
	public int GetLayerLength(){
		return nodesList.length;
	}
	
	public void adjustWeightsAndBias(Matrix outputs, Matrix layerError){
		// Iterate over the nodes in a given layer
		for(int k = 0; k < GetLayerLength(); ++k){
			Node node = nodesList[k];
			
			Matrix weightDeltas = new Matrix(outputs.getColumnDimension(), 1);
			
			// Iterate over the outputs of the previous layer
			for(int j = 0; j < outputs.getColumnDimension(); ++j){
				double weightDelta = outputs.get(0, j) * layerError.get(k, 0);
				weightDeltas.set(j, 0, weightDelta);
			}
			
			node.adjustWeights(weightDeltas);
			node.adjustBias(layerError.get(k, 0));
		}
	}
	
	public Matrix getWeightsMatrix(){
		double[][] weights = new double[nodesList.length][];
		
		for(int i = 0; i < nodesList.length; ++i) {
			weights[i] = nodesList[i].getWeights()[0];
		}
		
		Matrix ret = new Matrix(weights);
		
		return ret;
	}
}
