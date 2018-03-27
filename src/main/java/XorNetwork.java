/*
 * This Java source file was auto generated by running 'gradle buildInit --type java-library'
 * by 'colin_000' at '09/03/18 8:44 PM' with Gradle 2.14.1
 *
 * @author colin_000, @date 09/03/18 8:44 PM
 */

import Jama.Matrix;

public class XorNetwork {
	
	private double learningRate = 0.1;
	
	private class Node{
		// Weights is a column vector
		private Matrix weights;
		private double bias;
		private boolean isInput;
		
		public double[][] getWeights(){
			return weights.transpose().getArrayCopy();
		}
		
		public void adjustWeights(Matrix weightAdjustment){
			weights.plusEquals(weightAdjustment.times(learningRate));
		}
		
		public void adjustBias(double deltaBias){
			bias += learningRate * deltaBias;
		}
		
		public Node(int numWeights, boolean _isInput){
			
			isInput = _isInput;
			
			// If input node, no need to have bias or non-one weights
			if(isInput){
				weights = new Matrix(1, 1, 1);
				bias = 0;
			} else {
				// Set weights between -0.5 and 0.5
				weights = Matrix.random(numWeights, 1);
				weights.minusEquals(new Matrix(numWeights, 1, 0.5));
				
				bias = 0;
			}
		}
		
		public double takeInput(double inputs[]){
			if(isInput) {
				return inputs[0];
			}
			
			// The inputs become a row vector
			double inputsTemp[][] = new double[1][];
			inputsTemp[0] = inputs;
			
			Matrix inputsVector = new Matrix(inputsTemp);
			
			Matrix output = inputsVector.times(weights);
			
			return sigmoid(output.get(0, 0) + bias);
		}
		
		public double takeInputPrime(double inputs[]){
			if(isInput) {
				return inputs[0];
			}
			
			// The inputs become a row vector
			double inputsTemp[][] = new double[1][];
			inputsTemp[0] = inputs;
			
			Matrix inputsVector = new Matrix(inputsTemp);
			
			Matrix output = inputsVector.times(weights);
			
			return sigmoidPrime(output.get(0, 0) + bias);
		}
		
		private double sigmoid(double x){
			double ex = Math.exp(-x);
			return 1 / (1 + ex);
		}
		
		// First derivative of the sigmoid function
		private double sigmoidPrime(double x){
			double ex = Math.exp(-x);
			return ex / (1 + 2*ex + ex*ex);
		}
	}
	
	private Node[][] layersNodesWeightsBias;
	
	public XorNetwork(int numNodesInLayers[], double _learningRate) {
		
		learningRate = _learningRate;
		
		// Add correct number of layers
		layersNodesWeightsBias = new Node[numNodesInLayers.length][];
		
		// Add correct number of nodes, and then add weights and bias
		for(int i = 0; i < numNodesInLayers.length; ++i)
		{
			// Create correct number of nodes
			layersNodesWeightsBias[i] = new Node[numNodesInLayers[i]];
			
			// Initialize each node
			for(int k = 0; k < numNodesInLayers[i]; ++k)
			{
				if(i == 0){
					layersNodesWeightsBias[i][k] = new Node(1, true);
				} else {
					layersNodesWeightsBias[i][k] = new Node(numNodesInLayers[i - 1], false);
				}
			}
		}
    }
	
	// Get inputs to the network, return outputs of the network
	public double[] goThroughNetwork(double inputs[], boolean trainingMode, double idealOutputs[])
	{
		double outputs[][] = null;
		outputs = new double[layersNodesWeightsBias.length][];
		
		// Get the outputs of the first derivative of the Sigmoid functions
		double outputsPrime[][] = null;
		outputsPrime = new double[layersNodesWeightsBias.length][];
		
		//For each layer, multiply inputs by the weights + bias, then send outputs to next layer
		for(int layerNum = 0; layerNum < layersNodesWeightsBias.length; ++layerNum){
			Node[] layer = layersNodesWeightsBias[layerNum];
			
			outputs[layerNum] = new double[layer.length];
			outputsPrime[layerNum] = new double[layer.length];
			
			// For each node, multiple inputs by weights then add bias
			for(int nodeNum = 0; nodeNum < layer.length; ++nodeNum){
				
				// If this is the input node, then match each input with a node
				if(layerNum == 0){
					double singleInput[] = new double[1];
					singleInput[0] = inputs[nodeNum];
					
					outputs[layerNum][nodeNum] = layer[nodeNum].takeInput(singleInput);
				} else {
					outputs[layerNum][nodeNum] = layer[nodeNum].takeInput(outputs[layerNum - 1]);
					outputsPrime[layerNum][nodeNum] = layer[nodeNum].takeInputPrime(outputs[layerNum - 1]);
				}
			}
		}
		
		int outputLayerIndex = layersNodesWeightsBias.length - 1;
		
		if(trainingMode){
			doTraining(outputs, idealOutputs, outputsPrime, layersNodesWeightsBias);
		}
		
		return outputs[outputLayerIndex];
	}
	
	private void doTraining(double[][] outputs, double[] idealOutputs, double[][] outputsPrime, Node[][] layersNodesWeightsBias){
		
		int outputLayerIndex = layersNodesWeightsBias.length - 1;
		
		Matrix[] layerErrors = new Matrix[layersNodesWeightsBias.length];
		
		// Calculate error vector of the output layer
		double[] temp = gradientCostFunc(idealOutputs, outputs[outputLayerIndex]);
		double[][] gradientVector = new double[1][];
		gradientVector[0] = temp;
		Matrix gradientVectorMaxtrix = new Matrix(gradientVector);
		
		// Printing the difference between the ideal outputs and the actual outputs
		// gradientVectorMaxtrix.print( 2, 5);
		
		double[][] outputPrimeVector = new double[1][];
		outputPrimeVector[0] = outputsPrime[outputLayerIndex];
		Matrix outputVectorMatrix = new Matrix(outputPrimeVector);
		
		layerErrors[outputLayerIndex] = gradientVectorMaxtrix.arrayTimes(outputVectorMatrix);
		
		//Backpropogate the error
		for(int i = outputLayerIndex - 1; i > 0; --i){
			
			// Get the weights matrix, do w * err
			Matrix weightsMatrix = constructWeightsMatrix(layersNodesWeightsBias[i + 1]);
			Matrix applied = weightsMatrix.transpose().times(layerErrors[i + 1]);
			
			// Just get the outputs of the sigmoid primes from during the training
			double[][] layerOutputVector = new double[1][];
			layerOutputVector[0] = outputsPrime[i];
			Matrix layerOutputVectorMatrix = new Matrix(layerOutputVector).transpose();
			
			// Finally get the error for the layer
			layerErrors[i] = applied.arrayTimes(layerOutputVectorMatrix);
		}
		
		// Now use the error terms and the outputs to determine the how weights and bias should be changed
		for(int i = outputLayerIndex; i > 0; --i){
			
			Matrix layerError = layerErrors[i];
			
			// Iterate over the nodes in a given layer
			for(int k = 0; k < layersNodesWeightsBias[i].length; ++k){
				Node node = layersNodesWeightsBias[i][k];
				
				Matrix weightDeltas = new Matrix(outputs[i - 1].length, 1);
				
				// Iterate over the outputs of the previous layer
				for(int j = 0; j < outputs[i - 1].length; ++j){
					double weightDelta = outputs[i - 1][j] * layerError.get(k, 0);
					weightDeltas.set(j, 0, weightDelta);
				}
				
				node.adjustWeights(weightDeltas);
				node.adjustBias(layerError.get(k, 0));
			}
		}
	}
	
	private Matrix constructWeightsMatrix(Node layer[]){
		double[][] weights = new double[layer.length][];
		
		for(int i = 0; i < layer.length; ++i) {
			weights[i] = layer[i].getWeights()[0];
		}
		
		Matrix ret = new Matrix(weights);
		
		return ret;
	}
	
	// Calculate magnitude of the difference between two vectors
	private double magOfDifference(double vector1[], double vector2[]){
		
		double differenceVector[] = new double[vector1.length];
		
		// Get the difference vector
		for(int i = 0; i < vector1.length; ++i){
			differenceVector[i] = vector2[i] - vector1[i];
		}
		
		// Get the magnitude now
		double squareMagnitude = 0;
		for(int i = 0; i < vector1.length; ++i){
			squareMagnitude += differenceVector[i] * differenceVector[i];
		}
		
		return Math.sqrt(squareMagnitude);
	}
	
	private double costFunc(double idealOutputs[], double actualOutputs[]){
		
		double finalResult = 0;
		
		for(int i = 0; i < idealOutputs.length; ++i){
			double diff = idealOutputs[i] - actualOutputs[i];
			finalResult +=  diff * diff;
		}
		
		return finalResult/2;
	}
	
	// Calculate gradient of cost with respect to output layer
	private double[] gradientCostFunc(double idealOutputs[], double actualOutputs[]){
		
		double gradientVector[] = new double[idealOutputs.length];
		
		// Get the difference vector
		for(int i = 0; i < idealOutputs.length; ++i){
			gradientVector[i] = - ( actualOutputs[i] - idealOutputs[i] );
		}
		
		return gradientVector;
	}
	
	//Convenience function
	private double[] arrayCopy(double orig[]){
		double copy[] = new double[orig.length];
		
		for(int i = 0; i < orig.length; ++i){
			copy[i] = orig[i];
		}
		
		return copy;
	}
}
