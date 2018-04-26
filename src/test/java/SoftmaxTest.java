import org.junit.Test;

import com.neural.NeuralNetwork;

import junit.framework.AssertionFailedError;

import static org.junit.Assert.*;

/*
 * This Java source file was auto generated by running 'gradle init --type java-library'
 * by 'colin_000' at '09/03/18 8:44 PM' with Gradle 2.14.1
 *
 * @author colin_000, @date 09/03/18 8:44 PM
 */
public class SoftmaxTest {
	
	public void doTest(int numLayers[], 
								double inputs[][],
								double idealOutputs[][],
								NeuralNetwork.ACTIVATION_FUNC[] activationFuncs, 
								NeuralNetwork.WEIGHT_INIT_FUNC weightFunc,
								int numIterations){        
        for(int numAttempts = 0; numAttempts < 5; ++numAttempts){
        	NeuralNetwork network = new NeuralNetwork(numLayers, 0.1, 0.5, activationFuncs, weightFunc);
	        doTestInner(network, inputs, idealOutputs, numIterations);
	        
	        if(CheckNetworkPassed(network, inputs, idealOutputs)){
	        	return;
	        } else {
	        	System.out.println("Network did not converge on correct solution. Assuming it fell into a local minimum instead of global minimum");
	        }
		}
		
		fail("Neural network did not converge on the correct solution after 5 attempts. Test failed");
	}
	
	public void doTest(int numLayers[], 
								double inputs[][],
								double idealOutputs[][],
								NeuralNetwork.ACTIVATION_FUNC activationFunc, 
								NeuralNetwork.WEIGHT_INIT_FUNC weightFunc, 
								int numIterations){
		
		for(int numAttempts = 0; numAttempts < 5; ++numAttempts){
	        NeuralNetwork network = new NeuralNetwork(numLayers, 0.1, 0.5, activationFunc, weightFunc);
	        doTestInner(network, inputs, idealOutputs, numIterations);
	        
	        if(CheckNetworkPassed(network, inputs, idealOutputs)){
	        	return;
	        } else {
	        	System.out.println("Network did not converge on correct solution. Assuming it fell into a local minimum instead of global minimum");
	        }
		}
		
		fail("Neural network did not converge on the correct solution after 5 attempts. Test failed");
	}
	
	public NeuralNetwork doTestInner(NeuralNetwork network, double inputs[][], double idealOutputs[][], int numIterations){
		for(int i = 0; i < numIterations; ++i){
        	for(int k = 0; k < inputs.length; ++k){
        		double outputs[] = network.goThroughNetwork(inputs[k], true, idealOutputs[k]);
    	        
    	        assertEquals(outputs.length, idealOutputs[k].length);
        	}
        }
		
		return network;
	}
	
	public boolean CheckNetworkPassed(NeuralNetwork network, double inputs[][], double idealOutputs[][]){
		for(int k = 0; k < inputs.length; ++k){
    		double outputs[] = network.goThroughNetwork(inputs[k], false, null);
	        
	        assertEquals(outputs.length, idealOutputs[k].length);
	        
	        System.out.println("Round " + (k + 1) + ": " + outputs[0]);
	        
	        if(Math.abs(outputs[0] - idealOutputs[k][0]) >= 0.1){
	        	return false;
	        }
    	}
		
		return true;
	}
	
    @Test public void sanityTest() {
    	System.out.println("sanityTest beginning");
    	int numLayers[] = {2, 3, 3};
    	
    	NeuralNetwork.ACTIVATION_FUNC actFuncs[] = {
    			NeuralNetwork.ACTIVATION_FUNC.NONE,
    			NeuralNetwork.ACTIVATION_FUNC.RELU,
    			NeuralNetwork.ACTIVATION_FUNC.SOFTMAX
    	};
    	
        NeuralNetwork network = new NeuralNetwork(numLayers, 1, 0.1, actFuncs, null);
        
        double inputs[] = {1, 1};
        
        double outputs[] = network.goThroughNetwork(inputs, false, null);
        
        assertEquals(outputs.length, 3);
        
        System.out.println(outputs[0] + ", " + outputs[1] + ", " + outputs[2]);
        
        assertTrue(Math.abs((outputs[0] + outputs[1] + outputs[2]) - 1) <= 0.01);
    }
    
    @Test public void sanityTestHugeInputs() {
    	System.out.println("sanityTestHugeInputs beginning");
    	int numLayers[] = {2, 3, 3};
    	
    	NeuralNetwork.ACTIVATION_FUNC actFuncs[] = {
    			NeuralNetwork.ACTIVATION_FUNC.NONE,
    			NeuralNetwork.ACTIVATION_FUNC.RELU,
    			NeuralNetwork.ACTIVATION_FUNC.SOFTMAX
    	};
    	
        NeuralNetwork network = new NeuralNetwork(numLayers, 1, 0.1, actFuncs, null);
        
        double inputs[] = {65781, 122832};
        
        double outputs[] = network.goThroughNetwork(inputs, false, null);
        
        assertEquals(outputs.length, 3);
        
        System.out.println(outputs[0] + ", " + outputs[1] + ", " + outputs[2]);
        
        assertTrue(Math.abs((outputs[0] + outputs[1] + outputs[2]) - 1) <= 0.01);
    }
}
