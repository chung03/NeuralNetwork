import org.junit.Test;
import static org.junit.Assert.*;

/*
 * This Java source file was auto generated by running 'gradle init --type java-library'
 * by 'colin_000' at '09/03/18 8:44 PM' with Gradle 2.14.1
 *
 * @author colin_000, @date 09/03/18 8:44 PM
 */
public class XorTest {
    @Test public void sanityTest() {
    	int numLayers[] = {2, 3, 2};
        XorNetwork network = new XorNetwork(numLayers);
        
        double inputs[] = {1, 1};
        
        double outputs[] = network.goThroughNetwork(inputs, false, null);
        
        assertEquals(outputs.length, 2);
        assertTrue(outputs[0] >= 0 && outputs[0] <= 1);
        assertTrue(outputs[1] >= 0 && outputs[1] <= 1);
        
        System.out.println(outputs[0] + ", " + outputs[1]);
    }
    
    @Test public void trainingSimple() {
    	int numLayers[] = {2, 2, 1};
        XorNetwork network = new XorNetwork(numLayers);
        
        double inputs[][] = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
        double idealOutputs[][] = {{0}, {1}, {1}, {0}};
        
        for(int i = 0; i < 100000; ++i){
        	for(int k = 0; k < 4; ++k){
        		double outputs[] = network.goThroughNetwork(inputs[k], true, idealOutputs[k]);
    	        
    	        assertEquals(outputs.length, 1);
    	        assertTrue(outputs[0] >= 0 && outputs[0] <= 1);
    	        
    	        // System.out.println("Round " + (k + 1) + " of set " + (i + 1) + ": " + outputs[0]);
        	}
        }
        
        for(int k = 0; k < 4; ++k){
    		double outputs[] = network.goThroughNetwork(inputs[k], false, null);
	        
	        assertEquals(outputs.length, 1);
	        assertTrue(outputs[0] >= 0 && outputs[0] <= 1);
	        
	        System.out.println("Round " + (k + 1) + ": " + outputs[0]);
    	}
        
        for(int i = 0; i < 100000; ++i){
        	for(int k = 0; k < 4; ++k){
        		double outputs[] = network.goThroughNetwork(inputs[k], true, idealOutputs[k]);
    	        
    	        assertEquals(outputs.length, 1);
    	        assertTrue(outputs[0] >= 0 && outputs[0] <= 1);
    	        
    	        // System.out.println("Round " + (k + 1) + " of set " + (i + 1) + ": " + outputs[0]);
        	}
        }
        
        for(int k = 0; k < 4; ++k){
    		double outputs[] = network.goThroughNetwork(inputs[k], false, null);
	        
	        assertEquals(outputs.length, 1);
	        assertTrue(outputs[0] >= 0 && outputs[0] <= 1);
	        
	        System.out.println("Round " + (k + 1) + ": " + outputs[0]);
    	}
    }
}
