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
        
        double outputs[] = network.goThroughNetwork(inputs);
        
        assertEquals(outputs.length, 2);
        
        System.out.println(outputs[0] + ", " + outputs[1]);
    }
    
    @Test public void trainingSimple() {
    	int numLayers[] = {2, 3, 2};
        XorNetwork network = new XorNetwork(numLayers);
        
        double inputs[] = {1, 1};
        
        double outputs[] = network.goThroughNetwork(inputs);
        
        assertEquals(outputs.length, 2);
        
        System.out.println(outputs[0] + ", " + outputs[1]);
    }
}
