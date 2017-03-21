package Experiment;

/* Generate the experiment instance.
 * Created by Zehong on 3/14/2017 0014.
 */
public class Experiment {

    public static void main(String[] args) throws InterruptedException {
        Market_Simulator simulator = new Market_Simulator(20, 10);
        Active_Mechanism mechanism = new Active_Mechanism(simulator);
        mechanism.Run(30);
        System.out.println("----------------Finished!-----------------");
        System.out.println("Objective Function: " + mechanism.getFinalObjValue());
        //State result = mechanism.getFinalLabelMat();
        //result.Sparse_Print();
    }
}