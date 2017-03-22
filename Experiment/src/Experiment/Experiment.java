package Experiment;

import org.apache.commons.math3.linear.RealVector;

import java.lang.reflect.Array;
import java.util.*;

/* Generate the experiment instance.
 * Created by Zehong on 3/14/2017 0014.
 */
public class Experiment {

    private static double Calculate_Accuracy(List<Integer> learned_labels, RealVector true_labels)
    {
        int number_of_correct = 0;
        Iterator<Integer> it1 = learned_labels.iterator();
        double[] true_label_array = true_labels.toArray();
        int i = 0;
        while(it1.hasNext())
        {
            if(Math.abs(it1.next()-true_label_array[i])<0.5)
            {
                number_of_correct++;
            }
            ++i;
        }

        return (double)number_of_correct/learned_labels.size();
    }

    public static void main(String[] args) throws InterruptedException {
        Market_Simulator simulator = new Market_Simulator(20, 10);
        Active_Mechanism mechanism = new Active_Mechanism(simulator);
        Obj_Function obj = new Confidence_Obj();
        RL_Decision rl_module = new EpsGrad_RL(0, mechanism.getLabelMat());
        Prob_Model prob_model = new MJ_Model(simulator);
        mechanism.Mech_SetUp(obj, rl_module, prob_model);



        mechanism.Run(30);
        System.out.println("----------------Finished!-----------------");
        System.out.println("Objective Function: " + mechanism.getObjValue());
        System.out.println("Accuracy: " + Calculate_Accuracy(mechanism.getTaskLabel(), simulator.getTrue_label()));


        Active_Mechanism mechanism2 = new Active_Mechanism(simulator);
        Obj_Function obj2 = new Confidence_Obj();
        RL_Decision rl_module2 = new OptGrad_RL(0, mechanism2.getLabelMat());
        Prob_Model prob_model2 = new MJ_Model(simulator);
        mechanism2.Mech_SetUp(obj2, rl_module2, prob_model2);

        mechanism2.Run(30);
        System.out.println("----------------Finished!-----------------");
        System.out.println("Objective Function: " + mechanism2.getObjValue());
        System.out.println("Accuracy: " + Calculate_Accuracy(mechanism2.getTaskLabel(), simulator.getTrue_label()));
    }
}