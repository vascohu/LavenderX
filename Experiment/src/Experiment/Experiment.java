package Experiment;

import org.apache.commons.math3.linear.RealVector;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.*;

/* Generate the experiment instance.
 * Created by Zehong on 3/14/2017 0014.
 */
public class Experiment {


    private static List<Double> Exp1(Market_Simulator simulator, int T) throws InterruptedException {
        System.out.println("----------------EXPERIMENT-----------------");
        Active_Mechanism mechanism = new Active_Mechanism(simulator);
        Obj_Function obj = new Confidence_Obj();
        Random_RL rl_module = new Random_RL(0, mechanism.getLabelMat());
        MJ_Model prob_model = new MJ_Model(simulator.getTask_Num(), simulator.getWorker_Num(), simulator.getClass_Num());
        mechanism.Mechanism_SetUp(obj, rl_module, prob_model);
        Date t1 = new Date();
        mechanism.Run(T);
        Date t2 = new Date();
        System.out.println("----------------Finished!-----------------");
        System.out.println("Objective Function: " + mechanism.getObjValue());
        System.out.print("Accuracy: ");
        for(double acc:mechanism.accuracy_record)
        {
            System.out.print(acc+",");
        }
        System.out.println();
        System.out.println("Time: "+ (t2.getTime()-t1.getTime()));
        return(mechanism.accuracy_record);
    }

    private static List<Double> Exp2(Market_Simulator simulator, int T) throws InterruptedException {
        System.out.println("----------------EXPERIMENT-----------------");
        Active_Mechanism mechanism = new Active_Mechanism(simulator);
        Obj_Function obj = new Confidence_Obj();
        OptGrad_RL rl_module = new OptGrad_RL(0, mechanism.getLabelMat());
        BU_Model prob_model = new BU_Model(simulator.getTask_Num(), simulator.getWorker_Num(), simulator.getClass_Num());
        mechanism.Mechanism_SetUp(obj, rl_module, prob_model);
        Date t1 = new Date();
        mechanism.Run(T);
        Date t2 = new Date();
        System.out.println("----------------Finished!-----------------");
        System.out.println("Objective Function: " + mechanism.getObjValue());
        System.out.print("Accuracy: ");
        for(double acc:mechanism.accuracy_record)
        {
            System.out.print(acc+",");
        }
        System.out.println();
        System.out.println("Time: "+ (t2.getTime()-t1.getTime()));
        prob_model.PrintModel();
        return(mechanism.accuracy_record);
    }


    private static List<Double> Exp3(Market_Simulator simulator, int T) throws InterruptedException {
        System.out.println("----------------EXPERIMENT-----------------");
        Active_Mechanism mechanism = new Active_Mechanism(simulator);
        Obj_Function obj = new Confidence_Obj();
        RL_Decision rl_module = new EpsGrad_RL(0, mechanism.getLabelMat());
        EM_Model prob_model = new EM_Model(simulator.getTask_Num(), simulator.getWorker_Num(), simulator.getClass_Num());
        mechanism.Mechanism_SetUp(obj, rl_module, prob_model);
        Date t1 = new Date();
        mechanism.Run(T);
        Date t2 = new Date();
        System.out.println("----------------Finished!-----------------");
        System.out.println("Objective Function: " + mechanism.getObjValue());
        System.out.print("Accuracy: ");
        for(double acc:mechanism.accuracy_record)
        {
            System.out.print(acc+",");
        }
        System.out.println();
        System.out.println("Time: "+ (t2.getTime()-t1.getTime()));
        prob_model.PrintModel();
        return(mechanism.accuracy_record);
    }



    private static List<Double> Exp4(Market_Simulator simulator, int T) throws InterruptedException {
        System.out.println("----------------EXPERIMENT-----------------");
        Active_Mechanism mechanism = new Active_Mechanism(simulator);
        Obj_Function obj = new Confidence_Obj();
        RL_Decision rl_module = new OptGrad_RL(0, mechanism.getLabelMat());
        EM_Model prob_model = new EM_Model(simulator.getTask_Num(), simulator.getWorker_Num(), simulator.getClass_Num());
        mechanism.Mechanism_SetUp(obj, rl_module, prob_model);
        Date t1 = new Date();
        mechanism.Run(T);
        Date t2 = new Date();
        System.out.println("----------------Finished!-----------------");
        System.out.println("Objective Function: " + mechanism.getObjValue());
        System.out.print("Accuracy: ");
        for(double acc:mechanism.accuracy_record)
        {
            System.out.print(acc+",");
        }
        System.out.println();
        System.out.println("Time: "+ (t2.getTime()-t1.getTime()));
        return(mechanism.accuracy_record);
    }

    private static List<Double> Exp5(Market_Simulator simulator, int T) throws InterruptedException {
        System.out.println("----------------EXPERIMENT-----------------");
        Active_Mechanism mechanism = new Active_Mechanism(simulator);
        Obj_Function obj = new Confidence_Obj();
        RL_Decision rl_module = new Random_RL(0, mechanism.getLabelMat());
        EM_Model prob_model = new EM_Model(simulator.getTask_Num(), simulator.getWorker_Num(), simulator.getClass_Num());
        mechanism.Mechanism_SetUp(obj, rl_module, prob_model);
        Date t1 = new Date();
        mechanism.Run(T);
        Date t2 = new Date();
        System.out.println("----------------Finished!-----------------");
        System.out.println("Objective Function: " + mechanism.getObjValue());
        System.out.print("Accuracy: ");
        for(double acc:mechanism.accuracy_record)
        {
            System.out.print(acc+",");
        }
        System.out.println();
        System.out.println("Time: "+ (t2.getTime()-t1.getTime()));
        return(mechanism.accuracy_record);
    }

    public static void test(String[] args) throws InterruptedException {
        Market_Simulator simulator = new Market_Simulator(50, 10);
        int T = 400;//Integer.parseInt(args[0]);
        try(FileWriter fw = new FileWriter("result.txt", true);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter out = new PrintWriter(bw))
        {
            for(double acc: Exp1(simulator, T))
            {
                out.print(acc+",\t");
            }
            for(double acc: Exp2(simulator, T))
            {
                out.print(acc+",\t");
            }
            for(double acc: Exp3(simulator, T))
            {
                out.print(acc+",\t");
            }
            for(double acc: Exp4(simulator, T))
            {
                out.print(acc+",\t");
            }
            for(double acc: Exp5(simulator, T))
            {
                out.print(acc+",\t");
            }
            out.print("\n");
        } catch (IOException e) {
            System.out.println("IO Error: "+e.getMessage());
        }
    }

    public static void main(String[] args) throws InterruptedException {
        Market_Simulator simulator = new Market_Simulator(50, 10);
        Exp2(simulator, 50);
        Exp3(simulator, 50);
        System.out.println(simulator.getReliability_mat().toString());
//        for(int i=0; i<100; ++i)
//        {
//            System.out.println(">>>>>>>>>>>>>>>>>> "+i);
//            test(args);
//        }
    }
}