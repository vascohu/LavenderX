package Experiment;

import org.apache.commons.math3.linear.RealVector;

import static java.lang.Math.log;

/* The Objective Function calculates the desired accuracy metric.
 * Created by Zehong on 3/21/2017 0021.
 */
interface Obj_Function {
    double getObjValue(Prob_Model model);
}

/* The confidence objective function calculates the confidence sum[Pr(l^{*}_{i})]
 */
class Confidence_Obj implements Obj_Function {
    @Override
    public double getObjValue(Prob_Model model) {
        int task_num = model.getTask_Num();
        double confidence = 0;
        for(int i=0; i<task_num; ++i)
        {
            RealVector ProbVec = model.getTaskLabelProb(i, true);
            confidence += ProbVec.getMaxValue();
        }
        return confidence;
    }
}

/* The margin objective function calculates the margin sum[Pr(l^{*}_{i})-Pr(l^{*-1}_{i})]
 */
class Margin_Obj implements Obj_Function {
    @Override
    public double getObjValue(Prob_Model model) {
        int task_num = model.getTask_Num();
        double margin = 0;
        for(int i=0; i<task_num; ++i)
        {
            RealVector ProbVec = model.getTaskLabelProb(i, true);
            double max_confidence = ProbVec.getMaxValue();
            int index_of_max_conf = ProbVec.getMaxIndex();
            ProbVec.setEntry(index_of_max_conf, 0);
            double second_max_confidence = ProbVec.getMaxValue();
            margin += max_confidence - second_max_confidence;
        }
        return margin;
    }
}


/* The entropy objective function calculates the posterior entropy sum[Pr(l_{ik})*log(Pr(l_{ik}))]
 */
class Entropy_Obj implements Obj_Function {
    @Override
    public double getObjValue(Prob_Model model) {
        int task_num = model.getTask_Num();
        double entropy = 0;
        for(int i=0; i<task_num; ++i)
        {
            RealVector ProbVec = model.getTaskLabelProb(i, true);
            for(int k=0; k<ProbVec.getDimension(); ++k)
            {
                entropy += ProbVec.getEntry(k)*log(ProbVec.getEntry(k))-0.5*log(0.5);
            }
        }
        return entropy;
    }
}